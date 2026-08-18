"""Phase 11R-1 Degradation-Aware Restoration Script
Warm-starts from the best Phase 11 checkpoint.
Fine-tunes the model to handle randomized permutations and combinations of KLA degradations:
1. Gaussian noise (additive)
2. Multiplicative speckle noise
3. 2x downsampling
Uses a composite loss combining reconstruction, structural, perceptual, and teacher-distillation losses.
"""

import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import lpips

from functools import partial
print = partial(print, flush=True)

from dataset import KLADataset
from echo_model import BaselineECHOModel
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase10 import calculate_psnr
from utils import set_seed

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _stable_sample_index(sample_key, seed):
    if isinstance(sample_key, str):
        return hash((sample_key, seed)) & 0x7FFFFFFF
    return int(sample_key)

def _sample_uniform(generator, low, high):
    return low + (high - low) * torch.rand((), generator=generator).item()

def apply_kla_degradation(original_lr, target, sample_keys, epoch, cfg, device):
    """
    Apply randomized combination and ordering of official degradations:
    - Gaussian noise (additive)
    - Speckle noise (multiplicative)
    - Downsampling (2x)
    """
    deg = cfg["degradation"]
    seed = cfg["training"]["seed"]
    
    p_clean = deg.get("preserve_original_probability", 0.60)
    p_g = deg.get("p_gaussian", 0.05)
    p_s = deg.get("p_speckle", 0.10)
    p_d = deg.get("p_downsample", 0.05)
    p_gs = deg.get("p_gaussian_speckle", 0.05)
    p_gd = deg.get("p_gaussian_downsample", 0.05)
    p_sd = deg.get("p_speckle_downsample", 0.05)
    p_all = deg.get("p_all", 0.05)
    
    probs = [p_clean, p_g, p_s, p_d, p_gs, p_gd, p_sd, p_all]
    total_prob = sum(probs)
    probs = [p / total_prob for p in probs]
    
    noise_sigma_min = deg.get("noise_sigma_min", 0.01)
    noise_sigma_max = deg.get("noise_sigma_max", 0.06)
    speckle_sigma_min = deg.get("speckle_sigma_min", 0.01)
    speckle_sigma_max = deg.get("speckle_sigma_max", 0.08)
    
    out = original_lr.clone()
    stats = {
        "clean": 0,
        "gaussian": 0,
        "speckle": 0,
        "downsample": 0,
        "gaussian_speckle": 0,
        "gaussian_downsample": 0,
        "speckle_downsample": 0,
        "all_three": 0
    }
    
    for i, sample_key in enumerate(sample_keys):
        sample_index = _stable_sample_index(sample_key, seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 7919 + sample_index * 1009)
        
        cat = torch.multinomial(torch.tensor(probs), 1, generator=generator).item()
        
        # Draw noise parameter values
        g_sigma = _sample_uniform(generator, noise_sigma_min, noise_sigma_max)
        s_sigma = _sample_uniform(generator, speckle_sigma_min, speckle_sigma_max)
        
        img = original_lr[i : i + 1].to(device)
        tgt = target[i : i + 1].to(device)
        
        if cat == 0:
            stats["clean"] += 1
            out[i : i + 1] = img
            continue
            
        elif cat == 1:
            stats["gaussian"] += 1
            # Add Gaussian noise to original_lr (128x128)
            img = img + torch.randn(img.shape, generator=generator).to(device) * g_sigma
            out[i : i + 1] = img
            
        elif cat == 2:
            stats["speckle"] += 1
            # Add Speckle noise to original_lr (128x128)
            img = img * (1.0 + torch.randn(img.shape, generator=generator).to(device) * s_sigma)
            out[i : i + 1] = img
            
        elif cat == 3:
            stats["downsample"] += 1
            # Downsample target (256x256 -> 128x128)
            img = F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
            out[i : i + 1] = img
            
        elif cat == 4:
            stats["gaussian_speckle"] += 1
            # Gaussian + Speckle on original_lr (128x128)
            order = torch.rand((), generator=generator).item() < 0.5
            if order:
                img = img + torch.randn(img.shape, generator=generator).to(device) * g_sigma
                img = img * (1.0 + torch.randn(img.shape, generator=generator).to(device) * s_sigma)
            else:
                img = img * (1.0 + torch.randn(img.shape, generator=generator).to(device) * s_sigma)
                img = img + torch.randn(img.shape, generator=generator).to(device) * g_sigma
            out[i : i + 1] = img
            
        elif cat == 5:
            stats["gaussian_downsample"] += 1
            # Gaussian + Downsample on target (256x256)
            order = torch.rand((), generator=generator).item() < 0.5
            if order:
                tgt_noise = tgt + torch.randn(tgt.shape, generator=generator).to(device) * g_sigma
                img = F.interpolate(tgt_noise, scale_factor=0.5, mode="bicubic", align_corners=False)
            else:
                img = F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
                img = img + torch.randn(img.shape, generator=generator).to(device) * g_sigma
            out[i : i + 1] = img
            
        elif cat == 6:
            stats["speckle_downsample"] += 1
            # Speckle + Downsample on target (256x256)
            order = torch.rand((), generator=generator).item() < 0.5
            if order:
                tgt_noise = tgt * (1.0 + torch.randn(tgt.shape, generator=generator).to(device) * s_sigma)
                img = F.interpolate(tgt_noise, scale_factor=0.5, mode="bicubic", align_corners=False)
            else:
                img = F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
                img = img * (1.0 + torch.randn(img.shape, generator=generator).to(device) * s_sigma)
            out[i : i + 1] = img
            
        elif cat == 7:
            stats["all_three"] += 1
            perm = int(torch.rand((), generator=generator).item() * 6)
            if perm == 0:
                # G(256) -> S(256) -> D
                x = tgt + torch.randn(tgt.shape, generator=generator).to(device) * g_sigma
                x = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sigma)
                img = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
            elif perm == 1:
                # S(256) -> G(256) -> D
                x = tgt * (1.0 + torch.randn(tgt.shape, generator=generator).to(device) * s_sigma)
                x = x + torch.randn(x.shape, generator=generator).to(device) * g_sigma
                img = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
            elif perm == 2:
                # G(256) -> D -> S(128)
                x = tgt + torch.randn(tgt.shape, generator=generator).to(device) * g_sigma
                x = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
                img = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sigma)
            elif perm == 3:
                # S(256) -> D -> G(128)
                x = tgt * (1.0 + torch.randn(tgt.shape, generator=generator).to(device) * s_sigma)
                x = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
                img = x + torch.randn(x.shape, generator=generator).to(device) * g_sigma
            elif perm == 4:
                # D -> G(128) -> S(128)
                x = F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
                x = x + torch.randn(x.shape, generator=generator).to(device) * g_sigma
                img = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sigma)
            else:
                # D -> S(128) -> G(128)
                x = F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
                x = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sigma)
                img = x + torch.randn(x.shape, generator=generator).to(device) * g_sigma
            out[i : i + 1] = img
            
    return out, stats

def compute_total_loss(pred, target, pred_hf_branch, tgt_hf, p4_hr, l_weights, sobel_filter, lpips_model, decomp_helper):
    pred_lf, pred_mf, pred_hf = decomp_helper(pred)
    l_pixel = F.l1_loss(pred, target)
    l_ssim = 1.0 - ssim_pytorch(pred, target)
    l_fft = F.l1_loss(torch.fft.rfft2(pred, norm="ortho"), torch.fft.rfft2(target, norm="ortho"))
    l_hf_direct = F.l1_loss(pred_hf, tgt_hf)
    l_hf_comp = F.l1_loss(pred_hf_branch, tgt_hf)
    l_edge = F.l1_loss(sobel_filter(pred), sobel_filter(target))
    l_lpips = ssim_lpips_differentiable(pred, target, lpips_model)
    l_anchor = F.l1_loss(pred, p4_hr)
    total = (
        l_weights.get("pixel_l1", 2.0) * l_pixel
        + l_weights.get("ssim", 0.3) * l_ssim
        + l_weights.get("hf_direct", 0.15) * l_hf_direct
        + l_weights.get("sobel_edge", 0.15) * l_edge
        + l_weights.get("hf_component", 0.05) * l_hf_comp
        + l_weights.get("freq_fft", 0.05) * l_fft
        + l_weights.get("lpips", 0.010) * l_lpips
        + l_weights.get("p4_anchor", 0.02) * l_anchor
    )
    return total

def main():
    parser = argparse.ArgumentParser(description="Phase 11R-1 Degradation-Aware Restoration Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--sanity-only", action="store_true", help="Run sanity checks and overfit test then exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    
    # 1. Environment Safety Verification
    print("Check [1] CUDA availability...", end=" ")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! Phase 11R-1 training requires GPU execution.")
    device = torch.device("cuda")
    print("PASS")

    out_dir = cfg["output_dir"]
    checkpoints_dir = cfg["checkpoints_dir"]
    results_dir = cfg["results_dir"]
    evaluation_dir = cfg["evaluation_dir"]
    configs_dir = cfg["configs_dir"]

    for path in (checkpoints_dir, results_dir, evaluation_dir, configs_dir):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(configs_dir, os.path.basename(args.config)), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    l_weights = cfg["loss_weights"]
    lambda_robustness = l_weights.get("lambda_robustness", 0.5)

    print("=" * 60)
    print(f"PHASE 11R-1 DEGRADATION-AWARE RESTORATION — EXPERIMENT: {cfg['experiment_name'].upper()}")
    print(f"Device: {device}")
    print(f"Output Directory: {out_dir}")
    print(f"Learning Rate: {cfg['training']['lr']} | Epochs: {cfg['training']['epochs']}")
    print(f"Robustness Loss Weight (Distill): {lambda_robustness}")
    print("=" * 60)

    # 2. Checkpoint Safety Verification
    p9_ckpt_path = cfg["model"]["phase9_checkpoint"]
    print(f"Check [2] Phase 9 checkpoint exists at '{p9_ckpt_path}'...", end=" ")
    if not os.path.exists(p9_ckpt_path):
        raise FileNotFoundError(f"Phase 9 checkpoint not found at: {p9_ckpt_path}")
    print("PASS")

    p4_ckpt_path = cfg["model"]["phase4_checkpoint"]
    print(f"Check [3] Phase 4 checkpoint exists at '{p4_ckpt_path}'...", end=" ")
    if not os.path.exists(p4_ckpt_path):
        raise FileNotFoundError(f"Phase 4 checkpoint not found at: {p4_ckpt_path}")
    print("PASS")

    p11_ckpt_path = cfg["model"]["phase11_checkpoint"]
    print(f"Check [3.5] Phase 11 starting checkpoint exists at '{p11_ckpt_path}'...", end=" ")
    if not os.path.exists(p11_ckpt_path):
        raise FileNotFoundError(f"Phase 11 checkpoint not found at: {p11_ckpt_path}")
    print("PASS")

    # 3. Dataset Safety Verification
    dataset_root = cfg["dataset"]["dataset_root"]
    train_csv = cfg["dataset"]["train_csv"]
    val_csv = cfg["dataset"]["val_csv"]
    print(f"Check [4] Dataset paths valid...", end=" ")
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("CSV splits not found.")
    print("PASS")

    train_dataset = KLADataset(dataset_root, split="train", csv_path=train_csv)
    val_dataset = KLADataset(dataset_root, split="val", csv_path=val_csv)
    print(f"Check [5] Train dataset size: {len(train_dataset)} | Val dataset size: {len(val_dataset)}...", end=" ")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Loaded dataset is empty!")
    print("PASS")

    # Dataloaders
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg["training"]["seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 4. Model Loading and Dimensions
    print("Loading models...", end=" ")
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters():
        p.requires_grad = False

    teacher_9 = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)
    t9_chk = torch.load(p9_ckpt_path, map_location=device, weights_only=False)
    teacher_9.load_state_dict(t9_chk["model_state_dict"])
    teacher_9.eval()
    for p in teacher_9.parameters():
        p.requires_grad = False

    student = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)
    
    # Warm-start student from the best Phase 11 checkpoint
    p11_chk = torch.load(p11_ckpt_path, map_location=device, weights_only=False)
    student.load_state_dict(p11_chk["model_state_dict"])
    print("PASS")

    # Helpers
    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad = False
    decomp_helper = FrequencyDecompositionModule(
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"],
    ).to(device)

    # Forward check
    sample_batch = next(iter(train_loader))
    s_in = sample_batch["input"].to(device)
    s_tgt = sample_batch["target"].to(device)
    s_keys = list(sample_batch["input_path"])

    print("Check [6] Input shape [B, 1, 128, 128] & Target shape [B, 1, 256, 256]...", end=" ")
    if s_in.shape[1:] != (1, 128, 128) or s_tgt.shape[1:] != (1, 256, 256):
        raise ValueError(f"Incorrect tensor shape: input {s_in.shape}, target {s_tgt.shape}")
    print("PASS")

    # Apply degradation check
    print("Check [7] Degradation pipeline validation...", end=" ")
    deg_in, _ = apply_kla_degradation(s_in, s_tgt, s_keys, epoch=0, cfg=cfg, device=device)
    if deg_in.shape != s_in.shape:
        raise ValueError(f"Degraded output shape mismatch: {deg_in.shape} vs {s_in.shape}")
    if not torch.isfinite(deg_in).all():
        raise ValueError("Degraded inputs contain NaNs or Infs!")
    print("PASS")

    # Student forward pass
    print("Check [8] Student forward pass works and output dimensions match target...", end=" ")
    s_lr_up = F.interpolate(deg_in, scale_factor=2, mode="bicubic", align_corners=False)
    with torch.no_grad():
        s_p4_raw, _ = model_p4(deg_in)
        s_p4_hr = torch.clamp(s_p4_raw, 0.0, 1.0)
    s_pred_hr, _, _, _, _ = student(s_lr_up, s_p4_hr)
    # Clamp final output and verify
    s_pred_hr = torch.clamp(s_pred_hr, 0.0, 1.0)
    if s_pred_hr.shape != s_tgt.shape:
        raise ValueError(f"Student output shape mismatch: {s_pred_hr.shape} vs {s_tgt.shape}")
    if not torch.isfinite(s_pred_hr).all():
        raise ValueError("Student predictions contain NaNs or Infs!")
    print("PASS")

    # 5. Overfit test
    print("Check [9] Running 2-sample overfit test...", end=" ")
    overfit_subset = Subset(train_dataset, [0, 1])
    overfit_loader = DataLoader(overfit_subset, batch_size=2, shuffle=False)
    overfit_batch = next(iter(overfit_loader))
    o_in = overfit_batch["input"].to(device)
    o_tgt = overfit_batch["target"].to(device)
    o_keys = list(overfit_batch["input_path"])

    with torch.no_grad():
        o_p4_raw, _ = model_p4(o_in)
        o_p4_hr = torch.clamp(o_p4_raw, 0.0, 1.0)
        o_lr_up = F.interpolate(o_in, scale_factor=2, mode="bicubic", align_corners=False)
        _, _, o_tgt_hf = decomp_helper(o_tgt)

    o_opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    o_start_loss = 0.0
    o_end_loss = 0.0

    student.train()
    for step in range(30):
        o_opt.zero_grad()
        o_pred_hr, _, _, o_xhf, _ = student(o_lr_up, o_p4_hr)
        loss = compute_total_loss(
            o_pred_hr, o_tgt, o_xhf, o_tgt_hf, o_p4_hr, l_weights, sobel_filter, lpips_model, decomp_helper
        )
        loss.backward()
        o_opt.step()
        
        if step == 0:
            o_start_loss = loss.item()
        if step == 29:
            o_end_loss = loss.item()

    o_reduction = (o_start_loss - o_end_loss) / (o_start_loss + 1e-8) * 100.0
    if o_reduction < 1.0:
        raise ValueError(f"Overfit test failed to reduce loss! Start: {o_start_loss:.4f}, End: {o_end_loss:.4f}")
    print(f"PASS (Loss reduction: {o_start_loss:.6f} -> {o_end_loss:.6f}, {o_reduction:.2f}%)")

    # Restore warm start state
    student.load_state_dict(p11_chk["model_state_dict"])
    print("Sanity checks complete. Verification PASS.")

    if args.sanity_only:
        print("Sanity check mode complete. Exiting.")
        return

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["training"]["scheduler_gamma"])

    epochs = cfg["training"]["epochs"]
    best_score = -999.0
    history = []
    start_time = time.time()

    print("\n" + "=" * 50)
    print(f"STARTING {epochs} EPOCHS DEGRADATION-AWARE FINE-TUNING RUN")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        student.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_dist_sum = 0.0
        num_train_batches = 0
        epoch_aug_stats = {
            "clean": 0,
            "gaussian": 0,
            "speckle": 0,
            "downsample": 0,
            "gaussian_speckle": 0,
            "gaussian_downsample": 0,
            "speckle_downsample": 0,
            "all_three": 0
        }

        for batch in train_loader:
            b_orig = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            sample_keys = list(batch["input_path"])

            # Apply KLA degradation
            b_in, aug_stats = apply_kla_degradation(
                b_orig, b_tgt, sample_keys, epoch=epoch, cfg=cfg, device=device
            )
            for key in epoch_aug_stats:
                epoch_aug_stats[key] += aug_stats[key]

            # Forward passes
            with torch.no_grad():
                b_p4_raw, _ = model_p4(b_in)
                b_p4_hr = torch.clamp(b_p4_raw, 0.0, 1.0)
                b_lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
                _, _, b_tgt_hf = decomp_helper(b_tgt)
                
                # Phase 9 teacher clean reference predictions (to encourage fidelity preservation)
                b_orig_lr_up = F.interpolate(b_orig, scale_factor=2, mode="bicubic", align_corners=False)
                b_orig_p4_raw, _ = model_p4(b_orig)
                b_orig_p4_hr = torch.clamp(b_orig_p4_raw, 0.0, 1.0)
                b_t9_clean_hr, _, _, _, _ = teacher_9(b_orig_lr_up, b_orig_p4_hr)

            optimizer.zero_grad()
            b_pred_hr, _, _, b_xhf, _ = student(b_lr_up, b_p4_hr)

            # Compute primary reconstruction, structural, and perceptual loss
            loss_recon = compute_total_loss(
                b_pred_hr, b_tgt, b_xhf, b_tgt_hf, b_p4_hr, l_weights, sobel_filter, lpips_model, decomp_helper
            )

            # Compute Degradation-Robustness Loss (distillation against teacher clean predictions)
            loss_dist = F.l1_loss(b_pred_hr, b_t9_clean_hr)

            # Total loss
            loss = loss_recon + lambda_robustness * loss_dist

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += loss_recon.item()
            train_dist_sum += loss_dist.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / num_train_batches
        avg_recon_loss = train_recon_sum / num_train_batches
        avg_dist_loss = train_dist_sum / num_train_batches

        total_aug = sum(epoch_aug_stats.values()) or 1
        aug_pct = {k: 100.0 * v / total_aug for k, v in epoch_aug_stats.items()}

        # Print training statistics & observed distribution
        dist_str = ", ".join([f"{k}: {v:.1f}%" for k, v in aug_pct.items()])
        print(f"Epoch {epoch:02d} Augmentation Proportions:")
        print(f"  {dist_str}")

        # Validation loop (original clean validation set)
        student.eval()
        val_psnr_list, val_ssim_list, val_lpips_list = [], [], []
        val_mae_list, val_hf_err_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                v_in = batch["input"].to(device)
                v_tgt = batch["target"].to(device)
                v_p4_raw, _ = model_p4(v_in)
                v_p4_hr = torch.clamp(v_p4_raw, 0.0, 1.0)
                v_lr_up = F.interpolate(v_in, scale_factor=2, mode="bicubic", align_corners=False)
                v_pred_hr, _, _, _, _ = student(v_lr_up, v_p4_hr)
                
                # Enforce valid output range [0, 1]
                v_pred_hr = torch.clamp(v_pred_hr, 0.0, 1.0)
                assert torch.isfinite(v_pred_hr).all(), "Validation prediction contains NaN or Inf!"

                _, _, v_tgt_hf = decomp_helper(v_tgt)
                _, _, v_pred_hf = decomp_helper(v_pred_hr)

                val_psnr_list.append(calculate_psnr(v_pred_hr, v_tgt))
                val_ssim_list.append(ssim_pytorch(v_pred_hr, v_tgt).item())
                val_lpips_list.append(ssim_lpips_differentiable(v_pred_hr, v_tgt, lpips_model).item())
                val_mae_list.append(F.l1_loss(v_pred_hr, v_tgt).item())
                val_hf_err_list.append(F.l1_loss(v_pred_hf, v_tgt_hf).item())

        m_psnr = float(np.mean(val_psnr_list))
        m_ssim = float(np.mean(val_ssim_list))
        m_lpips = float(np.mean(val_lpips_list))
        m_mae = float(np.mean(val_mae_list))
        m_hf_err = float(np.mean(val_hf_err_list))

        # Composite score using Phase 9 baseline references
        score = (m_psnr - 28.2105) * 1.0 + (m_ssim - 0.7687) * 50.0 + (0.2749 - m_lpips) * 10.0 - (m_hf_err - 0.007854) * 100.0

        epoch_rec = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_recon_loss": avg_recon_loss,
            "train_dist_loss": avg_dist_loss,
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_rec.update({f"aug_{k}_pct": v for k, v in aug_pct.items()})
        history.append(epoch_rec)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Total Loss: {avg_train_loss:.4f} | Recon: {avg_recon_loss:.4f} | Dist: {avg_dist_loss:.4f} | "
            f"Val PSNR: {m_psnr:.4f} | SSIM: {m_ssim:.4f} | LPIPS: {m_lpips:.4f} | MAE: {m_mae:.4f} | HF: {m_hf_err:.6f} | "
            f"Score: {score:+.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err,
        }
        
        # Save last checkpoint
        torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase11r1_last.pth"))
        
        # Save best checkpoint
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase11r1_best.pth"))
            print(f"  --> Saved new best checkpoint (Score: {best_score:+.4f})")

    elapsed = time.time() - start_time
    hist_df = pd.DataFrame(history)
    hist_csv_path = os.path.join(results_dir, "phase11r1_history.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"\nTraining finished in {elapsed / 60.0:.2f} mins.")
    print(f"Saved history CSV to: {hist_csv_path}")

if __name__ == "__main__":
    main()
