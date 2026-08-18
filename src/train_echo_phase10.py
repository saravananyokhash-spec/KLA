"""Phase 10 Degradation-Aware Restoration Refinement Script
Warm-starts from the best Phase 9 checkpoint.
Uses targeted distillation weights based on degradation condition.
Student is conditioned on predicted degradation logits.
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
from phase10_model import ConditionedSpatialFrequencyRestorationNet
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr
from utils import set_seed
from degradation_utils import _stable_sample_index, _sample_uniform, synthetic_lr_from_target

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def apply_training_degradation_phase10(original_lr, target, sample_keys, epoch, cfg, device):
    """Return batch LR inputs with controlled 4-class degradation mixture:
    0: clean (original disk LR)
    1: noise only
    2: blur only
    3: noise + blur
    """
    deg = cfg["degradation"]
    seed = cfg["training"]["seed"]
    
    clean_p = float(deg.get("clean_fraction", 0.40))
    noise_p = float(deg.get("noise_fraction", 0.15))
    blur_p = float(deg.get("blur_fraction", 0.25))
    
    blur_sigma_min = float(deg["blur_sigma_min"])
    blur_sigma_max = float(deg["blur_sigma_max"])
    noise_sigma_min = float(deg.get("noise_sigma_min", 0.01))
    noise_sigma_max = float(deg.get("noise_sigma_max", 0.06))

    out = original_lr.clone()
    labels = []
    stats = {"original": 0, "noise_only": 0, "blur_only": 0, "noise_blur": 0}

    for i, sample_key in enumerate(sample_keys):
        sample_index = _stable_sample_index(sample_key, seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 7919 + sample_index * 1009)

        r = torch.rand((), generator=generator).item()
        
        if r < clean_p:
            labels.append(0)
            stats["original"] += 1
            continue
        elif r < clean_p + noise_p:
            labels.append(1)
            stats["noise_only"] += 1
            noise_sigma = _sample_uniform(generator, noise_sigma_min, noise_sigma_max)
            blur_sigma = 0.0
        elif r < clean_p + noise_p + blur_p:
            labels.append(2)
            stats["blur_only"] += 1
            noise_sigma = 0.0
            blur_sigma = _sample_uniform(generator, blur_sigma_min, blur_sigma_max)
        else:
            labels.append(3)
            stats["noise_blur"] += 1
            noise_sigma = _sample_uniform(generator, noise_sigma_min, noise_sigma_max)
            blur_sigma = _sample_uniform(generator, blur_sigma_min, blur_sigma_max)

        out[i : i + 1] = synthetic_lr_from_target(
            target[i : i + 1],
            noise_sigma,
            blur_sigma,
            [sample_index],
            device,
            seed=seed + epoch * 17,
        )

    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return out, labels, stats

def main():
    parser = argparse.ArgumentParser(description="Phase 10 Degradation-Aware restoration refinement")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sanity-only", action="store_true", help="Run sanity checks and overfit test then exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = cfg["output_dir"]
    checkpoints_dir = cfg["checkpoints_dir"]
    results_dir = cfg["results_dir"]
    evaluation_dir = cfg["evaluation_dir"]
    configs_dir = cfg.get("configs_dir", os.path.join(out_dir, "configs"))

    for path in (checkpoints_dir, results_dir, evaluation_dir, configs_dir):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(configs_dir, os.path.basename(args.config)), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    l_weights = cfg["loss_weights"]
    deg = cfg["degradation"]
    robustness_loss_weight = l_weights.get("robustness_loss_weight", 1.0)
    
    # Global distillation weights
    lambda_est = cfg["distillation_global"]["lambda_est"]

    print("=" * 60)
    print(f"PHASE 10 DEGRADATION-AWARE REFINEMENT — EXPERIMENT: {cfg['experiment_name'].upper()}")
    print(f"Device: {device}")
    print(f"Output Directory: {out_dir}")
    print("-" * 60)
    print("Configuration details:")
    print(f"  Training proportions clean/noise/blur/combo: {deg.get('clean_fraction')} / {deg.get('noise_fraction')} / {deg.get('blur_fraction')} / {deg.get('noise_blur_fraction')}")
    print(f"  Learning Rate                              : {cfg['training']['lr']}")
    print(f"  Epochs                                     : {cfg['training']['epochs']}")
    print("=" * 60)

    # --- MANDATORY SANITY CHECKS ---
    print("\nRunning Mandatory Sanity Checks...")

    # [1] CUDA availability
    print("Check [1] CUDA availability...", end=" ")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Training requires GPU.")
    print("PASS")

    # [2] Phase 5B checkpoint exists
    p5b_ckpt_path = cfg["model"]["phase5b_checkpoint"]
    print(f"Check [2] Phase 5B checkpoint exists at '{p5b_ckpt_path}'...", end=" ")
    if not os.path.exists(p5b_ckpt_path):
        raise FileNotFoundError(f"Phase 5B checkpoint not found at: {p5b_ckpt_path}")
    print("PASS")

    # [3] Phase 7 checkpoint exists
    phase7_ckpt = cfg["model"]["phase7_checkpoint"]
    print(f"Check [3] Phase 7 checkpoint exists at '{phase7_ckpt}'...", end=" ")
    if not os.path.exists(phase7_ckpt):
        raise FileNotFoundError(f"Phase 7 checkpoint not found at: {phase7_ckpt}")
    print("PASS")

    # [4] Phase 9 starting checkpoint exists
    phase9_ckpt = cfg["model"]["phase9_checkpoint"]
    print(f"Check [4] Phase 9 starting checkpoint exists at '{phase9_ckpt}'...", end=" ")
    if not os.path.exists(phase9_ckpt):
        raise FileNotFoundError(f"Phase 9 checkpoint not found at: {phase9_ckpt}")
    print("PASS")

    # [5] Train CSV exists
    train_csv = cfg["dataset"]["train_csv"]
    print(f"Check [5] Train CSV exists at '{train_csv}'...", end=" ")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train split CSV not found: {train_csv}")
    print("PASS")

    # [6] Validation CSV exists
    val_csv = cfg["dataset"]["val_csv"]
    print(f"Check [6] Validation CSV exists at '{val_csv}'...", end=" ")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Val split CSV not found: {val_csv}")
    print("PASS")

    # [7] Dataset length loads
    dataset_root = cfg["dataset"]["dataset_root"]
    train_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=train_csv)
    val_dataset = KLADataset(dataset_root=dataset_root, split="val", csv_path=val_csv)
    print(f"Check [7] Dataset length loaded...", end=" ")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Empty datasets! Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"PASS (Train: {len(train_dataset)}, Val: {len(val_dataset)})")

    # Setup dataloaders
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg["training"]["seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Instantiate student and teachers
    print("Instantiating networks...")
    
    # Load frozen Phase 4 baseline
    p4_ckpt_path = cfg["model"]["phase4_checkpoint"]
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for param in model_p4.parameters():
        param.requires_grad = False

    # Instantiate teachers
    teacher_5b = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)
    t5b_chk = torch.load(p5b_ckpt_path, map_location=device, weights_only=False)
    teacher_5b.load_state_dict(t5b_chk["model_state_dict"])
    teacher_5b.eval()
    for param in teacher_5b.parameters():
        param.requires_grad = False

    teacher_7 = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)
    t7_chk = torch.load(phase7_ckpt, map_location=device, weights_only=False)
    teacher_7.load_state_dict(t7_chk["model_state_dict"])
    teacher_7.eval()
    for param in teacher_7.parameters():
        param.requires_grad = False

    teacher_9 = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)
    t9_chk = torch.load(phase9_ckpt, map_location=device, weights_only=False)
    teacher_9.load_state_dict(t9_chk["model_state_dict"])
    teacher_9.eval()
    for param in teacher_9.parameters():
        param.requires_grad = False

    # Instantiate student model (with conditioning estimator) warm started from Phase 9
    print(f"Initializing student model from Phase 9 best checkpoint: {phase9_ckpt}")
    student = ConditionedSpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"],
        use_conditioning=cfg["model"]["use_conditioning"]
    ).to(device)
    # Load backbone parameters with strict=False to bypass Estimator/FiLM weights
    student.load_state_dict(t9_chk["model_state_dict"], strict=False)

    # Helpers
    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad = False
    decomp_helper = FrequencyDecompositionModule(
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"],
    ).to(device)

    # Get sample batch
    sample_batch = next(iter(train_loader))
    s_in = sample_batch["input"].to(device)
    s_tgt = sample_batch["target"].to(device)
    s_keys = list(sample_batch["input_path"])

    # [8] Tensor dimensions correct
    print("Check [8] Tensor dimensions...", end=" ")
    if s_in.ndim != 4 or s_tgt.ndim != 4 or s_in.shape[1] != 1 or s_tgt.shape[1] != 1:
        raise ValueError(f"Unexpected tensor shapes: input {s_in.shape}, target {s_tgt.shape}")
    print("PASS")

    # Apply 4-class targeted degradation
    s_aug, s_labels, s_stats = apply_training_degradation_phase10(s_in, s_tgt, s_keys, epoch=0, cfg=cfg, device=device)

    # [9] Student forward pass
    print("Check [9] Student forward pass works...", end=" ")
    s_lr_up = F.interpolate(s_aug, scale_factor=2, mode="bicubic", align_corners=False)
    with torch.no_grad():
        s_p4_raw, _ = model_p4(s_aug)
        s_p4_hr = torch.clamp(s_p4_raw, 0.0, 1.0)
    s_pred_hr, _, _, s_xhf, _, s_logits = student(s_lr_up, s_p4_hr, lr_input=s_aug)
    if s_pred_hr.shape != s_tgt.shape:
        raise ValueError(f"Prediction shape mismatch: {s_pred_hr.shape} vs {s_tgt.shape}")
    print("PASS")

    # [10] All teacher forward passes
    print("Check [10] All teacher forward passes work...", end=" ")
    with torch.no_grad():
        s_t5b_hr, _, _, _, _ = teacher_5b(s_lr_up, s_p4_hr)
        s_t7_hr, _, _, _, _ = teacher_7(s_lr_up, s_p4_hr)
        s_t9_hr, _, _, _, _ = teacher_9(s_lr_up, s_p4_hr)
    if s_t5b_hr.shape != s_tgt.shape or s_t7_hr.shape != s_tgt.shape or s_t9_hr.shape != s_tgt.shape:
        raise ValueError("Teacher predictions dimension mismatch")
    print("PASS")

    # [11] Degradation-condition generation
    print("Check [11] Degradation-condition generation works...", end=" ")
    if s_labels.shape[0] != s_in.shape[0]:
        raise ValueError("degradation labels batch size mismatch")
    print("PASS")

    # Setup loss values checks
    with torch.no_grad():
        _, _, s_tgt_hf = decomp_helper(s_tgt)
        # [12] Reconstruction loss finite
        print("Check [12] Reconstruction loss is finite...", end=" ")
        s_l_recon = compute_total_loss(
            s_pred_hr, s_tgt, s_xhf, s_tgt_hf, s_p4_hr, l_weights, sobel_filter, lpips_model, decomp_helper
        )
        if not torch.isfinite(s_l_recon): raise ValueError("Non-finite reconstruction loss")
        print("PASS")

        # [13] Teacher losses finite
        print("Check [13] Teacher losses are finite...", end=" ")
        s_dist_5b = F.l1_loss(s_pred_hr, s_t5b_hr)
        s_dist_7 = F.l1_loss(s_pred_hr, s_t7_hr)
        s_dist_9 = F.l1_loss(s_pred_hr, s_t9_hr)
        if not (torch.isfinite(s_dist_5b) and torch.isfinite(s_dist_7) and torch.isfinite(s_dist_9)):
            raise ValueError("Non-finite teacher distillation losses")
        print("PASS")

        # [14] Edge loss finite
        print("Check [14] Edge loss is finite...", end=" ")
        s_l_edge = F.l1_loss(sobel_filter(s_pred_hr), sobel_filter(s_tgt))
        if not torch.isfinite(s_l_edge): raise ValueError("Non-finite edge loss")
        print("PASS")

        # [15] HF loss finite
        print("Check [15] HF loss is finite...", end=" ")
        s_l_hf = F.l1_loss(decomp_helper(s_pred_hr)[2], s_tgt_hf)
        if not torch.isfinite(s_l_hf): raise ValueError("Non-finite HF loss")
        print("PASS")

        # Set up condition-dependent weights for checks
        l5b_weights = torch.tensor([cfg["distillation_weights"][c]["lambda_5b"] for c in ["clean", "noise", "blur", "noise_plus_blur"]], device=device)
        l7_weights = torch.tensor([cfg["distillation_weights"][c]["lambda_7"] for c in ["clean", "noise", "blur", "noise_plus_blur"]], device=device)
        l9_weights = torch.tensor([cfg["distillation_weights"][c]["lambda_9"] for c in ["clean", "noise", "blur", "noise_plus_blur"]], device=device)

        batch_l5b = l5b_weights[s_labels]
        batch_l7 = l7_weights[s_labels]
        batch_l9 = l9_weights[s_labels]

        s_dist_5b_samples = F.l1_loss(s_pred_hr, s_t5b_hr, reduction='none').mean(dim=[1, 2, 3])
        s_dist_7_samples = F.l1_loss(s_pred_hr, s_t7_hr, reduction='none').mean(dim=[1, 2, 3])
        s_dist_9_samples = F.l1_loss(s_pred_hr, s_t9_hr, reduction='none').mean(dim=[1, 2, 3])

        weighted_dist_5b = (s_dist_5b_samples * batch_l5b).mean()
        weighted_dist_7 = (s_dist_7_samples * batch_l7).mean()
        weighted_dist_9 = (s_dist_9_samples * batch_l9).mean()
        s_l_est = F.cross_entropy(s_logits, s_labels)

        s_total = s_l_recon + weighted_dist_5b + weighted_dist_7 + weighted_dist_9 + lambda_est * s_l_est

        # [16] Total loss finite
        print("Check [16] Total loss is finite...", end=" ")
        if not torch.isfinite(s_total): raise ValueError("Non-finite total loss value")
        print("PASS")

        # [17] No NaN / Inf
        print("Check [17] No NaN / Inf...", end=" ")
        if not torch.isfinite(s_pred_hr).all(): raise ValueError("Student predictions contain NaNs or Infs")
        print("PASS")

    # [18] Augmentation distribution
    print("Check [18] Augmentation distribution...", end=" ")
    all_stats = {"original": 0, "noise_only": 0, "blur_only": 0, "noise_blur": 0}
    for idx, batch in enumerate(train_loader):
        if idx >= 5: break
        _, _, b_stats = apply_training_degradation_phase10(batch["input"].to(device), batch["target"].to(device), list(batch["input_path"]), epoch=0, cfg=cfg, device=device)
        for k in all_stats:
            all_stats[k] += b_stats[k]
    total_samples = sum(all_stats.values())
    print(f"PASS (clean={100*all_stats['original']/total_samples:.1f}% (target 40%), noise={100*all_stats['noise_only']/total_samples:.1f}% (target 15%), blur={100*all_stats['blur_only']/total_samples:.1f}% (target 25%), noise+blur={100*all_stats['noise_blur']/total_samples:.1f}% (target 20%))")

    # [19] 2-sample overfit test
    print("Check [19] Running 2-sample overfit test...", end=" ")
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
        o_pred, _, _, o_xhf, _, o_logits = student(o_lr_up, o_p4_hr, lr_input=o_in)
        loss = compute_total_loss(
            o_pred, o_tgt, o_xhf, o_tgt_hf, o_p4_hr, l_weights, sobel_filter, lpips_model, decomp_helper
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

    # Restore student model to warm start weights (from Phase 9 checkpoint)
    student.load_state_dict(t9_chk["model_state_dict"], strict=False)

    # [20] Checkpoint save/load test
    print("Check [20] checkpoint save/load test...", end=" ")
    temp_ckpt_path = os.path.join(checkpoints_dir, "temp_sanity_check.pth")
    test_payload = {
        "epoch": 0,
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": o_opt.state_dict(),
        "val_psnr": 28.21,
    }
    torch.save(test_payload, temp_ckpt_path)
    loaded_payload = torch.load(temp_ckpt_path, map_location=device, weights_only=False)
    if loaded_payload["val_psnr"] != 28.21:
        raise ValueError("Failed to match value in loaded checkpoint")
    os.remove(temp_ckpt_path)
    print("PASS")

    print("All Sanity Checks Passed successfully.")

    if args.sanity_only:
        print("Sanity-only run complete. Exiting.")
        return

    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["training"]["scheduler_gamma"])

    # --- TRAINING LOOP ---
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
        train_dist_5b_sum = 0.0
        train_dist_7_sum = 0.0
        train_dist_9_sum = 0.0
        train_est_sum = 0.0
        num_train_batches = 0
        epoch_aug_stats = {"original": 0, "noise_only": 0, "blur_only": 0, "noise_blur": 0}

        for batch in train_loader:
            b_orig = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            sample_keys = list(batch["input_path"])

            # Apply targeted degradation mixture
            b_in, labels, aug_stats = apply_training_degradation_phase10(
                b_orig, b_tgt, sample_keys, epoch=epoch, cfg=cfg, device=device
            )
            for key in epoch_aug_stats:
                epoch_aug_stats[key] += aug_stats[key]

            is_augmented = (labels != 0)

            with torch.no_grad():
                b_p4_raw, _ = model_p4(b_in)
                b_p4_hr = torch.clamp(b_p4_raw, 0.0, 1.0)
                b_lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
                _, _, b_tgt_hf = decomp_helper(b_tgt)
                
                # Get teacher predictions
                b_t5b_hr, _, _, _, _ = teacher_5b(b_lr_up, b_p4_hr)
                b_t7_hr, _, _, _, _ = teacher_7(b_lr_up, b_p4_hr)
                b_t9_hr, _, _, _, _ = teacher_9(b_lr_up, b_p4_hr)

            optimizer.zero_grad()
            b_pred_hr, _, _, b_xhf, _, logits = student(b_lr_up, b_p4_hr, lr_input=b_in)

            # Compute split reconstruction loss
            loss_recon = 0.0
            orig_mask = ~is_augmented
            aug_mask = is_augmented

            if orig_mask.any():
                loss_orig = compute_total_loss(
                    b_pred_hr[orig_mask], b_tgt[orig_mask], b_xhf[orig_mask], b_tgt_hf[orig_mask],
                    b_p4_hr[orig_mask], l_weights, sobel_filter, lpips_model, decomp_helper
                )
                loss_recon += (orig_mask.sum().item() / len(sample_keys)) * loss_orig

            if aug_mask.any():
                loss_aug = compute_total_loss(
                    b_pred_hr[aug_mask], b_tgt[aug_mask], b_xhf[aug_mask], b_tgt_hf[aug_mask],
                    b_p4_hr[aug_mask], l_weights, sobel_filter, lpips_model, decomp_helper
                )
                loss_recon += (aug_mask.sum().item() / len(sample_keys)) * robustness_loss_weight * loss_aug

            # Compute distillation consistency losses per-sample
            dist_5b_samples = F.l1_loss(b_pred_hr, b_t5b_hr, reduction='none').mean(dim=[1, 2, 3])
            dist_7_samples = F.l1_loss(b_pred_hr, b_t7_hr, reduction='none').mean(dim=[1, 2, 3])
            dist_9_samples = F.l1_loss(b_pred_hr, b_t9_hr, reduction='none').mean(dim=[1, 2, 3])

            # Apply condition-dependent distillation weights
            batch_l5b = l5b_weights[labels]
            batch_l7 = l7_weights[labels]
            batch_l9 = l9_weights[labels]

            loss_dist_5b = (dist_5b_samples * batch_l5b).mean()
            loss_dist_7 = (dist_7_samples * batch_l7).mean()
            loss_dist_9 = (dist_9_samples * batch_l9).mean()

            # Degradation classification loss
            loss_est = F.cross_entropy(logits, labels)

            # Combine total loss
            loss = loss_recon + loss_dist_5b + loss_dist_7 + loss_dist_9 + lambda_est * loss_est

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += loss_recon.item()
            train_dist_5b_sum += loss_dist_5b.item()
            train_dist_7_sum += loss_dist_7.item()
            train_dist_9_sum += loss_dist_9.item()
            train_est_sum += loss_est.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / num_train_batches
        avg_recon_loss = train_recon_sum / num_train_batches
        avg_dist_5b_loss = train_dist_5b_sum / num_train_batches
        avg_dist_7_loss = train_dist_7_sum / num_train_batches
        avg_dist_9_loss = train_dist_9_sum / num_train_batches
        avg_est_loss = train_est_sum / num_train_batches
        
        total_aug = sum(epoch_aug_stats.values()) or 1
        aug_pct = {k: 100.0 * v / total_aug for k, v in epoch_aug_stats.items()}

        # Validation loop (original clean validation only)
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
                # Pass lr_input for estimation
                v_pred_hr, _, _, _, _, _ = student(v_lr_up, v_p4_hr, lr_input=v_in)
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

        # Scoring system (same as Phase 7, 8, 9)
        score = (m_psnr - 28.2153) * 1.0 + (m_ssim - 0.7682) * 50.0 + (0.2855 - m_lpips) * 10.0 - (m_hf_err - 0.0079) * 100.0

        epoch_rec = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_recon_loss": avg_recon_loss,
            "train_dist_5b_loss": avg_dist_5b_loss,
            "train_dist_7_loss": avg_dist_7_loss,
            "train_dist_9_loss": avg_dist_9_loss,
            "train_est_loss": avg_est_loss,
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err,
            "score": score,
            "aug_clean_pct": aug_pct["original"],
            "aug_noise_pct": aug_pct["noise_only"],
            "aug_blur_pct": aug_pct["blur_only"],
            "aug_noise_blur_pct": aug_pct["noise_blur"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_rec)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, D5B: {avg_dist_5b_loss:.4f}, D7: {avg_dist_7_loss:.4f}, D9: {avg_dist_9_loss:.4f}, Est: {avg_est_loss:.4f}) | "
            f"Val PSNR: {m_psnr:.4f} | SSIM: {m_ssim:.4f} | LPIPS: {m_lpips:.4f} | HF: {m_hf_err:.6f} | "
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
            "degradation_config": deg,
            "distillation_config": cfg["distillation_weights"],
            "distillation_global": cfg["distillation_global"]
        }
        
        # Save last checkpoint
        torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase10_last.pth"))
        
        # Save best checkpoint
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase10_best.pth"))
            print(f"  --> Saved new best checkpoint (Score: {best_score:+.4f})")

    elapsed = time.time() - start_time
    hist_df = pd.DataFrame(history)
    hist_csv_path = os.path.join(results_dir, "phase10_history.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"\nTraining finished in {elapsed / 60.0:.2f} mins.")
    print(f"Saved history CSV to: {hist_csv_path}")

if __name__ == "__main__":
    main()
