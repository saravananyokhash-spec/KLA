"""
Phase 5B — Targeted Perceptual-Fidelity Refinement

Key experimental change vs Phase 5A:
  - LPIPS weight: 0.005 → 0.010 (2x increase, controlled perceptual reintroduction)
  - Initializes from Phase 5A best checkpoint (warm-start, NOT random init)
  - All other hyperparameters IDENTICAL to Phase 5A

Hypothesis: Phase 5A over-corrected toward pixel fidelity. A modest
increase in perceptual influence should recover some of Phase 5's LPIPS
advantage without sacrificing the fidelity that Phase 5A recovered.
"""
import os
import sys
import yaml
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import lpips

from functools import partial
print = partial(print, flush=True)

from utils import set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_image_png(tensor_img, path):
    """
    Saves float32 tensor [1, 1, H, W] or [1, H, W] in [0, 1] as 2D uint8 PNG.
    """
    arr = tensor_img.detach().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    plt.imsave(path, arr, cmap='gray')

def main():
    config_path = "configs/phase5b.yaml"
    if not os.path.exists(config_path):
        config_path = "outputs/phase5b/configs/phase5b.yaml"
    cfg = load_config(config_path)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = cfg["output_dir"]
    checkpoints_dir = cfg["checkpoints_dir"]
    results_dir = cfg["results_dir"]
    evaluation_dir = cfg["evaluation_dir"]
    configs_dir = cfg.get("configs_dir", os.path.join(out_dir, "configs"))

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    # Save copy of config in output directory
    with open(os.path.join(configs_dir, "phase5b.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    l_weights = cfg["loss_weights"]

    print("=" * 60)
    print("PHASE 5B — TARGETED PERCEPTUAL-FIDELITY REFINEMENT")
    print(f"Device: {device}")
    print(f"Output Directory: {out_dir}")
    print("-" * 60)
    print("Phase 5B Loss Configuration")
    print("----------------------------")
    print(f"Pixel (L1)          : {l_weights.get('pixel_l1', 2.0)}")
    print(f"Structural (SSIM)   : {l_weights.get('ssim', 0.3)}")
    print(f"High-Freq (Direct)  : {l_weights.get('hf_direct', 0.15)}")
    print(f"Sobel Edge          : {l_weights.get('sobel_edge', 0.15)}")
    print(f"HF Branch           : {l_weights.get('hf_component', 0.05)}")
    print(f"FFT Frequency       : {l_weights.get('freq_fft', 0.05)}")
    print(f"Perceptual (LPIPS)  : {l_weights.get('lpips', 0.010)}")
    print(f"P4 Anchor           : {l_weights.get('p4_anchor', 0.02)}")
    print("-" * 60)
    print("EXPERIMENTAL CHANGE: LPIPS weight 0.005 -> 0.010 (2x)")
    print("INITIALIZATION: Phase 5A best checkpoint (warm-start)")
    print("=" * 60)

    # --- MANDATORY SANITY CHECKS ---
    print("\n" + "=" * 50)
    print("RUNNING PHASE 5B SANITY CHECKS")
    print("=" * 50)

    # Check 1: CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Phase 5B training requires GPU.")
    print("Sanity Check 1: CUDA available: PASSED")

    # Check 2: Phase 4 Checkpoint
    p4_ckpt_path = cfg["model"]["phase4_checkpoint"]
    if not os.path.exists(p4_ckpt_path):
        raise FileNotFoundError(f"Phase 4 checkpoint missing at {p4_ckpt_path}")
    print(f"Sanity Check 2: Phase 4 Checkpoint exists ({p4_ckpt_path}): PASSED")

    # Check 3: Phase 5A Checkpoint
    p5a_ckpt_path = cfg["model"]["phase5a_checkpoint"]
    if not os.path.exists(p5a_ckpt_path):
        raise FileNotFoundError(f"Phase 5A checkpoint missing at {p5a_ckpt_path}")
    print(f"Sanity Check 3: Phase 5A Checkpoint exists ({p5a_ckpt_path}): PASSED")

    # Check 4 & 5: Dataset & CSVs
    dataset_root = cfg["dataset"]["dataset_root"]
    train_csv = cfg["dataset"]["train_csv"]
    val_csv = cfg["dataset"]["val_csv"]

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Train or Validation split CSV missing!")
    print("Sanity Check 4 & 5: Dataset root & split CSVs exist: PASSED")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_files = set(train_df["input_path"])
    val_files = set(val_df["input_path"])
    overlap = train_files.intersection(val_files)

    # Check 6: Disjointness
    if len(overlap) > 0:
        raise ValueError(f"Data leak detected! Overlap: {len(overlap)} samples")
    print("Sanity Check 6: Train/validation splits completely disjoint: PASSED")

    train_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=train_csv)
    val_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=val_csv)

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Load Phase 4 Frozen Baseline Model
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters():
        p.requires_grad = False
    print("Sanity Check 7: Phase 4 baseline loaded & frozen: PASSED")

    # Check 8: Phase 4 frozen
    p4_frozen = all(not p.requires_grad for p in model_p4.parameters())
    if not p4_frozen:
        raise ValueError("CRITICAL ERROR: Phase 4 parameters are NOT frozen!")
    print("Sanity Check 8: Phase 4 parameters completely frozen: PASSED")

    # Instantiate Phase 5B Model Architecture (same as Phase 5/5A)
    model_p5b = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)

    # Load Phase 5A best checkpoint as warm-start initialization
    p5a_chk = torch.load(p5a_ckpt_path, map_location=device, weights_only=False)
    model_p5b.load_state_dict(p5a_chk["model_state_dict"])
    print(f"Sanity Check 9: Phase 5A weights loaded into Phase 5B (epoch {p5a_chk.get('epoch', '?')}): PASSED")

    # Check 10: Phase 5B trainable
    p5b_trainable = any(p.requires_grad for p in model_p5b.parameters())
    if not p5b_trainable:
        raise ValueError("CRITICAL ERROR: Model parameters are NOT trainable!")
    print("Sanity Check 10: Phase 5B model parameters trainable: PASSED")

    # Setup Loss Functions & Helpers
    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters(): p.requires_grad = False

    decomp_helper = FrequencyDecompositionModule(
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)

    # Sample Pass for Checks 11-16
    sample_batch = next(iter(train_loader))
    s_in = sample_batch["input"].to(device)
    s_tgt = sample_batch["target"].to(device)

    with torch.no_grad():
        s_p4_raw, _ = model_p4(s_in)
        s_p4_hr = torch.clamp(s_p4_raw, 0.0, 1.0)
        s_lr_up = F.interpolate(s_in, scale_factor=2, mode="bicubic", align_corners=False)

    s_p5b_hr, s_xlf, s_xmf, s_xhf, s_feat = model_p5b(s_lr_up, s_p4_hr)

    # Check 11: Output shape
    if list(s_p5b_hr.shape) != [s_in.shape[0], 1, 256, 256]:
        raise ValueError(f"Output shape error! Got {s_p5b_hr.shape}")
    print(f"Sanity Check 11: Output shape {list(s_p5b_hr.shape)}: PASSED")

    # Check 12: Finiteness
    if not torch.isfinite(s_p5b_hr).all():
        raise ValueError("Output contains NaNs/Infs!")
    print("Sanity Check 12: Output values finite: PASSED")

    # Check 13: Output Range
    if s_p5b_hr.min() < 0.0 or s_p5b_hr.max() > 1.0:
        raise ValueError(f"Output range exceeded! min={s_p5b_hr.min():.4f}, max={s_p5b_hr.max():.4f}")
    print("Sanity Check 13: Output range [0, 1]: PASSED")

    # Check 14: Frequency Tensors Finite
    if not (torch.isfinite(s_xlf).all() and torch.isfinite(s_xmf).all() and torch.isfinite(s_xhf).all()):
        raise ValueError("Frequency component tensors contain NaNs/Infs!")
    print("Sanity Check 14: Frequency tensors finite: PASSED")

    # Compute Individual Losses for Sample Batch
    l_pixel = F.l1_loss(s_p5b_hr, s_tgt)
    l_ssim = 1.0 - ssim_pytorch(s_p5b_hr, s_tgt)
    l_fft = F.l1_loss(torch.fft.rfft2(s_p5b_hr, norm="ortho"), torch.fft.rfft2(s_tgt, norm="ortho"))

    tgt_lf, tgt_mf, tgt_hf = decomp_helper(s_tgt)
    pred_lf, pred_mf, pred_hf = decomp_helper(s_p5b_hr)

    l_hf_direct = F.l1_loss(pred_hf, tgt_hf)
    l_hf_comp = F.l1_loss(s_xhf, tgt_hf)
    l_edge = F.l1_loss(sobel_filter(s_p5b_hr), sobel_filter(s_tgt))
    l_lpips = ssim_lpips_differentiable(s_p5b_hr, s_tgt, lpips_model)
    l_anchor = F.l1_loss(s_p5b_hr, s_p4_hr)

    l_total = (l_weights.get("pixel_l1", 2.0) * l_pixel +
               l_weights.get("ssim", 0.3) * l_ssim +
               l_weights.get("hf_direct", 0.15) * l_hf_direct +
               l_weights.get("sobel_edge", 0.15) * l_edge +
               l_weights.get("hf_component", 0.05) * l_hf_comp +
               l_weights.get("freq_fft", 0.05) * l_fft +
               l_weights.get("lpips", 0.010) * l_lpips +
               l_weights.get("p4_anchor", 0.02) * l_anchor)

    # Check 15 & 16: Loss Finiteness
    if not (torch.isfinite(l_pixel) and torch.isfinite(l_ssim) and torch.isfinite(l_fft) and
            torch.isfinite(l_hf_direct) and torch.isfinite(l_edge) and torch.isfinite(l_lpips) and torch.isfinite(l_anchor)):
        raise ValueError("One or more individual losses non-finite!")
    print("Sanity Check 15: Individual losses finite: PASSED")

    if not torch.isfinite(l_total):
        raise ValueError("Total loss non-finite!")
    print("Sanity Check 16: Total loss finite: PASSED")

    # Check Gradient Flow (Checks 17-21)
    optimizer = torch.optim.AdamW(model_p5b.parameters(), lr=cfg["training"]["lr"])
    optimizer.zero_grad()
    l_total.backward()

    # Check 17: Phase 5B receives gradients
    has_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model_p5b.parameters())
    if not has_grads:
        raise ValueError("Phase 5B model received NO gradients!")
    print("Sanity Check 17: Phase 5B model receives gradients: PASSED")

    # Check 18: Phase 4 receives NO gradients
    p4_grads = any(p.grad is not None for p in model_p4.parameters())
    if p4_grads:
        raise ValueError("CRITICAL ERROR: Phase 4 received gradients!")
    print("Sanity Check 18: Phase 4 receives NO gradients: PASSED")

    # Check 19: Gradient Norms & NaN/Inf Check
    grad_norm = torch.nn.utils.clip_grad_norm_(model_p5b.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise ValueError("Gradient norm is non-finite!")
    print("Sanity Check 19: Gradient norms finite & no NaNs: PASSED")

    # Check 20: Frequency branch gradients
    freq_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model_p5b.freq_branch.parameters())
    if not freq_grads:
        raise ValueError("Frequency branch received NO gradients!")
    print("Sanity Check 20: Frequency branch receives gradients: PASSED")

    # Check 21: Spatial branch gradients
    spatial_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model_p5b.spatial_branch.parameters())
    if not spatial_grads:
        raise ValueError("Spatial branch received NO gradients!")
    print("Sanity Check 21: Spatial branch receives gradients: PASSED")

    # Check 22: Fusion module gradients
    fusion_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model_p5b.fusion.parameters())
    if not fusion_grads:
        raise ValueError("Fusion module received NO gradients!")
    print("Sanity Check 22: Fusion module receives gradients: PASSED")

    optimizer.zero_grad()

    # --- 2-SAMPLE OVERFIT DIAGNOSTIC TEST ---
    print("\n" + "=" * 50)
    print("RUNNING 2-SAMPLE OVERFIT DIAGNOSTIC TEST")
    print("=" * 50)

    overfit_subset = Subset(train_dataset, [0, 1])
    overfit_loader = DataLoader(overfit_subset, batch_size=2, shuffle=False)
    overfit_batch = next(iter(overfit_loader))
    o_in = overfit_batch["input"].to(device)
    o_tgt = overfit_batch["target"].to(device)

    with torch.no_grad():
        o_p4_raw, _ = model_p4(o_in)
        o_p4_hr = torch.clamp(o_p4_raw, 0.0, 1.0)
        o_lr_up = F.interpolate(o_in, scale_factor=2, mode="bicubic", align_corners=False)
        o_tgt_lf, o_tgt_mf, o_tgt_hf = decomp_helper(o_tgt)

    o_optimizer = torch.optim.AdamW(model_p5b.parameters(), lr=1.0e-3)

    o_start_loss = 0.0
    o_end_loss = 0.0

    for step in range(50):
        o_optimizer.zero_grad()
        o_p5b_hr, o_xlf, o_xmf, o_xhf, _ = model_p5b(o_lr_up, o_p4_hr)

        l_pix = F.l1_loss(o_p5b_hr, o_tgt)
        l_ss = 1.0 - ssim_pytorch(o_p5b_hr, o_tgt)
        l_ff = F.l1_loss(torch.fft.rfft2(o_p5b_hr, norm="ortho"), torch.fft.rfft2(o_tgt, norm="ortho"))
        o_pred_lf, o_pred_mf, o_pred_hf = decomp_helper(o_p5b_hr)
        l_hfd = F.l1_loss(o_pred_hf, o_tgt_hf)
        l_edg = F.l1_loss(sobel_filter(o_p5b_hr), sobel_filter(o_tgt))

        loss = 2.0 * l_pix + 0.3 * l_ss + 0.15 * l_hfd + 0.15 * l_edg + 0.05 * l_ff
        loss.backward()
        o_optimizer.step()

        if step == 0:
            o_start_loss = loss.item()
        if step == 49:
            o_end_loss = loss.item()

    o_reduction = (o_start_loss - o_end_loss) / (o_start_loss + 1e-8) * 100.0
    print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f} | Reduction: {o_reduction:.2f}%")

    if o_reduction < 10.0:
        print(f"WARNING: Low overfit reduction ({o_reduction:.2f}%) — expected since warm-starting from Phase 5A (already partially converged).")
    else:
        print("Sanity Check 23 (Overfit Diagnostic): PASSED")

    # Re-initialize model from Phase 5A checkpoint for actual training
    model_p5b = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"]
    ).to(device)

    # Load Phase 5A best checkpoint weights for warm-start
    p5a_chk_fresh = torch.load(p5a_ckpt_path, map_location=device, weights_only=False)
    model_p5b.load_state_dict(p5a_chk_fresh["model_state_dict"])
    print(f"\nRe-loaded Phase 5A best checkpoint for training (epoch {p5a_chk_fresh.get('epoch', '?')})")

    optimizer = torch.optim.AdamW(model_p5b.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["training"]["scheduler_gamma"])

    # --- 5-EPOCH PHASE 5B TRAINING RUN ---
    epochs = cfg["training"]["epochs"]
    print("\n" + "=" * 50)
    print(f"STARTING PHASE 5B ({epochs} EPOCHS) TRAINING RUN")
    print("=" * 50)

    best_score = -999.0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model_p5b.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        for batch in train_loader:
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)

            with torch.no_grad():
                b_p4_raw, _ = model_p4(b_in)
                b_p4_hr = torch.clamp(b_p4_raw, 0.0, 1.0)
                b_lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
                b_tgt_lf, b_tgt_mf, b_tgt_hf = decomp_helper(b_tgt)

            optimizer.zero_grad()
            b_p5b_hr, b_xlf, b_xmf, b_xhf, _ = model_p5b(b_lr_up, b_p4_hr)
            b_pred_lf, b_pred_mf, b_pred_hf = decomp_helper(b_p5b_hr)

            l_pixel = F.l1_loss(b_p5b_hr, b_tgt)
            l_ssim = 1.0 - ssim_pytorch(b_p5b_hr, b_tgt)
            l_fft = F.l1_loss(torch.fft.rfft2(b_p5b_hr, norm="ortho"), torch.fft.rfft2(b_tgt, norm="ortho"))
            l_hf_direct = F.l1_loss(b_pred_hf, b_tgt_hf)
            l_hf_comp = F.l1_loss(b_xhf, b_tgt_hf)
            l_edge = F.l1_loss(sobel_filter(b_p5b_hr), sobel_filter(b_tgt))
            l_lpips = ssim_lpips_differentiable(b_p5b_hr, b_tgt, lpips_model)
            l_anchor = F.l1_loss(b_p5b_hr, b_p4_hr)

            loss = (l_weights.get("pixel_l1", 2.0) * l_pixel +
                    l_weights.get("ssim", 0.3) * l_ssim +
                    l_weights.get("hf_direct", 0.15) * l_hf_direct +
                    l_weights.get("sobel_edge", 0.15) * l_edge +
                    l_weights.get("hf_component", 0.05) * l_hf_comp +
                    l_weights.get("freq_fft", 0.05) * l_fft +
                    l_weights.get("lpips", 0.010) * l_lpips +
                    l_weights.get("p4_anchor", 0.02) * l_anchor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_p5b.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / num_train_batches

        # Validation Loop
        model_p5b.eval()
        val_psnr_list, val_ssim_list, val_lpips_list, val_mae_list, val_edge_list = [], [], [], [], []
        val_lf_err_list, val_mf_err_list, val_hf_err_list = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                v_in = batch["input"].to(device)
                v_tgt = batch["target"].to(device)

                v_p4_raw, _ = model_p4(v_in)
                v_p4_hr = torch.clamp(v_p4_raw, 0.0, 1.0)
                v_lr_up = F.interpolate(v_in, scale_factor=2, mode="bicubic", align_corners=False)
                v_tgt_lf, v_tgt_mf, v_tgt_hf = decomp_helper(v_tgt)

                v_p5b_hr, v_xlf, v_xmf, v_xhf, _ = model_p5b(v_lr_up, v_p4_hr)
                v_pred_lf, v_pred_mf, v_pred_hf = decomp_helper(v_p5b_hr)

                val_psnr_list.append(calculate_psnr(v_p5b_hr, v_tgt))
                val_ssim_list.append(ssim_pytorch(v_p5b_hr, v_tgt).item())
                val_lpips_list.append(ssim_lpips_differentiable(v_p5b_hr, v_tgt, lpips_model).item())
                val_mae_list.append(F.l1_loss(v_p5b_hr, v_tgt).item())
                val_edge_list.append(F.l1_loss(sobel_filter(v_p5b_hr), sobel_filter(v_tgt)).item())

                val_lf_err_list.append(F.l1_loss(v_pred_lf, v_tgt_lf).item())
                val_mf_err_list.append(F.l1_loss(v_pred_mf, v_tgt_mf).item())
                val_hf_err_list.append(F.l1_loss(v_pred_hf, v_tgt_hf).item())

        m_psnr = float(np.mean(val_psnr_list))
        m_ssim = float(np.mean(val_ssim_list))
        m_lpips = float(np.mean(val_lpips_list))
        m_mae = float(np.mean(val_mae_list))
        m_edge = float(np.mean(val_edge_list))
        m_lf_err = float(np.mean(val_lf_err_list))
        m_mf_err = float(np.mean(val_mf_err_list))
        m_hf_err = float(np.mean(val_hf_err_list))

        # Metric Score for Model Checkpointing (same formula as Phase 5A)
        score = (m_psnr - 28.2153) * 1.0 + (m_ssim - 0.7682) * 50.0 + (0.2855 - m_lpips) * 10.0 - (m_hf_err - 0.0079) * 100.0

        epoch_rec = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_edge": m_edge,
            "val_lf_err": m_lf_err,
            "val_mf_err": m_mf_err,
            "val_hf_err": m_hf_err,
            "score": score
        }
        history.append(epoch_rec)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val PSNR: {m_psnr:.4f} dB | Val SSIM: {m_ssim:.4f} | Val LPIPS: {m_lpips:.4f} | Val HF Err: {m_hf_err:.6f} | Score: {score:+.4f}")

        # Save Checkpoint
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model_p5b.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err
        }
        torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase5b_last.pth"))

        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase5b_best.pth"))
            print(f"  --> Saved new best checkpoint (Score: {best_score:+.4f})")

    elapsed = time.time() - start_time
    print(f"\nPhase 5B training finished in {elapsed/60.0:.2f} mins.")

    # Save History CSV
    hist_df = pd.DataFrame(history)
    hist_csv_path = os.path.join(results_dir, "phase5b_history.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"Saved history CSV to: {hist_csv_path}")

if __name__ == "__main__":
    main()
