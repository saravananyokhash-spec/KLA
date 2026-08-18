"""
Phase 7 Fine-Tuning — Robustness & Fidelity Optimization
Supports:
  - Custom checkpoint loading (warm-start from either Phase 5B or Phase 7 checkpoint)
  - Configurable clean-data mixing ratio (preserve_original_probability)
  - Robustness loss scaling (robustness_loss_weight)
  - Custom learning rate tuning
  - Full evaluation metric selection
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
from degradation_utils import apply_training_degradation
from echo_model import BaselineECHOModel
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr
from utils import set_seed

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

def main():
    parser = argparse.ArgumentParser(description="Phase 7 Fine-Tuning Script")
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

    # Save copy of configuration file
    with open(os.path.join(configs_dir, os.path.basename(args.config)), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    l_weights = cfg["loss_weights"]
    deg = cfg["degradation"]
    robustness_loss_weight = l_weights.get("robustness_loss_weight", 1.0)

    print("=" * 60)
    print(f"PHASE 7 FINE-TUNE — EXPERIMENT: {cfg['experiment_name'].upper()}")
    print(f"Device: {device}")
    print(f"Output Directory: {out_dir}")
    print("-" * 60)
    print("Configuration details:")
    print(f"  Preserve original probability: {deg['preserve_original_probability']}")
    print(f"  Robustness loss weight       : {robustness_loss_weight}")
    print(f"  Learning Rate                : {cfg['training']['lr']}")
    print(f"  Epochs                       : {cfg['training']['epochs']}")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Training requires GPU.")

    # Load frozen Phase 4 baseline
    p4_ckpt_path = cfg["model"]["phase4_checkpoint"]
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for param in model_p4.parameters():
        param.requires_grad = False

    # Instantiate Fine-tuning model (SpatialFrequencyRestorationNet)
    model_ft = SpatialFrequencyRestorationNet(
        spatial_channels=cfg["model"]["spatial_channels"],
        freq_channels=cfg["model"]["freq_channels"],
        fusion_channels=cfg["model"]["fusion_channels"],
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"],
    ).to(device)

    # Determine initialization checkpoint
    phase7_ckpt = cfg["model"].get("phase7_checkpoint", "")
    if phase7_ckpt and os.path.exists(phase7_ckpt):
        print(f"Warm-starting from Phase 7 Best Checkpoint: {phase7_ckpt}")
        init_chk = torch.load(phase7_ckpt, map_location=device, weights_only=False)
    else:
        p5b_ckpt_path = cfg["model"]["phase5b_checkpoint"]
        print(f"Warm-starting from Phase 5B Best Checkpoint: {p5b_ckpt_path}")
        init_chk = torch.load(p5b_ckpt_path, map_location=device, weights_only=False)

    model_ft.load_state_dict(init_chk["model_state_dict"])

    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for param in lpips_model.parameters():
        param.requires_grad = False
    decomp_helper = FrequencyDecompositionModule(
        cutoff_low=cfg["model"]["cutoff_low"],
        cutoff_high=cfg["model"]["cutoff_high"],
    ).to(device)

    dataset_root = cfg["dataset"]["dataset_root"]
    train_csv = cfg["dataset"]["train_csv"]
    val_csv = cfg["dataset"]["val_csv"]

    train_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=train_csv)
    val_dataset = KLADataset(dataset_root=dataset_root, split="val", csv_path=val_csv)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg["training"]["seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # --- SANITY CHECKS ---
    print("\nRunning Sanity Checks...")
    sample_batch = next(iter(train_loader))
    s_in = sample_batch["input"].to(device)
    s_tgt = sample_batch["target"].to(device)
    s_keys = list(sample_batch["input_path"])
    s_aug, s_stats = apply_training_degradation(s_in, s_tgt, s_keys, epoch=0, cfg=cfg, device=device)
    with torch.no_grad():
        s_p4_raw, _ = model_p4(s_aug)
        s_p4_hr = torch.clamp(s_p4_raw, 0.0, 1.0)
        s_lr_up = F.interpolate(s_aug, scale_factor=2, mode="bicubic", align_corners=False)
    s_ft_hr, _, _, s_xhf, _ = model_ft(s_lr_up, s_p4_hr)

    if list(s_ft_hr.shape) != [s_in.shape[0], 1, 256, 256]:
        raise ValueError(f"Output shape mismatch! Got {s_ft_hr.shape}")
    if not torch.isfinite(s_ft_hr).all():
        raise ValueError("Outputs are non-finite!")
    print(f"Sanity Check passed. Augmentation sample batch stats: {s_stats}")

    optimizer = torch.optim.AdamW(
        model_ft.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg["training"]["scheduler_gamma"])

    if args.sanity_only:
        # --- 2-SAMPLE OVERFIT DIAGNOSTIC ---
        print("\nRunning 2-Sample Overfit Test...")
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

        o_opt = torch.optim.AdamW(model_ft.parameters(), lr=1e-3)
        o_start_loss = 0.0
        o_end_loss = 0.0

        for step in range(30):
            o_opt.zero_grad()
            o_pred, _, _, o_xhf, _ = model_ft(o_lr_up, o_p4_hr)
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
        print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f} | Reduction: {o_reduction:.2f}%")
        print("Sanity-only run complete.")
        return

    # --- TRAINING LOOP ---
    epochs = cfg["training"]["epochs"]
    best_score = -999.0
    history = []
    start_time = time.time()

    print("\n" + "=" * 50)
    print(f"STARTING {epochs} EPOCHS FINE-TUNING RUN")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model_ft.train()
        train_loss_sum = 0.0
        num_train_batches = 0
        epoch_aug_stats = {"original": 0, "blur_only": 0, "noise_blur": 0}

        for batch in train_loader:
            b_orig = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            sample_keys = list(batch["input_path"])

            # Apply controlled degradation
            b_in, aug_stats = apply_training_degradation(
                b_orig, b_tgt, sample_keys, epoch=epoch, cfg=cfg, device=device
            )
            for key in epoch_aug_stats:
                epoch_aug_stats[key] += aug_stats[key]

            # Determine which samples in the batch were augmented vs original
            is_augmented = []
            for i in range(len(sample_keys)):
                # If modified, the input is different from original
                is_augmented.append(not torch.equal(b_in[i], b_orig[i]))
            
            is_augmented = torch.tensor(is_augmented, dtype=torch.bool, device=device)

            with torch.no_grad():
                b_p4_raw, _ = model_p4(b_in)
                b_p4_hr = torch.clamp(b_p4_raw, 0.0, 1.0)
                b_lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
                _, _, b_tgt_hf = decomp_helper(b_tgt)

            optimizer.zero_grad()
            b_pred_hr, _, _, b_xhf, _ = model_ft(b_lr_up, b_p4_hr)

            # Compute split loss
            loss = 0.0
            orig_mask = ~is_augmented
            aug_mask = is_augmented

            if orig_mask.any():
                loss_orig = compute_total_loss(
                    b_pred_hr[orig_mask], b_tgt[orig_mask], b_xhf[orig_mask], b_tgt_hf[orig_mask],
                    b_p4_hr[orig_mask], l_weights, sobel_filter, lpips_model, decomp_helper
                )
                loss += (orig_mask.sum().item() / len(sample_keys)) * loss_orig

            if aug_mask.any():
                loss_aug = compute_total_loss(
                    b_pred_hr[aug_mask], b_tgt[aug_mask], b_xhf[aug_mask], b_tgt_hf[aug_mask],
                    b_p4_hr[aug_mask], l_weights, sobel_filter, lpips_model, decomp_helper
                )
                loss += (aug_mask.sum().item() / len(sample_keys)) * robustness_loss_weight * loss_aug

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / num_train_batches
        total_aug = sum(epoch_aug_stats.values()) or 1
        aug_pct = {k: 100.0 * v / total_aug for k, v in epoch_aug_stats.items()}

        # Validation loop (original clean validation)
        model_ft.eval()
        val_psnr_list, val_ssim_list, val_lpips_list = [], [], []
        val_mae_list, val_hf_err_list = [], []

        with torch.no_grad():
            for batch in val_loader:
                v_in = batch["input"].to(device)
                v_tgt = batch["target"].to(device)
                v_p4_raw, _ = model_p4(v_in)
                v_p4_hr = torch.clamp(v_p4_raw, 0.0, 1.0)
                v_lr_up = F.interpolate(v_in, scale_factor=2, mode="bicubic", align_corners=False)
                v_pred_hr, _, _, _, _ = model_ft(v_lr_up, v_p4_hr)
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

        # Scoring system prioritizing original PSNR/SSIM and LPIPS preservation
        score = (m_psnr - 28.2153) * 1.0 + (m_ssim - 0.7682) * 50.0 + (0.2855 - m_lpips) * 10.0 - (m_hf_err - 0.0079) * 100.0

        epoch_rec = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err,
            "score": score,
            "aug_original_pct": aug_pct["original"],
            "aug_blur_only_pct": aug_pct["blur_only"],
            "aug_noise_blur_pct": aug_pct["noise_blur"],
        }
        history.append(epoch_rec)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_train_loss:.4f} | "
            f"Val PSNR: {m_psnr:.4f} | SSIM: {m_ssim:.4f} | LPIPS: {m_lpips:.4f} | HF: {m_hf_err:.6f} | "
            f"Aug orig/blur/combo: {aug_pct['original']:.1f}/{aug_pct['blur_only']:.1f}/{aug_pct['noise_blur']:.1f}% | "
            f"Score: {score:+.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model_ft.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": m_psnr,
            "val_ssim": m_ssim,
            "val_lpips": m_lpips,
            "val_mae": m_mae,
            "val_hf_err": m_hf_err,
            "degradation_config": deg,
        }
        torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase7_ft_last.pth"))
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, os.path.join(checkpoints_dir, "echo_phase7_ft_best.pth"))
            print(f"  --> Saved new best checkpoint (Score: {best_score:+.4f})")

    elapsed = time.time() - start_time
    hist_df = pd.DataFrame(history)
    hist_csv_path = os.path.join(results_dir, "phase7_ft_history.csv")
    hist_df.to_csv(hist_csv_path, index=False)
    print(f"\nFine-tuning finished in {elapsed / 60.0:.2f} mins.")
    print(f"Saved history CSV to: {hist_csv_path}")

if __name__ == "__main__":
    main()
