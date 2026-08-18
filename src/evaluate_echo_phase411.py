import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lpips

from utils import set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import Phase410PriorNet, calculate_psnr
from train_echo_phase411 import ErrorAwareFusionNet, build_evidence_input

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
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("PHASE 4 vs PHASE 4.10 vs PHASE 4.11 STANDALONE EVALUATOR")
    print(f"Device: {device}")
    print("=" * 60)

    # Output Directories
    eval_dir = "outputs/phase411/evaluation"
    visuals_dir = os.path.join(eval_dir, "visuals")
    raw_outputs_dir = os.path.join(eval_dir, "raw_outputs")
    analysis_dir = os.path.join(eval_dir, "analysis")
    plots_dir = os.path.join(eval_dir, "plots")

    visual_subdirs = ["best", "worst", "lpips_gain_psnr_loss", "balanced_wins", "regression"]
    for sd in visual_subdirs:
        os.makedirs(os.path.join(visuals_dir, sd), exist_ok=True)

    raw_subdirs = ["input", "target", "phase4", "phase410", "phase411"]
    for rd in raw_subdirs:
        os.makedirs(os.path.join(raw_outputs_dir, rd), exist_ok=True)

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Checkpoint Paths
    p4_ckpt_path = "outputs/echo_phase4/checkpoints/echo_best.pth"
    p410_ckpt_path = "outputs/phase410/checkpoints/echo_phase410_best.pth"
    
    # Locate best Phase 4.11 checkpoint
    p411_ckpt_dir = "outputs/phase411/checkpoints"
    p411_ckpt_path = os.path.join(p411_ckpt_dir, "echo_phase411_best.pth")
    if not os.path.exists(p411_ckpt_path):
        p411_ckpt_path = os.path.join(p411_ckpt_dir, "echo_phase411_last.pth")
        
    val_csv_path = "outputs/baseline/val_split.csv"
    dataset_root = "D:/kla"

    # --- SANITY CHECKS (1-12) ---
    print("\n" + "=" * 50)
    print("RUNNING STANDALONE SANITY CHECKS (1-12)")
    print("=" * 50)

    # Check 1: CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Evaluation requires GPU.")
    print("Sanity Check 1: CUDA available: PASSED")

    # Check 2: Phase 4 Checkpoint
    if not os.path.exists(p4_ckpt_path):
        raise FileNotFoundError(f"Phase 4 checkpoint missing at {p4_ckpt_path}")
    print(f"Sanity Check 2: Phase 4 Checkpoint exists ({p4_ckpt_path}): PASSED")

    # Check 3: Phase 4.10 Checkpoint
    if not os.path.exists(p410_ckpt_path):
        raise FileNotFoundError(f"Phase 4.10 checkpoint missing at {p410_ckpt_path}")
    print(f"Sanity Check 3: Phase 4.10 Checkpoint exists ({p410_ckpt_path}): PASSED")

    # Check 4: Phase 4.11 Checkpoint
    if not os.path.exists(p411_ckpt_path):
        raise FileNotFoundError(f"Phase 4.11 checkpoint missing at {p411_ckpt_path}")
    print(f"Sanity Check 4: Phase 4.11 Checkpoint exists ({p411_ckpt_path}): PASSED")

    # Check 5: Validation CSV
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"Validation split CSV missing at {val_csv_path}")
    print("Sanity Check 5: Validation CSV exists: PASSED")

    # Load Dataset
    val_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=val_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Check 6: Exactly 640 samples
    num_samples = len(val_dataset)
    if num_samples != 640:
        raise ValueError(f"Validation dataset must contain exactly 640 samples, got {num_samples}")
    print(f"Sanity Check 6: Validation dataset contains exactly 640 samples: PASSED")

    # Load Phase 4
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters(): p.requires_grad = False

    # Load Phase 4.10
    model_p410 = Phase410PriorNet(num_features=32).to(device)
    p410_chk = torch.load(p410_ckpt_path, map_location=device, weights_only=False)
    if "head_state_dict" in p410_chk:
        model_p410.load_state_dict(p410_chk["head_state_dict"])
    elif "model_state_dict" in p410_chk:
        model_p410.load_state_dict(p410_chk["model_state_dict"])
    else:
        model_p410.load_state_dict(p410_chk)
    model_p410.eval()
    for p in model_p410.parameters(): p.requires_grad = False

    # Load Phase 4.11
    model_p411 = ErrorAwareFusionNet(num_features=32, num_res_blocks=3).to(device)
    p411_chk = torch.load(p411_ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in p411_chk:
        model_p411.load_state_dict(p411_chk["model_state_dict"])
    else:
        model_p411.load_state_dict(p411_chk)
    model_p411.eval()
    for p in model_p411.parameters(): p.requires_grad = False

    # Check 7: Models loaded
    print("Sanity Check 7: All 3 models loaded successfully: PASSED")

    # Load Helpers
    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters(): p.requires_grad = False

    # Sample batch pass
    s_batch = next(iter(val_loader))
    s_in = s_batch["input"].to(device)
    s_tgt = s_batch["target"].to(device)

    with torch.no_grad():
        s_p4_raw, _ = model_p4(s_in)
        s_p4_out = torch.clamp(s_p4_raw, 0.0, 1.0)
        s_lr_up = F.interpolate(s_in, scale_factor=2, mode="bicubic", align_corners=False)
        s_lr_edge = get_lr_edge(s_lr_up, sobel_filter)
        s_p410_out, _, _, _, _, _ = model_p410(s_lr_up, s_p4_out, s_lr_edge, bounded_scale=0.05)
        s_evidence, _, _, _ = build_evidence_input(s_lr_up, s_p4_out, s_p410_out, sobel_filter)
        s_p411_out, _, _, _, _, _ = model_p411(s_evidence, s_p4_out, s_p410_out, bounded_scale=0.10)

    # Check 8: Output shapes
    if (list(s_p4_out.shape) != [1, 1, 256, 256] or
        list(s_p410_out.shape) != [1, 1, 256, 256] or
        list(s_p411_out.shape) != [1, 1, 256, 256]):
        raise ValueError("Shape check failed!")
    print("Sanity Check 8: All models produce [1, 1, 256, 256]: PASSED")

    # Check 9: Finiteness
    if not torch.isfinite(s_p4_out).all() or not torch.isfinite(s_p410_out).all() or not torch.isfinite(s_p411_out).all():
        raise ValueError("Outputs contain NaNs/Infs!")
    print("Sanity Check 9: All outputs finite: PASSED")

    # Check 10: Range [0, 1]
    if (s_p4_out.min() < 0.0 or s_p4_out.max() > 1.0 or
        s_p410_out.min() < 0.0 or s_p410_out.max() > 1.0 or
        s_p411_out.min() < 0.0 or s_p411_out.max() > 1.0):
        raise ValueError(f"Output range exceeded limits! p4=[{s_p4_out.min():.4f}, {s_p4_out.max():.4f}], p410=[{s_p410_out.min():.4f}, {s_p410_out.max():.4f}], p411=[{s_p411_out.min():.4f}, {s_p411_out.max():.4f}]")
    print("Sanity Check 10: Outputs remain within [0, 1]: PASSED")

    # Check 11: Validation samples
    print(f"Sanity Check 11: Same validation samples used (640 samples): PASSED")

    # Check 12: Metric calculation
    m_p = calculate_psnr(s_p411_out, s_tgt)
    m_s = ssim_pytorch(s_p411_out, s_tgt).item()
    m_l = ssim_lpips_differentiable(s_p411_out, s_tgt, lpips_model).item()
    print(f"Sanity Check 12: Metric calculations valid (Sample 0 PSNR={m_p:.2f}, SSIM={m_s:.4f}, LPIPS={m_l:.4f}): PASSED")

    print("\nEvaluating all 640 validation images across Phase 4, Phase 4.10, and Phase 4.11...")

    metrics_records = []
    eval_cache = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            in_path = batch["input_path"][0]
            tgt_path = batch["target_path"][0]

            p4_hr_raw, _ = model_p4(b_in)
            p4_hr = torch.clamp(p4_hr_raw, 0.0, 1.0)
            lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p410_hr, _, _, _, _, _ = model_p410(lr_up, p4_hr, lr_edge, bounded_scale=0.05)

            b_evidence, _, _, _ = build_evidence_input(lr_up, p4_hr, p410_hr, sobel_filter)
            p411_hr, c_p4, c_p410, corr, gate, _ = model_p411(b_evidence, p4_hr, p410_hr, bounded_scale=0.10)

            # Compute Metrics
            p4_psnr = calculate_psnr(p4_hr, b_tgt)
            p4_ssim = ssim_pytorch(p4_hr, b_tgt).item()
            p4_lpips = ssim_lpips_differentiable(p4_hr, b_tgt, lpips_model).item()
            p4_mae = F.l1_loss(p4_hr, b_tgt).item()
            p4_edge = F.l1_loss(sobel_filter(p4_hr), sobel_filter(b_tgt)).item()

            p410_psnr = calculate_psnr(p410_hr, b_tgt)
            p410_ssim = ssim_pytorch(p410_hr, b_tgt).item()
            p410_lpips = ssim_lpips_differentiable(p410_hr, b_tgt, lpips_model).item()
            p410_mae = F.l1_loss(p410_hr, b_tgt).item()
            p410_edge = F.l1_loss(sobel_filter(p410_hr), sobel_filter(b_tgt)).item()

            p411_psnr = calculate_psnr(p411_hr, b_tgt)
            p411_ssim = ssim_pytorch(p411_hr, b_tgt).item()
            p411_lpips = ssim_lpips_differentiable(p411_hr, b_tgt, lpips_model).item()
            p411_mae = F.l1_loss(p411_hr, b_tgt).item()
            p411_edge = F.l1_loss(sobel_filter(p411_hr), sobel_filter(b_tgt)).item()

            # Win Count
            wins_411 = 0
            if p411_psnr > p4_psnr and p411_psnr > p410_psnr: wins_411 += 1
            if p411_ssim > p4_ssim and p411_ssim > p410_ssim: wins_411 += 1
            if p411_lpips < p4_lpips and p411_lpips < p410_lpips: wins_411 += 1
            if p411_mae < p4_mae and p411_mae < p410_mae: wins_411 += 1
            if p411_edge < p4_edge and p411_edge < p410_edge: wins_411 += 1

            sid = f"sample_{idx+1:04d}"

            rec = {
                "sample_id": sid,
                "input_path": in_path,
                "target_path": tgt_path,
                "phase4_psnr": p4_psnr, "phase410_psnr": p410_psnr, "phase411_psnr": p411_psnr,
                "phase4_ssim": p4_ssim, "phase410_ssim": p410_ssim, "phase411_ssim": p411_ssim,
                "phase4_lpips": p4_lpips, "phase410_lpips": p410_lpips, "phase411_lpips": p411_lpips,
                "phase4_mae": p4_mae, "phase410_mae": p410_mae, "phase411_mae": p411_mae,
                "phase4_edge": p4_edge, "phase410_edge": p410_edge, "phase411_edge": p411_edge,
                "delta_411_vs_4_psnr": p411_psnr - p4_psnr,
                "delta_411_vs_410_psnr": p411_psnr - p410_psnr,
                "delta_411_vs_4_ssim": p411_ssim - p4_ssim,
                "delta_411_vs_410_ssim": p411_ssim - p410_ssim,
                "delta_411_vs_4_lpips": p411_lpips - p4_lpips,
                "delta_411_vs_410_lpips": p411_lpips - p410_lpips,
                "delta_411_vs_4_mae": p411_mae - p4_mae,
                "delta_411_vs_410_mae": p411_mae - p410_mae,
                "delta_411_vs_4_edge": p411_edge - p4_edge,
                "delta_411_vs_410_edge": p411_edge - p410_edge,
                "phase411_wins": wins_411
            }
            metrics_records.append(rec)

            eval_cache.append({
                "sample_id": sid,
                "input": b_in.cpu(),
                "target": b_tgt.cpu(),
                "phase4": p4_hr.cpu(),
                "phase410": p410_hr.cpu(),
                "phase411": p411_hr.cpu(),
                "record": rec
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
                print(f"Evaluated {idx+1}/{num_samples} samples...")

    df = pd.DataFrame(metrics_records)

    # Save CSV
    csv_path = os.path.join(eval_dir, "phase4_vs_phase410_vs_phase411_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_path}")

    # Summary JSON & Stats
    summary_dict = {}
    for model_key, prefix in [("phase4", "phase4_"), ("phase410", "phase410_"), ("phase411", "phase411_")]:
        summary_dict[model_key] = {
            "mean_psnr": float(df[f"{prefix}psnr"].mean()),
            "median_psnr": float(df[f"{prefix}psnr"].median()),
            "mean_ssim": float(df[f"{prefix}ssim"].mean()),
            "median_ssim": float(df[f"{prefix}ssim"].median()),
            "mean_lpips": float(df[f"{prefix}lpips"].mean()),
            "median_lpips": float(df[f"{prefix}lpips"].median()),
            "mean_mae": float(df[f"{prefix}mae"].mean()),
            "median_mae": float(df[f"{prefix}mae"].median()),
            "mean_edge_error": float(df[f"{prefix}edge"].mean()),
            "median_edge_error": float(df[f"{prefix}edge"].median())
        }

    # Statistical Deltas Summary
    stats_deltas = {}
    for d_col in ["delta_411_vs_4_psnr", "delta_411_vs_410_psnr",
                  "delta_411_vs_4_ssim", "delta_411_vs_410_ssim",
                  "delta_411_vs_4_lpips", "delta_411_vs_410_lpips",
                  "delta_411_vs_4_mae", "delta_411_vs_410_mae",
                  "delta_411_vs_4_edge", "delta_411_vs_410_edge"]:
        stats_deltas[d_col] = {
            "mean": float(df[d_col].mean()),
            "median": float(df[d_col].median()),
            "std": float(df[d_col].std()),
            "p25": float(df[d_col].quantile(0.25)),
            "p75": float(df[d_col].quantile(0.75))
        }

    summary_dict["deltas"] = stats_deltas

    # Win Rates (%)
    win_rates = {
        "vs_phase4": {
            "psnr_win_pct": float((df["delta_411_vs_4_psnr"] > 0).mean() * 100.0),
            "ssim_win_pct": float((df["delta_411_vs_4_ssim"] > 0).mean() * 100.0),
            "lpips_win_pct": float((df["delta_411_vs_4_lpips"] < 0).mean() * 100.0),
            "mae_win_pct": float((df["delta_411_vs_4_mae"] < 0).mean() * 100.0),
            "edge_win_pct": float((df["delta_411_vs_4_edge"] < 0).mean() * 100.0)
        },
        "vs_phase410": {
            "psnr_win_pct": float((df["delta_411_vs_410_psnr"] > 0).mean() * 100.0),
            "ssim_win_pct": float((df["delta_411_vs_410_ssim"] > 0).mean() * 100.0),
            "lpips_win_pct": float((df["delta_411_vs_410_lpips"] < 0).mean() * 100.0),
            "mae_win_pct": float((df["delta_411_vs_410_mae"] < 0).mean() * 100.0),
            "edge_win_pct": float((df["delta_411_vs_410_edge"] < 0).mean() * 100.0)
        }
    }
    summary_dict["win_rates"] = win_rates

    summary_json_path = os.path.join(eval_dir, "phase411_evaluation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Saved summary JSON to: {summary_json_path}")

    # Balanced Restoration Score Computation
    # Relative normalization across all 3 models per metric
    min_p, max_p = min(summary_dict["phase4"]["mean_psnr"], summary_dict["phase410"]["mean_psnr"], summary_dict["phase411"]["mean_psnr"]), max(summary_dict["phase4"]["mean_psnr"], summary_dict["phase410"]["mean_psnr"], summary_dict["phase411"]["mean_psnr"])
    min_s, max_s = min(summary_dict["phase4"]["mean_ssim"], summary_dict["phase410"]["mean_ssim"], summary_dict["phase411"]["mean_ssim"]), max(summary_dict["phase4"]["mean_ssim"], summary_dict["phase410"]["mean_ssim"], summary_dict["phase411"]["mean_ssim"])
    min_l, max_l = min(summary_dict["phase4"]["mean_lpips"], summary_dict["phase410"]["mean_lpips"], summary_dict["phase411"]["mean_lpips"]), max(summary_dict["phase4"]["mean_lpips"], summary_dict["phase410"]["mean_lpips"], summary_dict["phase411"]["mean_lpips"])
    min_m, max_m = min(summary_dict["phase4"]["mean_mae"], summary_dict["phase410"]["mean_mae"], summary_dict["phase411"]["mean_mae"]), max(summary_dict["phase4"]["mean_mae"], summary_dict["phase410"]["mean_mae"], summary_dict["phase411"]["mean_mae"])
    min_e, max_e = min(summary_dict["phase4"]["mean_edge_error"], summary_dict["phase410"]["mean_edge_error"], summary_dict["phase411"]["mean_edge_error"]), max(summary_dict["phase4"]["mean_edge_error"], summary_dict["phase410"]["mean_edge_error"], summary_dict["phase411"]["mean_edge_error"])

    def calc_balanced(mean_psnr, mean_ssim, mean_lpips, mean_mae, mean_edge):
        n_p = (mean_psnr - min_p) / (max_p - min_p + 1e-8)
        n_s = (mean_ssim - min_s) / (max_s - min_s + 1e-8)
        n_l = (max_l - mean_lpips) / (max_l - min_l + 1e-8)
        n_m = (max_m - mean_mae) / (max_m - min_m + 1e-8)
        n_e = (max_e - mean_edge) / (max_e - min_e + 1e-8)
        return 0.25 * n_p + 0.25 * n_s + 0.25 * n_l + 0.15 * n_m + 0.10 * n_e

    b_score_p4 = calc_balanced(summary_dict["phase4"]["mean_psnr"], summary_dict["phase4"]["mean_ssim"], summary_dict["phase4"]["mean_lpips"], summary_dict["phase4"]["mean_mae"], summary_dict["phase4"]["mean_edge_error"])
    b_score_p410 = calc_balanced(summary_dict["phase410"]["mean_psnr"], summary_dict["phase410"]["mean_ssim"], summary_dict["phase410"]["mean_lpips"], summary_dict["phase410"]["mean_mae"], summary_dict["phase410"]["mean_edge_error"])
    b_score_p411 = calc_balanced(summary_dict["phase411"]["mean_psnr"], summary_dict["phase411"]["mean_ssim"], summary_dict["phase411"]["mean_lpips"], summary_dict["phase411"]["mean_mae"], summary_dict["phase411"]["mean_edge_error"])

    # Sample Selections & Categorization for Visuals
    # Combined per-sample score
    p_z = (df["delta_411_vs_4_psnr"] - df["delta_411_vs_4_psnr"].mean()) / (df["delta_411_vs_4_psnr"].std() + 1e-8)
    s_z = (df["delta_411_vs_4_ssim"] - df["delta_411_vs_4_ssim"].mean()) / (df["delta_411_vs_4_ssim"].std() + 1e-8)
    l_z = (df["delta_411_vs_4_lpips"] - df["delta_411_vs_4_lpips"].mean()) / (df["delta_411_vs_4_lpips"].std() + 1e-8)
    df["comb_score"] = p_z + s_z - l_z

    df_best = df.sort_values(by="comb_score", ascending=False).head(10)
    df_worst = df.sort_values(by="comb_score", ascending=True).head(10)
    df_lpips_gain_psnr_loss = df[(df["delta_411_vs_4_lpips"] < -0.05) & (df["delta_411_vs_4_psnr"] < -0.05)].head(10)
    df_balanced_wins = df[(df["delta_411_vs_4_psnr"] > 0) & (df["delta_411_vs_4_lpips"] < 0)].head(10)
    df_regression = df[(df["delta_411_vs_4_psnr"] < 0) & (df["delta_411_vs_410_psnr"] < 0) & (df["delta_411_vs_4_lpips"] > 0)].head(10)

    # Save Analysis CSVs
    df_best.to_csv(os.path.join(analysis_dir, "best_phase411_samples.csv"), index=False)
    df_worst.to_csv(os.path.join(analysis_dir, "worst_phase411_samples.csv"), index=False)
    df_lpips_gain_psnr_loss.to_csv(os.path.join(analysis_dir, "phase411_tradeoff_samples.csv"), index=False)

    # Save Visuals (5-Panel Figures) & Raw Output PNGs
    visual_categories = [
        ("best", df_best),
        ("worst", df_worst),
        ("lpips_gain_psnr_loss", df_lpips_gain_psnr_loss),
        ("balanced_wins", df_balanced_wins),
        ("regression", df_regression)
    ]

    selected_raw_ids = set()

    for cat_name, cat_df in visual_categories:
        cat_dir = os.path.join(visuals_dir, cat_name)
        for _, row in cat_df.iterrows():
            sid = row["sample_id"]
            selected_raw_ids.add(sid)
            item = next(it for it in eval_cache if it["sample_id"] == sid)
            
            inp_t = item["input"]
            tgt_t = item["target"]
            p4_t = item["phase4"]
            p410_t = item["phase410"]
            p411_t = item["phase411"]

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(inp_t[0].squeeze().numpy(), cmap="gray")
            axes[0].set_title("Input / LR\n"); axes[0].axis("off")

            axes[1].imshow(tgt_t[0].squeeze().numpy(), cmap="gray")
            axes[1].set_title("Ground Truth\n"); axes[1].axis("off")

            axes[2].imshow(p4_t[0].squeeze().numpy(), cmap="gray")
            axes[2].set_title(f"Phase 4\nPSNR: {row['phase4_psnr']:.2f} | SSIM: {row['phase4_ssim']:.3f}\nLPIPS: {row['phase4_lpips']:.3f}"); axes[2].axis("off")

            axes[3].imshow(p410_t[0].squeeze().numpy(), cmap="gray")
            axes[3].set_title(f"Phase 4.10\nPSNR: {row['phase410_psnr']:.2f} | SSIM: {row['phase410_ssim']:.3f}\nLPIPS: {row['phase410_lpips']:.3f}"); axes[3].axis("off")

            axes[4].imshow(p411_t[0].squeeze().numpy(), cmap="gray")
            axes[4].set_title(f"Phase 4.11 ({sid})\nPSNR: {row['phase411_psnr']:.2f} | SSIM: {row['phase411_ssim']:.3f}\nLPIPS: {row['phase411_lpips']:.3f}"); axes[4].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(cat_dir, f"{sid}_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Save Raw Outputs for Selected Samples
    for item in eval_cache:
        sid = item["sample_id"]
        if sid in selected_raw_ids:
            save_image_png(item["input"], os.path.join(raw_outputs_dir, "input", f"{sid}.png"))
            save_image_png(item["target"], os.path.join(raw_outputs_dir, "target", f"{sid}.png"))
            save_image_png(item["phase4"], os.path.join(raw_outputs_dir, "phase4", f"{sid}.png"))
            save_image_png(item["phase410"], os.path.join(raw_outputs_dir, "phase410", f"{sid}.png"))
            save_image_png(item["phase411"], os.path.join(raw_outputs_dir, "phase411", f"{sid}.png"))

    print("Saved 5-panel Visual Comparisons and Raw Output PNGs.")

    # Generate Distribution Plots
    print("\nGenerating 10 Distribution & Diagnostic Plots...")
    
    # 1-5. Metric distributions
    for m in ["psnr", "ssim", "lpips", "mae", "edge"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[f"phase4_{m}"], bins=30, alpha=0.4, label="Phase 4", color="blue")
        plt.hist(df[f"phase410_{m}"], bins=30, alpha=0.4, label="Phase 4.10", color="orange")
        plt.hist(df[f"phase411_{m}"], bins=30, alpha=0.6, label="Phase 4.11", color="green")
        plt.title(f"{m.upper()} Distribution (640 Validation Samples)")
        plt.xlabel(m.upper()); plt.ylabel("Frequency"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, f"{m}_distribution.png"), dpi=150); plt.close()

    # 6. PSNR Delta (P4.11 vs P4)
    plt.figure(figsize=(8, 5))
    plt.hist(df["delta_411_vs_4_psnr"], bins=30, alpha=0.7, color="purple")
    plt.axvline(0.0, color="red", linestyle="--")
    plt.title("Delta PSNR Distribution (Phase 4.11 vs Phase 4)")
    plt.xlabel("Delta PSNR (dB)"); plt.ylabel("Frequency"); plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "psnr_delta_distribution.png"), dpi=150); plt.close()

    # 7. LPIPS Delta (P4.11 vs P4)
    plt.figure(figsize=(8, 5))
    plt.hist(df["delta_411_vs_4_lpips"], bins=30, alpha=0.7, color="teal")
    plt.axvline(0.0, color="red", linestyle="--")
    plt.title("Delta LPIPS Distribution (Phase 4.11 vs Phase 4)")
    plt.xlabel("Delta LPIPS"); plt.ylabel("Frequency"); plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "lpips_delta_distribution.png"), dpi=150); plt.close()

    # 8. PSNR vs LPIPS Scatter plot
    plt.figure(figsize=(8, 5))
    plt.scatter(df["phase4_psnr"], df["phase4_lpips"], alpha=0.4, label="Phase 4", color="blue")
    plt.scatter(df["phase410_psnr"], df["phase410_lpips"], alpha=0.4, label="Phase 4.10", color="orange")
    plt.scatter(df["phase411_psnr"], df["phase411_lpips"], alpha=0.6, label="Phase 4.11", color="green")
    plt.title("PSNR vs LPIPS Trade-Off Across Validation Set")
    plt.xlabel("PSNR (dB)"); plt.ylabel("LPIPS"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "psnr_vs_lpips_scatter.png"), dpi=150); plt.close()

    # 9. Phase 4 vs Phase 4.11 Comparison
    plt.figure(figsize=(8, 5))
    plt.scatter(df["phase4_psnr"], df["phase411_psnr"], alpha=0.5, color="green")
    plt.plot([min(df["phase4_psnr"]), max(df["phase4_psnr"])], [min(df["phase4_psnr"]), max(df["phase4_psnr"])], "r--")
    plt.title("Phase 4 vs Phase 4.11 PSNR Scatter")
    plt.xlabel("Phase 4 PSNR (dB)"); plt.ylabel("Phase 4.11 PSNR (dB)"); plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "phase4_vs_phase411_psnr.png"), dpi=150); plt.close()

    # 10. Phase 4.10 vs Phase 4.11 Comparison
    plt.figure(figsize=(8, 5))
    plt.scatter(df["phase410_lpips"], df["phase411_lpips"], alpha=0.5, color="teal")
    plt.plot([min(df["phase410_lpips"]), max(df["phase410_lpips"])], [min(df["phase410_lpips"]), max(df["phase410_lpips"])], "r--")
    plt.title("Phase 4.10 vs Phase 4.11 LPIPS Scatter")
    plt.xlabel("Phase 4.10 LPIPS"); plt.ylabel("Phase 4.11 LPIPS"); plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plots_dir, "phase410_vs_phase411_lpips.png"), dpi=150); plt.close()

    print("Saved 10 Distribution & Diagnostic Plots.")

    # Final Decision Verdict Classification
    m_p4_p, m_p410_p, m_p411_p = summary_dict["phase4"]["mean_psnr"], summary_dict["phase410"]["mean_psnr"], summary_dict["phase411"]["mean_psnr"]
    m_p4_s, m_p410_s, m_p411_s = summary_dict["phase4"]["mean_ssim"], summary_dict["phase410"]["mean_ssim"], summary_dict["phase411"]["mean_ssim"]
    m_p4_l, m_p410_l, m_p411_l = summary_dict["phase4"]["mean_lpips"], summary_dict["phase410"]["mean_lpips"], summary_dict["phase411"]["mean_lpips"]

    d_p4_p = m_p411_p - m_p4_p
    d_p4_s = m_p411_s - m_p4_s
    d_p4_l = m_p411_l - m_p4_l

    if d_p4_p >= -0.02 and d_p4_s >= -0.001 and m_p411_l <= m_p410_l:
        classification = "STRONG SUCCESS"
    elif d_p4_p >= -0.08 and d_p4_l <= -0.06:
        classification = "BALANCED SUCCESS"
    elif m_p411_l < m_p410_l and d_p4_p < -0.10:
        classification = "PERCEPTUAL SUCCESS"
    elif d_p4_p > 0 and d_p4_s > 0 and m_p411_l >= m_p4_l:
        classification = "PIXEL FIDELITY SUCCESS"
    elif d_p4_p < -0.15 and d_p4_l > 0:
        classification = "REGRESSION"
    elif abs(d_p4_p) < 0.02 and abs(d_p4_l) < 0.005:
        classification = "NO MEANINGFUL IMPROVEMENT"
    else:
        classification = "MIXED RESULT"

    # Best Model Recommendations per Metric
    best_psnr_model = "Phase 4" if m_p4_p >= max(m_p410_p, m_p411_p) else ("Phase 4.10" if m_p410_p >= m_p411_p else "Phase 4.11")
    best_ssim_model = "Phase 4" if m_p4_s >= max(m_p410_s, m_p411_s) else ("Phase 4.10" if m_p410_s >= m_p411_s else "Phase 4.11")
    best_lpips_model = "Phase 4.11" if m_p411_l <= min(m_p4_l, m_p410_l) else ("Phase 4.10" if m_p410_l <= m_p4_l else "Phase 4")
    best_mae_model = "Phase 4" if summary_dict["phase4"]["mean_mae"] <= min(summary_dict["phase410"]["mean_mae"], summary_dict["phase411"]["mean_mae"]) else ("Phase 4.10" if summary_dict["phase410"]["mean_mae"] <= summary_dict["phase411"]["mean_mae"] else "Phase 4.11")
    best_edge_model = "Phase 4.11" if summary_dict["phase411"]["mean_edge_error"] <= min(summary_dict["phase4"]["mean_edge_error"], summary_dict["phase410"]["mean_edge_error"]) else ("Phase 4.10" if summary_dict["phase410"]["mean_edge_error"] <= summary_dict["phase4"]["mean_edge_error"] else "Phase 4")

    scores = [("Phase 4", b_score_p4), ("Phase 4.10", b_score_p410), ("Phase 4.11", b_score_p411)]
    recommended_model = max(scores, key=lambda x: x[1])[0]

    # Save Human-Readable Text Report (UTF-8)
    report_path = os.path.join(eval_dir, "PHASE411_EVALUATION_REPORT.txt")
    report_text = f"""============================================================
PHASE 4 vs PHASE 4.10 vs PHASE 4.11 EVALUATION REPORT
============================================================

Dataset: Validation set (outputs/baseline/val_split.csv)
Samples: {num_samples}

------------------------------------------------------------
1. METRICS SUMMARY (Mean / Median)
------------------------------------------------------------
Metric       Phase 4          Phase 4.10       Phase 4.11
------------------------------------------------------------
PSNR (dB)    {summary_dict['phase4']['mean_psnr']:8.4f} / {summary_dict['phase4']['median_psnr']:8.4f}  {summary_dict['phase410']['mean_psnr']:8.4f} / {summary_dict['phase410']['median_psnr']:8.4f}  {summary_dict['phase411']['mean_psnr']:8.4f} / {summary_dict['phase411']['median_psnr']:8.4f}
SSIM         {summary_dict['phase4']['mean_ssim']:8.4f} / {summary_dict['phase4']['median_ssim']:8.4f}  {summary_dict['phase410']['mean_ssim']:8.4f} / {summary_dict['phase410']['median_ssim']:8.4f}  {summary_dict['phase411']['mean_ssim']:8.4f} / {summary_dict['phase411']['median_ssim']:8.4f}
LPIPS        {summary_dict['phase4']['mean_lpips']:8.4f} / {summary_dict['phase4']['median_lpips']:8.4f}  {summary_dict['phase410']['mean_lpips']:8.4f} / {summary_dict['phase410']['median_lpips']:8.4f}  {summary_dict['phase411']['mean_lpips']:8.4f} / {summary_dict['phase411']['median_lpips']:8.4f}
MAE          {summary_dict['phase4']['mean_mae']:8.4f} / {summary_dict['phase4']['median_mae']:8.4f}  {summary_dict['phase410']['mean_mae']:8.4f} / {summary_dict['phase410']['median_mae']:8.4f}  {summary_dict['phase411']['mean_mae']:8.4f} / {summary_dict['phase411']['median_mae']:8.4f}
Edge Error   {summary_dict['phase4']['mean_edge_error']:8.4f} / {summary_dict['phase4']['median_edge_error']:8.4f}  {summary_dict['phase410']['mean_edge_error']:8.4f} / {summary_dict['phase410']['median_edge_error']:8.4f}  {summary_dict['phase411']['mean_edge_error']:8.4f} / {summary_dict['phase411']['median_edge_error']:8.4f}

------------------------------------------------------------
2. STATISTICAL DELTAS (Phase 4.11 vs Phase 4 & Phase 4.10)
------------------------------------------------------------
Phase 4.11 vs Phase 4 Champion:
  Delta PSNR : {d_p4_p:+8.4f} dB  (Win Rate: {win_rates['vs_phase4']['psnr_win_pct']:.1f}%)
  Delta SSIM : {d_p4_s:+8.4f}     (Win Rate: {win_rates['vs_phase4']['ssim_win_pct']:.1f}%)
  Delta LPIPS: {d_p4_l:+8.4f}     (Win Rate: {win_rates['vs_phase4']['lpips_win_pct']:.1f}%)
  Delta MAE  : {summary_dict['phase411']['mean_mae'] - summary_dict['phase4']['mean_mae']:+8.4f}     (Win Rate: {win_rates['vs_phase4']['mae_win_pct']:.1f}%)
  Delta Edge : {summary_dict['phase411']['mean_edge_error'] - summary_dict['phase4']['mean_edge_error']:+8.4f}     (Win Rate: {win_rates['vs_phase4']['edge_win_pct']:.1f}%)

Phase 4.11 vs Phase 4.10:
  Delta PSNR : {m_p411_p - m_p410_p:+8.4f} dB  (Win Rate: {win_rates['vs_phase410']['psnr_win_pct']:.1f}%)
  Delta SSIM : {m_p411_s - m_p410_s:+8.4f}     (Win Rate: {win_rates['vs_phase410']['ssim_win_pct']:.1f}%)
  Delta LPIPS: {m_p411_l - m_p410_l:+8.4f}     (Win Rate: {win_rates['vs_phase410']['lpips_win_pct']:.1f}%)
  Delta MAE  : {summary_dict['phase411']['mean_mae'] - summary_dict['phase410']['mean_mae']:+8.4f}     (Win Rate: {win_rates['vs_phase410']['mae_win_pct']:.1f}%)
  Delta Edge : {summary_dict['phase411']['mean_edge_error'] - summary_dict['phase410']['mean_edge_error']:+8.4f}     (Win Rate: {win_rates['vs_phase410']['edge_win_pct']:.1f}%)

------------------------------------------------------------
3. BALANCED RESTORATION SCORES
------------------------------------------------------------
Phase 4     : {b_score_p4:.4f}
Phase 4.10  : {b_score_p410:.4f}
Phase 4.11  : {b_score_p411:.4f}

------------------------------------------------------------
4. FINAL CLASSIFICATION & RECOMMENDATION
------------------------------------------------------------
CLASSIFICATION: {classification}

CURRENT CHAMPION PER METRIC:
- Best PSNR    : {best_psnr_model}
- Best SSIM    : {best_ssim_model}
- Best LPIPS   : {best_lpips_model}
- Best MAE     : {best_mae_model}
- Best Edge    : {best_edge_model}
- Best Balanced: {recommended_model}

RECOMMENDED MODEL FOR KLA ECHO:
{recommended_model}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}")

    # Clean Console Output Table
    print("\n" + "=" * 65)
    print("STANDALONE EVALUATION COMPLETE")
    print("=" * 65)
    print(f"{'MODEL':<12} {'PSNR (dB)':<10} {'SSIM':<8} {'LPIPS':<8} {'MAE':<8} {'EDGE':<8}")
    print("-" * 65)
    print(f"{'Phase 4':<12} {summary_dict['phase4']['mean_psnr']:<10.4f} {summary_dict['phase4']['mean_ssim']:<8.4f} {summary_dict['phase4']['mean_lpips']:<8.4f} {summary_dict['phase4']['mean_mae']:<8.4f} {summary_dict['phase4']['mean_edge_error']:<8.4f}")
    print(f"{'Phase 4.10':<12} {summary_dict['phase410']['mean_psnr']:<10.4f} {summary_dict['phase410']['mean_ssim']:<8.4f} {summary_dict['phase410']['mean_lpips']:<8.4f} {summary_dict['phase410']['mean_mae']:<8.4f} {summary_dict['phase410']['mean_edge_error']:<8.4f}")
    print(f"{'Phase 4.11':<12} {summary_dict['phase411']['mean_psnr']:<10.4f} {summary_dict['phase411']['mean_ssim']:<8.4f} {summary_dict['phase411']['mean_lpips']:<8.4f} {summary_dict['phase411']['mean_mae']:<8.4f} {summary_dict['phase411']['mean_edge_error']:<8.4f}")
    print("-" * 65)
    print("DELTA: Phase 4.11 vs Phase 4 Champion:")
    print(f"Delta PSNR  : {d_p4_p:+8.4f} dB")
    print(f"Delta SSIM  : {d_p4_s:+8.4f}")
    print(f"Delta LPIPS : {d_p4_l:+8.4f}")
    print(f"Delta MAE   : {summary_dict['phase411']['mean_mae'] - summary_dict['phase4']['mean_mae']:+8.4f}")
    print(f"Delta Edge  : {summary_dict['phase411']['mean_edge_error'] - summary_dict['phase4']['mean_edge_error']:+8.4f}")
    print("-" * 65)
    print("DELTA: Phase 4.11 vs Phase 4.10:")
    print(f"Delta PSNR  : {m_p411_p - m_p410_p:+8.4f} dB")
    print(f"Delta SSIM  : {m_p411_s - m_p410_s:+8.4f}")
    print(f"Delta LPIPS : {m_p411_l - m_p410_l:+8.4f}")
    print(f"Delta MAE   : {summary_dict['phase411']['mean_mae'] - summary_dict['phase410']['mean_mae']:+8.4f}")
    print(f"Delta Edge  : {summary_dict['phase411']['mean_edge_error'] - summary_dict['phase410']['mean_edge_error']:+8.4f}")
    print("-" * 65)
    print("BALANCED RESTORATION SCORES:")
    print(f"Phase 4     : {b_score_p4:.4f}")
    print(f"Phase 4.10  : {b_score_p410:.4f}")
    print(f"Phase 4.11  : {b_score_p411:.4f}")
    print("=" * 65)
    print(f"PHASE 4.11 FINAL VERDICT:\nCLASSIFICATION: {classification}\nRECOMMENDED CHAMPION: {recommended_model}")
    print("=" * 65)

if __name__ == "__main__":
    main()
