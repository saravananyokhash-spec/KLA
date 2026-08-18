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
from train_echo_phase410 import calculate_psnr
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule

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
    print("PHASE 4 vs PHASE 5 vs PHASE 5A EVALUATION")
    print(f"Device: {device}")
    print("=" * 60)

    # Output Directories
    eval_dir = "outputs/phase5a/evaluation"
    visuals_dir = os.path.join(eval_dir, "visuals")
    raw_outputs_dir = os.path.join(eval_dir, "raw_outputs")
    analysis_dir = os.path.join(eval_dir, "analysis")
    plots_dir = os.path.join(eval_dir, "plots")

    visual_subdirs = ["best", "worst", "tradeoff", "hf_wins", "regression"]
    for sd in visual_subdirs:
        os.makedirs(os.path.join(visuals_dir, sd), exist_ok=True)

    raw_subdirs = ["input", "target", "phase4", "phase5", "phase5a"]
    for rd in raw_subdirs:
        os.makedirs(os.path.join(raw_outputs_dir, rd), exist_ok=True)

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Checkpoints
    p4_ckpt_path = "outputs/echo_phase4/checkpoints/echo_best.pth"
    p5_ckpt_path = "outputs/phase5/checkpoints/echo_phase5_best.pth"
    if not os.path.exists(p5_ckpt_path):
        p5_ckpt_path = "outputs/phase5/checkpoints/echo_phase5_last.pth"
    
    p5a_ckpt_path = "outputs/phase5a/checkpoints/echo_phase5a_best.pth"
    if not os.path.exists(p5a_ckpt_path):
        p5a_ckpt_path = "outputs/phase5a/checkpoints/echo_phase5a_last.pth"

    val_csv_path = "outputs/baseline/val_split.csv"
    dataset_root = "D:/kla"

    # --- SANITY CHECKS ---
    print("\nRunning Evaluator Sanity Checks...")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required!")
    if not os.path.exists(p4_ckpt_path): raise FileNotFoundError(f"Phase 4 checkpoint missing: {p4_ckpt_path}")
    if not os.path.exists(p5_ckpt_path): raise FileNotFoundError(f"Phase 5 checkpoint missing: {p5_ckpt_path}")
    if not os.path.exists(p5a_ckpt_path): raise FileNotFoundError(f"Phase 5A checkpoint missing: {p5a_ckpt_path}")
    if not os.path.exists(val_csv_path): raise FileNotFoundError(f"Val CSV missing: {val_csv_path}")

    val_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=val_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    num_samples = len(val_dataset)
    if num_samples != 640: raise ValueError(f"Expected 640 samples, got {num_samples}")
    print("Sanity Checks PASSED: 640 validation samples loaded.")

    # Load Phase 4
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters(): p.requires_grad = False

    # Load Phase 5
    model_p5 = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    p5_chk = torch.load(p5_ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in p5_chk:
        model_p5.load_state_dict(p5_chk["model_state_dict"])
    else:
        model_p5.load_state_dict(p5_chk)
    model_p5.eval()
    for p in model_p5.parameters(): p.requires_grad = False

    # Load Phase 5A
    model_p5a = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    p5a_chk = torch.load(p5a_ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in p5a_chk:
        model_p5a.load_state_dict(p5a_chk["model_state_dict"])
    else:
        model_p5a.load_state_dict(p5a_chk)
    model_p5a.eval()
    for p in model_p5a.parameters(): p.requires_grad = False

    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters(): p.requires_grad = False

    decomp_helper = FrequencyDecompositionModule(cutoff_low=0.15, cutoff_high=0.40).to(device)

    print("\nEvaluating all 640 validation images across Phase 4, Phase 5, Phase 5A...")
    metrics_records = []
    eval_cache = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            in_path = batch["input_path"][0]
            tgt_path = batch["target_path"][0]

            p4_raw, _ = model_p4(b_in)
            p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
            lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)

            p5_hr, _, _, _, _ = model_p5(lr_up, p4_hr)
            p5a_hr, _, _, _, _ = model_p5a(lr_up, p4_hr)

            tgt_lf, tgt_mf, tgt_hf = decomp_helper(b_tgt)
            p4_lf, p4_mf, p4_hf = decomp_helper(p4_hr)
            p5_lf, p5_mf, p5_hf = decomp_helper(p5_hr)
            p5a_lf, p5a_mf, p5a_hf = decomp_helper(p5a_hr)

            # Phase 4 Metrics
            p4_psnr = calculate_psnr(p4_hr, b_tgt)
            p4_ssim = ssim_pytorch(p4_hr, b_tgt).item()
            p4_lpips = ssim_lpips_differentiable(p4_hr, b_tgt, lpips_model).item()
            p4_mae = F.l1_loss(p4_hr, b_tgt).item()
            p4_edge = F.l1_loss(sobel_filter(p4_hr), sobel_filter(b_tgt)).item()
            p4_hf_err = F.l1_loss(p4_hf, tgt_hf).item()

            # Phase 5 Metrics
            p5_psnr = calculate_psnr(p5_hr, b_tgt)
            p5_ssim = ssim_pytorch(p5_hr, b_tgt).item()
            p5_lpips = ssim_lpips_differentiable(p5_hr, b_tgt, lpips_model).item()
            p5_mae = F.l1_loss(p5_hr, b_tgt).item()
            p5_edge = F.l1_loss(sobel_filter(p5_hr), sobel_filter(b_tgt)).item()
            p5_hf_err = F.l1_loss(p5_hf, tgt_hf).item()

            # Phase 5A Metrics
            p5a_psnr = calculate_psnr(p5a_hr, b_tgt)
            p5a_ssim = ssim_pytorch(p5a_hr, b_tgt).item()
            p5a_lpips = ssim_lpips_differentiable(p5a_hr, b_tgt, lpips_model).item()
            p5a_mae = F.l1_loss(p5a_hr, b_tgt).item()
            p5a_edge = F.l1_loss(sobel_filter(p5a_hr), sobel_filter(b_tgt)).item()
            p5a_hf_err = F.l1_loss(p5a_hf, tgt_hf).item()

            sid = f"sample_{idx+1:04d}"

            rec = {
                "sample_id": sid,
                "input_path": in_path,
                "target_path": tgt_path,

                "phase4_psnr": p4_psnr, "phase5_psnr": p5_psnr, "phase5a_psnr": p5a_psnr,
                "phase4_ssim": p4_ssim, "phase5_ssim": p5_ssim, "phase5a_ssim": p5a_ssim,
                "phase4_lpips": p4_lpips, "phase5_lpips": p5_lpips, "phase5a_lpips": p5a_lpips,
                "phase4_mae": p4_mae, "phase5_mae": p5_mae, "phase5a_mae": p5a_mae,
                "phase4_edge": p4_edge, "phase5_edge": p5_edge, "phase5a_edge": p5a_edge,
                "phase4_hf_err": p4_hf_err, "phase5_hf_err": p5_hf_err, "phase5a_hf_err": p5a_hf_err,

                "delta_psnr_p5a_vs_p4": p5a_psnr - p4_psnr,
                "delta_ssim_p5a_vs_p4": p5a_ssim - p4_ssim,
                "delta_lpips_p5a_vs_p4": p5a_lpips - p4_lpips,
                "delta_mae_p5a_vs_p4": p5a_mae - p4_mae,
                "delta_edge_p5a_vs_p4": p5a_edge - p4_edge,
                "delta_hf_err_p5a_vs_p4": p5a_hf_err - p4_hf_err,

                "delta_psnr_p5a_vs_p5": p5a_psnr - p5_psnr,
                "delta_ssim_p5a_vs_p5": p5a_ssim - p5_ssim,
                "delta_lpips_p5a_vs_p5": p5a_lpips - p5_lpips,
                "delta_mae_p5a_vs_p5": p5a_mae - p5_mae,
                "delta_hf_err_p5a_vs_p5": p5a_hf_err - p5_hf_err
            }
            metrics_records.append(rec)

            eval_cache.append({
                "sample_id": sid,
                "input": b_in.cpu(),
                "target": b_tgt.cpu(),
                "phase4": p4_hr.cpu(),
                "phase5": p5_hr.cpu(),
                "phase5a": p5a_hr.cpu(),
                "record": rec
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
                print(f"Evaluated {idx+1}/{num_samples} samples...")

    df = pd.DataFrame(metrics_records)

    # Save Metrics CSV
    csv_path = os.path.join(eval_dir, "phase4_vs_phase5_vs_phase5a_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_path}")

    # Summary JSON
    summary_dict = {
        "phase4": {
            "mean_psnr": float(df["phase4_psnr"].mean()), "median_psnr": float(df["phase4_psnr"].median()),
            "mean_ssim": float(df["phase4_ssim"].mean()), "median_ssim": float(df["phase4_ssim"].median()),
            "mean_lpips": float(df["phase4_lpips"].mean()), "median_lpips": float(df["phase4_lpips"].median()),
            "mean_mae": float(df["phase4_mae"].mean()), "median_mae": float(df["phase4_mae"].median()),
            "mean_edge": float(df["phase4_edge"].mean()), "median_edge": float(df["phase4_edge"].median()),
            "mean_hf_err": float(df["phase4_hf_err"].mean()), "median_hf_err": float(df["phase4_hf_err"].median())
        },
        "phase5": {
            "mean_psnr": float(df["phase5_psnr"].mean()), "median_psnr": float(df["phase5_psnr"].median()),
            "mean_ssim": float(df["phase5_ssim"].mean()), "median_ssim": float(df["phase5_ssim"].median()),
            "mean_lpips": float(df["phase5_lpips"].mean()), "median_lpips": float(df["phase5_lpips"].median()),
            "mean_mae": float(df["phase5_mae"].mean()), "median_mae": float(df["phase5_mae"].median()),
            "mean_edge": float(df["phase5_edge"].mean()), "median_edge": float(df["phase5_edge"].median()),
            "mean_hf_err": float(df["phase5_hf_err"].mean()), "median_hf_err": float(df["phase5_hf_err"].median())
        },
        "phase5a": {
            "mean_psnr": float(df["phase5a_psnr"].mean()), "median_psnr": float(df["phase5a_psnr"].median()),
            "mean_ssim": float(df["phase5a_ssim"].mean()), "median_ssim": float(df["phase5a_ssim"].median()),
            "mean_lpips": float(df["phase5a_lpips"].mean()), "median_lpips": float(df["phase5a_lpips"].median()),
            "mean_mae": float(df["phase5a_mae"].mean()), "median_mae": float(df["phase5a_mae"].median()),
            "mean_edge": float(df["phase5a_edge"].mean()), "median_edge": float(df["phase5a_edge"].median()),
            "mean_hf_err": float(df["phase5a_hf_err"].mean()), "median_hf_err": float(df["phase5a_hf_err"].median())
        },
        "deltas_vs_phase4": {
            "delta_psnr": float(df["phase5a_psnr"].mean() - df["phase4_psnr"].mean()),
            "delta_ssim": float(df["phase5a_ssim"].mean() - df["phase4_ssim"].mean()),
            "delta_lpips": float(df["phase5a_lpips"].mean() - df["phase4_lpips"].mean()),
            "delta_mae": float(df["phase5a_mae"].mean() - df["phase4_mae"].mean()),
            "delta_hf_err": float(df["phase5a_hf_err"].mean() - df["phase4_hf_err"].mean())
        },
        "win_rates_p5a_vs_p4": {
            "psnr_win_pct": float((df["delta_psnr_p5a_vs_p4"] > 0).mean() * 100.0),
            "ssim_win_pct": float((df["delta_ssim_p5a_vs_p4"] > 0).mean() * 100.0),
            "lpips_win_pct": float((df["delta_lpips_p5a_vs_p4"] < 0).mean() * 100.0),
            "mae_win_pct": float((df["delta_mae_p5a_vs_p4"] < 0).mean() * 100.0),
            "hf_err_win_pct": float((df["delta_hf_err_p5a_vs_p4"] < 0).mean() * 100.0)
        }
    }

    summary_json_path = os.path.join(eval_dir, "phase5a_evaluation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Saved summary JSON to: {summary_json_path}")

    # Visual Sample Selection
    p_z = (df["delta_psnr_p5a_vs_p4"] - df["delta_psnr_p5a_vs_p4"].mean()) / (df["delta_psnr_p5a_vs_p4"].std() + 1e-8)
    l_z = (df["delta_lpips_p5a_vs_p4"] - df["delta_lpips_p5a_vs_p4"].mean()) / (df["delta_lpips_p5a_vs_p4"].std() + 1e-8)
    df["comb_score"] = p_z - l_z

    df_best = df.sort_values(by="comb_score", ascending=False).head(5)
    df_worst = df.sort_values(by="comb_score", ascending=True).head(5)
    df_tradeoff = df[(df["delta_lpips_p5a_vs_p4"] < 0) & (df["delta_psnr_p5a_vs_p4"] < 0)].head(5)
    df_hf_wins = df[df["delta_hf_err_p5a_vs_p4"] < 0].sort_values(by="delta_hf_err_p5a_vs_p4").head(5)
    df_regression = df[(df["delta_psnr_p5a_vs_p4"] < -0.1)].head(5)

    visual_categories = [
        ("best", df_best),
        ("worst", df_worst),
        ("tradeoff", df_tradeoff),
        ("hf_wins", df_hf_wins),
        ("regression", df_regression)
    ]

    selected_raw_ids = set()

    for cat_name, cat_df in visual_categories:
        cat_dir = os.path.join(visuals_dir, cat_name)
        for _, row in cat_df.iterrows():
            sid = row["sample_id"]
            selected_raw_ids.add(sid)
            item = next(it for it in eval_cache if it["sample_id"] == sid)

            inp_t = item["input"][0].squeeze().numpy()
            tgt_t = item["target"][0].squeeze().numpy()
            p4_t = item["phase4"][0].squeeze().numpy()
            p5_t = item["phase5"][0].squeeze().numpy()
            p5a_t = item["phase5a"][0].squeeze().numpy()

            err_p4 = np.abs(p4_t - tgt_t)
            err_p5 = np.abs(p5_t - tgt_t)
            err_p5a = np.abs(p5a_t - tgt_t)

            # 8-Panel Comparison Figure
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes[0, 0].imshow(inp_t, cmap="gray"); axes[0, 0].set_title("Input LR"); axes[0, 0].axis("off")
            axes[0, 1].imshow(tgt_t, cmap="gray"); axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")
            axes[0, 2].imshow(p4_t, cmap="gray"); axes[0, 2].set_title(f"Phase 4 Baseline\nPSNR: {row['phase4_psnr']:.2f} | SSIM: {row['phase4_ssim']:.3f}\nLPIPS: {row['phase4_lpips']:.3f}"); axes[0, 2].axis("off")
            axes[0, 3].imshow(p5a_t, cmap="gray"); axes[0, 3].set_title(f"Phase 5A ({sid})\nPSNR: {row['phase5a_psnr']:.2f} | SSIM: {row['phase5a_ssim']:.3f}\nLPIPS: {row['phase5a_lpips']:.3f}"); axes[0, 3].axis("off")

            axes[1, 0].imshow(p5_t, cmap="gray"); axes[1, 0].set_title(f"Phase 5\nPSNR: {row['phase5_psnr']:.2f} | SSIM: {row['phase5_ssim']:.3f}\nLPIPS: {row['phase5_lpips']:.3f}"); axes[1, 0].axis("off")
            axes[1, 1].imshow(err_p4, cmap="magma", vmin=0, vmax=0.15); axes[1, 1].set_title("|Phase 4 - GT|"); axes[1, 1].axis("off")
            axes[1, 2].imshow(err_p5, cmap="magma", vmin=0, vmax=0.15); axes[1, 2].set_title("|Phase 5 - GT|"); axes[1, 2].axis("off")
            axes[1, 3].imshow(err_p5a, cmap="magma", vmin=0, vmax=0.15); axes[1, 3].set_title("|Phase 5A - GT|"); axes[1, 3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(cat_dir, f"{sid}_8panel_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Save Raw PNG Outputs
    for item in eval_cache:
        sid = item["sample_id"]
        if sid in selected_raw_ids:
            save_image_png(item["input"], os.path.join(raw_outputs_dir, "input", f"{sid}.png"))
            save_image_png(item["target"], os.path.join(raw_outputs_dir, "target", f"{sid}.png"))
            save_image_png(item["phase4"], os.path.join(raw_outputs_dir, "phase4", f"{sid}.png"))
            save_image_png(item["phase5"], os.path.join(raw_outputs_dir, "phase5", f"{sid}.png"))
            save_image_png(item["phase5a"], os.path.join(raw_outputs_dir, "phase5a", f"{sid}.png"))

    print("Saved 8-panel Visual Comparisons and Raw Output PNGs.")

    # Generate Distribution Plots
    print("\nGenerating Diagnostic Plots...")
    for m in ["psnr", "ssim", "lpips", "mae", "edge", "hf_err"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[f"phase4_{m}"], bins=30, alpha=0.4, label="Phase 4", color="blue")
        plt.hist(df[f"phase5_{m}"], bins=30, alpha=0.4, label="Phase 5", color="red")
        plt.hist(df[f"phase5a_{m}"], bins=30, alpha=0.4, label="Phase 5A", color="green")
        plt.title(f"{m.upper()} Distribution (Phase 4 vs Phase 5 vs Phase 5A)")
        plt.xlabel(m.upper()); plt.ylabel("Frequency"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, f"{m}_distribution.png"), dpi=150); plt.close()

    # Scientific Decision Rule Criteria
    m_p4_p, m_p5a_p = summary_dict["phase4"]["mean_psnr"], summary_dict["phase5a"]["mean_psnr"]
    m_p4_s, m_p5a_s = summary_dict["phase4"]["mean_ssim"], summary_dict["phase5a"]["mean_ssim"]
    m_p4_l, m_p5a_l = summary_dict["phase4"]["mean_lpips"], summary_dict["phase5a"]["mean_lpips"]
    m_p4_m, m_p5a_m = summary_dict["phase4"]["mean_mae"], summary_dict["phase5a"]["mean_mae"]
    m_p4_hf, m_p5a_hf = summary_dict["phase4"]["mean_hf_err"], summary_dict["phase5a"]["mean_hf_err"]

    m_p5_p = summary_dict["phase5"]["mean_psnr"]

    d_p = m_p5a_p - m_p4_p
    d_s = m_p5a_s - m_p4_s
    d_l = m_p5a_l - m_p4_l
    d_m = m_p5a_m - m_p4_m
    d_hf = m_p5a_hf - m_p4_hf

    if d_p >= 0.0 and d_s >= 0.0 and d_l < 0.0:
        verdict_str = "CASE A: ACCEPT Phase 5A — Phase 5A surpassed Phase 4 across PSNR, SSIM, and LPIPS!"
        champion = "Phase 5A"
    elif d_p > (m_p5_p - m_p4_p) and d_p < 0.0:
        verdict_str = "CASE B: Phase 5A improved over Phase 5, but did not fully beat Phase 4 benchmark."
        champion = "Phase 4"
    else:
        verdict_str = "CASE C: REJECT Phase 5A — Loss modification failed to resolve regression."
        champion = "Phase 4"

    # Report Text
    report_path = os.path.join(eval_dir, "PHASE5A_EVALUATION_REPORT.txt")
    report_text = f"""============================================================
PHASE 4 vs PHASE 5 vs PHASE 5A RESTORATION REPORT
============================================================

Dataset: Validation split (640 samples)

------------------------------------------------------------
1. METRICS SUMMARY (Mean / Median)
------------------------------------------------------------
MODEL        PSNR (dB)   SSIM      LPIPS     MAE       HF ERR
------------------------------------------------------------
Phase 4      {summary_dict['phase4']['mean_psnr']:8.4f}   {summary_dict['phase4']['mean_ssim']:8.4f}    {summary_dict['phase4']['mean_lpips']:8.4f}    {summary_dict['phase4']['mean_mae']:8.4f}    {summary_dict['phase4']['mean_hf_err']:8.4f}
Phase 5      {summary_dict['phase5']['mean_psnr']:8.4f}   {summary_dict['phase5']['mean_ssim']:8.4f}    {summary_dict['phase5']['mean_lpips']:8.4f}    {summary_dict['phase5']['mean_mae']:8.4f}    {summary_dict['phase5']['mean_hf_err']:8.4f}
Phase 5A     {summary_dict['phase5a']['mean_psnr']:8.4f}   {summary_dict['phase5a']['mean_ssim']:8.4f}    {summary_dict['phase5a']['mean_lpips']:8.4f}    {summary_dict['phase5a']['mean_mae']:8.4f}    {summary_dict['phase5a']['mean_hf_err']:8.4f}
------------------------------------------------------------

DELTAS RELATIVE TO PHASE 4:
  ΔPSNR   : {d_p:+8.4f} dB  (Win Rate: {summary_dict['win_rates_p5a_vs_p4']['psnr_win_pct']:.1f}%)
  ΔSSIM   : {d_s:+8.4f}     (Win Rate: {summary_dict['win_rates_p5a_vs_p4']['ssim_win_pct']:.1f}%)
  ΔLPIPS  : {d_l:+8.4f}     (Win Rate: {summary_dict['win_rates_p5a_vs_p4']['lpips_win_pct']:.1f}%)
  ΔMAE    : {d_m:+8.4f}     (Win Rate: {summary_dict['win_rates_p5a_vs_p4']['mae_win_pct']:.1f}%)
  ΔHF ERR : {d_hf:+8.4f}     (Win Rate: {summary_dict['win_rates_p5a_vs_p4']['hf_err_win_pct']:.1f}%)

------------------------------------------------------------
2. EVALUATION DECISION
------------------------------------------------------------
VERDICT : {verdict_str}
CHAMPION: {champion}
============================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}")

    # Console Summary
    print("\n" + "=" * 65)
    print("PHASE 5A EVALUATION COMPLETE")
    print("=" * 65)
    print(f"{'MODEL':<12} {'PSNR (dB)':<10} {'SSIM':<8} {'LPIPS':<8} {'MAE':<8} {'HF ERR':<8}")
    print("-" * 65)
    print(f"{'Phase 4':<12} {summary_dict['phase4']['mean_psnr']:<10.4f} {summary_dict['phase4']['mean_ssim']:<8.4f} {summary_dict['phase4']['mean_lpips']:<8.4f} {summary_dict['phase4']['mean_mae']:<8.4f} {summary_dict['phase4']['mean_hf_err']:<8.4f}")
    print(f"{'Phase 5':<12} {summary_dict['phase5']['mean_psnr']:<10.4f} {summary_dict['phase5']['mean_ssim']:<8.4f} {summary_dict['phase5']['mean_lpips']:<8.4f} {summary_dict['phase5']['mean_mae']:<8.4f} {summary_dict['phase5']['mean_hf_err']:<8.4f}")
    print(f"{'Phase 5A':<12} {summary_dict['phase5a']['mean_psnr']:<10.4f} {summary_dict['phase5a']['mean_ssim']:<8.4f} {summary_dict['phase5a']['mean_lpips']:<8.4f} {summary_dict['phase5a']['mean_mae']:<8.4f} {summary_dict['phase5a']['mean_hf_err']:<8.4f}")
    print("-" * 65)
    print(f"VERDICT : {verdict_str}")
    print(f"CHAMPION: {champion}")
    print("=" * 65)

if __name__ == "__main__":
    main()
