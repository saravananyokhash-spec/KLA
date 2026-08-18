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
    print("PHASE 4 vs PHASE 5 STANDALONE EVALUATOR")
    print(f"Device: {device}")
    print("=" * 60)

    # Output Directories
    eval_dir = "outputs/phase5/evaluation"
    visuals_dir = os.path.join(eval_dir, "visuals")
    raw_outputs_dir = os.path.join(eval_dir, "raw_outputs")
    analysis_dir = os.path.join(eval_dir, "analysis")
    plots_dir = os.path.join(eval_dir, "plots")

    visual_subdirs = ["best", "worst", "tradeoff", "hf_wins", "regression"]
    for sd in visual_subdirs:
        os.makedirs(os.path.join(visuals_dir, sd), exist_ok=True)

    raw_subdirs = ["input", "target", "phase4", "phase5", "high_freq"]
    for rd in raw_subdirs:
        os.makedirs(os.path.join(raw_outputs_dir, rd), exist_ok=True)

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Checkpoints
    p4_ckpt_path = "outputs/echo_phase4/checkpoints/echo_best.pth"
    p5_ckpt_dir = "outputs/phase5/checkpoints"
    p5_ckpt_path = os.path.join(p5_ckpt_dir, "echo_phase5_best.pth")
    if not os.path.exists(p5_ckpt_path):
        p5_ckpt_path = os.path.join(p5_ckpt_dir, "echo_phase5_last.pth")

    val_csv_path = "outputs/baseline/val_split.csv"
    dataset_root = "D:/kla"

    # --- SANITY CHECKS ---
    print("\nRunning Evaluator Sanity Checks...")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required!")
    if not os.path.exists(p4_ckpt_path): raise FileNotFoundError(f"Phase 4 checkpoint missing: {p4_ckpt_path}")
    if not os.path.exists(p5_ckpt_path): raise FileNotFoundError(f"Phase 5 checkpoint missing: {p5_ckpt_path}")
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

    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters(): p.requires_grad = False

    decomp_helper = FrequencyDecompositionModule(cutoff_low=0.15, cutoff_high=0.40).to(device)

    print("\nEvaluating all 640 validation images...")
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

            p5_hr, p5_lf, p5_mf, p5_hf, _ = model_p5(lr_up, p4_hr)
            tgt_lf, tgt_mf, tgt_hf = decomp_helper(b_tgt)
            p4_lf, p4_mf, p4_hf = decomp_helper(p4_hr)

            # Phase 4 Metrics
            p4_psnr = calculate_psnr(p4_hr, b_tgt)
            p4_ssim = ssim_pytorch(p4_hr, b_tgt).item()
            p4_lpips = ssim_lpips_differentiable(p4_hr, b_tgt, lpips_model).item()
            p4_mae = F.l1_loss(p4_hr, b_tgt).item()
            p4_edge = F.l1_loss(sobel_filter(p4_hr), sobel_filter(b_tgt)).item()

            p4_lf_err = F.l1_loss(p4_lf, tgt_lf).item()
            p4_mf_err = F.l1_loss(p4_mf, tgt_mf).item()
            p4_hf_err = F.l1_loss(p4_hf, tgt_hf).item()

            # Phase 5 Metrics
            p5_psnr = calculate_psnr(p5_hr, b_tgt)
            p5_ssim = ssim_pytorch(p5_hr, b_tgt).item()
            p5_lpips = ssim_lpips_differentiable(p5_hr, b_tgt, lpips_model).item()
            p5_mae = F.l1_loss(p5_hr, b_tgt).item()
            p5_edge = F.l1_loss(sobel_filter(p5_hr), sobel_filter(b_tgt)).item()

            p5_lf_err = F.l1_loss(p5_lf, tgt_lf).item()
            p5_mf_err = F.l1_loss(p5_mf, tgt_mf).item()
            p5_hf_err = F.l1_loss(p5_hf, tgt_hf).item()

            # Wins
            wins = 0
            if p5_psnr > p4_psnr: wins += 1
            if p5_ssim > p4_ssim: wins += 1
            if p5_lpips < p4_lpips: wins += 1
            if p5_mae < p4_mae: wins += 1
            if p5_edge < p4_edge: wins += 1

            sid = f"sample_{idx+1:04d}"

            rec = {
                "sample_id": sid,
                "input_path": in_path,
                "target_path": tgt_path,
                "phase4_psnr": p4_psnr, "phase5_psnr": p5_psnr,
                "phase4_ssim": p4_ssim, "phase5_ssim": p5_ssim,
                "phase4_lpips": p4_lpips, "phase5_lpips": p5_lpips,
                "phase4_mae": p4_mae, "phase5_mae": p5_mae,
                "phase4_edge": p4_edge, "phase5_edge": p5_edge,
                "phase4_lf_err": p4_lf_err, "phase5_lf_err": p5_lf_err,
                "phase4_mf_err": p4_mf_err, "phase5_mf_err": p5_mf_err,
                "phase4_hf_err": p4_hf_err, "phase5_hf_err": p5_hf_err,
                "delta_psnr": p5_psnr - p4_psnr,
                "delta_ssim": p5_ssim - p4_ssim,
                "delta_lpips": p5_lpips - p4_lpips,
                "delta_mae": p5_mae - p4_mae,
                "delta_edge": p5_edge - p4_edge,
                "delta_hf_err": p5_hf_err - p4_hf_err,
                "phase5_wins": wins
            }
            metrics_records.append(rec)

            eval_cache.append({
                "sample_id": sid,
                "input": b_in.cpu(),
                "target": b_tgt.cpu(),
                "phase4": p4_hr.cpu(),
                "phase5": p5_hr.cpu(),
                "p5_hf": p5_hf.cpu(),
                "tgt_hf": tgt_hf.cpu(),
                "record": rec
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
                print(f"Evaluated {idx+1}/{num_samples} samples...")

    df = pd.DataFrame(metrics_records)

    # Save Metrics CSV
    csv_path = os.path.join(eval_dir, "phase4_vs_phase5_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_path}")

    # Summary JSON & Win Rates
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
        "win_rates": {
            "psnr_win_pct": float((df["delta_psnr"] > 0).mean() * 100.0),
            "ssim_win_pct": float((df["delta_ssim"] > 0).mean() * 100.0),
            "lpips_win_pct": float((df["delta_lpips"] < 0).mean() * 100.0),
            "mae_win_pct": float((df["delta_mae"] < 0).mean() * 100.0),
            "edge_win_pct": float((df["delta_edge"] < 0).mean() * 100.0),
            "hf_err_win_pct": float((df["delta_hf_err"] < 0).mean() * 100.0)
        }
    }

    summary_json_path = os.path.join(eval_dir, "phase5_evaluation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Saved summary JSON to: {summary_json_path}")

    # Sample Selection for 8-Panel Visuals
    p_z = (df["delta_psnr"] - df["delta_psnr"].mean()) / (df["delta_psnr"].std() + 1e-8)
    l_z = (df["delta_lpips"] - df["delta_lpips"].mean()) / (df["delta_lpips"].std() + 1e-8)
    df["comb_score"] = p_z - l_z

    df_best = df.sort_values(by="comb_score", ascending=False).head(5)
    df_worst = df.sort_values(by="comb_score", ascending=True).head(5)
    df_tradeoff = df[(df["delta_lpips"] < 0) & (df["delta_psnr"] < 0)].head(5)
    df_hf_wins = df[df["delta_hf_err"] < 0].sort_values(by="delta_hf_err").head(5)
    df_regression = df[(df["delta_psnr"] < -0.1) & (df["delta_lpips"] > 0)].head(5)

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
            p5_hf_t = item["p5_hf"][0].squeeze().numpy()
            tgt_hf_t = item["tgt_hf"][0].squeeze().numpy()

            err_p4 = np.abs(p4_t - tgt_t)
            err_p5 = np.abs(p5_t - tgt_t)

            # FFT Spectrum calculation
            fft_p5 = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(p5_t))))
            fft_tgt = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(tgt_t))))
            fft_diff = np.abs(fft_p5 - fft_tgt)

            # 8-Panel Figure
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes[0, 0].imshow(inp_t, cmap="gray"); axes[0, 0].set_title("Input LR"); axes[0, 0].axis("off")
            axes[0, 1].imshow(tgt_t, cmap="gray"); axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")
            axes[0, 2].imshow(p4_t, cmap="gray"); axes[0, 2].set_title(f"Phase 4\nPSNR: {row['phase4_psnr']:.2f} | SSIM: {row['phase4_ssim']:.3f}\nLPIPS: {row['phase4_lpips']:.3f}"); axes[0, 2].axis("off")
            axes[0, 3].imshow(p5_t, cmap="gray"); axes[0, 3].set_title(f"Phase 5 ({sid})\nPSNR: {row['phase5_psnr']:.2f} | SSIM: {row['phase5_ssim']:.3f}\nLPIPS: {row['phase5_lpips']:.3f}"); axes[0, 3].axis("off")

            axes[1, 0].imshow(err_p4, cmap="magma", vmin=0, vmax=0.15); axes[1, 0].set_title("|Phase 4 - GT|"); axes[1, 0].axis("off")
            axes[1, 1].imshow(err_p5, cmap="magma", vmin=0, vmax=0.15); axes[1, 1].set_title("|Phase 5 - GT|"); axes[1, 1].axis("off")
            axes[1, 2].imshow(np.abs(p5_hf_t - tgt_hf_t), cmap="plasma", vmin=0, vmax=0.10); axes[1, 2].set_title("High-Freq Error"); axes[1, 2].axis("off")
            axes[1, 3].imshow(fft_diff, cmap="inferno"); axes[1, 3].set_title("FFT Spectrum Error"); axes[1, 3].axis("off")

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
            save_image_png(item["p5_hf"], os.path.join(raw_outputs_dir, "high_freq", f"{sid}.png"))

    print("Saved 8-panel Visual Comparisons and Raw Output PNGs.")

    # Generate Distribution Plots
    print("\nGenerating Diagnostic Plots...")
    for m in ["psnr", "ssim", "lpips", "mae", "edge", "hf_err"]:
        plt.figure(figsize=(8, 5))
        plt.hist(df[f"phase4_{m}"], bins=30, alpha=0.5, label="Phase 4", color="blue")
        plt.hist(df[f"phase5_{m}"], bins=30, alpha=0.5, label="Phase 5", color="green")
        plt.title(f"{m.upper()} Distribution (Phase 4 vs Phase 5)")
        plt.xlabel(m.upper()); plt.ylabel("Frequency"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, f"{m}_distribution.png"), dpi=150); plt.close()

    # Verdict Decision
    m_p4_p, m_p5_p = summary_dict["phase4"]["mean_psnr"], summary_dict["phase5"]["mean_psnr"]
    m_p4_s, m_p5_s = summary_dict["phase4"]["mean_ssim"], summary_dict["phase5"]["mean_ssim"]
    m_p4_l, m_p5_l = summary_dict["phase4"]["mean_lpips"], summary_dict["phase5"]["mean_lpips"]

    d_p = m_p5_p - m_p4_p
    d_s = m_p5_s - m_p4_s
    d_l = m_p5_l - m_p4_l

    if d_p >= 0.0 and d_s >= 0.0 and d_l < 0.0:
        verdict_str = "STRONG SUCCESS: Phase 5 improved LPIPS while preserving/improving Phase 4 PSNR and SSIM!"
        champion = "Phase 5"
    elif d_p >= -0.05 and d_l <= -0.04:
        verdict_str = "BALANCED SUCCESS: Phase 5 achieved major perceptual gains with minimal PSNR trade-off."
        champion = "Phase 5"
    elif d_l < -0.05 and d_p < -0.10:
        verdict_str = "PERCEPTUAL SUCCESS ONLY: Phase 5 improved LPIPS but suffered PSNR degradation."
        champion = "Phase 4"
    else:
        verdict_str = "REJECTED: Phase 5 did not surpass Phase 4 benchmark."
        champion = "Phase 4"

    # Final Text Report (UTF-8)
    report_path = os.path.join(eval_dir, "PHASE5_EVALUATION_REPORT.txt")
    report_text = f"""============================================================
PHASE 4 vs PHASE 5 MULTI-SCALE RESTORATION REPORT
============================================================

Dataset: Validation split (640 samples)

------------------------------------------------------------
1. METRICS SUMMARY (Mean / Median)
------------------------------------------------------------
Metric       Phase 4 Champion     Phase 5 Model        Delta (P5 - P4)
------------------------------------------------------------
PSNR (dB)    {summary_dict['phase4']['mean_psnr']:8.4f} / {summary_dict['phase4']['median_psnr']:8.4f}  {summary_dict['phase5']['mean_psnr']:8.4f} / {summary_dict['phase5']['median_psnr']:8.4f}  {d_p:+8.4f} (Win: {summary_dict['win_rates']['psnr_win_pct']:.1f}%)
SSIM         {summary_dict['phase4']['mean_ssim']:8.4f} / {summary_dict['phase4']['median_ssim']:8.4f}  {summary_dict['phase5']['mean_ssim']:8.4f} / {summary_dict['phase5']['median_ssim']:8.4f}  {d_s:+8.4f} (Win: {summary_dict['win_rates']['ssim_win_pct']:.1f}%)
LPIPS        {summary_dict['phase4']['mean_lpips']:8.4f} / {summary_dict['phase4']['median_lpips']:8.4f}  {summary_dict['phase5']['mean_lpips']:8.4f} / {summary_dict['phase5']['median_lpips']:8.4f}  {d_l:+8.4f} (Win: {summary_dict['win_rates']['lpips_win_pct']:.1f}%)
MAE          {summary_dict['phase4']['mean_mae']:8.4f} / {summary_dict['phase4']['median_mae']:8.4f}  {summary_dict['phase5']['mean_mae']:8.4f} / {summary_dict['phase5']['median_mae']:8.4f}  {summary_dict['phase5']['mean_mae'] - summary_dict['phase4']['mean_mae']:+8.4f} (Win: {summary_dict['win_rates']['mae_win_pct']:.1f}%)
Edge Error   {summary_dict['phase4']['mean_edge']:8.4f} / {summary_dict['phase4']['median_edge']:8.4f}  {summary_dict['phase5']['mean_edge']:8.4f} / {summary_dict['phase5']['median_edge']:8.4f}  {summary_dict['phase5']['mean_edge'] - summary_dict['phase4']['mean_edge']:+8.4f} (Win: {summary_dict['win_rates']['edge_win_pct']:.1f}%)
High-Freq    {summary_dict['phase4']['mean_hf_err']:8.4f} / {summary_dict['phase4']['median_hf_err']:8.4f}  {summary_dict['phase5']['mean_hf_err']:8.4f} / {summary_dict['phase5']['median_hf_err']:8.4f}  {summary_dict['phase5']['mean_hf_err'] - summary_dict['phase4']['mean_hf_err']:+8.4f} (Win: {summary_dict['win_rates']['hf_err_win_pct']:.1f}%)

------------------------------------------------------------
2. EVALUATION QUESTIONS
------------------------------------------------------------
1. Did Phase 5 beat Phase 4 in PSNR?        {"YES" if d_p > 0 else "NO"} ({d_p:+8.4f} dB)
2. Did Phase 5 beat Phase 4 in SSIM?        {"YES" if d_s > 0 else "NO"} ({d_s:+8.4f})
3. Did Phase 5 beat Phase 4 in LPIPS?       {"YES" if d_l < 0 else "NO"} ({d_l:+8.4f})
4. Did Phase 5 beat Phase 4 in MAE?         {"YES" if summary_dict['phase5']['mean_mae'] < summary_dict['phase4']['mean_mae'] else "NO"}
5. Did Phase 5 recover High-Freq details?  {"YES" if summary_dict['phase5']['mean_hf_err'] < summary_dict['phase4']['mean_hf_err'] else "NO"}

------------------------------------------------------------
3. FINAL VERDICT & CHAMPION
------------------------------------------------------------
VERDICT : {verdict_str}
CHAMPION: {champion}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}")

    # Console Summary
    print("\n" + "=" * 65)
    print("PHASE 5 STANDALONE EVALUATION COMPLETE")
    print("=" * 65)
    print(f"{'MODEL':<12} {'PSNR (dB)':<10} {'SSIM':<8} {'LPIPS':<8} {'MAE':<8} {'HF ERR':<8}")
    print("-" * 65)
    print(f"{'Phase 4':<12} {summary_dict['phase4']['mean_psnr']:<10.4f} {summary_dict['phase4']['mean_ssim']:<8.4f} {summary_dict['phase4']['mean_lpips']:<8.4f} {summary_dict['phase4']['mean_mae']:<8.4f} {summary_dict['phase4']['mean_hf_err']:<8.4f}")
    print(f"{'Phase 5':<12} {summary_dict['phase5']['mean_psnr']:<10.4f} {summary_dict['phase5']['mean_ssim']:<8.4f} {summary_dict['phase5']['mean_lpips']:<8.4f} {summary_dict['phase5']['mean_mae']:<8.4f} {summary_dict['phase5']['mean_hf_err']:<8.4f}")
    print("-" * 65)
    print(f"VERDICT : {verdict_str}")
    print(f"CHAMPION: {champion}")
    print("=" * 65)

if __name__ == "__main__":
    main()
