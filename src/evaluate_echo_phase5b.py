"""
Phase 5B Evaluator — Compares Phase 4, Phase 5, Phase 5A, Phase 5B
on the same 640 validation samples using identical metrics.
"""
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

from functools import partial
print = partial(print, flush=True)

from utils import set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule

def save_image_png(tensor_img, path):
    arr = tensor_img.detach().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    plt.imsave(path, arr, cmap='gray')

def load_sfr_model(ckpt_path, device):
    """Load a SpatialFrequencyRestorationNet from checkpoint."""
    model = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64,
        cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    chk = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in chk:
        model.load_state_dict(chk["model_state_dict"])
    else:
        model.load_state_dict(chk)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("PHASE 4 vs PHASE 5 vs PHASE 5A vs PHASE 5B EVALUATION")
    print(f"Device: {device}")
    print("=" * 70)

    # Output Directories
    eval_dir = "outputs/phase5b/evaluation"
    visuals_dir = os.path.join(eval_dir, "visuals")
    raw_outputs_dir = os.path.join(eval_dir, "raw_outputs")
    plots_dir = os.path.join(eval_dir, "plots")

    visual_subdirs = ["best", "worst", "tradeoff", "hf_wins", "regression"]
    for sd in visual_subdirs:
        os.makedirs(os.path.join(visuals_dir, sd), exist_ok=True)

    raw_subdirs = ["input", "target", "phase4", "phase5", "phase5a", "phase5b"]
    for rd in raw_subdirs:
        os.makedirs(os.path.join(raw_outputs_dir, rd), exist_ok=True)

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Checkpoints
    p4_ckpt = "outputs/echo_phase4/checkpoints/echo_best.pth"
    p5_ckpt = "outputs/phase5/checkpoints/echo_phase5_best.pth"
    if not os.path.exists(p5_ckpt):
        p5_ckpt = "outputs/phase5/checkpoints/echo_phase5_last.pth"
    p5a_ckpt = "outputs/phase5a/checkpoints/echo_phase5a_best.pth"
    if not os.path.exists(p5a_ckpt):
        p5a_ckpt = "outputs/phase5a/checkpoints/echo_phase5a_last.pth"
    p5b_ckpt = "outputs/phase5b/checkpoints/echo_phase5b_best.pth"
    if not os.path.exists(p5b_ckpt):
        p5b_ckpt = "outputs/phase5b/checkpoints/echo_phase5b_last.pth"

    val_csv_path = "outputs/baseline/val_split.csv"
    dataset_root = "D:/kla"

    # Sanity checks
    print("\nRunning Evaluator Sanity Checks...")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required!")
    for tag, path in [("Phase 4", p4_ckpt), ("Phase 5", p5_ckpt), ("Phase 5A", p5a_ckpt), ("Phase 5B", p5b_ckpt)]:
        if not os.path.exists(path): raise FileNotFoundError(f"{tag} checkpoint missing: {path}")
        print(f"  {tag} checkpoint: {path} — OK")
    if not os.path.exists(val_csv_path): raise FileNotFoundError(f"Val CSV missing: {val_csv_path}")

    val_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=val_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    num_samples = len(val_dataset)
    if num_samples != 640: raise ValueError(f"Expected 640 samples, got {num_samples}")
    print(f"Sanity Checks PASSED: {num_samples} validation samples loaded.\n")

    # Load Phase 4
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_ckpt, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters(): p.requires_grad = False

    # Load Phase 5, 5A, 5B
    model_p5 = load_sfr_model(p5_ckpt, device)
    model_p5a = load_sfr_model(p5a_ckpt, device)
    model_p5b = load_sfr_model(p5b_ckpt, device)

    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters(): p.requires_grad = False

    decomp_helper = FrequencyDecompositionModule(cutoff_low=0.15, cutoff_high=0.40).to(device)

    print("Evaluating all 640 validation images across Phase 4, 5, 5A, 5B...")
    metrics_records = []
    eval_cache = []

    def compute_metrics(pred_hr, target, sobel_f, lpips_m, decomp, tgt_hf):
        psnr = calculate_psnr(pred_hr, target)
        ssim_val = ssim_pytorch(pred_hr, target).item()
        lpips_val = ssim_lpips_differentiable(pred_hr, target, lpips_m).item()
        mae_val = F.l1_loss(pred_hr, target).item()
        edge_val = F.l1_loss(sobel_f(pred_hr), sobel_f(target)).item()
        p_lf, p_mf, p_hf = decomp(pred_hr)
        hf_err_val = F.l1_loss(p_hf, tgt_hf).item()
        return psnr, ssim_val, lpips_val, mae_val, edge_val, hf_err_val

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)

            p4_raw, _ = model_p4(b_in)
            p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
            lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)

            p5_hr, _, _, _, _ = model_p5(lr_up, p4_hr)
            p5a_hr, _, _, _, _ = model_p5a(lr_up, p4_hr)
            p5b_hr, _, _, _, _ = model_p5b(lr_up, p4_hr)

            tgt_lf, tgt_mf, tgt_hf = decomp_helper(b_tgt)

            m4 = compute_metrics(p4_hr, b_tgt, sobel_filter, lpips_model, decomp_helper, tgt_hf)
            m5 = compute_metrics(p5_hr, b_tgt, sobel_filter, lpips_model, decomp_helper, tgt_hf)
            m5a = compute_metrics(p5a_hr, b_tgt, sobel_filter, lpips_model, decomp_helper, tgt_hf)
            m5b = compute_metrics(p5b_hr, b_tgt, sobel_filter, lpips_model, decomp_helper, tgt_hf)

            sid = f"sample_{idx+1:04d}"

            rec = {
                "sample_id": sid,
                "input_path": batch["input_path"][0],
                "target_path": batch["target_path"][0],
            }
            for label, vals in [("phase4", m4), ("phase5", m5), ("phase5a", m5a), ("phase5b", m5b)]:
                rec[f"{label}_psnr"] = vals[0]
                rec[f"{label}_ssim"] = vals[1]
                rec[f"{label}_lpips"] = vals[2]
                rec[f"{label}_mae"] = vals[3]
                rec[f"{label}_edge"] = vals[4]
                rec[f"{label}_hf_err"] = vals[5]

            # Deltas vs Phase 4
            for label, vals in [("phase5b", m5b)]:
                rec[f"delta_psnr_{label}_vs_p4"] = vals[0] - m4[0]
                rec[f"delta_ssim_{label}_vs_p4"] = vals[1] - m4[1]
                rec[f"delta_lpips_{label}_vs_p4"] = vals[2] - m4[2]
                rec[f"delta_mae_{label}_vs_p4"] = vals[3] - m4[3]
                rec[f"delta_hf_err_{label}_vs_p4"] = vals[5] - m4[5]

            metrics_records.append(rec)

            eval_cache.append({
                "sample_id": sid,
                "input": b_in.cpu(),
                "target": b_tgt.cpu(),
                "phase4": p4_hr.cpu(),
                "phase5": p5_hr.cpu(),
                "phase5a": p5a_hr.cpu(),
                "phase5b": p5b_hr.cpu(),
                "record": rec
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
                print(f"  Evaluated {idx+1}/{num_samples} samples...")

    df = pd.DataFrame(metrics_records)

    # Save Metrics CSV
    csv_path = os.path.join(eval_dir, "phase4_vs_phase5_vs_phase5a_vs_phase5b_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_path}")

    # Build Summary
    phases = ["phase4", "phase5", "phase5a", "phase5b"]
    summary_dict = {}
    for ph in phases:
        summary_dict[ph] = {}
        for m in ["psnr", "ssim", "lpips", "mae", "edge", "hf_err"]:
            col = f"{ph}_{m}"
            summary_dict[ph][f"mean_{m}"] = float(df[col].mean())
            summary_dict[ph][f"median_{m}"] = float(df[col].median())

    # Deltas
    summary_dict["deltas_5b_vs_p4"] = {
        "delta_psnr": summary_dict["phase5b"]["mean_psnr"] - summary_dict["phase4"]["mean_psnr"],
        "delta_ssim": summary_dict["phase5b"]["mean_ssim"] - summary_dict["phase4"]["mean_ssim"],
        "delta_lpips": summary_dict["phase5b"]["mean_lpips"] - summary_dict["phase4"]["mean_lpips"],
        "delta_mae": summary_dict["phase5b"]["mean_mae"] - summary_dict["phase4"]["mean_mae"],
        "delta_hf_err": summary_dict["phase5b"]["mean_hf_err"] - summary_dict["phase4"]["mean_hf_err"],
    }

    # Win rates P5B vs P4
    summary_dict["win_rates_p5b_vs_p4"] = {
        "psnr_win_pct": float((df["delta_psnr_phase5b_vs_p4"] > 0).mean() * 100.0),
        "ssim_win_pct": float((df["delta_ssim_phase5b_vs_p4"] > 0).mean() * 100.0),
        "lpips_win_pct": float((df["delta_lpips_phase5b_vs_p4"] < 0).mean() * 100.0),
        "mae_win_pct": float((df["delta_mae_phase5b_vs_p4"] < 0).mean() * 100.0),
        "hf_err_win_pct": float((df["delta_hf_err_phase5b_vs_p4"] < 0).mean() * 100.0),
    }

    summary_json_path = os.path.join(eval_dir, "phase5b_evaluation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4)
    print(f"Saved summary JSON to: {summary_json_path}")

    # Visual Sample Selection (P5B vs P4 focus)
    p_z = (df["delta_psnr_phase5b_vs_p4"] - df["delta_psnr_phase5b_vs_p4"].mean()) / (df["delta_psnr_phase5b_vs_p4"].std() + 1e-8)
    l_z = (df["delta_lpips_phase5b_vs_p4"] - df["delta_lpips_phase5b_vs_p4"].mean()) / (df["delta_lpips_phase5b_vs_p4"].std() + 1e-8)
    df["comb_score"] = p_z - l_z

    df_best = df.sort_values(by="comb_score", ascending=False).head(5)
    df_worst = df.sort_values(by="comb_score", ascending=True).head(5)
    df_tradeoff = df[(df["delta_lpips_phase5b_vs_p4"] < 0) & (df["delta_psnr_phase5b_vs_p4"] < 0)].head(5)
    df_hf_wins = df[df["delta_hf_err_phase5b_vs_p4"] < 0].sort_values(by="delta_hf_err_phase5b_vs_p4").head(5)
    df_regression = df[(df["delta_psnr_phase5b_vs_p4"] < -0.1)].head(5)

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
            p5b_t = item["phase5b"][0].squeeze().numpy()

            err_p4 = np.abs(p4_t - tgt_t)
            err_p5a = np.abs(p5a_t - tgt_t)
            err_p5b = np.abs(p5b_t - tgt_t)

            # 8-Panel: Input, GT, P4, P5B (top); P5, P5A, |P4-GT|, |P5B-GT| (bottom)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes[0, 0].imshow(inp_t, cmap="gray"); axes[0, 0].set_title("Input LR"); axes[0, 0].axis("off")
            axes[0, 1].imshow(tgt_t, cmap="gray"); axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")
            axes[0, 2].imshow(p4_t, cmap="gray"); axes[0, 2].set_title(f"Phase 4\nPSNR:{row['phase4_psnr']:.2f} SSIM:{row['phase4_ssim']:.3f}\nLPIPS:{row['phase4_lpips']:.3f}"); axes[0, 2].axis("off")
            axes[0, 3].imshow(p5b_t, cmap="gray"); axes[0, 3].set_title(f"Phase 5B ({sid})\nPSNR:{row['phase5b_psnr']:.2f} SSIM:{row['phase5b_ssim']:.3f}\nLPIPS:{row['phase5b_lpips']:.3f}"); axes[0, 3].axis("off")

            axes[1, 0].imshow(p5_t, cmap="gray"); axes[1, 0].set_title(f"Phase 5\nPSNR:{row['phase5_psnr']:.2f} SSIM:{row['phase5_ssim']:.3f}\nLPIPS:{row['phase5_lpips']:.3f}"); axes[1, 0].axis("off")
            axes[1, 1].imshow(p5a_t, cmap="gray"); axes[1, 1].set_title(f"Phase 5A\nPSNR:{row['phase5a_psnr']:.2f} SSIM:{row['phase5a_ssim']:.3f}\nLPIPS:{row['phase5a_lpips']:.3f}"); axes[1, 1].axis("off")
            axes[1, 2].imshow(err_p4, cmap="magma", vmin=0, vmax=0.15); axes[1, 2].set_title("|Phase 4 - GT|"); axes[1, 2].axis("off")
            axes[1, 3].imshow(err_p5b, cmap="magma", vmin=0, vmax=0.15); axes[1, 3].set_title("|Phase 5B - GT|"); axes[1, 3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(cat_dir, f"{sid}_8panel_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # Save Raw PNG Outputs
    for item in eval_cache:
        sid = item["sample_id"]
        if sid in selected_raw_ids:
            for label in ["input", "target", "phase4", "phase5", "phase5a", "phase5b"]:
                save_image_png(item[label], os.path.join(raw_outputs_dir, label, f"{sid}.png"))

    print("Saved 8-panel Visual Comparisons and Raw Output PNGs.")

    # Generate Distribution Plots
    print("\nGenerating Diagnostic Plots...")
    for m in ["psnr", "ssim", "lpips", "mae", "edge", "hf_err"]:
        plt.figure(figsize=(9, 5))
        colors = {"phase4": "blue", "phase5": "red", "phase5a": "orange", "phase5b": "green"}
        for ph, color in colors.items():
            plt.hist(df[f"{ph}_{m}"], bins=30, alpha=0.35, label=ph.replace("phase", "Phase ").replace("5a", "5A").replace("5b", "5B"), color=color)
        plt.title(f"{m.upper()} Distribution (Phase 4 / 5 / 5A / 5B)")
        plt.xlabel(m.upper()); plt.ylabel("Frequency"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, f"{m}_distribution.png"), dpi=150); plt.close()

    # Extract summary values
    d = summary_dict["deltas_5b_vs_p4"]
    d_p, d_s, d_l, d_m, d_hf = d["delta_psnr"], d["delta_ssim"], d["delta_lpips"], d["delta_mae"], d["delta_hf_err"]
    wr = summary_dict["win_rates_p5b_vs_p4"]

    # Decision logic
    if d_p >= 0.0 and d_s >= 0.0 and d_l <= 0.0:
        verdict_str = "CASE A — STRONG SUCCESS: Phase 5B improved or maintained all fidelity metrics AND improved perceptual quality."
        champion = "Phase 5B"
    elif d_p >= -0.02 and d_s >= 0.0 and d_l < -0.01:
        verdict_str = "CASE A (marginal) — Phase 5B maintained fidelity with meaningful perceptual improvement."
        champion = "Phase 5B"
    elif d_l < 0.0 and (d_p < -0.05 or d_hf > 0.0005):
        verdict_str = "CASE C — PERCEPTUAL GAIN BUT FIDELITY REGRESSION: LPIPS improved but PSNR/HF degraded."
        champion = "Phase 4"
    elif abs(d_p) < 0.02 and abs(d_l) < 0.005:
        verdict_str = "CASE D — NO MEANINGFUL CHANGE: Metrics effectively unchanged."
        champion = "Phase 4"
    elif d_p > (summary_dict["phase5"]["mean_psnr"] - summary_dict["phase4"]["mean_psnr"]):
        verdict_str = "CASE B — PARTIAL SUCCESS: Phase 5B improved over Phase 5 but did not fully beat Phase 4."
        champion = "Phase 4"
    elif d_p < (summary_dict["phase5a"]["mean_psnr"] - summary_dict["phase4"]["mean_psnr"]):
        verdict_str = "CASE E — WORSE THAN PHASE 5A: Phase 5B is clearly worse overall."
        champion = "Phase 5A"
    else:
        verdict_str = "CASE B — PARTIAL SUCCESS: Phase 5B shows mixed results."
        champion = "Phase 4"

    # Report
    report_path = os.path.join(eval_dir, "PHASE5B_EVALUATION_REPORT.txt")
    s = summary_dict
    report_text = f"""============================================================
PHASE 4 vs PHASE 5 vs PHASE 5A vs PHASE 5B EVALUATION REPORT
============================================================

Dataset: Validation split (640 samples)

------------------------------------------------------------
METRICS SUMMARY (Mean)
------------------------------------------------------------
MODEL        PSNR (dB)   SSIM      LPIPS     MAE       HF ERR
------------------------------------------------------------
Phase 4      {s['phase4']['mean_psnr']:8.4f}   {s['phase4']['mean_ssim']:8.4f}    {s['phase4']['mean_lpips']:8.4f}    {s['phase4']['mean_mae']:8.4f}    {s['phase4']['mean_hf_err']:8.6f}
Phase 5      {s['phase5']['mean_psnr']:8.4f}   {s['phase5']['mean_ssim']:8.4f}    {s['phase5']['mean_lpips']:8.4f}    {s['phase5']['mean_mae']:8.4f}    {s['phase5']['mean_hf_err']:8.6f}
Phase 5A     {s['phase5a']['mean_psnr']:8.4f}   {s['phase5a']['mean_ssim']:8.4f}    {s['phase5a']['mean_lpips']:8.4f}    {s['phase5a']['mean_mae']:8.4f}    {s['phase5a']['mean_hf_err']:8.6f}
Phase 5B     {s['phase5b']['mean_psnr']:8.4f}   {s['phase5b']['mean_ssim']:8.4f}    {s['phase5b']['mean_lpips']:8.4f}    {s['phase5b']['mean_mae']:8.4f}    {s['phase5b']['mean_hf_err']:8.6f}
------------------------------------------------------------

PHASE 5B DELTAS vs PHASE 4:
  ΔPSNR   : {d_p:+8.4f} dB  (Win Rate: {wr['psnr_win_pct']:.1f}%)
  ΔSSIM   : {d_s:+8.4f}     (Win Rate: {wr['ssim_win_pct']:.1f}%)
  ΔLPIPS  : {d_l:+8.4f}     (Win Rate: {wr['lpips_win_pct']:.1f}%)
  ΔMAE    : {d_m:+8.4f}     (Win Rate: {wr['mae_win_pct']:.1f}%)
  ΔHF ERR : {d_hf:+8.6f}     (Win Rate: {wr['hf_err_win_pct']:.1f}%)

------------------------------------------------------------
VERDICT : {verdict_str}
CHAMPION: {champion}
============================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved evaluation report to: {report_path}")

    # Console Summary
    print("\n" + "=" * 70)
    print("PHASE 5B EVALUATION COMPLETE")
    print("=" * 70)
    print(f"{'MODEL':<12} {'PSNR (dB)':<10} {'SSIM':<8} {'LPIPS':<8} {'MAE':<8} {'HF ERR':<10}")
    print("-" * 70)
    for ph, label in [("phase4", "Phase 4"), ("phase5", "Phase 5"), ("phase5a", "Phase 5A"), ("phase5b", "Phase 5B")]:
        print(f"{label:<12} {s[ph]['mean_psnr']:<10.4f} {s[ph]['mean_ssim']:<8.4f} {s[ph]['mean_lpips']:<8.4f} {s[ph]['mean_mae']:<8.4f} {s[ph]['mean_hf_err']:<10.6f}")
    print("-" * 70)
    print(f"VERDICT : {verdict_str}")
    print(f"CHAMPION: {champion}")
    print("=" * 70)

if __name__ == "__main__":
    main()
