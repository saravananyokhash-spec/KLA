"""Phase 11 Evaluation Script
Compares Phase 11 against Phase 5B, Phase 7, Phase 8, Phase 9, and Phase 10 (if available).
Runs original validation and official KLA degradation stress tests:
- Gaussian noise (low, medium, high)
- Speckle noise (low, medium, high)
- Downsampling (low, medium, high)
- Combined degradations:
  - Gaussian + Speckle
  - Gaussian + Downsampling
  - Speckle + Downsampling
  - Gaussian + Speckle + Downsampling
Generates line plots, visual comparison panels, win rates, and final report.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import lpips
import matplotlib.pyplot as plt

from functools import partial
print = partial(print, flush=True)

from dataset import KLADataset
from echo_model import BaselineECHOModel
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule
from phase10_model import ConditionedSpatialFrequencyRestorationNet
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase10 import calculate_psnr
from utils import set_seed

# Teacher paths and outputs setup
P4_CKPT = "outputs/echo_phase4/checkpoints/echo_best.pth"
P5B_CKPT = "outputs/phase5b/checkpoints/echo_phase5b_best.pth"
P7_CKPT = "outputs/phase7_finetune/exp_b/checkpoints/echo_phase7_ft_best.pth"
P8_CKPT = "outputs/phase8_hybrid/checkpoints/echo_phase8_hybrid_best.pth"
P9_CKPT = "outputs/phase9_targeted/checkpoints/echo_phase9_best.pth"
P10_CKPT = "outputs/phase10_degradation_aware/checkpoints/echo_phase10_best.pth"
P11_CKPT = "outputs/phase11_degradation_aware/checkpoints/echo_phase11_best.pth"

OUT_ROOT = "outputs/phase11_degradation_aware"
VAL_CSV = "outputs/baseline/val_split.csv"
DATASET_ROOT = "D:/kla"

# Degradation conditions for evaluation
CONDITIONS = [
    "gaussian",
    "speckle",
    "downsample",
    "gaussian_speckle",
    "gaussian_downsample",
    "speckle_downsample",
    "gaussian_speckle_downsample"
]
SEVERITIES = ["low", "medium", "high"]
METRICS = ["psnr", "ssim", "lpips", "mae", "hf_err"]

def run_inference(p4, model, degraded_lr, is_p10=False):
    p4_raw, _ = p4(degraded_lr)
    p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
    lr_up = F.interpolate(degraded_lr, scale_factor=2, mode="bicubic", align_corners=False)
    if is_p10:
        out, *_ = model(lr_up, p4_hr, lr_input=degraded_lr)
    else:
        out, *_ = model(lr_up, p4_hr)
    return out

def get_eval_degraded(target, original_lr, condition, severity, index, device):
    """
    Generate specific degraded input for evaluation stress test.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42 + index * 1009)
    
    img = original_lr.clone().to(device)
    tgt = target.clone().to(device)
    
    if condition == "gaussian":
        sigmas = {"low": 0.01, "medium": 0.03, "high": 0.06}
        sigma = sigmas[severity]
        return img + torch.randn(img.shape, generator=generator).to(device) * sigma
        
    elif condition == "speckle":
        sigmas = {"low": 0.01, "medium": 0.04, "high": 0.08}
        sigma = sigmas[severity]
        return img * (1.0 + torch.randn(img.shape, generator=generator).to(device) * sigma)
        
    elif condition == "downsample":
        # low (bilinear = OOD), medium (bicubic = ID), high (box/area = OOD)
        if severity == "low":
            return F.interpolate(tgt, scale_factor=0.5, mode="bilinear", align_corners=False)
        elif severity == "medium":
            return F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
        else:
            return F.interpolate(tgt, scale_factor=0.5, mode="area")
            
    elif condition == "gaussian_speckle":
        sigmas = {
            "low": (0.01, 0.01),
            "medium": (0.03, 0.04),
            "high": (0.06, 0.08)
        }
        g_sig, s_sig = sigmas[severity]
        x = img + torch.randn(img.shape, generator=generator).to(device) * g_sig
        return x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sig)
        
    elif condition == "gaussian_downsample":
        sigmas = {"low": 0.01, "medium": 0.03, "high": 0.06}
        sigma = sigmas[severity]
        x = tgt + torch.randn(tgt.shape, generator=generator).to(device) * sigma
        return F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
        
    elif condition == "speckle_downsample":
        sigmas = {"low": 0.01, "medium": 0.04, "high": 0.08}
        sigma = sigmas[severity]
        x = tgt * (1.0 + torch.randn(tgt.shape, generator=generator).to(device) * sigma)
        return F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
        
    elif condition == "gaussian_speckle_downsample":
        sigmas = {
            "low": (0.01, 0.01),
            "medium": (0.03, 0.04),
            "high": (0.06, 0.08)
        }
        g_sig, s_sig = sigmas[severity]
        x = tgt + torch.randn(tgt.shape, generator=generator).to(device) * g_sig
        x = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sig)
        return F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
        
    return img

def metrics_for_one(pred, target, lpips_model, decomposition, target_hf):
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    _, _, pred_hf = decomposition(pred_clamped)
    
    psnr = calculate_psnr(pred_clamped, target)
    ssim = ssim_pytorch(pred_clamped, target).item()
    l_lpips = ssim_lpips_differentiable(pred_clamped, target, lpips_model).item()
    mae = F.l1_loss(pred_clamped, target).item()
    hf_err = F.l1_loss(pred_hf, target_hf).item()
    
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "lpips": float(l_lpips),
        "mae": float(mae),
        "hf_err": float(hf_err),
    }

def add_records(records, condition, severity, sample_indices, p5b_out, p7_out, p8_out, p9_out, p10_out, p11_out, target, lpips_model, decomposition):
    for i, sample_index in enumerate(sample_indices):
        target_i = target[i : i + 1]
        target_hf = decomposition(target_i)[2]
        
        common = {
            "sample_index": sample_index,
            "condition": condition,
            "severity": severity,
        }
        
        models = [
            ("phase5b", p5b_out[i : i + 1]),
            ("phase7", p7_out[i : i + 1]),
            ("phase8", p8_out[i : i + 1]),
            ("phase9", p9_out[i : i + 1]),
            ("phase11", p11_out[i : i + 1])
        ]
        if p10_out is not None:
            models.append(("phase10", p10_out[i : i + 1]))
            
        for name, output in models:
            record = dict(common, model=name)
            record.update(metrics_for_one(output, target_i, lpips_model, decomposition, target_hf))
            records.append(record)

def make_summary(df):
    rows = []
    for (model, condition, severity), group in df.groupby(["model", "condition", "severity"], sort=False):
        item = {"model": model, "condition": condition, "severity": severity}
        for metric in METRICS:
            values = group[metric]
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        rows.append(item)

    # Compute win rates against Phase 9 and Phase 5B
    win_rates = {}
    for (condition, severity), group in df.groupby(["condition", "severity"], sort=False):
        pivot = group.pivot(index="sample_index", columns="model", values=METRICS)
        
        rates = {
            "p11_vs_p9": {
                "psnr": float((pivot["psnr"]["phase11"] > pivot["psnr"]["phase9"]).mean() * 100),
                "ssim": float((pivot["ssim"]["phase11"] > pivot["ssim"]["phase9"]).mean() * 100),
                "lpips": float((pivot["lpips"]["phase11"] < pivot["lpips"]["phase9"]).mean() * 100),
                "mae": float((pivot["mae"]["phase11"] < pivot["mae"]["phase9"]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["phase11"] < pivot["hf_err"]["phase9"]).mean() * 100),
            },
            "p11_vs_p5b": {
                "psnr": float((pivot["psnr"]["phase11"] > pivot["psnr"]["phase5b"]).mean() * 100),
                "ssim": float((pivot["ssim"]["phase11"] > pivot["ssim"]["phase5b"]).mean() * 100),
                "lpips": float((pivot["lpips"]["phase11"] < pivot["lpips"]["phase5b"]).mean() * 100),
                "mae": float((pivot["mae"]["phase11"] < pivot["mae"]["phase5b"]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["phase11"] < pivot["hf_err"]["phase5b"]).mean() * 100),
            }
        }
        win_rates[f"{condition}|{severity}"] = rates
    return pd.DataFrame(rows), win_rates

def build_report(summary_df, baseline_df, win_rates, verdict, qualitative_notes=None):
    base = baseline_df.groupby("model")[METRICS].mean()
    
    p5b_orig = base.loc["phase5b"]
    p7_orig = base.loc["phase7"]
    p8_orig = base.loc["phase8"]
    p9_orig = base.loc["phase9"]
    p11_orig = base.loc["phase11"]
    has_p10 = "phase10" in base.index
    p10_orig = base.loc["phase10"] if has_p10 else None

    report = []
    report.append("============================================================")
    report.append("PHASE 11 DEGRADATION-AWARE RESTORATION REPORT")
    report.append("============================================================")
    report.append("\nORIGINAL VALIDATION\n")
    report.append(f"{'MODEL':<12}{'PSNR':<9}{'SSIM':<9}{'LPIPS':<10}{'MAE':<9}{'HF ERR'}")
    
    models = ["phase5b", "phase7", "phase8", "phase9"]
    if has_p10:
        models.append("phase10")
    models.append("phase11")
    
    for m in models:
        vals = base.loc[m]
        label = m.upper().replace("PHASE", "Phase ")
        report.append(f"{label:<12}{vals.psnr:<9.4f}{vals.ssim:<9.4f}{vals.lpips:<10.4f}{vals.mae:<9.4f}{vals.hf_err:.6f}")

    report.append("\n------------------------------------------------------------\n")
    report.append("PHASE 11 vs PHASE 9 BASES DELTAS\n")
    for metric in METRICS:
        abs_delta = p11_orig[metric] - p9_orig[metric]
        pct_delta = (abs_delta / (p9_orig[metric] + 1e-8)) * 100.0
        report.append(f"{metric.upper():<7}: Absolute: {abs_delta:+.6f} | Percentage: {pct_delta:+.4f}%")

    # Bilinear/Area Downsamplings represent OOD tests
    report.append("\n------------------------------------------------------------\n")
    report.append("ID vs OOD RESTORATION GAP\n")
    
    # ID: downsample medium (bicubic)
    id_m = summary_df[(summary_df.model == "phase11") & (summary_df.condition == "downsample") & (summary_df.severity == "medium")].iloc[0]
    # OOD: downsample low (bilinear) and high (area)
    ood_l = summary_df[(summary_df.model == "phase11") & (summary_df.condition == "downsample") & (summary_df.severity == "low")].iloc[0]
    ood_h = summary_df[(summary_df.model == "phase11") & (summary_df.condition == "downsample") & (summary_df.severity == "high")].iloc[0]
    
    report.append("ID (Downsample Bicubic):")
    report.append(f"  PSNR: {id_m.psnr_mean:.4f} | SSIM: {id_m.ssim_mean:.4f} | LPIPS: {id_m.lpips_mean:.4f}")
    report.append("OOD Low (Downsample Bilinear):")
    report.append(f"  PSNR: {ood_l.psnr_mean:.4f} | SSIM: {ood_l.ssim_mean:.4f} | LPIPS: {ood_l.lpips_mean:.4f}")
    report.append("OOD High (Downsample Box/Area):")
    report.append(f"  PSNR: {ood_h.psnr_mean:.4f} | SSIM: {ood_h.ssim_mean:.4f} | LPIPS: {ood_h.lpips_mean:.4f}")
    
    gap_l = id_m.psnr_mean - ood_l.psnr_mean
    gap_h = id_m.psnr_mean - ood_h.psnr_mean
    report.append(f"ID -> OOD Low PSNR Gap: {gap_l:+.4f} dB")
    report.append(f"ID -> OOD High PSNR Gap: {gap_h:+.4f} dB")

    report.append("\n------------------------------------------------------------\n")
    report.append("STRESS TEST SUMMARY (PSNR mean)\n")
    
    header = f"{'Condition':<30} | {'P5B':<6} | {'P7':<6} | {'P8':<6} | {'P9':<6}"
    if has_p10:
        header += " | {'P10':<6}"
    header += " | {'P11':<6} | Best"
    report.append(header)
    report.append("-" * 95)

    for cond in CONDITIONS:
        for sev in SEVERITIES:
            vals_dict = {}
            for m in models:
                subset = summary_df[(summary_df.model == m) & (summary_df.condition == cond) & (summary_df.severity == sev)]
                if len(subset) > 0:
                    vals_dict[m] = float(subset.iloc[0]["psnr_mean"])
            
            p5b_val = vals_dict.get("phase5b", 0.0)
            p7_val = vals_dict.get("phase7", 0.0)
            p8_val = vals_dict.get("phase8", 0.0)
            p9_val = vals_dict.get("phase9", 0.0)
            p11_val = vals_dict.get("phase11", 0.0)
            
            best_model = max(vals_dict, key=vals_dict.get).upper().replace("PHASE", "Phase ")
            cond_label = f"{cond.replace('_', ' ').title()} ({sev.capitalize()})"
            row_str = f"{cond_label:<30} | {p5b_val:<6.2f} | {p7_val:<6.2f} | {p8_val:<6.2f} | {p9_val:<6.2f}"
            if has_p10:
                row_str += f" | {vals_dict.get('phase10', 0.0):<6.2f}"
            row_str += f" | {p11_val:<6.2f} | {best_model}"
            report.append(row_str)

    report.append("\n------------------------------------------------------------\n")
    report.append("PHASE 11 WIN RATES AGAINST Phase 9\n")
    for k, v in win_rates.items():
        w = v["p11_vs_p9"]
        report.append(f"  {k:<35}: PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%")

    report.append("\n------------------------------------------------------------\n")
    report.append("QUALITATIVE EVALUATION RESULTS\n")
    if qualitative_notes is None:
        qualitative_notes = {
            "ringing": "[Pending visual inspection]",
            "halos": "[Pending visual inspection]",
            "hallucinated texture": "[Pending visual inspection]",
            "oversharpening": "[Pending visual inspection]",
            "smoothing": "[Pending visual inspection]",
            "edge artifacts": "[Pending visual inspection]"
        }
    for k, v in qualitative_notes.items():
        report.append(f"- {k}: {v}")

    report.append("\n------------------------------------------------------------\n")
    report.append("FINAL VERDICT\n")
    report.append(verdict)
    report.append("\nCURRENT CHAMPION:")
    report.append("Phase 11" if "CASE A" in verdict else "Phase 9")
    report.append("\nREASON:")
    if "CASE A" in verdict:
        report.append("Phase 11 improved degradation robustness across KLA Gaussian noise, Speckle noise, and Downsampling stress tests,")
        report.append("while preserving clean validation PSNR/SSIM/LPIPS baseline metrics within tight margins.")
    elif "CASE B" in verdict:
        report.append("Phase 11 shows robustness gains on stress tasks, but regresses slightly beyond the tight 0.05 dB margin on clean validation.")
    else:
        report.append("Phase 11 introduces regression on clean validation (>0.08 dB) or fails to improve robustness.")
    report.append("============================================================")
    
    return "\n".join(report)

def save_plots(summary_df, plots_dir, has_p10=True):
    labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS", "mae": "MAE", "hf_err": "HF Error"}
    for metric in METRICS:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes_flat = axes.flatten()
        
        for idx, condition in enumerate(CONDITIONS):
            axis = axes_flat[idx]
            
            models = ["phase5b", "phase7", "phase8", "phase9"]
            colors = ["#3568b8", "#c45a11", "#9467bd", "#2ca02c"]
            labels_m = ["Phase 5B", "Phase 7", "Phase 8", "Phase 9"]
            if has_p10:
                models.append("phase10")
                colors.append("#bcbd22")
                labels_m.append("Phase 10")
            models.append("phase11")
            colors.append("#d62728")
            labels_m.append("Phase 11")

            for m, col, lbl in zip(models, colors, labels_m):
                subset = summary_df[(summary_df.model == m) & (summary_df.condition == condition)].set_index("severity").loc[SEVERITIES]
                axis.plot(
                    SEVERITIES,
                    subset[f"{metric}_mean"],
                    marker="o",
                    linewidth=2,
                    color=col,
                    label=lbl,
                )
            axis.set_title(condition.replace("_", " ").title())
            axis.set_xlabel("Severity")
            axis.grid(alpha=0.3)
            
        axes_flat[-1].axis("off") # hide extra axes
        axes_flat[0].set_ylabel(labels[metric])
        axes_flat[0].legend()
        fig.suptitle(f"{labels[metric]} vs Degradation Severity", y=1.02, fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_vs_severity.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

def save_visuals(df, dataset, p4, p5b, p7, p8, p9, p11, device, visual_root):
    # Visualize representative scenarios
    scenarios = [
        ("gaussian", "high"),
        ("speckle", "high"),
        ("downsample", "high"),
        ("gaussian_speckle_downsample", "high")
    ]
    
    for condition, severity in scenarios:
        pair = df[(df.condition == condition) & (df.severity == severity)].pivot(index="sample_index", columns="model", values="psnr")
        pair["delta"] = pair["phase11"] - pair["phase9"]
        
        chosen = [
            ("best", int(pair["delta"].idxmax())),
            ("typical", int((pair["delta"] - pair["delta"].median()).abs().idxmin())),
            ("worst", int(pair["delta"].idxmin())),
        ]
        
        for label, index in chosen:
            item = dataset[index]
            target = item["target"].unsqueeze(0).to(device)
            original = item["input"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                degraded = get_eval_degraded(target, original, condition, severity, index, device)
                p5b_out = run_inference(p4, p5b, degraded, is_p10=False)
                p7_out = run_inference(p4, p7, degraded, is_p10=False)
                p8_out = run_inference(p4, p8, degraded, is_p10=False)
                p9_out = run_inference(p4, p9, degraded, is_p10=False)
                p11_out = run_inference(p4, p11, degraded, is_p10=False)
                
            imgs = [
                target,
                F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False),
                p5b_out,
                p7_out,
                p9_out,
                p11_out,
                (p9_out - target).abs(),
                (p11_out - target).abs()
            ]
            titles = [
                "Ground Truth", f"Degraded Input ({condition})", 
                "Phase 5B", "Phase 7",
                "Phase 9 (Champ)", "Phase 11 (Candidate)",
                "|Phase 9 - GT|", "|Phase 11 - GT|"
            ]
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes.flat[8].axis("off")
            
            for axis, image, title in zip(axes.flat[:8], imgs, titles):
                axis.imshow(
                    image[0, 0].detach().cpu(),
                    cmap="magma" if "|" in title else "gray",
                    vmin=0 if "|" in title else None,
                    vmax=0.15 if "|" in title else None,
                )
                axis.set_title(title)
                axis.axis("off")
                
            fig.suptitle(f"{condition} / {severity} / {label} - sample {index + 1:04d}", fontsize=14)
            fig.tight_layout()
            
            directory = os.path.join(visual_root, severity, condition)
            os.makedirs(directory, exist_ok=True)
            filename = f"{condition}_{severity}_{label}_sample_{index + 1:04d}.png"
            fig.savefig(os.path.join(directory, filename), dpi=160, bbox_inches="tight")
            
            if label == "worst":
                os.makedirs(os.path.join(visual_root, "worst_cases"), exist_ok=True)
                fig.savefig(os.path.join(visual_root, "worst_cases", filename), dpi=160, bbox_inches="tight")
                
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--qualitative-json", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = KLADataset(DATASET_ROOT, split="val", csv_path=VAL_CSV)
    
    # Load P4 model
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(P4_CKPT, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters():
        p.requires_grad = False
        
    def load_sfr(ckpt_path):
        m = SpatialFrequencyRestorationNet(
            spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.4
        ).to(device)
        chk = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(chk["model_state_dict"])
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        return m

    p5b = load_sfr(P5B_CKPT)
    p7 = load_sfr(P7_CKPT)
    p8 = load_sfr(P8_CKPT)
    p9 = load_sfr(P9_CKPT)
    p11 = load_sfr(P11_CKPT)
    
    has_p10 = os.path.exists(P10_CKPT)
    p10 = None
    if has_p10:
        print(f"Loading Phase 10 model from: {P10_CKPT}")
        p10 = ConditionedSpatialFrequencyRestorationNet(
            spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.4, use_conditioning=True
        ).to(device)
        p10_chk = torch.load(P10_CKPT, map_location=device, weights_only=False)
        p10.load_state_dict(p10_chk["model_state_dict"])
        p10.eval()
        for p in p10.parameters():
            p.requires_grad = False

    # Check forward pass
    sample = dataset[0]
    inp = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = run_inference(model_p4, p11, inp, is_p10=False)
    assert out.shape == target.shape, "Inference shape mismatch!"
    print("Inference verification PASS.")

    if args.sanity_only:
        print("Sanity-only evaluation pass complete.")
        return

    limit = min(args.max_samples or len(dataset), len(dataset))
    loader = DataLoader(Subset(dataset, range(limit)), batch_size=args.batch_size, shuffle=False)

    for path in (
        os.path.join(OUT_ROOT, "evaluation"),
        os.path.join(OUT_ROOT, "plots"),
        os.path.join(OUT_ROOT, "visual_comparisons"),
        os.path.join(OUT_ROOT, "visual_comparisons", "worst_cases")
    ):
        os.makedirs(path, exist_ok=True)

    decomposition = FrequencyDecompositionModule(0.15, 0.40).to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_model.parameters():
        p.requires_grad = False

    records, baseline_records = [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            start = batch_idx * args.batch_size
            indices = list(range(start, start + len(batch["input"])))
            
            original = batch["input"].to(device)
            target = batch["target"].to(device)
            
            p5b_orig = run_inference(model_p4, p5b, original, is_p10=False)
            p7_orig = run_inference(model_p4, p7, original, is_p10=False)
            p8_orig = run_inference(model_p4, p8, original, is_p10=False)
            p9_orig = run_inference(model_p4, p9, original, is_p10=False)
            p10_orig = run_inference(model_p4, p10, original, is_p10=True) if has_p10 else None
            p11_orig = run_inference(model_p4, p11, original, is_p10=False)
            
            add_records(
                baseline_records,
                "original_validation",
                "original",
                indices,
                p5b_orig,
                p7_orig,
                p8_orig,
                p9_orig,
                p10_orig,
                p11_orig,
                target,
                lpips_model,
                decomposition
            )
            
            for cond in CONDITIONS:
                for sev in SEVERITIES:
                    degraded = get_eval_degraded(target, original, cond, sev, start, device)
                    p5b_out = run_inference(model_p4, p5b, degraded, is_p10=False)
                    p7_out = run_inference(model_p4, p7, degraded, is_p10=False)
                    p8_out = run_inference(model_p4, p8, degraded, is_p10=False)
                    p9_out = run_inference(model_p4, p9, degraded, is_p10=False)
                    p10_out = run_inference(model_p4, p10, degraded, is_p10=True) if has_p10 else None
                    p11_out = run_inference(model_p4, p11, degraded, is_p10=False)
                    
                    add_records(
                        records,
                        cond,
                        sev,
                        indices,
                        p5b_out,
                        p7_out,
                        p8_out,
                        p9_out,
                        p10_out,
                        p11_out,
                        target,
                        lpips_model,
                        decomposition
                    )
            
            completed = min(start + len(indices), limit)
            if completed % 100 == 0 or completed == limit:
                print(f"Evaluated {completed}/{limit}")

    df = pd.DataFrame(records)
    baseline_df = pd.DataFrame(baseline_records)
    summary_df, win_rates = make_summary(df)
    
    # Save validation metrics CSV
    eval_csv_path = os.path.join(OUT_ROOT, "results", "phase11_evaluation.csv")
    df.to_csv(eval_csv_path, index=False)
    
    # Verdict computation
    base = baseline_df.groupby("model")[METRICS].mean()
    p9_base = base.loc["phase9"]
    p11_base = base.loc["phase11"]
    
    psnr_diff = p11_base.psnr - p9_base.psnr
    ssim_diff = p11_base.ssim - p9_base.ssim
    
    # Baseline constraints check
    orig_preserved_a = (psnr_diff >= -0.05) and (ssim_diff >= -0.001)
    orig_preserved_b = (psnr_diff >= -0.08) and (ssim_diff >= -0.0015)
    
    # Stress test robustness change check
    p11_overall_stress = summary_df[summary_df.model == "phase11"]["psnr_mean"].mean()
    p9_overall_stress = summary_df[summary_df.model == "phase9"]["psnr_mean"].mean()
    stress_gain = p11_overall_stress - p9_overall_stress
    
    robustness_improved = stress_gain >= 0.05 # at least 0.05 dB overall stress test gain
    
    if orig_preserved_a and robustness_improved:
        verdict = "CASE A — NEW CHAMPION"
    elif orig_preserved_b and robustness_improved:
        verdict = "CASE B — PROMISING BUT NOT CHAMPION"
    else:
        verdict = "CASE C — REGRESSION"

    qualitative_notes = None
    if args.qualitative_json and os.path.exists(args.qualitative_json):
        try:
            with open(args.qualitative_json, "r", encoding="utf-8") as f:
                qualitative_notes = json.load(f)
            print(f"Loaded qualitative analysis from: {args.qualitative_json}")
        except Exception as e:
            print(f"Error loading qualitative JSON: {e}")

    report = build_report(summary_df, baseline_df, win_rates, verdict, qualitative_notes)
    
    # Save Report
    with open(os.path.join(OUT_ROOT, "results", "PHASE11_REPORT.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Generating comparison plots...")
    save_plots(summary_df, os.path.join(OUT_ROOT, "plots"), has_p10=has_p10)
    
    print("Generating qualitative comparison panels...")
    save_visuals(df, dataset, model_p4, p5b, p7, p8, p9, p11, device, os.path.join(OUT_ROOT, "visual_comparisons"))
    
    print(f"\nPhase 11 evaluation complete. Verdict: {verdict}")
    print(report)

if __name__ == "__main__":
    main()
