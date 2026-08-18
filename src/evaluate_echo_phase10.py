"""Phase 10 evaluation - Phase 5B vs Phase 7 vs Phase 8 vs Phase 9 vs Phase 10 Comparison.
Runs the same validation protocol (640 samples, seed 42) and stress tests.
Generates full metric tables, win-rates, visual comparison panels, and final report.
Imports utility functions and parameters from src/evaluate_echo_phase7.py to reuse metric logic.
"""

import argparse
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import lpips

# Import configuration and common utilities from evaluate_echo_phase7 to reuse logic
from evaluate_echo_phase7 import (
    SEED,
    DATASET_ROOT,
    VAL_CSV,
    P4_CKPT,
    P5B_CKPT,
    P7_CKPT,
    CONDITIONS,
    SEVERITIES,
    METRICS,
    make_json_safe,
    load_sfr_model,
    load_p4,
    metrics_for_one,
    pct_change
)

from utils import set_seed
from dataset import KLADataset
from degradation_utils import synthetic_lr_from_target
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule
from phase10_model import ConditionedSpatialFrequencyRestorationNet

P8_CKPT = "outputs/phase8_hybrid/checkpoints/echo_phase8_hybrid_best.pth"
P9_CKPT = "outputs/phase9_targeted/checkpoints/echo_phase9_best.pth"
P10_CKPT = "outputs/phase10_degradation_aware/checkpoints/echo_phase10_best.pth"
OUT_ROOT = "outputs/phase10_degradation_aware"

def run_inference_p10(p4, model, degraded_lr, is_p10=False):
    p4_raw, _ = p4(degraded_lr)
    p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
    lr_up = F.interpolate(degraded_lr, scale_factor=2, mode="bicubic", align_corners=False)
    if is_p10:
        out, *_ = model(lr_up, p4_hr, lr_input=degraded_lr)
    else:
        out, *_ = model(lr_up, p4_hr)
    return out

def add_records(records, condition, severity, sample_indices, input_paths, target_paths, p5b_out, p7_out, p8_out, p9_out, p10_out, target, lpips_model, decomposition):
    for i, sample_index in enumerate(sample_indices):
        target_i = target[i : i + 1]
        target_hf = decomposition(target_i)[2]
        common = {
            "sample_index": sample_index,
            "sample_id": f"sample_{sample_index + 1:04d}",
            "condition": condition,
            "severity": severity,
            "input_path": input_paths[i],
            "target_path": target_paths[i],
        }
        for model_name, output in (
            ("phase5b", p5b_out[i : i + 1]),
            ("phase7", p7_out[i : i + 1]),
            ("phase8", p8_out[i : i + 1]),
            ("phase9", p9_out[i : i + 1]),
            ("phase10", p10_out[i : i + 1])
        ):
            record = dict(common, model=model_name)
            record.update(metrics_for_one(output, target_i, lpips_model, decomposition, target_hf))
            records.append(record)

def make_summary(df):
    rows = []
    for (model, condition, severity), group in df.groupby(["model", "condition", "severity"], sort=False):
        item = {"model": model, "condition": condition, "severity": severity}
        for metric in METRICS:
            values = group[metric]
            for stat, value in (
                ("mean", values.mean()),
                ("median", values.median()),
                ("std", values.std(ddof=0)),
            ):
                item[f"{metric}_{stat}"] = float(value)
        rows.append(item)

    # Compute win rates
    win_rates = {}
    for (condition, severity), group in df.groupby(["condition", "severity"], sort=False):
        pivot = group.pivot(index="sample_index", columns="model", values=METRICS)
        win_rates[f"{condition}|{severity}"] = {
            # Phase 10 vs Phase 9
            "p10_vs_p9": {
                "psnr": float((pivot["psnr"]["phase10"] > pivot["psnr"]["phase9"]).mean() * 100),
                "ssim": float((pivot["ssim"]["phase10"] > pivot["ssim"]["phase9"]).mean() * 100),
                "lpips": float((pivot["lpips"]["phase10"] < pivot["lpips"]["phase9"]).mean() * 100),
                "mae": float((pivot["mae"]["phase10"] < pivot["mae"]["phase9"]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["phase10"] < pivot["hf_err"]["phase9"]).mean() * 100),
            },
            # Phase 10 vs Phase 5B
            "p10_vs_p5b": {
                "psnr": float((pivot["psnr"]["phase10"] > pivot["psnr"]["phase5b"]).mean() * 100),
                "ssim": float((pivot["ssim"]["phase10"] > pivot["ssim"]["phase5b"]).mean() * 100),
                "lpips": float((pivot["lpips"]["phase10"] < pivot["lpips"]["phase5b"]).mean() * 100),
                "mae": float((pivot["mae"]["phase10"] < pivot["mae"]["phase5b"]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["phase10"] < pivot["hf_err"]["phase5b"]).mean() * 100),
            },
            # Phase 10 vs Phase 7
            "p10_vs_p7": {
                "psnr": float((pivot["psnr"]["phase10"] > pivot["psnr"]["phase7"]).mean() * 100),
                "ssim": float((pivot["ssim"]["phase10"] > pivot["ssim"]["phase7"]).mean() * 100),
                "lpips": float((pivot["lpips"]["phase10"] < pivot["lpips"]["phase7"]).mean() * 100),
                "mae": float((pivot["mae"]["phase10"] < pivot["mae"]["phase7"]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["phase10"] < pivot["hf_err"]["phase7"]).mean() * 100),
            }
        }
    return pd.DataFrame(rows), win_rates

def build_verdict(summary_df, baseline_df):
    base = baseline_df.groupby("model")[METRICS].mean()
    p9_base = base.loc["phase9"]
    p10_base = base.loc["phase10"]

    # Clean validation check: within 0.015 dB of Phase 9
    orig_psnr_ok = p10_base.psnr >= p9_base.psnr - 0.015
    orig_ssim_ok = p10_base.ssim >= p9_base.ssim - 0.0015
    orig_lpips_ok = p10_base.lpips <= p9_base.lpips + 0.0025

    orig_preserved = orig_psnr_ok and orig_ssim_ok and orig_lpips_ok

    # Robustness checks: Gaussian blur medium/high and noise+blur medium/high
    blur_m_p10 = summary_df[(summary_df.model == "phase10") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "medium")].iloc[0]
    blur_m_p9 = summary_df[(summary_df.model == "phase9") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "medium")].iloc[0]
    
    blur_h_p10 = summary_df[(summary_df.model == "phase10") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "high")].iloc[0]
    blur_h_p9 = summary_df[(summary_df.model == "phase9") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "high")].iloc[0]

    combo_h_p10 = summary_df[(summary_df.model == "phase10") & (summary_df.condition == "noise_plus_blur") & (summary_df.severity == "high")].iloc[0]
    combo_h_p9 = summary_df[(summary_df.model == "phase9") & (summary_df.condition == "noise_plus_blur") & (summary_df.severity == "high")].iloc[0]

    avg_p10_stress = (blur_m_p10.psnr_mean + blur_h_p10.psnr_mean + combo_h_p10.psnr_mean) / 3.0
    avg_p9_stress = (blur_m_p9.psnr_mean + blur_h_p9.psnr_mean + combo_h_p9.psnr_mean) / 3.0
    stress_gain = avg_p10_stress - avg_p9_stress

    robustness_improved = stress_gain >= 0.03

    if orig_preserved and robustness_improved:
        verdict = "CASE A — NEW CHAMPION"
    elif robustness_improved:
        verdict = "CASE B — PROMISING SPECIALIST"
    elif stress_gain < 0.01:
        verdict = "CASE D — NO MEANINGFUL CHANGE"
    else:
        verdict = "CASE C — REGRESSION"

    return verdict, orig_preserved, robustness_improved

def build_report(summary_df, baseline_df, win_rates, verdict, qualitative_notes=None):
    base = baseline_df.groupby("model")[METRICS].mean()
    p9_orig = base.loc["phase9"]
    p10_orig = base.loc["phase10"]

    cond_mapping = {
        "gaussian_noise": "Gaussian Noise",
        "gaussian_blur": "Gaussian Blur",
        "noise_plus_blur": "Noise + Blur"
    }

    report_lines = []
    report_lines.append("============================================================")
    report_lines.append("PHASE 10 DEGRADATION-AWARE RESTORATION REPORT")
    report_lines.append("============================================================")
    report_lines.append("\nORIGINAL VALIDATION\n")
    report_lines.append(f"{'MODEL':<12}{'PSNR':<9}{'SSIM':<9}{'LPIPS':<10}{'MAE':<9}{'HF ERR'}")
    
    for model_key, label in [
        ("phase5b", "Phase 5B"), 
        ("phase7", "Phase 7"), 
        ("phase8", "Phase 8"), 
        ("phase9", "Phase 9"), 
        ("phase10", "Phase 10")
    ]:
        vals = base.loc[model_key]
        report_lines.append(f"{label:<12}{vals.psnr:<9.4f}{vals.ssim:<9.4f}{vals.lpips:<10.4f}{vals.mae:<9.4f}{vals.hf_err:.6f}")

    report_lines.append("\n------------------------------------------------------------\n")
    report_lines.append("PHASE 10 vs PHASE 9\n")
    for metric in METRICS:
        delta = p10_orig[metric] - p9_orig[metric]
        report_lines.append(f"{metric.upper():<7}: {delta:+.6f}")

    report_lines.append("\n------------------------------------------------------------\n")
    report_lines.append("STRESS TEST SUMMARY\n")
    report_lines.append(f"{'Condition':<25} | {'P5B':<7} | {'P7':<7} | {'P8':<7} | {'P9':<7} | {'P10':<7} | Best")
    report_lines.append("-" * 80)

    for condition in ["gaussian_noise", "gaussian_blur", "noise_plus_blur"]:
        for severity in ["low", "medium", "high"]:
            p5b_row = summary_df[(summary_df.model == "phase5b") & (summary_df.condition == condition) & (summary_df.severity == severity)].iloc[0]
            p7_row = summary_df[(summary_df.model == "phase7") & (summary_df.condition == condition) & (summary_df.severity == severity)].iloc[0]
            p8_row = summary_df[(summary_df.model == "phase8") & (summary_df.condition == condition) & (summary_df.severity == severity)].iloc[0]
            p9_row = summary_df[(summary_df.model == "phase9") & (summary_df.condition == condition) & (summary_df.severity == severity)].iloc[0]
            p10_row = summary_df[(summary_df.model == "phase10") & (summary_df.condition == condition) & (summary_df.severity == severity)].iloc[0]

            val_5b = p5b_row["psnr_mean"]
            val_7 = p7_row["psnr_mean"]
            val_8 = p8_row["psnr_mean"]
            val_9 = p9_row["psnr_mean"]
            val_10 = p10_row["psnr_mean"]

            max_val = max(val_5b, val_7, val_8, val_9, val_10)
            if max_val == val_10:
                best_model = "Phase 10"
            elif max_val == val_9:
                best_model = "Phase 9"
            elif max_val == val_8:
                best_model = "Phase 8"
            elif max_val == val_7:
                best_model = "Phase 7"
            else:
                best_model = "Phase 5B"

            cond_name = f"{cond_mapping[condition]} — {severity.capitalize()}"
            report_lines.append(f"{cond_name:<25} | {val_5b:<7.3f} | {val_7:<7.3f} | {val_8:<7.3f} | {val_9:<7.3f} | {val_10:<7.3f} | {best_model}")

    report_lines.append("\n------------------------------------------------------------\n")
    report_lines.append("PHASE 10 WIN RATES\n")
    
    # Phase 10 vs Phase 9
    report_lines.append("Phase 10 win rates against Phase 9:")
    for name, wins in win_rates.items():
        w = wins["p10_vs_p9"]
        report_lines.append(
            f"  {name:<20}: PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%"
        )
        
    # Phase 10 vs Phase 5B
    report_lines.append("\nPhase 10 win rates against Phase 5B:")
    for name, wins in win_rates.items():
        w = wins["p10_vs_p5b"]
        report_lines.append(
            f"  {name:<20}: PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%"
        )

    # Phase 10 vs Phase 7
    report_lines.append("\nPhase 10 win rates against Phase 7:")
    for name, wins in win_rates.items():
        w = wins["p10_vs_p7"]
        report_lines.append(
            f"  {name:<20}: PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%"
        )

    report_lines.append("\n------------------------------------------------------------\n")
    report_lines.append("QUALITATIVE RESULTS\n")
    report_lines.append("Report whether Phase 10 introduces:\n")
    
    if qualitative_notes is None:
        qualitative_notes = {
            "ringing": "[To be updated after visual inspection]",
            "halos": "[To be updated after visual inspection]",
            "hallucinated texture": "[To be updated after visual inspection]",
            "oversharpening": "[To be updated after visual inspection]",
            "smoothing": "[To be updated after visual inspection]",
            "edge artifacts": "[To be updated after visual inspection]"
        }
        
    for k, v in qualitative_notes.items():
        report_lines.append(f"- {k}: {v}")

    report_lines.append("\n------------------------------------------------------------\n")
    report_lines.append("FINAL VERDICT\n")
    report_lines.append(verdict)
    report_lines.append("\nCURRENT CHAMPION:")
    report_lines.append("Phase 10" if "CASE A" in verdict else "Phase 9")
    report_lines.append("\nREASON:")
    
    explanation = []
    if "CASE A" in verdict:
        explanation.append("Phase 10 successfully refinemented Phase 9. It preserved Phase 9's original fidelity")
        explanation.append(f"(clean PSNR is {p10_orig.psnr:.4f} vs Phase 9's {p9_orig.psnr:.4f}, a difference of only {p9_orig.psnr - p10_orig.psnr:.4f} dB)")
        explanation.append("while successfully improving blur/noise+blur robustness using targeted conditioning.")
        explanation.append("Therefore, Phase 10 is declared the NEW CHAMPION.")
    elif "CASE B" in verdict:
        explanation.append("Phase 10 shows promising gains on stress tasks, but regresses beyond acceptable margins on clean validation.")
        explanation.append(f"PSNR regressed from {p9_orig.psnr:.4f} to {p10_orig.psnr:.4f} ({p9_orig.psnr - p10_orig.psnr:.4f} dB regression).")
        explanation.append("Therefore, Phase 9 remains the champion, and Phase 10 serves as a specialised robustness model.")
    elif "CASE D" in verdict:
        explanation.append("Phase 10 has negligible change compared to Phase 9. Therefore, Phase 9 is kept as champion.")
    else:
        explanation.append("Phase 10 damaged clean validation or visual quality without sufficient robustness gains.")
        explanation.append("Therefore, Phase 9 remains the champion.")
    
    report_lines.extend(explanation)
    report_lines.append("============================================================")
    
    return "\n".join(report_lines)

def save_plots(summary_df, plots_dir):
    labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS", "mae": "MAE", "hf_err": "HF Error"}
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, condition in zip(axes, CONDITIONS):
            for model, color, label in (
                ("phase5b", "#3568b8", "Phase 5B"), 
                ("phase7", "#c45a11", "Phase 7"), 
                ("phase8", "#9467bd", "Phase 8"), 
                ("phase9", "#2ca02c", "Phase 9"),
                ("phase10", "#d62728", "Phase 10")
            ):
                subset = summary_df[(summary_df.model == model) & (summary_df.condition == condition)].set_index("severity").loc[SEVERITIES]
                axis.plot(
                    SEVERITIES,
                    subset[f"{metric}_mean"],
                    marker="o",
                    linewidth=2,
                    color=color,
                    label=label,
                )
            axis.set_title(condition.replace("_", " ").title())
            axis.set_xlabel("Severity")
            axis.grid(alpha=0.3)
        axes[0].set_ylabel(labels[metric])
        axes[0].legend()
        fig.suptitle(f"{labels[metric]} vs degradation severity", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_vs_severity.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

def save_visuals(df, dataset, p4, p5b, p7, p8, p9, p10, device, visual_root):
    scenarios = [
        ("gaussian_noise", "high"),
        ("gaussian_blur", "high"),
        ("noise_plus_blur", "high")
    ]
    
    for condition, severity in scenarios:
        levels = CONDITIONS[condition]
        noise_sigma, blur_sigma = levels[severity]
        
        pair = df[(df.condition == condition) & (df.severity == severity)].pivot(index="sample_index", columns="model", values="psnr")
        pair["delta"] = pair["phase10"] - pair["phase9"]
        
        chosen = [
            ("best", int(pair["delta"].idxmax())),
            ("typical", int((pair["delta"] - pair["delta"].median()).abs().idxmin())),
            ("worst", int(pair["delta"].idxmin())),
        ]
        
        for label, index in chosen:
            item = dataset[index]
            target = item["target"].unsqueeze(0).to(device)
            with torch.no_grad():
                degraded = synthetic_lr_from_target(target, noise_sigma, blur_sigma, [index], device, seed=SEED)
                p5b_out = run_inference_p10(p4, p5b, degraded, is_p10=False)
                p7_out = run_inference_p10(p4, p7, degraded, is_p10=False)
                p8_out = run_inference_p10(p4, p8, degraded, is_p10=False)
                p9_out = run_inference_p10(p4, p9, degraded, is_p10=False)
                p10_out = run_inference_p10(p4, p10, degraded, is_p10=True)
            
            # Setup a beautiful 3x3 panel containing predictions and error maps
            imgs = [
                target,
                F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False),
                p5b_out,
                p7_out,
                p9_out,
                p10_out,
                (p9_out - target).abs(),
                (p10_out - target).abs(),
            ]
            titles = [
                "Ground Truth", "Degraded Input (2x)", 
                "Phase 5B (Fid)", "Phase 7 (Rob)",
                "Phase 9 (Prev Champ)", "Phase 10 (Candidate)",
                "|Phase 9 - GT|", "|Phase 10 - GT|"
            ]
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            
            # Hide the last axes slot (index 8) since we have 8 images
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

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 10 evaluation.")

    device = torch.device("cuda")
    dataset = KLADataset(DATASET_ROOT, split="val", csv_path=VAL_CSV)
    p4 = load_p4(device)
    p5b = load_sfr_model(P5B_CKPT, device)
    p7 = load_sfr_model(P7_CKPT, device)
    p8 = load_sfr_model(P8_CKPT, device)
    p9 = load_sfr_model(P9_CKPT, device)
    
    print(f"Loading Phase 10 model from: {P10_CKPT}")
    p10 = ConditionedSpatialFrequencyRestorationNet(
        spatial_channels=32,
        freq_channels=32,
        fusion_channels=64,
        cutoff_low=0.15,
        cutoff_high=0.40,
        use_conditioning=True
    ).to(device)
    p10_chk = torch.load(P10_CKPT, map_location=device, weights_only=False)
    p10.load_state_dict(p10_chk["model_state_dict"])
    p10.eval()

    # Minimal validation checks
    print("Sanity checking model shapes...")
    sample = dataset[0]
    inp = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    with torch.no_grad():
        p10_out = run_inference_p10(p4, p10, inp, is_p10=True)
    assert p10_out.shape == target.shape, f"Output shape mismatch: {p10_out.shape} vs {target.shape}"
    print("Sanity checks passed.")

    if args.sanity_only:
        print("Sanity-only evaluation pass complete.")
        return

    limit = min(args.max_samples or len(dataset), len(dataset))
    loader = DataLoader(Subset(dataset, range(limit)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    for path in (
        os.path.join(OUT_ROOT, "evaluation"),
        os.path.join(OUT_ROOT, "plots"),
        os.path.join(OUT_ROOT, "visual_comparisons"),
        os.path.join(OUT_ROOT, "visual_comparisons", "worst_cases"),
    ):
        os.makedirs(path, exist_ok=True)

    decomposition = FrequencyDecompositionModule(0.15, 0.40).to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad_(False)

    records, baseline_records = [], []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            start = batch_number * args.batch_size
            indices = list(range(start, start + len(batch["input"])))
            original, target = batch["input"].to(device), batch["target"].to(device)
            
            p5b_orig = run_inference_p10(p4, p5b, original, is_p10=False)
            p7_orig = run_inference_p10(p4, p7, original, is_p10=False)
            p8_orig = run_inference_p10(p4, p8, original, is_p10=False)
            p9_orig = run_inference_p10(p4, p9, original, is_p10=False)
            p10_orig = run_inference_p10(p4, p10, original, is_p10=True)
            
            add_records(
                baseline_records,
                "original_validation",
                "original",
                indices,
                batch["input_path"],
                batch["target_path"],
                p5b_orig,
                p7_orig,
                p8_orig,
                p9_orig,
                p10_orig,
                target,
                lpips_model,
                decomposition,
            )
            
            for condition, levels in CONDITIONS.items():
                for severity, (noise_sigma, blur_sigma) in levels.items():
                    degraded = synthetic_lr_from_target(target, noise_sigma, blur_sigma, indices, device, seed=SEED)
                    p5b_out = run_inference_p10(p4, p5b, degraded, is_p10=False)
                    p7_out = run_inference_p10(p4, p7, degraded, is_p10=False)
                    p8_out = run_inference_p10(p4, p8, degraded, is_p10=False)
                    p9_out = run_inference_p10(p4, p9, degraded, is_p10=False)
                    p10_out = run_inference_p10(p4, p10, degraded, is_p10=True)
                    
                    add_records(
                        records,
                        condition,
                        severity,
                        indices,
                        batch["input_path"],
                        batch["target_path"],
                        p5b_out,
                        p7_out,
                        p8_out,
                        p9_out,
                        p10_out,
                        target,
                        lpips_model,
                        decomposition,
                    )
            completed = min(start + len(indices), limit)
            if completed % 100 == 0 or completed == limit:
                print(f"Evaluated {completed}/{limit}")

    df = pd.DataFrame(records)
    baseline_df = pd.DataFrame(baseline_records)
    summary_df, win_rates = make_summary(df)
    
    evaluation_dir = os.path.join(OUT_ROOT, "evaluation")
    all_records_df = pd.concat([baseline_df, df], ignore_index=True)
    all_records_df.to_csv(os.path.join(evaluation_dir, "phase10_metrics.csv"), index=False)

    verdict, orig_ok, robustness_ok = build_verdict(summary_df, baseline_df)
    
    # Load qualitative notes if available
    qualitative_notes = None
    if args.qualitative_json and os.path.exists(args.qualitative_json):
        try:
            with open(args.qualitative_json, "r", encoding="utf-8") as f:
                qualitative_notes = json.load(f)
            print(f"Successfully loaded qualitative analysis from: {args.qualitative_json}")
        except Exception as e:
            print(f"Error loading qualitative JSON: {e}")

    report = build_report(summary_df, baseline_df, win_rates, verdict, qualitative_notes)
    
    summary = {
        "seed": SEED,
        "sample_count": limit,
        "verdict": verdict,
        "original_preserved": orig_ok,
        "robustness_improved": robustness_ok,
        "win_rates_pct": win_rates,
        "degradation_parameters": CONDITIONS,
    }
    
    results_dir = os.path.join(OUT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "phase10_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(summary), handle, indent=2)
    with open(os.path.join(results_dir, "PHASE10_REPORT.txt"), "w", encoding="utf-8") as handle:
        handle.write(report)

    print("Generating comparison plots...")
    save_plots(summary_df, os.path.join(OUT_ROOT, "plots"))
    
    print("Generating qualitative comparison panels...")
    save_visuals(df, dataset, p4, p5b, p7, p8, p9, p10, device, os.path.join(OUT_ROOT, "visual_comparisons"))
    
    print(f"\nPhase 10 evaluation complete. Verdict: {verdict}")
    print(report)

if __name__ == "__main__":
    main()
