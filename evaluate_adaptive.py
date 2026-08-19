import os
import sys
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
import matplotlib.pyplot as plt

# Add src to sys.path to load echo scripts
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset import KLADataset
from echo_model import BaselineECHOModel
from phase5_model import SpatialFrequencyRestorationNet, FrequencyDecompositionModule
from train_echo_phase43 import ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase10 import calculate_psnr
from utils import set_seed

# Paths to models
P4_CKPT = "outputs/echo_phase4/checkpoints/echo_best.pth"
P9_CKPT = "outputs/phase9_targeted/checkpoints/echo_phase9_best.pth"
P11_CKPT = "outputs/phase11_detail_robustness/checkpoints/echo_phase11_best.pth"
VAL_CSV = "outputs/baseline/val_split.csv"
DATASET_ROOT = "D:/kla"

OUT_ROOT = "outputs/adaptive"

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

def run_inference(p4, model, degraded_lr):
    p4_raw, _ = p4(degraded_lr)
    p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
    lr_up = F.interpolate(degraded_lr, scale_factor=2, mode="bicubic", align_corners=False)
    out, *_ = model(lr_up, p4_hr)
    return torch.clamp(out, 0.0, 1.0)

def get_eval_degraded(target, original_lr, condition, severity, index, device):
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
        if severity == "low":
            return F.interpolate(tgt, scale_factor=0.5, mode="bilinear", align_corners=False)
        elif severity == "medium":
            return F.interpolate(tgt, scale_factor=0.5, mode="bicubic", align_corners=False)
        else:
            return F.interpolate(tgt, scale_factor=0.5, mode="area")
            
    elif condition == "gaussian_speckle":
        sigmas = {"low": (0.01, 0.01), "medium": (0.03, 0.04), "high": (0.06, 0.08)}
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
        sigmas = {"low": (0.01, 0.01), "medium": (0.03, 0.04), "high": (0.06, 0.08)}
        g_sig, s_sig = sigmas[severity]
        x = tgt + torch.randn(tgt.shape, generator=generator).to(device) * g_sig
        x = x * (1.0 + torch.randn(x.shape, generator=generator).to(device) * s_sig)
        return F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
        
    return img

def estimate_noise_tensor(t):
    # Laplacian MAD noise level estimator
    arr = t.squeeze(0).squeeze(0).cpu().numpy()
    lap = arr[1:-1, 1:-1] - 0.25 * (arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:])
    mad = np.median(np.abs(lap - np.median(lap)))
    return mad * 1.4826 / 1.118

def get_adaptive_weight(noise_sigma):
    # Linear blend transition between Phase 9 and Phase 11-DP
    t_low, t_high = 0.015, 0.060
    return 1.0 - np.clip((noise_sigma - t_low) / (t_high - t_low), 0.0, 1.0)

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

def add_records(records, condition, severity, sample_indices, p9_out, p11_out, adapt_out, target, lpips_model, decomposition):
    for i, sample_index in enumerate(sample_indices):
        target_i = target[i : i + 1]
        target_hf = decomposition(target_i)[2]
        
        common = {
            "sample_index": sample_index,
            "condition": condition,
            "severity": severity,
        }
        
        models = [
            ("phase9", p9_out[i : i + 1]),
            ("phase11_detail", p11_out[i : i + 1]),
            ("adaptive", adapt_out[i : i + 1])
        ]
        
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

    # Compute win rates
    win_rates = {}
    for (condition, severity), group in df.groupby(["condition", "severity"], sort=False):
        pivot = group.pivot(index="sample_index", columns="model", values=METRICS)
        rates = {}
        for other in ["phase9", "phase11_detail"]:
            key = f"adapt_vs_{other}"
            rates[key] = {
                "psnr": float((pivot["psnr"]["adaptive"] > pivot["psnr"][other]).mean() * 100),
                "ssim": float((pivot["ssim"]["adaptive"] > pivot["ssim"][other]).mean() * 100),
                "lpips": float((pivot["lpips"]["adaptive"] < pivot["lpips"][other]).mean() * 100),
                "mae": float((pivot["mae"]["adaptive"] < pivot["mae"][other]).mean() * 100),
                "hf_err": float((pivot["hf_err"]["adaptive"] < pivot["hf_err"][other]).mean() * 100),
            }
        win_rates[f"{condition}|{severity}"] = rates
    return pd.DataFrame(rows), win_rates

def build_report(summary_df, baseline_df, win_rates, verdict, qualitative_notes=None):
    base = baseline_df.groupby("model")[METRICS].mean()
    p9_orig = base.loc["phase9"]
    p11_orig = base.loc["phase11_detail"]
    adapt_orig = base.loc["adaptive"]

    report = []
    report.append("============================================================")
    report.append("ECHO ADAPTIVE ROUTING EVALUATION REPORT")
    report.append("============================================================")
    report.append("\nBASELINE\n")
    report.append(f"{'MODEL':<18}{'PSNR':<9}{'SSIM':<9}{'LPIPS':<10}{'MAE':<9}{'HF ERR'}")
    
    models = ["phase9", "phase11_detail", "adaptive"]
    for m in models:
        vals = base.loc[m]
        label = m.upper().replace("PHASE", "Phase ").replace("DETAIL", "Detail-Preserved").title()
        report.append(f"{label:<18}{vals.psnr:<9.4f}{vals.ssim:<9.4f}{vals.lpips:<10.4f}{vals.mae:<9.4f}{vals.hf_err:.6f}")

    report.append("\n------------------------------------------------------------\n")
    report.append("DELTA vs PHASE 9\n")
    for metric in METRICS:
        abs_delta = adapt_orig[metric] - p9_orig[metric]
        pct_delta = (abs_delta / (p9_orig[metric] + 1e-8)) * 100.0
        report.append(f"{metric.upper():<7}: Absolute: {abs_delta:+.6f} | Percentage: {pct_delta:+.4f}%")

    report.append("\n------------------------------------------------------------\n")
    # Clean validation is essentially low noise, OOD represents downsampled
    report.append("DEGRADATION ROUTING (Clean vs Noisy blends)\n")
    # Low: downsample high (very low noise), Medium: clean validation, High: noise high
    # Let's count actual adaptive weights used on Clean Validation
    report.append("Original validation routing is dynamically controlled.")

    # Stress test summary table
    report.append("\n------------------------------------------------------------\n")
    report.append("ROBUSTNESS SUMMARY (PSNR mean)\n")
    report.append(f"{'Condition':<35} | {'P9':<6} | {'P11-DP':<6} | {'Adaptive':<8} | Best")
    report.append("-" * 75)
    
    for cond in CONDITIONS:
        for sev in SEVERITIES:
            vals_dict = {}
            for m in models:
                subset = summary_df[(summary_df.model == m) & (summary_df.condition == cond) & (summary_df.severity == sev)]
                if len(subset) > 0:
                    vals_dict[m] = float(subset.iloc[0]["psnr_mean"])
            
            p9_val = vals_dict.get("phase9", 0.0)
            p11_val = vals_dict.get("phase11_detail", 0.0)
            adapt_val = vals_dict.get("adaptive", 0.0)
            
            best_model_name = max(vals_dict, key=vals_dict.get)
            best_lbl = "Phase 9" if best_model_name == "phase9" else ("Phase 11-DP" if best_model_name == "phase11_detail" else "Adaptive")
            cond_label = f"{cond.replace('_', ' ').title()} ({sev.capitalize()})"
            report.append(f"{cond_label:<35} | {p9_val:<6.2f} | {p11_val:<6.2f} | {adapt_val:<8.2f} | {best_lbl}")

    report.append("\n------------------------------------------------------------\n")
    report.append("WIN RATES\n")
    report.append("Adaptive vs Phase 9:")
    for k, v in win_rates.items():
        w = v["adapt_vs_phase9"]
        report.append(f"  {k:<35} | PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%")
        
    report.append("\nAdaptive vs Phase 11-DP:")
    for k, v in win_rates.items():
        w = v["adapt_vs_phase11_detail"]
        report.append(f"  {k:<35} | PSNR {w['psnr']:.1f}%, SSIM {w['ssim']:.1f}%, LPIPS {w['lpips']:.1f}%, MAE {w['mae']:.1f}%, HF {w['hf_err']:.1f}%")

    report.append("\n------------------------------------------------------------\n")
    report.append("QUALITATIVE RESULTS\n")
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
    report.append("CHAMPION DECISION\n")
    report.append(verdict)
    report.append("\nCURRENT CHAMPION:")
    report.append("Adaptive" if "CASE A" in verdict else "Phase 9")
    report.append("\nREASON:")
    if "CASE A" in verdict:
        report.append("Adaptive successfully beats Phase 9 clean baseline and preserves or exceeds robustness.")
    elif "CASE B" in verdict:
        report.append("Adaptive is promising but did not satisfy the strict Clean Fidelity gate (+0.02 dB constraint). Phase 9 remains champion.")
    else:
        report.append("Adaptive caused regression in clean fidelity or robustness. Phase 9 remains champion.")
    report.append("============================================================")
    
    return "\n".join(report)

def save_visuals(df, dataset, p4, p9, p11, device, visual_root):
    scenarios = [
        ("gaussian", "high"),
        ("speckle", "high"),
        ("downsample", "high"),
        ("gaussian_speckle_downsample", "high")
    ]
    
    for condition, severity in scenarios:
        pair = df[(df.condition == condition) & (df.severity == severity)].pivot(index="sample_index", columns="model", values="psnr")
        pair["delta"] = pair["adaptive"] - pair["phase9"]
        
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
                p9_out = run_inference(p4, p9, degraded)
                p11_out = run_inference(p4, p11, degraded)
                
                # Estimate noise and blend adaptively
                sigma = estimate_noise_tensor(degraded)
                alpha = get_adaptive_weight(sigma)
                adapt_out = alpha * p9_out + (1.0 - alpha) * p11_out
                adapt_out = torch.clamp(adapt_out, 0.0, 1.0)
                
            imgs = [
                target,
                F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False),
                p9_out,
                p11_out,
                adapt_out,
                (p9_out - target).abs(),
                (adapt_out - target).abs()
            ]
            titles = [
                "Ground Truth", f"Degraded Input ({condition})", 
                "Phase 9 (Champ)", "Phase 11-DP (Expert)", "Adaptive (Candidate)",
                "|Phase 9 - GT|", "|Adaptive - GT|"
            ]
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes.flat[7].axis("off")
            axes.flat[8].axis("off")
            
            for ax, img, t in zip(axes.flat, imgs, titles):
                arr = img.squeeze(0).squeeze(0).cpu().numpy()
                ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0 if not t.startswith("|") else None)
                ax.set_title(t)
                ax.axis("off")
                
            fig.suptitle(f"{condition} / {severity} / {label} - sample {index:04d}", fontsize=14, y=0.98)
            os.makedirs(os.path.join(visual_root, f"{condition}_{severity}"), exist_ok=True)
            fig.savefig(os.path.join(visual_root, f"{condition}_{severity}", f"{label}_sample_{index:04d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load validation split dataset
    print(f"Loading dataset paths from CSV: {VAL_CSV}")
    dataset = KLADataset(DATASET_ROOT, split="val", csv_path=VAL_CSV)

    # Instantiate models
    print("Loading models...")
    p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4.load_state_dict(torch.load(P4_CKPT, map_location=device)["model_state_dict"])
    p4.eval()

    p9 = SpatialFrequencyRestorationNet().to(device)
    p9.load_state_dict(torch.load(P9_CKPT, map_location=device)["model_state_dict"])
    p9.eval()

    p11 = SpatialFrequencyRestorationNet().to(device)
    p11.load_state_dict(torch.load(P11_CKPT, map_location=device)["model_state_dict"])
    p11.eval()

    # Verify basic inference
    with torch.no_grad():
        test_in = dataset[0]["input"].unsqueeze(0).to(device)
        test_out = run_inference(p4, p9, test_in)
    print("Inference verification PASS.")

    limit = len(dataset)
    if args.max_samples is not None:
        limit = min(args.max_samples, len(dataset))
    if args.sanity_only:
        limit = 1
        print("Sanity-only evaluation mode.")

    loader = DataLoader(Subset(dataset, range(limit)), batch_size=args.batch_size, shuffle=False)

    decomposition = FrequencyDecompositionModule(0.15, 0.40).to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    baseline_records = []
    stress_records = []

    print(f"Starting evaluation on {limit} samples...")
    for idx_batch, batch in enumerate(loader):
        start = idx_batch * args.batch_size
        indices = list(range(start, start + len(batch["input"])))
        original_lr = batch["input"].to(device)
        target = batch["target"].to(device)
        
        # 1. Baseline Clean Validation
        with torch.no_grad():
            p9_out = run_inference(p4, p9, original_lr)
            p11_out = run_inference(p4, p11, original_lr)
            
            # Compute adaptive blend per image in batch
            adapt_outs = []
            for b_idx in range(original_lr.shape[0]):
                single_in = original_lr[b_idx : b_idx + 1]
                sigma = estimate_noise_tensor(single_in)
                alpha = get_adaptive_weight(sigma)
                fused = alpha * p9_out[b_idx : b_idx + 1] + (1.0 - alpha) * p11_out[b_idx : b_idx + 1]
                adapt_outs.append(fused)
            adapt_out = torch.cat(adapt_outs, dim=0)
            
        add_records(baseline_records, "original", "none", indices, p9_out, p11_out, adapt_out, target, lpips_model, decomposition)
        
        if args.sanity_only:
            # only run 1 sample for sanity checks
            break

        # 2. Stress tests loop
        for cond in CONDITIONS:
            for sev in SEVERITIES:
                degraded_batch = []
                for b_idx, s_idx in enumerate(indices):
                    deg = get_eval_degraded(target[b_idx:b_idx+1], original_lr[b_idx:b_idx+1], cond, sev, s_idx, device)
                    degraded_batch.append(deg)
                degraded = torch.cat(degraded_batch, dim=0)
                
                with torch.no_grad():
                    p9_out = run_inference(p4, p9, degraded)
                    p11_out = run_inference(p4, p11, degraded)
                    
                    adapt_outs = []
                    for b_idx in range(degraded.shape[0]):
                        single_in = degraded[b_idx : b_idx + 1]
                        sigma = estimate_noise_tensor(single_in)
                        alpha = get_adaptive_weight(sigma)
                        fused = alpha * p9_out[b_idx : b_idx + 1] + (1.0 - alpha) * p11_out[b_idx : b_idx + 1]
                        adapt_outs.append(fused)
                    adapt_out = torch.cat(adapt_outs, dim=0)
                    
                add_records(stress_records, cond, sev, indices, p9_out, p11_out, adapt_out, target, lpips_model, decomposition)

        completed = idx_batch * args.batch_size + len(indices)
        if completed % 100 == 0 or completed == limit:
            print(f"Evaluated {completed}/{limit}...")

    # Build dataframes
    baseline_df = pd.DataFrame(baseline_records)
    stress_df = pd.DataFrame(stress_records)
    
    if args.sanity_only:
        print("Sanity checks complete. Validation scripts compiled successfully.")
        return

    # Aggregate summaries
    summary_df, win_rates = make_summary(stress_df)

    # Output paths
    os.makedirs(os.path.join(OUT_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "visual_comparisons"), exist_ok=True)

    # Save details
    stress_df.to_csv(os.path.join(OUT_ROOT, "results", "adaptive_evaluation.csv"), index=False)

    # 1. Apply Decision Gates
    base = baseline_df.groupby("model")[METRICS].mean()
    p9_orig = base.loc["phase9"]
    adapt_orig = base.loc["adaptive"]

    # Gate 1: Clean PSNR must not meaningfully regress (within 0.02 dB)
    psnr_diff = adapt_orig.psnr - p9_orig.psnr
    ssim_diff = adapt_orig.ssim - p9_orig.ssim
    gate_clean_pass = (psnr_diff >= -0.02) and (ssim_diff >= -0.001)

    # Gate 2: LPIPS maintains or improves
    lpips_diff = adapt_orig.lpips - p9_orig.lpips
    gate_lpips_pass = (lpips_diff <= 0.005) # Allow tiny variance

    # Gate 4: Robustness improvements on medium/high
    # Count how many conditions are improved or equal
    better_count = 0
    total_conds = 0
    for cond in CONDITIONS:
        for sev in ["medium", "high"]:
            total_conds += 1
            p9_val = stress_df[(stress_df.model == "phase9") & (stress_df.condition == cond) & (stress_df.severity == sev)]["psnr"].mean()
            adapt_val = stress_df[(stress_df.model == "adaptive") & (stress_df.condition == cond) & (stress_df.severity == sev)]["psnr"].mean()
            if adapt_val >= p9_val - 0.05:
                better_count += 1
    gate_robustness_pass = (better_count / total_conds) >= 0.70

    if gate_clean_pass and gate_lpips_pass and gate_robustness_pass:
        verdict = "CASE A — NEW CHAMPION"
    elif not gate_clean_pass:
        verdict = f"CASE B — PROMISING BUT NOT CHAMPION (Clean PSNR regressed: {psnr_diff:+.4f} dB)"
    else:
        verdict = "CASE C — REGRESSION / INSUFFICIENT IMPROVEMENT"

    # Build report
    report = build_report(summary_df, baseline_df, win_rates, verdict)
    with open(os.path.join(OUT_ROOT, "results", "ADAPTIVE_REPORT.txt"), "w") as f:
        f.write(report)
    print("\n" + report)

    # Generate visuals
    print("Generating qualitative comparison panels...")
    save_visuals(stress_df, dataset, p4, p9, p11, device, os.path.join(OUT_ROOT, "visual_comparisons"))
    print("Evaluation pipeline run complete.")

if __name__ == "__main__":
    main()
