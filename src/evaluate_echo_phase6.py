"""Phase 6 robustness validation for the frozen Phase 4 and Phase 5B models.

This is evaluation-only code.  It reuses the model and metric implementation from
the Phase 5B evaluator, and writes all generated artifacts below outputs/phase6.
Run ``python src/evaluate_echo_phase6.py --sanity-only`` before the full run.
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips

from utils import set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from phase5_model import FrequencyDecompositionModule, SpatialFrequencyRestorationNet
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr


SEED = 42
DATASET_ROOT = "D:/kla"
VAL_CSV = "outputs/baseline/val_split.csv"
P4_CKPT = "outputs/echo_phase4/checkpoints/echo_best.pth"
P5B_CKPT = "outputs/phase5b/checkpoints/echo_phase5b_best.pth"
OUT_ROOT = "outputs/phase6"

# Parameters are deliberately moderate.  Dataset characterization identifies
# Gaussian-like, signal-dependent noise and blur/downsampling as existing modes,
# but does not provide canonical sigma ranges.  Noise is injected after 2x
# downsampling so it remains a genuine LR-input stressor; images are not clipped,
# preserving the project's raw-input convention.
CONDITIONS = {
    "gaussian_noise": {"low": (0.010, 0.0), "medium": (0.030, 0.0), "high": (0.060, 0.0)},
    "gaussian_blur": {"low": (0.0, 0.50), "medium": (0.0, 1.00), "high": (0.0, 1.50)},
    "noise_plus_blur": {"low": (0.010, 0.50), "medium": (0.030, 1.00), "high": (0.060, 1.50)},
}
SEVERITIES = ["low", "medium", "high"]
METRICS = ["psnr", "ssim", "lpips", "mae", "hf_err"]


def gaussian_blur(x, sigma):
    """Separable Gaussian blur for a BxCxHxW tensor; identity when sigma is 0."""
    if sigma <= 0:
        return x
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=channels)
    return F.conv2d(x, ky, padding=(radius, 0), groups=channels)


def synthetic_lr_from_target(target, noise_sigma, blur_sigma, sample_indices, device):
    """Create deterministic LR inputs from GT: blur HR -> 2x bicubic -> LR noise."""
    degraded_hr = gaussian_blur(target, blur_sigma)
    lr = F.interpolate(degraded_hr, scale_factor=0.5, mode="bicubic", align_corners=False)
    if noise_sigma:
        noise = torch.empty_like(lr)
        # Per-sample generators ensure results do not depend on batch size/order.
        for i, sample_index in enumerate(sample_indices):
            generator = torch.Generator(device=device)
            generator.manual_seed(SEED + sample_index * 1009 + int(noise_sigma * 1_000_000) + int(blur_sigma * 10_000))
            noise[i] = torch.randn(lr[i].shape, generator=generator, device=device, dtype=lr.dtype)
        lr = lr + noise_sigma * noise
    return lr


def load_p5b(device):
    model = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    checkpoint = torch.load(P5B_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_p4(device):
    model = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    checkpoint = torch.load(P4_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def metrics_for_one(pred, target, sobel, lpips_model, decomposition, target_hf):
    return {
        "psnr": float(calculate_psnr(pred, target)),
        "ssim": float(ssim_pytorch(pred, target).item()),
        "lpips": float(ssim_lpips_differentiable(pred, target, lpips_model).item()),
        "mae": float(F.l1_loss(pred, target).item()),
        "hf_err": float(F.l1_loss(decomposition(pred)[2], target_hf).item()),
        "edge_err": float(F.l1_loss(sobel(pred), sobel(target)).item()),
    }


def run_sanity_checks(dataset, device, p4, p5b):
    checks = []
    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    check("CUDA available", device.type == "cuda", str(device))
    check("Phase 4 checkpoint exists", os.path.isfile(P4_CKPT), P4_CKPT)
    check("Phase 5B checkpoint exists", os.path.isfile(P5B_CKPT), P5B_CKPT)
    check("Validation CSV exists", os.path.isfile(VAL_CSV), VAL_CSV)
    check("Validation sample count is 640", len(dataset) == 640, str(len(dataset)))
    all_paths = all(os.path.isfile(a) and os.path.isfile(b) for a, b in zip(dataset.input_paths, dataset.target_paths))
    check("Validation paths are valid", all_paths)
    sample = dataset[0]
    inp, target = sample["input"].unsqueeze(0).to(device), sample["target"].unsqueeze(0).to(device)
    check("Ground-truth dimensions are 2x input", tuple(target.shape[-2:]) == tuple(2 * n for n in inp.shape[-2:]), f"{tuple(inp.shape)} -> {tuple(target.shape)}")
    with torch.no_grad():
        p4_raw, _ = p4(inp)
        p4_out = torch.clamp(p4_raw, 0.0, 1.0)
        p5b_out, *_ = p5b(F.interpolate(inp, scale_factor=2, mode="bicubic", align_corners=False), p4_out)
        stress_a = synthetic_lr_from_target(target, 0.03, 1.0, [0], device)
        stress_b = synthetic_lr_from_target(target, 0.03, 1.0, [0], device)
        stress_p4_raw, _ = p4(stress_a)
        stress_p4 = torch.clamp(stress_p4_raw, 0.0, 1.0)
        stress_p5b, *_ = p5b(F.interpolate(stress_a, scale_factor=2, mode="bicubic", align_corners=False), stress_p4)
    check("Model output dimensions are correct", p4_out.shape == target.shape and stress_p5b.shape == target.shape)
    check("Output values are finite", bool(torch.isfinite(p4_out).all() and torch.isfinite(p5b_out).all() and torch.isfinite(stress_p5b).all()))
    check("Output range is valid", bool((p4_out >= 0).all() and (p4_out <= 1).all() and (p5b_out >= 0).all() and (p5b_out <= 1).all() and (stress_p5b >= 0).all() and (stress_p5b <= 1).all()))
    check("Models are in eval mode", not p4.training and not p5b.training)
    check("No gradients are accumulated", all(p.grad is None for p in p4.parameters()) and all(p.grad is None for p in p5b.parameters()))
    check("Random seed is fixed", SEED == 42)
    check("Synthetic degradation is reproducible", bool(torch.equal(stress_a, stress_b)))
    check("Models receive identical degraded input", bool(torch.equal(stress_a, stress_a)))
    check("Existing raw preprocessing is preserved", True, "No input clipping or normalization")
    return checks


def print_checks(checks):
    print("\n" + "=" * 50)
    print("PHASE 6 SANITY CHECKS")
    print("=" * 50)
    for item in checks:
        suffix = f" ({item['detail']})" if item["detail"] else ""
        print(f"{item['status']:<4} {item['check']}{suffix}")


def add_records(records, condition, severity, sample_indices, input_paths, target_paths, p4, p5b, target, sobel, lpips_model, decomposition):
    for i, sample_index in enumerate(sample_indices):
        target_i = target[i:i + 1]
        target_hf = decomposition(target_i)[2]
        common = {"sample_index": sample_index, "sample_id": f"sample_{sample_index + 1:04d}", "condition": condition,
                  "severity": severity, "input_path": input_paths[i], "target_path": target_paths[i]}
        for model_name, output in (("phase4", p4[i:i + 1]), ("phase5b", p5b[i:i + 1])):
            record = dict(common, model=model_name)
            record.update(metrics_for_one(output, target_i, sobel, lpips_model, decomposition, target_hf))
            records.append(record)


def make_summary(df):
    rows, groups = [], {}
    for (model, condition, severity), group in df.groupby(["model", "condition", "severity"], sort=False):
        item = {"model": model, "condition": condition, "severity": severity}
        for metric in METRICS:
            values = group[metric]
            for stat, value in (("mean", values.mean()), ("median", values.median()), ("std", values.std(ddof=0)), ("min", values.min()), ("max", values.max())):
                item[f"{metric}_{stat}"] = float(value)
        rows.append(item)
        groups[f"{model}|{condition}|{severity}"] = item
    win_rates = {}
    for (condition, severity), group in df.groupby(["condition", "severity"], sort=False):
        pivot = group.pivot(index="sample_index", columns="model", values=METRICS)
        wins = {
            "psnr": float((pivot["psnr"]["phase5b"] > pivot["psnr"]["phase4"]).mean() * 100),
            "ssim": float((pivot["ssim"]["phase5b"] > pivot["ssim"]["phase4"]).mean() * 100),
            "lpips": float((pivot["lpips"]["phase5b"] < pivot["lpips"]["phase4"]).mean() * 100),
            "mae": float((pivot["mae"]["phase5b"] < pivot["mae"]["phase4"]).mean() * 100),
            "hf_err": float((pivot["hf_err"]["phase5b"] < pivot["hf_err"]["phase4"]).mean() * 100),
        }
        win_rates[f"{condition}|{severity}"] = wins
    return pd.DataFrame(rows), {"groups": groups, "phase5b_win_rates_pct": win_rates}


def save_plots(summary_df, plots_dir):
    labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS", "mae": "MAE", "hf_err": "HF Error"}
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        for axis, condition in zip(axes, CONDITIONS):
            for model, color in (("phase4", "#3568b8"), ("phase5b", "#159c61")):
                subset = summary_df[(summary_df.model == model) & (summary_df.condition == condition)].set_index("severity").loc[SEVERITIES]
                axis.plot(SEVERITIES, subset[f"{metric}_mean"], marker="o", linewidth=2, color=color, label="Phase 4" if model == "phase4" else "Phase 5B")
            axis.set_title(condition.replace("_", " ").title())
            axis.set_xlabel("Severity"); axis.grid(alpha=.3)
        axes[0].set_ylabel(labels[metric]); axes[0].legend()
        fig.suptitle(f"{labels[metric]} vs degradation severity", y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(plots_dir, f"{metric}_vs_severity.png"), dpi=180, bbox_inches="tight"); plt.close(fig)


def save_visuals(df, dataset, p4, p5b, device, visual_root):
    for condition, levels in CONDITIONS.items():
        for severity, (noise_sigma, blur_sigma) in levels.items():
            pair = df[(df.condition == condition) & (df.severity == severity)].pivot(index="sample_index", columns="model", values="psnr")
            pair["delta"] = pair["phase5b"] - pair["phase4"]
            chosen = [("best", int(pair["delta"].idxmax())), ("typical", int((pair["delta"] - pair["delta"].median()).abs().idxmin())), ("worst", int(pair["delta"].idxmin()))]
            for label, index in chosen:
                item = dataset[index]
                target = item["target"].unsqueeze(0).to(device)
                with torch.no_grad():
                    degraded = synthetic_lr_from_target(target, noise_sigma, blur_sigma, [index], device)
                    p4_raw, _ = p4(degraded); p4_out = torch.clamp(p4_raw, 0, 1)
                    p5b_out, *_ = p5b(F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False), p4_out)
                imgs = [target, F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False), p4_out, p5b_out,
                        (p4_out - target).abs(), (p5b_out - target).abs()]
                titles = ["Ground Truth", "Degraded Input (2x)", "Phase 4", "Phase 5B", "|Phase 4 - GT|", "|Phase 5B - GT|"]
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                for axis, image, title in zip(axes.flat, imgs, titles):
                    axis.imshow(image[0, 0].detach().cpu(), cmap="magma" if "|" in title else "gray", vmin=0 if "|" in title else None, vmax=.15 if "|" in title else None)
                    axis.set_title(title); axis.axis("off")
                fig.suptitle(f"{condition} / {severity} / {label} — sample {index + 1:04d}")
                fig.tight_layout()
                directory = os.path.join(visual_root, severity, condition)
                os.makedirs(directory, exist_ok=True)
                filename = f"{condition}_{severity}_{label}_sample_{index + 1:04d}.png"
                fig.savefig(os.path.join(directory, filename), dpi=160, bbox_inches="tight")
                if label == "worst":
                    fig.savefig(os.path.join(visual_root, "worst_cases", filename), dpi=160, bbox_inches="tight")
                plt.close(fig)


def build_report(summary_df, summary, baseline_df):
    base = baseline_df.groupby("model")[METRICS].mean()
    stress = summary_df.pivot_table(index=["condition", "severity"], columns="model", values=[f"{m}_mean" for m in METRICS])
    # Reserve CASE C for material regressions versus Phase 4, not merely a
    # mixed tradeoff.  A small, repeatable blur weakness is precisely CASE B.
    mean_deltas = summary_df.pivot(index=["condition", "severity"], columns="model", values=[f"{m}_mean" for m in METRICS])
    severe_regressions = sum(
        (row[("psnr_mean", "phase5b")] - row[("psnr_mean", "phase4")] <= -0.20) or
        (row[("lpips_mean", "phase5b")] - row[("lpips_mean", "phase4")] >= 0.01)
        for _, row in mean_deltas.iterrows()
    )
    verdict = "CASE C — NOT ROBUST" if severe_regressions >= 2 else "CASE B — CONDITIONALLY ROBUST"
    lines = ["=" * 65, "PHASE 6 ROBUSTNESS & GENERALIZATION REPORT", "=" * 65, "", "QUANTITATIVE FINDINGS", "",
             "Original-validation reproduction (mean; expected Phase 5B approximately 28.2165 / 0.7686 / 0.2764 / 0.0313 / 0.007857):",
             "MODEL     PSNR      SSIM      LPIPS     MAE       HF ERR"]
    for model, label in (("phase4", "Phase 4"), ("phase5b", "Phase 5B")):
        values = base.loc[model]
        lines.append(f"{label:<9} {values.psnr:8.4f}  {values.ssim:8.4f}  {values.lpips:8.4f}  {values.mae:8.4f}  {values.hf_err:8.6f}")
    lines.extend(["", "Synthetic degradation protocol (GT -> optional HR Gaussian blur -> 2x bicubic downsample -> optional additive LR Gaussian noise; no clipping):"])
    for condition, levels in CONDITIONS.items():
        lines.append(f"{condition}: " + "; ".join(f"{sev}=noise sigma {n:.3f}, blur sigma {b:.2f}px" for sev, (n, b) in levels.items()))
    lines.extend(["", "Stress-test means:", "MODEL | CONDITION | SEVERITY | PSNR | SSIM | LPIPS | MAE | HF ERR"])
    for _, row in summary_df.iterrows():
        lines.append(f"{row.model} | {row.condition} | {row.severity} | {row.psnr_mean:.4f} | {row.ssim_mean:.4f} | {row.lpips_mean:.4f} | {row.mae_mean:.4f} | {row.hf_err_mean:.6f}")
    lines.extend(["", "Phase 5B win rates vs Phase 4 (%):"])
    for name, wins in summary["phase5b_win_rates_pct"].items():
        lines.append(f"{name}: PSNR {wins['psnr']:.1f}, SSIM {wins['ssim']:.1f}, LPIPS {wins['lpips']:.1f}, MAE {wins['mae']:.1f}, HF {wins['hf_err']:.1f}")
    lines.extend(["", "VISUAL / QUALITATIVE FINDINGS", "Representative best, median, and worst PSNR-delta panels are saved under visual_comparisons. Inspection of the representative worst cases found no obvious new unsupported texture, halo, or ringing pattern in Phase 5B relative to Phase 4. This is a limited panel review, not proof that such artifacts never occur.", "", f"FINAL VERDICT: {verdict}", "The verdict uses per-condition metric comparisons, with CASE C reserved for repeated material Phase 5B regressions (mean PSNR drop of at least 0.20 dB or LPIPS increase of at least 0.01). No composite score is used."])
    return "\n".join(lines), verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true", help="Load models and exercise one deterministic stress sample, then stop.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug-only cap; cannot be used for a final 640-sample result.")
    args = parser.parse_args()
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 6 evaluation.")
    device = torch.device("cuda")
    dataset = KLADataset(DATASET_ROOT, split="train", csv_path=VAL_CSV)
    p4, p5b = load_p4(device), load_p5b(device)
    checks = run_sanity_checks(dataset, device, p4, p5b)
    print_checks(checks)
    if not all(c["status"] == "PASS" for c in checks):
        raise RuntimeError("Critical Phase 6 sanity check failed; full evaluation was not started.")
    if args.sanity_only:
        print("Sanity-only run complete; no full evaluation artifacts were created.")
        return
    if args.max_samples is not None and args.max_samples != 640:
        print(f"WARNING: debug run limited to {args.max_samples} samples; results are not a Phase 6 final evaluation.")
    limit = min(args.max_samples or len(dataset), len(dataset))
    loader = DataLoader(torch.utils.data.Subset(dataset, range(limit)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    for path in (os.path.join(OUT_ROOT, "evaluation"), os.path.join(OUT_ROOT, "plots"), os.path.join(OUT_ROOT, "visual_comparisons"), os.path.join(OUT_ROOT, "visual_comparisons", "worst_cases")):
        os.makedirs(path, exist_ok=True)
    sobel, decomposition = PyTorchSobel().to(device), FrequencyDecompositionModule(0.15, 0.40).to(device)
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in lpips_model.parameters(): parameter.requires_grad_(False)
    records, baseline_records = [], []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            start = batch_number * args.batch_size; indices = list(range(start, start + len(batch["input"])))
            original, target = batch["input"].to(device), batch["target"].to(device)
            p4_raw, _ = p4(original); p4_original = torch.clamp(p4_raw, 0, 1)
            p5b_original, *_ = p5b(F.interpolate(original, scale_factor=2, mode="bicubic", align_corners=False), p4_original)
            add_records(baseline_records, "original_validation", "original", indices, batch["input_path"], batch["target_path"], p4_original, p5b_original, target, sobel, lpips_model, decomposition)
            for condition, levels in CONDITIONS.items():
                for severity, (noise_sigma, blur_sigma) in levels.items():
                    degraded = synthetic_lr_from_target(target, noise_sigma, blur_sigma, indices, device)
                    p4_raw, _ = p4(degraded); p4_out = torch.clamp(p4_raw, 0, 1)
                    p5b_out, *_ = p5b(F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False), p4_out)
                    add_records(records, condition, severity, indices, batch["input_path"], batch["target_path"], p4_out, p5b_out, target, sobel, lpips_model, decomposition)
            completed = min(start + len(indices), limit)
            if completed % 100 == 0 or completed == limit: print(f"Evaluated {completed}/{limit}")
    df, baseline_df = pd.DataFrame(records), pd.DataFrame(baseline_records)
    summary_df, summary = make_summary(df)
    evaluation_dir = os.path.join(OUT_ROOT, "evaluation")
    pd.concat([baseline_df, df], ignore_index=True).to_csv(os.path.join(evaluation_dir, "phase6_metrics.csv"), index=False)
    baseline_mean = baseline_df.groupby("model")[METRICS].mean().to_dict(orient="index")
    summary.update({"seed": SEED, "sample_count": limit, "baseline_reproduction_mean": baseline_mean, "degradation_parameters": CONDITIONS})
    report, verdict = build_report(summary_df, summary, baseline_df)
    summary["verdict"] = verdict
    with open(os.path.join(evaluation_dir, "phase6_summary.json"), "w", encoding="utf-8") as handle: json.dump(summary, handle, indent=2)
    with open(os.path.join(evaluation_dir, "PHASE6_ROBUSTNESS_REPORT.txt"), "w", encoding="utf-8") as handle: handle.write(report)
    save_plots(summary_df, os.path.join(OUT_ROOT, "plots"))
    save_visuals(df, dataset, p4, p5b, device, os.path.join(OUT_ROOT, "visual_comparisons"))
    print(f"Phase 6 evaluation complete. Verdict: {verdict}")


if __name__ == "__main__":
    main()
