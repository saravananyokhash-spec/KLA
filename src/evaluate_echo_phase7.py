"""Phase 7 evaluation - Phase 7 vs Phase 5B robustness comparison.

Reuses the Phase 6 degradation protocol and metrics.  Phase 5B is the baseline.
Run ``python src/evaluate_echo_phase7.py --sanity-only`` before the full run.
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

from utils import set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from degradation_utils import PHASE6_CONDITIONS, synthetic_lr_from_target
from phase5_model import FrequencyDecompositionModule, SpatialFrequencyRestorationNet
from train_echo_phase43 import PyTorchSobel, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import calculate_psnr


SEED = 42
DATASET_ROOT = "D:/kla"
VAL_CSV = "outputs/baseline/val_split.csv"
P4_CKPT = "outputs/echo_phase4/checkpoints/echo_best.pth"
P5B_CKPT = "outputs/phase5b/checkpoints/echo_phase5b_best.pth"
# Default Phase 7 checkpoint – use the fine‑tuned EXP_B best checkpoint
P7_DEFAULT_CKPT = "outputs/phase7_finetune/exp_b/checkpoints/echo_phase7_ft_best.pth"
# Alias for backward compatibility (used throughout the script)
P7_CKPT = P7_DEFAULT_CKPT
# Output root for evaluation results – keep them under the EXP_B directory
OUT_ROOT = "outputs/phase7_finetune/exp_b"


CONDITIONS = PHASE6_CONDITIONS
SEVERITIES = ["low", "medium", "high"]
METRICS = ["psnr", "ssim", "lpips", "mae", "hf_err"]

P5B_BASELINE = {
    "original_validation": {"psnr": 28.2165, "ssim": 0.7686, "lpips": 0.2764, "mae": 0.0313, "hf_err": 0.007857},
    "gaussian_blur_medium": {"psnr": 27.8966, "ssim": 0.7330, "lpips": 0.4739, "mae": 0.0325, "hf_err": 0.008075},
    "gaussian_blur_high": {"psnr": 26.3443, "ssim": 0.6920, "lpips": 0.5398, "mae": 0.0372, "hf_err": 0.008240},
    "gaussian_noise_high": {"psnr": 27.2416, "ssim": 0.6922, "lpips": 0.3669, "mae": 0.0354, "hf_err": 0.008358},
    "noise_plus_blur_high": {"psnr": 24.9947, "ssim": 0.6019, "lpips": 0.4878, "mae": 0.0439, "hf_err": 0.008871},
}


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    elif torch.is_tensor(obj):
        return make_json_safe(obj.tolist()) if obj.ndim > 0 else make_json_safe(obj.item())
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    else:
        return obj


def load_sfr_model(ckpt_path, device):
    model = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def run_inference(p4, sfr_model, degraded_lr):
    p4_raw, _ = p4(degraded_lr)
    p4_hr = torch.clamp(p4_raw, 0.0, 1.0)
    lr_up = F.interpolate(degraded_lr, scale_factor=2, mode="bicubic", align_corners=False)
    out, *_ = sfr_model(lr_up, p4_hr)
    return out


def metrics_for_one(pred, target, lpips_model, decomposition, target_hf):
    return {
        "psnr": float(calculate_psnr(pred, target)),
        "ssim": float(ssim_pytorch(pred, target).item()),
        "lpips": float(ssim_lpips_differentiable(pred, target, lpips_model).item()),
        "mae": float(F.l1_loss(pred, target).item()),
        "hf_err": float(F.l1_loss(decomposition(pred)[2], target_hf).item()),
    }


def add_records(records, condition, severity, sample_indices, input_paths, target_paths, p5b_out, p7_out, target, lpips_model, decomposition):
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
        for model_name, output in (("phase5b", p5b_out[i : i + 1]), ("phase7", p7_out[i : i + 1])):
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

    win_rates = {}
    for (condition, severity), group in df.groupby(["condition", "severity"], sort=False):
        pivot = group.pivot(index="sample_index", columns="model", values=METRICS)
        win_rates[f"{condition}|{severity}"] = {
            "psnr": float((pivot["psnr"]["phase7"] > pivot["psnr"]["phase5b"]).mean() * 100),
            "ssim": float((pivot["ssim"]["phase7"] > pivot["ssim"]["phase5b"]).mean() * 100),
            "lpips": float((pivot["lpips"]["phase7"] < pivot["lpips"]["phase5b"]).mean() * 100),
            "mae": float((pivot["mae"]["phase7"] < pivot["mae"]["phase5b"]).mean() * 100),
            "hf_err": float((pivot["hf_err"]["phase7"] < pivot["hf_err"]["phase5b"]).mean() * 100),
        }
    return pd.DataFrame(rows), win_rates


def pct_change(new_val, old_val):
    if abs(old_val) < 1e-12:
        return 0.0
    return 100.0 * (new_val - old_val) / abs(old_val)


def build_verdict(summary_df, baseline_df):
    base = baseline_df.groupby("model")[METRICS].mean()
    p5b_base = base.loc["phase5b"]
    p7_base = base.loc["phase7"]

    orig_ok = (
        p7_base.psnr >= p5b_base.psnr - 0.02
        and p7_base.ssim >= p5b_base.ssim - 0.002
        and p7_base.lpips <= p5b_base.lpips + 0.005
        and p7_base.mae <= p5b_base.mae + 0.001
        and p7_base.hf_err <= p5b_base.hf_err + 0.0001
    )

    blur_improved = True
    for severity in ["medium", "high"]:
        p7_row = summary_df[
            (summary_df.model == "phase7") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == severity)
        ].iloc[0]
        p5b_row = summary_df[
            (summary_df.model == "phase5b") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == severity)
        ].iloc[0]
        improvements = 0
        if p7_row.psnr_mean > p5b_row.psnr_mean + 0.02:
            improvements += 1
        if p7_row.ssim_mean > p5b_row.ssim_mean + 0.001:
            improvements += 1
        if p7_row.lpips_mean < p5b_row.lpips_mean - 0.005:
            improvements += 1
        if p7_row.mae_mean < p5b_row.mae_mean - 0.0005:
            improvements += 1
        if p7_row.hf_err_mean < p5b_row.hf_err_mean - 0.00003:
            improvements += 1
        if improvements < 2:
            blur_improved = False

    noise_high = summary_df[
        (summary_df.model == "phase7") & (summary_df.condition == "gaussian_noise") & (summary_df.severity == "high")
    ].iloc[0]
    p5b_noise_high = summary_df[
        (summary_df.model == "phase5b") & (summary_df.condition == "gaussian_noise") & (summary_df.severity == "high")
    ].iloc[0]
    noise_ok = (
        noise_high.psnr_mean >= p5b_noise_high.psnr_mean - 0.03
        and noise_high.ssim_mean >= p5b_noise_high.ssim_mean - 0.003
        and noise_high.lpips_mean <= p5b_noise_high.lpips_mean + 0.005
    )

    combo_high = summary_df[
        (summary_df.model == "phase7") & (summary_df.condition == "noise_plus_blur") & (summary_df.severity == "high")
    ].iloc[0]
    p5b_combo_high = summary_df[
        (summary_df.model == "phase5b") & (summary_df.condition == "noise_plus_blur") & (summary_df.severity == "high")
    ].iloc[0]
    combo_ok = combo_high.lpips_mean <= p5b_combo_high.lpips_mean + 0.005 and combo_high.hf_err_mean <= p5b_combo_high.hf_err_mean + 0.0001

    material_regression = (
        p7_base.psnr < p5b_base.psnr - 0.05
        or p7_base.lpips > p5b_base.lpips + 0.01
        or not noise_ok
    )

    if material_regression:
        verdict = "CASE C - REGRESSION"
    elif orig_ok and blur_improved and noise_ok and combo_ok:
        verdict = "CASE A - IMPROVED"
    elif orig_ok and not material_regression:
        verdict = "CASE B - NO MATERIAL IMPROVEMENT"
    else:
        verdict = "CASE C - REGRESSION"

    return verdict, orig_ok, blur_improved, noise_ok, combo_ok


def build_report(summary_df, baseline_df, win_rates, verdict):
    base = baseline_df.groupby("model")[METRICS].mean()
    lines = [
        "=" * 70,
        "PHASE 7 EVALUATION REPORT - Phase 7 vs Phase 5B",
        "=" * 70,
        "",
        "PHASE 5B BASELINE (reference)",
        f"  Original validation: PSNR={P5B_BASELINE['original_validation']['psnr']:.4f}  SSIM={P5B_BASELINE['original_validation']['ssim']:.4f}  "
        f"LPIPS={P5B_BASELINE['original_validation']['lpips']:.4f}  MAE={P5B_BASELINE['original_validation']['mae']:.4f}  "
        f"HF ERR={P5B_BASELINE['original_validation']['hf_err']:.6f}",
        "",
        "ORIGINAL VALIDATION (640 samples, seed 42)",
        "MODEL     PSNR      SSIM      LPIPS     MAE       HF ERR",
    ]
    for model, label in (("phase5b", "Phase 5B"), ("phase7", "Phase 7")):
        values = base.loc[model]
        lines.append(
            f"{label:<9} {values.psnr:8.4f}  {values.ssim:8.4f}  {values.lpips:8.4f}  {values.mae:8.4f}  {values.hf_err:8.6f}"
        )

    p5b_orig = base.loc["phase5b"]
    p7_orig = base.loc["phase7"]
    lines.extend(["", "ORIGINAL VALIDATION DELTAS (Phase 7 - Phase 5B):"])
    for metric in METRICS:
        delta = p7_orig[metric] - p5b_orig[metric]
        lines.append(f"  {metric.upper():<8}: {delta:+.6f}  ({pct_change(p7_orig[metric], p5b_orig[metric]):+.2f}%)")

    lines.extend(["", "STRESS-TEST COMPARISON (mean metrics)", "CONDITION | SEVERITY | METRIC | Phase 5B | Phase 7 | ABS DELTA | PCT DELTA"])
    for condition in CONDITIONS:
        for severity in SEVERITIES:
            p5b_row = summary_df[
                (summary_df.model == "phase5b") & (summary_df.condition == condition) & (summary_df.severity == severity)
            ]
            p7_row = summary_df[
                (summary_df.model == "phase7") & (summary_df.condition == condition) & (summary_df.severity == severity)
            ]
            if p5b_row.empty or p7_row.empty:
                continue
            p5b_row = p5b_row.iloc[0]
            p7_row = p7_row.iloc[0]
            for metric in METRICS:
                old_v = p5b_row[f"{metric}_mean"]
                new_v = p7_row[f"{metric}_mean"]
                delta = new_v - old_v
                lines.append(
                    f"{condition} | {severity} | {metric} | {old_v:.4f} | {new_v:.4f} | {delta:+.4f} | {pct_change(new_v, old_v):+.2f}%"
                )

    lines.extend(["", "Phase 7 win rates vs Phase 5B (%):"])
    for name, wins in win_rates.items():
        lines.append(
            f"  {name}: PSNR {wins['psnr']:.1f}, SSIM {wins['ssim']:.1f}, LPIPS {wins['lpips']:.1f}, "
            f"MAE {wins['mae']:.1f}, HF {wins['hf_err']:.1f}"
        )

    blur_medium = summary_df[
        (summary_df.model == "phase7") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "medium")
    ].iloc[0]
    blur_high = summary_df[
        (summary_df.model == "phase7") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "high")
    ].iloc[0]
    p5b_blur_m = summary_df[
        (summary_df.model == "phase5b") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "medium")
    ].iloc[0]
    p5b_blur_h = summary_df[
        (summary_df.model == "phase5b") & (summary_df.condition == "gaussian_blur") & (summary_df.severity == "high")
    ].iloc[0]

    lines.extend(
        [
            "",
            "FINDINGS",
            f"  Original validation preserved : {'YES' if p7_orig.psnr >= p5b_orig.psnr - 0.02 and p7_orig.lpips <= p5b_orig.lpips + 0.005 else 'NO'}",
            f"  Blur medium PSNR delta        : {blur_medium.psnr_mean - p5b_blur_m.psnr_mean:+.4f} dB",
            f"  Blur high PSNR delta          : {blur_high.psnr_mean - p5b_blur_h.psnr_mean:+.4f} dB",
            f"  Blur medium LPIPS delta       : {blur_medium.lpips_mean - p5b_blur_m.lpips_mean:+.4f}",
            f"  Blur high LPIPS delta         : {blur_high.lpips_mean - p5b_blur_h.lpips_mean:+.4f}",
            "",
            "QUALITATIVE",
            "  Representative panels saved under visual_comparisons/. Review worst-case blur and noise+blur panels for ringing, halos, or unsupported texture.",
            "",
            f"FINAL VERDICT: {verdict}",
            "=" * 70,
        ]
    )
    return "\n".join(lines)


def save_plots(summary_df, plots_dir):
    labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS", "mae": "MAE", "hf_err": "HF Error"}
    for metric in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, condition in zip(axes, CONDITIONS):
            for model, color in (("phase5b", "#3568b8"), ("phase7", "#c45a11")):
                subset = summary_df[(summary_df.model == model) & (summary_df.condition == condition)].set_index("severity").loc[SEVERITIES]
                axis.plot(
                    SEVERITIES,
                    subset[f"{metric}_mean"],
                    marker="o",
                    linewidth=2,
                    color=color,
                    label="Phase 5B" if model == "phase5b" else "Phase 7",
                )
            axis.set_title(condition.replace("_", " ").title())
            axis.set_xlabel("Severity")
            axis.grid(alpha=0.3)
        axes[0].set_ylabel(labels[metric])
        axes[0].legend()
        fig.suptitle(f"{labels[metric]} vs degradation severity (Phase 7 vs Phase 5B)", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{metric}_vs_severity.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_visuals(df, dataset, p4, p5b, p7, device, visual_root):
    for condition, levels in CONDITIONS.items():
        for severity, (noise_sigma, blur_sigma) in levels.items():
            pair = df[(df.condition == condition) & (df.severity == severity)].pivot(index="sample_index", columns="model", values="psnr")
            pair["delta"] = pair["phase7"] - pair["phase5b"]
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
                    p5b_out = run_inference(p4, p5b, degraded)
                    p7_out = run_inference(p4, p7, degraded)
                imgs = [
                    target,
                    F.interpolate(degraded, scale_factor=2, mode="bicubic", align_corners=False),
                    p5b_out,
                    p7_out,
                    (p5b_out - target).abs(),
                    (p7_out - target).abs(),
                ]
                titles = ["Ground Truth", "Degraded Input (2x)", "Phase 5B", "Phase 7", "|Phase 5B - GT|", "|Phase 7 - GT|"]
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                for axis, image, title in zip(axes.flat, imgs, titles):
                    axis.imshow(
                        image[0, 0].detach().cpu(),
                        cmap="magma" if "|" in title else "gray",
                        vmin=0 if "|" in title else None,
                        vmax=0.15 if "|" in title else None,
                    )
                    axis.set_title(title)
                    axis.axis("off")
                fig.suptitle(f"{condition} / {severity} / {label} - sample {index + 1:04d}")
                fig.tight_layout()
                directory = os.path.join(visual_root, severity, condition)
                os.makedirs(directory, exist_ok=True)
                filename = f"{condition}_{severity}_{label}_sample_{index + 1:04d}.png"
                fig.savefig(os.path.join(directory, filename), dpi=160, bbox_inches="tight")
                if label == "worst":
                    os.makedirs(os.path.join(visual_root, "worst_cases"), exist_ok=True)
                    fig.savefig(os.path.join(visual_root, "worst_cases", filename), dpi=160, bbox_inches="tight")
                plt.close(fig)


def run_sanity_checks(dataset, device, p4, p5b, p7):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    check("CUDA available", device.type == "cuda", str(device))
    check("Phase 4 checkpoint exists", os.path.isfile(P4_CKPT), P4_CKPT)
    check("Phase 5B checkpoint exists", os.path.isfile(P5B_CKPT), P5B_CKPT)
    check("Phase 7 checkpoint exists", os.path.isfile(P7_CKPT), P7_CKPT)
    check("Validation sample count is 640", len(dataset) == 640, str(len(dataset)))
    sample = dataset[0]
    inp, target = sample["input"].unsqueeze(0).to(device), sample["target"].unsqueeze(0).to(device)
    with torch.no_grad():
        p5b_out = run_inference(p4, p5b, inp)
        p7_out = run_inference(p4, p7, inp)
        stress = synthetic_lr_from_target(target, 0.03, 1.0, [0], device, seed=SEED)
        stress_p7 = run_inference(p4, p7, stress)
    check("Output dimensions correct", p5b_out.shape == target.shape and p7_out.shape == target.shape)
    check("Outputs finite", bool(torch.isfinite(p5b_out).all() and torch.isfinite(p7_out).all() and torch.isfinite(stress_p7).all()))
    check("Seed fixed", SEED == 42)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 7 evaluation.")

    device = torch.device("cuda")
    dataset = KLADataset(DATASET_ROOT, split="val", csv_path=VAL_CSV)
    p4 = load_p4(device)
    p5b = load_sfr_model(P5B_CKPT, device)
    p7 = load_sfr_model(P7_CKPT, device)

    checks = run_sanity_checks(dataset, device, p4, p5b, p7)
    print("\nPHASE 7 EVAL SANITY CHECKS")
    for item in checks:
        print(f"{item['status']:<4} {item['check']}")
    if not all(c["status"] == "PASS" for c in checks):
        raise RuntimeError("Phase 7 sanity checks failed.")

    if args.sanity_only:
        print("Sanity-only run complete.")
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
            p5b_orig = run_inference(p4, p5b, original)
            p7_orig = run_inference(p4, p7, original)
            add_records(
                baseline_records,
                "original_validation",
                "original",
                indices,
                batch["input_path"],
                batch["target_path"],
                p5b_orig,
                p7_orig,
                target,
                lpips_model,
                decomposition,
            )
            for condition, levels in CONDITIONS.items():
                for severity, (noise_sigma, blur_sigma) in levels.items():
                    degraded = synthetic_lr_from_target(target, noise_sigma, blur_sigma, indices, device, seed=SEED)
                    p5b_out = run_inference(p4, p5b, degraded)
                    p7_out = run_inference(p4, p7, degraded)
                    add_records(
                        records,
                        condition,
                        severity,
                        indices,
                        batch["input_path"],
                        batch["target_path"],
                        p5b_out,
                        p7_out,
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
    pd.concat([baseline_df, df], ignore_index=True).to_csv(os.path.join(evaluation_dir, "phase7_metrics.csv"), index=False)

    verdict, orig_ok, blur_improved, noise_ok, combo_ok = build_verdict(summary_df, baseline_df)
    report = build_report(summary_df, baseline_df, win_rates, verdict)
    summary = {
        "seed": SEED,
        "sample_count": limit,
        "verdict": verdict,
        "original_preserved": orig_ok,
        "blur_improved": blur_improved,
        "noise_preserved": noise_ok,
        "combo_preserved": combo_ok,
        "phase7_win_rates_pct": win_rates,
        "degradation_parameters": CONDITIONS,
    }
    with open(os.path.join(evaluation_dir, "phase7_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(summary), handle, indent=2)
    with open(os.path.join(evaluation_dir, "PHASE7_REPORT.txt"), "w", encoding="utf-8") as handle:
        handle.write(report)

    save_plots(summary_df, os.path.join(OUT_ROOT, "plots"))
    save_visuals(df, dataset, p4, p5b, p7, device, os.path.join(OUT_ROOT, "visual_comparisons"))
    print(f"\nPhase 7 evaluation complete. Verdict: {verdict}")
    print(report)


if __name__ == "__main__":
    main()
