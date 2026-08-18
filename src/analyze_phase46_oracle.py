import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import scipy.ndimage
from skimage.filters import sobel
import matplotlib.pyplot as plt
import lpips

from utils import load_config, set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from metrics import compute_psnr, compute_ssim, compute_lpips

def decompose_frequencies(img, r_low=15, r_mid=64):
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r2 = x**2 + y**2
    
    mask_low = r2 < r_low**2
    mask_mid = (r2 >= r_low**2) & (r2 < r_mid**2)
    mask_high = r2 >= r_mid**2
    
    img_low = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_low)))
    img_mid = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_mid)))
    img_high = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_high)))
    
    return img_low, img_mid, img_high

def safe_pearson(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    var_a = a_flat.var()
    var_b = b_flat.var()
    if var_a < 1e-12 or var_b < 1e-12:
        return 0.0
    cov = np.cov(a_flat, b_flat)[0, 1]
    std_a = np.sqrt(var_a)
    std_b = np.sqrt(var_b)
    return float(cov / (std_a * std_b + 1e-12))

def normalize_map(m):
    m_min = m.min()
    m_max = m.max()
    return (m - m_min) / (m_max - m_min + 1e-8)

def main():
    phase46_dir = "outputs/phase46_oracle"
    galleries_dir = os.path.join(phase46_dir, "galleries")
    
    os.makedirs(phase46_dir, exist_ok=True)
    os.makedirs(os.path.join(galleries_dir, "highly_recoverable"), exist_ok=True)
    os.makedirs(os.path.join(galleries_dir, "partially_recoverable"), exist_ok=True)
    os.makedirs(os.path.join(galleries_dir, "ambiguous"), exist_ok=True)
    os.makedirs(os.path.join(galleries_dir, "failures"), exist_ok=True)
    
    # 1. Safety Checks
    p4_checkpoint_path = "outputs/echo_phase4/checkpoints/echo_best.pth"
    train_split_path = "outputs/baseline/train_split.csv"
    val_split_path = "outputs/baseline/val_split.csv"
    p45_csv_path = "outputs/phase45_recoverability/sample_analysis.csv"
    
    for p in [p4_checkpoint_path, train_split_path, val_split_path, p45_csv_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Safety Error: required path {p} does not exist.")
    print("1. File existence checks: PASSED")
    
    train_split = pd.read_csv(train_split_path)
    val_split = pd.read_csv(val_split_path)
    train_fns = set(os.path.basename(p) for p in train_split["input_path"])
    val_fns = set(os.path.basename(p) for p in val_split["input_path"])
    if len(train_fns.intersection(val_fns)) > 0:
        raise ValueError("Safety Error: train/validation splits are not disjoint!")
    print("2. Split disjointness verification: PASSED")
    
    df_p45 = pd.read_csv(p45_csv_path)[["image_id", "recoverability_group"]]
    
    config_path = "configs/echo.yaml"
    config = load_config(config_path)
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load LPIPS
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Load validation split dataset
    val_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path=val_split_path
    )
    
    # Load Phase 4 model
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablation", {})
    model_p4 = BaselineECHOModel(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 6),
        ablation=ablation_cfg
    ).to(device)
    p4_chk = torch.load(p4_checkpoint_path, map_location=device)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    
    records = []
    print("\nRunning oracle and headroom analysis on all 640 validation images...")
    
    # We will test Oracle Gates and Headroom
    # To keep execution efficient, we process validation split.
    # For every image, we save metrics for Phase 4 baseline, Oracle Gates, and headroom.
    for idx in range(len(val_dataset)):
        batch = val_dataset[idx]
        
        inp_tensor = batch["input"]
        tgt_tensor = batch["target"]
        inp_path = batch["input_path"]
        filename = os.path.basename(inp_path)
        
        inp_batch = inp_tensor.unsqueeze(0).to(device)
        tgt_batch = tgt_tensor.unsqueeze(0).to(device)
        
        # LR upsampled
        lr_up_batch = torch.nn.functional.interpolate(
            inp_batch, scale_factor=2, mode="bicubic", align_corners=False
        )
        lr_up_arr = np.clip(lr_up_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        # GT
        gt_arr = tgt_tensor.squeeze(0).numpy()
        
        # Phase 4 prediction
        with torch.no_grad():
            p4_batch, _ = model_p4(inp_batch)
        p4_arr = np.clip(p4_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        # 1. Base Metrics
        p4_psnr = compute_psnr(p4_batch.squeeze(0), tgt_tensor)
        p4_ssim = compute_ssim(p4_batch.squeeze(0), tgt_tensor)
        p4_lpips = compute_lpips(p4_batch, tgt_batch, lpips_model, device)
        
        # 2. Decompose GT and Phase 4
        _, _, hf_lr = decompose_frequencies(lr_up_arr)
        _, _, hf_gt = decompose_frequencies(gt_arr)
        _, _, hf_p4 = decompose_frequencies(p4_arr)
        
        edge_lr = sobel(lr_up_arr)
        edge_gt = sobel(gt_arr)
        edge_p4 = sobel(p4_arr)
        
        # 3. Create Oracle Gates
        # Gate A: Pixel Residual
        g_a = normalize_map(np.abs(gt_arr - p4_arr))
        # Gate B: HF Residual
        g_b = normalize_map(np.abs(hf_gt - hf_p4))
        # Gate C: Edge Residual
        g_c = normalize_map(np.abs(edge_gt - edge_p4))
        # Gate D: Combined
        g_d = normalize_map(0.4 * g_a + 0.4 * g_b + 0.2 * g_c)
        
        # 4. Oracle Correction using LR HF (Alpha = 0.10)
        corrected_arr = np.clip(p4_arr + 0.10 * g_d * hf_lr, 0.0, 1.0)
        corr_tensor = torch.from_numpy(corrected_arr).unsqueeze(0).unsqueeze(0).to(device)
        
        oracle_hf_psnr = compute_psnr(corr_tensor.squeeze(0), tgt_tensor)
        oracle_hf_ssim = compute_ssim(corr_tensor.squeeze(0), tgt_tensor)
        oracle_hf_lpips = compute_lpips(corr_tensor, tgt_batch, lpips_model, device)
        
        # 5. Second Oracle: Ideal direction (Alpha = 0.50)
        # corrected = p4 + alpha * Gate * (GT - p4)
        ideal_corrected_arr = np.clip(p4_arr + 0.50 * g_d * (gt_arr - p4_arr), 0.0, 1.0)
        ideal_tensor = torch.from_numpy(ideal_corrected_arr).unsqueeze(0).unsqueeze(0).to(device)
        
        oracle_ideal_psnr = compute_psnr(ideal_tensor.squeeze(0), tgt_tensor)
        oracle_ideal_ssim = compute_ssim(ideal_tensor.squeeze(0), tgt_tensor)
        oracle_ideal_lpips = compute_lpips(ideal_tensor, tgt_batch, lpips_model, device)
        
        # 6. Candidate Evidence Maps
        # Edge Evidence
        ev_edge = normalize_map(edge_lr)
        # HF Evidence
        ev_hf = normalize_map(np.abs(hf_lr))
        # Local Texture (using PyTorch uniform filter for variance)
        mean_lr = torch.nn.functional.avg_pool2d(lr_up_batch, 5, stride=1, padding=2)
        mean_lr_sq = torch.nn.functional.avg_pool2d(lr_up_batch**2, 5, stride=1, padding=2)
        ev_tex = torch.clamp(mean_lr_sq - mean_lr**2, 0.0).squeeze(0).squeeze(0).cpu().numpy()
        ev_tex = normalize_map(ev_tex)
        
        # Target GT support mask
        gt_support = normalize_map(0.5 * np.abs(hf_gt - hf_p4) + 0.5 * np.abs(edge_gt - edge_p4))
        
        # Pearson correlations with support mask
        ev_edge_corr = safe_pearson(ev_edge, gt_support)
        ev_hf_corr = safe_pearson(ev_hf, gt_support)
        ev_tex_corr = safe_pearson(ev_tex, gt_support)
        
        # Headroom
        headroom_psnr = oracle_ideal_psnr - p4_psnr
        headroom_ssim = oracle_ideal_ssim - p4_ssim
        headroom_lpips = p4_lpips - oracle_ideal_lpips # Positive is improvement
        
        records.append({
            "image_id": filename,
            "phase4_psnr": p4_psnr,
            "phase4_ssim": p4_ssim,
            "phase4_lpips": p4_lpips,
            "oracle_hf_psnr": oracle_hf_psnr,
            "oracle_hf_ssim": oracle_hf_ssim,
            "oracle_hf_lpips": oracle_hf_lpips,
            "oracle_combined_psnr": oracle_ideal_psnr,
            "oracle_combined_ssim": oracle_ideal_ssim,
            "oracle_combined_lpips": oracle_ideal_lpips,
            "evidence_edge_correlation": ev_edge_corr,
            "evidence_hf_correlation": ev_hf_corr,
            "evidence_texture_correlation": ev_tex_corr,
            "recoverable_headroom_psnr": headroom_psnr,
            "recoverable_headroom_ssim": headroom_ssim,
            "recoverable_headroom_lpips": headroom_lpips,
            "p4_arr": p4_arr,
            "lr_up_arr": lr_up_arr,
            "gt_arr": gt_arr,
            "ev_edge": ev_edge,
            "ev_hf": ev_hf,
            "gt_support": gt_support,
            "ideal_corrected_arr": ideal_corrected_arr
        })
        
        if (idx + 1) % 150 == 0:
            print(f"Evaluated {idx + 1}/640 samples.")
            
    df = pd.DataFrame(records)
    
    # 7. Merge with Phase 4.5 Group classification
    df = df.merge(df_p45, on="image_id")
    
    # Save CSV tables
    df_csv = df.drop(columns=["p4_arr", "lr_up_arr", "gt_arr", "ev_edge", "ev_hf", "gt_support", "ideal_corrected_arr"])
    df_csv.to_csv(os.path.join(phase46_dir, "oracle_results.csv"), index=False)
    
    # Group results
    group_stats = []
    for g_name in ["Group A", "Group B", "Group C"]:
        df_g = df[df["recoverability_group"] == g_name]
        g_count = len(df_g)
        
        g_p4_psnr = df_g["phase4_psnr"].mean()
        g_p4_ssim = df_g["phase4_ssim"].mean()
        g_p4_lpips = df_g["phase4_lpips"].mean()
        
        g_ideal_psnr = df_g["oracle_combined_psnr"].mean()
        g_ideal_ssim = df_g["oracle_combined_ssim"].mean()
        g_ideal_lpips = df_g["oracle_combined_lpips"].mean()
        
        g_hr_psnr = df_g["recoverable_headroom_psnr"].mean()
        g_hr_ssim = df_g["recoverable_headroom_ssim"].mean()
        g_hr_lpips = df_g["recoverable_headroom_lpips"].mean()
        
        group_stats.append({
            "Group": g_name,
            "Count": g_count,
            "P4_PSNR": g_p4_psnr, "P4_SSIM": g_p4_ssim, "P4_LPIPS": g_p4_lpips,
            "Oracle_PSNR": g_ideal_psnr, "Oracle_SSIM": g_ideal_ssim, "Oracle_LPIPS": g_ideal_lpips,
            "Headroom_PSNR": g_hr_psnr, "Headroom_SSIM": g_hr_ssim, "Headroom_LPIPS": g_hr_lpips
        })
    df_groups = pd.DataFrame(group_stats)
    df_groups.to_csv(os.path.join(phase46_dir, "group_results.csv"), index=False)
    
    # Evidence quality
    ev_edge_mean = df["evidence_edge_correlation"].mean()
    ev_hf_mean = df["evidence_hf_correlation"].mean()
    ev_tex_mean = df["evidence_texture_correlation"].mean()
    
    df_ev = pd.DataFrame([{
        "edge_evidence_correlation": ev_edge_mean,
        "hf_evidence_correlation": ev_hf_mean,
        "texture_evidence_correlation": ev_tex_mean
    }])
    df_ev.to_csv(os.path.join(phase46_dir, "evidence_quality.csv"), index=False)
    
    # Global Averages
    base_psnr = float(df["phase4_psnr"].mean())
    base_ssim = float(df["phase4_ssim"].mean())
    base_lpips = float(df["phase4_lpips"].mean())
    
    oracle_psnr = float(df["oracle_combined_psnr"].mean())
    oracle_ssim = float(df["oracle_combined_ssim"].mean())
    oracle_lpips = float(df["oracle_combined_lpips"].mean())
    
    hr_psnr = oracle_psnr - base_psnr
    hr_ssim = oracle_ssim - base_ssim
    hr_lpips = base_lpips - oracle_lpips
    
    # --- VISUAL GALLERIES (9 panels landscape format) ---
    print("\nGenerating 9-panel comparative galleries...")
    best_a = df[df["recoverability_group"] == "Group A"].sort_values(by="phase4_psnr", ascending=False).iloc[0]
    best_b = df[df["recoverability_group"] == "Group B"].sort_values(by="phase4_psnr", ascending=False).iloc[0]
    best_c = df[df["recoverability_group"] == "Group C"].sort_values(by="phase4_psnr", ascending=False).iloc[0]
    worst_p4 = df.sort_values(by="phase4_psnr", ascending=True).iloc[0]
    
    gallery_samples = [
        ("highly_recoverable", best_a),
        ("partially_recoverable", best_b),
        ("ambiguous", best_c),
        ("failures", worst_p4)
    ]
    
    for folder, row in gallery_samples:
        fn = row.image_id
        
        p4_arr = row.p4_arr
        lr_up_arr = row.lr_up_arr
        gt_arr = row.gt_arr
        
        ev_edge = row.ev_edge
        ev_hf = row.ev_hf
        gt_support = row.gt_support
        ideal_corrected_arr = row.ideal_corrected_arr
        
        abs_err = np.abs(p4_arr - gt_arr)
        
        # 9 Panels Plot
        fig, axes = plt.subplots(1, 9, figsize=(27, 3.5))
        
        axes[0].imshow(lr_up_arr, cmap="gray")
        axes[0].set_title("1. NoisyLR")
        axes[0].axis("off")
        
        axes[1].imshow(lr_up_arr, cmap="gray") # Bicubic is upsampled
        axes[1].set_title("2. Bicubic")
        axes[1].axis("off")
        
        axes[2].imshow(p4_arr, cmap="gray")
        axes[2].set_title(f"3. Phase 4\nPSNR: {row.phase4_psnr:.2f}")
        axes[2].axis("off")
        
        axes[3].imshow(gt_arr, cmap="gray")
        axes[3].set_title("4. Ground Truth")
        axes[3].axis("off")
        
        axes[4].imshow(ev_edge, cmap="gray")
        axes[4].set_title("5. Edge Evidence")
        axes[4].axis("off")
        
        axes[5].imshow(ev_hf, cmap="gray")
        axes[5].set_title("6. HF Evidence")
        axes[5].axis("off")
        
        axes[6].imshow(gt_support, cmap="gray")
        axes[6].set_title("7. Oracle Support")
        axes[6].axis("off")
        
        axes[7].imshow(ideal_corrected_arr, cmap="gray")
        axes[7].set_title("8. Oracle Corrected")
        axes[7].axis("off")
        
        axes[8].imshow(abs_err, cmap="hot")
        axes[8].set_title("9. P4 Abs Error")
        axes[8].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(galleries_dir, folder, f"9_panel_{fn.replace('.npy', '.png')}"), dpi=150)
        plt.close()
        
    # --- DECISION LOGIC & READYNESS TEST ---
    # Case A: Oracle Headroom is High, Evidence is High
    # Let's say:
    # High Headroom: PSNR Headroom >= 1.0 dB
    # High Evidence: Edge Evidence correlation >= 0.40
    
    oracle_headroom_level = "LOW"
    if hr_psnr >= 1.5:
        oracle_headroom_level = "HIGH"
    elif hr_psnr >= 0.5:
        oracle_headroom_level = "MEDIUM"
        
    evidence_availability = "LOW"
    if ev_edge_mean >= 0.40:
        evidence_availability = "HIGH"
    elif ev_edge_mean >= 0.25:
        evidence_availability = "MEDIUM"
        
    p47_justified = "NOT JUSTIFIED"
    if oracle_headroom_level in ["HIGH", "MEDIUM"] and evidence_availability == "HIGH":
        p47_justified = "YES"
    elif oracle_headroom_level in ["HIGH", "MEDIUM"]:
        p47_justified = "CONDITIONAL"
        
    # Recommendation text
    if p47_justified == "YES":
        verdict_conclusion = "There is significant recoverable headroom and degraded input gradients provide reliable structural evidence."
        recommendation = (
            "We recommend proceeding directly to **Phase 4.7: Evidence-Gated learned HF Recovery**. "
            "Since the degraded input contains enough evidence to identify structural boundaries (Edge correlation: "
            f"{ev_edge_mean:.4f}), we should implement a dual-branch architecture. Branch 1 learns the restoration, "
            "while Branch 2 (Evidence Gate) estimates local confidence from gradients, adaptively controlling HF scaling."
        )
    elif p47_justified == "CONDITIONAL":
        verdict_conclusion = "Significant headroom exists, but degraded input gradients correlate poorly with actual residual support."
        recommendation = (
            "An evidence-gated residual head is only conditionally justified. Because the degraded input correlation is low "
            f"(Edge correlation: {ev_edge_mean:.4f}), a simple spatial gate might fail. We should prioritize learned regularized "
            "priors in the network to reconstruct boundaries safely without relying solely on input edge evidence."
        )
    else:
        verdict_conclusion = "Even with an ideal oracle, localized correction yields negligible performance improvements."
        recommendation = (
            "Do NOT proceed with gated residual heads. Keep Phase 4 as the official champion, and focus future work on "
            "refining downsampling and noise simulation in the data pipeline."
        )
        
    # Write Report
    report_md = f"""# Phase 4.6: Evidence-Gated Oracle Analysis Report

This report documents the evidence-gated oracle analysis and Phase 4.7 readiness evaluation.

## 1. Executive Summary
- **Oracle Headroom:** **{oracle_headroom_level}** (PSNR improvement: **{hr_psnr:+.4f}** dB)
- **Evidence Availability:** **{evidence_availability}** (Sobel edge correlation: **{ev_edge_mean:.4f}**)
- **Phase 4.7 Verdict:** **{p47_justified}**
- **Conclusion:** {verdict_conclusion}

---

## 2. Quantitative Group Results

| Group | Count | Phase 4 PSNR | Oracle PSNR | Headroom PSNR | Phase 4 LPIPS | Oracle LPIPS | Headroom LPIPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_groups.iterrows():
        report_md += f"| **{r['Group']}** | {r['Count']} | {r['P4_PSNR']:.4f} | {r['Oracle_PSNR']:.4f} | {r['Headroom_PSNR']:+.4f} | {r['P4_LPIPS']:.4f} | {r['Oracle_LPIPS']:.4f} | {r['Headroom_LPIPS']:+.4f} |\n"
        
    report_md += f"""
---

## 3. Evidence Quality Mapping
- **Sobel Edge Correlation:** **{ev_edge_mean:.4f}**
- **High-Frequency Evidence Correlation:** **{ev_hf_mean:.4f}**
- **Local Texture Correlation:** **{ev_tex_mean:.4f}**

---

## 4. Final Verdict and Recommendation
{verdict_conclusion}

**Recommended Action:**
{recommendation}
"""
    with open(os.path.join(phase46_dir, "phase46_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    # --- CONSOLE OUTPUT ---
    print("\n" + "="*60)
    print("PHASE 4.6 ORACLE ANALYSIS COMPLETE")
    print("="*60)
    print(f"Phase 4 Champion:")
    print(f"PSNR  = {base_psnr:.4f}")
    print(f"SSIM  = {base_ssim:.4f}")
    print(f"LPIPS = {base_lpips:.4f}")
    
    print("\n" + "-"*60)
    print("ORACLE HEADROOM")
    print("-"*60)
    print(f"Oracle Best:")
    print(f"PSNR  = {oracle_psnr:.4f}")
    print(f"SSIM  = {oracle_ssim:.4f}")
    print(f"LPIPS = {oracle_lpips:.4f}")
    print(f"\nPSNR Headroom:  {hr_psnr:+.4f} dB")
    print(f"SSIM Headroom:  {hr_ssim:+.4f}")
    print(f"LPIPS Headroom: {hr_lpips:+.4f}")
    
    print("\n" + "-"*60)
    print("GROUP RESULTS")
    print("-"*60)
    for _, r in df_groups.iterrows():
        print(f"{r['Group']}:")
        print(f"  Phase 4 = PSNR {r['P4_PSNR']:.4f} | LPIPS {r['P4_LPIPS']:.4f}")
        print(f"  Oracle  = PSNR {r['Oracle_PSNR']:.4f} | LPIPS {r['Oracle_LPIPS']:.4f}")
        
    print("\n" + "-"*60)
    print("EVIDENCE QUALITY")
    print("-"*60)
    print(f"Edge Evidence Correlation:    {ev_edge_mean:.4f}")
    print(f"HF Evidence Correlation:      {ev_hf_mean:.4f}")
    print(f"Texture Evidence Correlation: {ev_tex_mean:.4f}")
    
    print("\n" + "-"*60)
    print("FINAL DECISION")
    print("-"*60)
    print(f"Oracle Headroom:       {oracle_headroom_level}")
    print(f"Evidence Availability: {evidence_availability}")
    print(f"Phase 4.7:             {p47_justified}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(recommendation)
    print("="*60)

if __name__ == "__main__":
    main()
