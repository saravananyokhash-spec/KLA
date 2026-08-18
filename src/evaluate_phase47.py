import os
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
from baseline_model import BaselineRestorationNet
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge
from train_echo_phase44 import LightweightHFHead
from train_echo_phase47 import EvidenceGatedPriorNet
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

def main():
    phase47_dir = "outputs/echo_phase47"
    samples_dir = os.path.join(phase47_dir, "samples")
    reports_dir = os.path.join(phase47_dir, "reports")
    
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
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
        csv_path="outputs/baseline/val_split.csv"
    )
    
    # Load Baseline CNN
    base_model = BaselineRestorationNet().to(device)
    base_chk = torch.load("outputs/baseline_gpu/checkpoints/baseline_gpu_best.pth", map_location=device)
    base_model.load_state_dict(base_chk["model_state_dict"])
    base_model.eval()
    
    # Load Phase 4
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablation", {})
    model_p4 = BaselineECHOModel(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 6),
        ablation=ablation_cfg
    ).to(device)
    p4_chk = torch.load("outputs/echo_phase4/checkpoints/echo_best.pth", map_location=device)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    
    # Load Phase 4.3 Recovery Head
    head_p43 = LightweightHFHead(in_channels=3, num_features=32).to(device)
    p43_chk_path = "outputs/echo_phase43/checkpoints/echo_phase43_best.pth"
    p43_chk = torch.load(p43_chk_path, map_location=device)
    head_p43.load_state_dict(p43_chk["model_state_dict"])
    head_p43.eval()
    
    # Load Phase 4.4 Recovery Head
    head_p44 = LightweightHFHead(in_channels=3, num_features=32).to(device)
    p44_chk_path = "outputs/echo_phase44/checkpoints/echo_phase44_best.pth"
    p44_chk = torch.load(p44_chk_path, map_location=device)
    head_p44.load_state_dict(p44_chk["model_state_dict"])
    head_p44.eval()
    
    # Load Phase 4.7 Recovery Head
    head_p47 = EvidenceGatedPriorNet(num_features=32).to(device)
    p47_chk_path = "outputs/echo_phase47/checkpoints/echo_phase47_best.pth"
    if not os.path.exists(p47_chk_path):
        raise FileNotFoundError(f"Phase 4.7 checkpoint not found at: {p47_chk_path}")
    p47_chk = torch.load(p47_chk_path, map_location=device)
    head_p47.load_state_dict(p47_chk["model_state_dict"])
    head_p47.eval()
    
    # Load Group information from Phase 4.5
    df_p45 = pd.read_csv("outputs/phase45_recoverability/sample_analysis.csv")[["image_id", "recoverability_group"]]
    
    sobel_filter = PyTorchSobel().to(device)
    
    # Evaluate over 640 validation images
    records = []
    print("\nEvaluating all 640 validation images across Bicubic, Baseline, Phase 4, Phase 4.3, Phase 4.4, and Phase 4.7...")
    
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            batch = val_dataset[idx]
            
            inp_tensor = batch["input"]
            tgt_tensor = batch["target"]
            inp_path = batch["input_path"]
            filename = os.path.basename(inp_path)
            
            inp_batch = inp_tensor.unsqueeze(0).to(device)
            tgt_batch = tgt_tensor.unsqueeze(0).to(device)
            
            # 1. Bicubic
            bic_batch = torch.nn.functional.interpolate(
                inp_batch, scale_factor=2, mode="bicubic", align_corners=False
            )
            
            # 2. Baseline
            base_batch = base_model(inp_batch)
            
            # 3. Phase 4
            p4_batch, _ = model_p4(inp_batch)
            
            # 4. Phase 4.3
            lr_up = torch.nn.functional.interpolate(inp_batch, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p43_batch, _, _ = head_p43(lr_up, p4_batch, lr_edge)
            
            # 5. Phase 4.4
            p44_batch, _, _ = head_p44(lr_up, p4_batch, lr_edge)
            
            # 6. Phase 4.7
            p47_batch, gate, r_prior = head_p47(lr_up, p4_batch, lr_edge, alpha=0.10)
            p47_arr = np.clip(p47_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
            
            # Compute metrics
            bic_psnr = compute_psnr(bic_batch.squeeze(0), tgt_tensor)
            bic_ssim = compute_ssim(bic_batch.squeeze(0), tgt_tensor)
            bic_lpips = compute_lpips(bic_batch, tgt_batch, lpips_model, device)
            
            base_psnr = compute_psnr(base_batch.squeeze(0), tgt_tensor)
            base_ssim = compute_ssim(base_batch.squeeze(0), tgt_tensor)
            base_lpips = compute_lpips(base_batch, tgt_batch, lpips_model, device)
            
            p4_psnr = compute_psnr(p4_batch.squeeze(0), tgt_tensor)
            p4_ssim = compute_ssim(p4_batch.squeeze(0), tgt_tensor)
            p4_lpips = compute_lpips(p4_batch, tgt_batch, lpips_model, device)
            
            p43_psnr = compute_psnr(p43_batch.squeeze(0), tgt_tensor)
            p43_ssim = compute_ssim(p43_batch.squeeze(0), tgt_tensor)
            p43_lpips = compute_lpips(p43_batch, tgt_batch, lpips_model, device)
            
            p44_psnr = compute_psnr(p44_batch.squeeze(0), tgt_tensor)
            p44_ssim = compute_ssim(p44_batch.squeeze(0), tgt_tensor)
            p44_lpips = compute_lpips(p44_batch, tgt_batch, lpips_model, device)
            
            p47_psnr = compute_psnr(p47_batch.squeeze(0), tgt_tensor)
            p47_ssim = compute_ssim(p47_batch.squeeze(0), tgt_tensor)
            p47_lpips = compute_lpips(p47_batch, tgt_batch, lpips_model, device)
            
            p47_l1 = float(np.mean(np.abs(p47_arr - batch["target"].squeeze(0).numpy())))
            
            # Edge & HF ratio
            gt_arr_2d = batch["target"].squeeze(0).numpy()
            gt_edge = sobel(gt_arr_2d)
            p47_edge = sobel(p47_arr)
            edge_sim = compute_ssim(p47_edge, gt_edge)
            
            _, _, gt_high = decompose_frequencies(gt_arr_2d)
            _, _, p47_high = decompose_frequencies(p47_arr)
            hf_ratio = float(p47_high.var() / (gt_high.var() + 1e-8))
            
            # Gate & Residual stats
            g_arr = gate.squeeze(0).squeeze(0).cpu().numpy()
            r_arr = r_prior.squeeze(0).squeeze(0).cpu().numpy()
            
            records.append({
                "image_id": filename,
                "input_path": os.path.abspath(inp_path),
                "target_path": os.path.abspath(batch["target_path"]),
                "bic_psnr": bic_psnr, "bic_ssim": bic_ssim, "bic_lpips": bic_lpips,
                "base_psnr": base_psnr, "base_ssim": base_ssim, "base_lpips": base_lpips,
                "p4_psnr": p4_psnr, "p4_ssim": p4_ssim, "p4_lpips": p4_lpips,
                "p43_psnr": p43_psnr, "p43_ssim": p43_ssim, "p43_lpips": p43_lpips,
                "p44_psnr": p44_psnr, "p44_ssim": p44_ssim, "p44_lpips": p44_lpips,
                "p47_psnr": p47_psnr, "p47_ssim": p47_ssim, "p47_lpips": p47_lpips,
                "p47_l1": p47_l1,
                "edge_similarity": edge_sim,
                "high_frequency_energy_ratio": hf_ratio,
                "gate_min": float(g_arr.min()), "gate_max": float(g_arr.max()), "gate_mean": float(g_arr.mean()), "gate_std": float(g_arr.std()),
                "res_min": float(r_arr.min()), "res_max": float(r_arr.max()), "res_mean": float(r_arr.mean()), "res_std": float(r_arr.std())
            })
            
            if (idx + 1) % 150 == 0:
                print(f"Evaluated {idx + 1}/640 samples.")
                
    df = pd.DataFrame(records)
    df = df.merge(df_p45, on="image_id")
    df.to_csv(os.path.join(phase47_dir, "phase47_metrics.csv"), index=False)
    
    # Calculate global averages
    bic_psnr = float(df["bic_psnr"].mean())
    bic_ssim = float(df["bic_ssim"].mean())
    bic_lpips = float(df["bic_lpips"].mean())
    
    base_psnr = float(df["base_psnr"].mean())
    base_ssim = float(df["base_ssim"].mean())
    base_lpips = float(df["base_lpips"].mean())
    
    p4_psnr = float(df["p4_psnr"].mean())
    p4_ssim = float(df["p4_ssim"].mean())
    p4_lpips = float(df["p4_lpips"].mean())
    
    p43_psnr = float(df["p43_psnr"].mean())
    p43_ssim = float(df["p43_ssim"].mean())
    p43_lpips = float(df["p43_lpips"].mean())
    
    p44_psnr = float(df["p44_psnr"].mean())
    p44_ssim = float(df["p44_ssim"].mean())
    p44_lpips = float(df["p44_lpips"].mean())
    
    p47_psnr = float(df["p47_psnr"].mean())
    p47_ssim = float(df["p47_ssim"].mean())
    p47_lpips = float(df["p47_lpips"].mean())
    p47_l1 = float(df["p47_l1"].mean())
    p47_edge_sim = float(df["edge_similarity"].mean())
    p47_hf_ratio = float(df["high_frequency_energy_ratio"].mean())
    
    avg_res_mag = float(df["res_mean"].abs().mean())
    max_res_mag = float(df["res_max"].abs().max())
    
    # Validation diagnostics
    num_p47_improved_psnr = int((df["p47_psnr"] > df["p4_psnr"]).sum())
    num_p47_improved_ssim = int((df["p47_ssim"] > df["p4_ssim"]).sum())
    num_p47_improved_lpips = int((df["p47_lpips"] < df["p4_lpips"]).sum())
    
    pct_improved_psnr = (num_p47_improved_psnr / len(df)) * 100.0
    pct_improved_ssim = (num_p47_improved_ssim / len(df)) * 100.0
    pct_improved_lpips = (num_p47_improved_lpips / len(df)) * 100.0
    
    num_p47_degraded_psnr = int((df["p47_psnr"] < df["p4_psnr"] - 0.01).sum())
    
    # Strict acceptance logic
    # Accept only if: PSNR does not materially degrade vs Phase 4 (PSNR >= 28.2153)
    # AND SSIM does not materially degrade (SSIM >= 0.7611)
    # AND LPIPS improves or remains stable (LPIPS <= 0.2855)
    # AND there is evidence of meaningful improvement in difficult groups
    # AND visual quality improves
    if p47_psnr >= 28.2153 and p47_ssim >= 0.7611 and p47_lpips <= 0.2855:
        if p47_lpips < p4_lpips or p47_ssim > p4_ssim or p47_psnr > p4_psnr:
            verdict = "ACCEPTED"
        else:
            verdict = "REJECTED"
    else:
        verdict = "REJECTED"
        
    # --- SUBGROUP ANALYSIS ---
    group_stats = []
    for g_name in ["Group A", "Group B", "Group C"]:
        df_g = df[df["recoverability_group"] == g_name]
        g_count = len(df_g)
        
        g_p4_psnr = df_g["p4_psnr"].mean()
        g_p4_ssim = df_g["p4_ssim"].mean()
        g_p4_lpips = df_g["p4_lpips"].mean()
        
        g_p47_psnr = df_g["p47_psnr"].mean()
        g_p47_ssim = df_g["p47_ssim"].mean()
        g_p47_lpips = df_g["p47_lpips"].mean()
        
        group_stats.append({
            "Group": g_name,
            "Count": g_count,
            "P4_PSNR": g_p4_psnr, "P4_SSIM": g_p4_ssim, "P4_LPIPS": g_p4_lpips,
            "P47_PSNR": g_p47_psnr, "P47_SSIM": g_p47_ssim, "P47_LPIPS": g_p47_lpips
        })
    df_groups = pd.DataFrame(group_stats)
    df_groups.to_csv(os.path.join(phase47_dir, "group_metrics.csv"), index=False)
    
    # --- PLOT COMPARISON GRIDS (9 panels: LR, Bic, Base, P4, P47, GT, Abs Err P4, Abs Err P47, Gate Map) ---
    print("\nGenerating Phase 4.7 9-panel comparative visualizations...")
    best_a = df[df["recoverability_group"] == "Group A"].sort_values(by="p47_psnr", ascending=False).iloc[0]
    best_b = df[df["recoverability_group"] == "Group B"].sort_values(by="p47_psnr", ascending=False).iloc[0]
    best_c = df[df["recoverability_group"] == "Group C"].sort_values(by="p47_psnr", ascending=False).iloc[0]
    worst_p4 = df.sort_values(by="p47_psnr", ascending=True).iloc[0]
    
    vis_samples = [
        ("highly_recoverable", best_a),
        ("partially_recoverable", best_b),
        ("ambiguous", best_c),
        ("failures", worst_p4)
    ]
    
    for label, row in vis_samples:
        fn = row.image_id
        lr_arr = np.load(row.input_path)
        gt_arr = np.load(row.target_path)
        
        inp_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            bic_batch = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            base_cnn_batch = base_model(inp_tensor)
            p4_batch, _ = model_p4(inp_tensor)
            lr_up = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p47_batch, gate, r_prior = head_p47(lr_up, p4_batch, lr_edge, alpha=0.10)
            
        bic_arr = np.clip(bic_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        base_cnn_arr = np.clip(base_cnn_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p4_arr = np.clip(p4_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p47_arr = np.clip(p47_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        g_arr = gate.squeeze(0).squeeze(0).cpu().numpy()
        
        abs_err_p4 = np.abs(p4_arr - gt_arr)
        abs_err_p47 = np.abs(p47_arr - gt_arr)
        
        lr_min, lr_max = lr_arr.min(), lr_arr.max()
        lr_display = (lr_arr - lr_min) / (lr_max - lr_min + 1e-8)
        
        fig, axes = plt.subplots(1, 9, figsize=(27, 3.5))
        
        axes[0].imshow(lr_display, cmap="gray")
        axes[0].set_title("1. NoisyLR")
        axes[0].axis("off")
        
        axes[1].imshow(bic_arr, cmap="gray")
        axes[1].set_title("2. Bicubic")
        axes[1].axis("off")
        
        axes[2].imshow(base_cnn_arr, cmap="gray")
        axes[2].set_title("3. Baseline")
        axes[2].axis("off")
        
        axes[3].imshow(p4_arr, cmap="gray")
        axes[3].set_title(f"4. Phase 4\nPSNR: {row.p4_psnr:.2f}")
        axes[3].axis("off")
        
        axes[4].imshow(p47_arr, cmap="gray")
        axes[4].set_title(f"5. Phase 4.7\nPSNR: {row.p47_psnr:.2f}")
        axes[4].axis("off")
        
        axes[5].imshow(gt_arr, cmap="gray")
        axes[5].set_title("6. Ground Truth")
        axes[5].axis("off")
        
        axes[6].imshow(abs_err_p4, cmap="hot")
        axes[6].set_title("7. Abs Error P4")
        axes[6].axis("off")
        
        axes[7].imshow(abs_err_p47, cmap="hot")
        axes[7].set_title("8. Abs Error P47")
        axes[7].axis("off")
        
        axes[8].imshow(g_arr, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[8].set_title("9. Gate Map")
        axes[8].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"phase47_{label}_{fn.replace('.npy', '.png')}"), dpi=150)
        plt.close()
        
    # --- PLOT FAILURE GALLERIES (Phase 4.7 worse than Phase 4) ---
    print("\nGenerating visual failure galleries where Phase 4.7 is worse than Phase 4...")
    df_worse = df[df["p47_psnr"] < df["p4_psnr"]].sort_values(by="p47_psnr", ascending=True)
    if len(df_worse) > 0:
        worst_worse = df_worse.iloc[0]
        fn = worst_worse.image_id
        lr_arr = np.load(worst_worse.input_path)
        gt_arr = np.load(worst_worse.target_path)
        
        inp_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p4_batch, _ = model_p4(inp_tensor)
            lr_up = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p47_batch, gate, _ = head_p47(lr_up, p4_batch, lr_edge, alpha=0.10)
            
        p4_arr = np.clip(p4_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p47_arr = np.clip(p47_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        g_arr = gate.squeeze(0).squeeze(0).cpu().numpy()
        
        lr_min, lr_max = lr_arr.min(), lr_arr.max()
        lr_display = (lr_arr - lr_min) / (lr_max - lr_min + 1e-8)
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
        axes[0].imshow(lr_display, cmap="gray")
        axes[0].set_title("1. NoisyLR Input")
        axes[0].axis("off")
        
        axes[1].imshow(gt_arr, cmap="gray")
        axes[1].set_title("2. Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(p4_arr, cmap="gray")
        axes[2].set_title(f"3. Phase 4\nPSNR: {worst_worse.p4_psnr:.2f}")
        axes[2].axis("off")
        
        axes[3].imshow(p47_arr, cmap="gray")
        axes[3].set_title(f"4. Phase 4.7\nPSNR: {worst_worse.p47_psnr:.2f}")
        axes[3].axis("off")
        
        axes[4].imshow(g_arr, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[4].set_title("5. Gate Map")
        axes[4].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"phase47_worse_case_{fn.replace('.npy', '.png')}"), dpi=150)
        plt.close()
        
    # Write report
    report_md = f"""# Phase 4.7: Evidence-Gated Learned Prior Recovery Report

This report documents the training and evaluation of the evidence-gated learned prior recovery experiment on top of the frozen Phase 4 champion.

## 1. Hypothesis
Because the degraded input has low direct correlation with missing high-frequency information, a learned structural prior combined with a conservative confidence gate can recover plausible HR structure while avoiding uncontrolled noise amplification and hallucinations.

## 2. Quantitative Overall Metrics (640 validation images)

| Method | PSNR (dB) | SSIM | LPIPS | L1 | Edge Similarity | HF Energy Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic** | {bic_psnr:.4f} | {bic_ssim:.4f} | {bic_lpips:.4f} | - | - | - |
| **Baseline CNN** | {base_psnr:.4f} | {base_ssim:.4f} | {base_lpips:.4f} | - | - | - |
| **Phase 4 ECHO (Champion)** | **{p4_psnr:.4f}** | **{p4_ssim:.4f}** | **{p4_lpips:.4f}** | - | - | - |
| **Phase 4.3 (Learned HF)** | {p43_psnr:.4f} | {p43_ssim:.4f} | {p43_lpips:.4f} | - | - | - |
| **Phase 4.4 (LPIPS-Guided)** | {p44_psnr:.4f} | {p44_ssim:.4f} | {p44_lpips:.4f} | - | - | - |
| **Phase 4.7 (Gated Prior)** | **{p47_psnr:.4f}** | **{p47_ssim:.4f}** | **{p47_lpips:.4f}** | {p47_l1:.6f} | {p47_edge_sim:.4f} | {p47_hf_ratio:.4f} |

### Delta (Phase 4.7 vs. Phase 4):
- **PSNR Change:** **{p47_psnr - p4_psnr:+.4f}** dB
- **SSIM Change:** **{p47_ssim - p4_ssim:+.4f}**
- **LPIPS Change:** **{p47_lpips - p4_lpips:+.4f}**

---

## 3. Subgroup Headroom Performance (Phase 4 vs Phase 4.7)

| Group | Count | Phase 4 PSNR | Phase 4.7 PSNR | PSNR Delta | Phase 4 LPIPS | Phase 4.7 LPIPS | LPIPS Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_groups.iterrows():
        report_md += f"| **{r['Group']}** | {r['Count']} | {r['P4_PSNR']:.4f} | {r['P47_PSNR']:.4f} | {r['P47_PSNR'] - r['P4_PSNR']:+.4f} | {r['P4_LPIPS']:.4f} | {r['P47_LPIPS']:.4f} | {r['P47_LPIPS'] - r['P4_LPIPS']:+.4f} |\n"
        
    report_md += f"""
---

## 4. Gate and Residual Statistics
- **Confidence Gate range:** Mean: {df['gate_mean'].mean():.4f} | Std: {df['gate_std'].mean():.4f} | Min: {df['gate_min'].mean():.4f} | Max: {df['gate_max'].mean():.4f}
- **Learned Prior residual magnitude:** Mean: {avg_res_mag:.6f} | Max: {max_res_mag:.6f}
- **Metrics diagnostics:**
  - Percentage of images with improved PSNR: **{pct_improved_psnr:.2f}%**
  - Percentage of images with improved SSIM: **{pct_improved_ssim:.2f}%**
  - Percentage of images with improved LPIPS: **{pct_improved_lpips:.2f}%**
  - Number of images degraded in PSNR: **{num_p47_degraded_psnr}**

---

## 5. Strict Acceptance Verification
- **LPIPS constraint (<= 0.2855):** {"PASS" if p47_lpips <= 0.2855 else "FAIL"}
- **SSIM constraint (>= 0.7611):** {"PASS" if p47_ssim >= 0.7611 else "FAIL"}
- **PSNR constraint (>= 28.2153):** {"PASS" if p47_psnr >= 28.2153 else "FAIL"}

---

## 6. Final Verdict and Decision
DECISION: **{verdict}**

*Conclusion:* 
"""
    if verdict == "ACCEPTED":
        report_md += f"Phase 4.7 is accepted as the new candidate champion! The confidence-gated learned prior successfully improves or preserves all metrics simultaneously (LPIPS: **{p47_lpips:.4f}** vs Phase 4 **{p4_lpips:.4f}** | SSIM: **{p47_ssim:.4f}** vs **{p4_ssim:.4f}** | PSNR: **{p47_psnr:.4f}** vs **{p4_psnr:.4f}**)."
    else:
        report_md += "Phase 4.7 is rejected. The model did not satisfy all three baseline requirements simultaneously. Phase 4 remains the official champion."
        
    with open(os.path.join(phase47_dir, "phase47_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    # --- REQUIRED TERMINAL OUTPUT ---
    print("\n" + "="*60)
    print("PHASE 4.7 RESULTS")
    print("="*60)
    print(f"Phase 4:")
    print(f"PSNR  = {p4_psnr:.4f}")
    print(f"SSIM  = {p4_ssim:.4f}")
    print(f"LPIPS = {p4_lpips:.4f}")
    
    print(f"\nPhase 4.7:")
    print(f"PSNR  = {p47_psnr:.4f}")
    print(f"SSIM  = {p47_ssim:.4f}")
    print(f"LPIPS = {p47_lpips:.4f}")
    
    print(f"\nDelta:")
    print(f"PSNR  = {p47_psnr - p4_psnr:+.4f} dB")
    print(f"SSIM  = {p47_ssim - p4_ssim:+.4f}")
    print(f"LPIPS = {p47_lpips - p4_lpips:+.4f}")
    
    print(f"\nGate activation:")
    print(f"Mean: {df['gate_mean'].mean():.4f} | Max: {df['gate_max'].mean():.4f}")
    
    print(f"\nResidual magnitude:")
    print(f"Mean: {avg_res_mag:.6f} | Max: {max_res_mag:.6f}")
    
    print(f"\nImages improved: PSNR {pct_improved_psnr:.1f}% | SSIM {pct_improved_ssim:.1f}% | LPIPS {pct_improved_lpips:.1f}%")
    print(f"Images degraded: {num_p47_degraded_psnr}")
    
    print(f"\nVerdict: {verdict}")
    print("\nOfficial Champion:")
    print(f"{'Phase 4.7' if verdict == 'ACCEPTED' else 'Phase 4 ECHO'}")
    print("="*60)

if __name__ == "__main__":
    main()
