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
from train_echo_phase43 import LightweightHFHead, PyTorchSobel, get_lr_edge
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
    phase44_dir = "outputs/echo_phase44"
    samples_dir = os.path.join(phase44_dir, "samples")
    reports_dir = os.path.join(phase44_dir, "reports")
    
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
    if not os.path.exists(p44_chk_path):
        raise FileNotFoundError(f"Phase 4.4 checkpoint not found at: {p44_chk_path}")
    p44_chk = torch.load(p44_chk_path, map_location=device)
    head_p44.load_state_dict(p44_chk["model_state_dict"])
    head_p44.eval()
    
    sobel_filter = PyTorchSobel().to(device)
    
    # Evaluate over 640 validation images
    records = []
    print("\nEvaluating all 640 validation images across Bicubic, Baseline, Phase 4, Phase 4.3, and Phase 4.4...")
    
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
            p44_batch, gate, pred_res = head_p44(lr_up, p4_batch, lr_edge)
            p44_arr = np.clip(p44_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
            
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
            
            p44_l1 = float(np.mean(np.abs(p44_arr - batch["target"].squeeze(0).numpy())))
            
            # Edge & HF ratio
            gt_arr_2d = batch["target"].squeeze(0).numpy()
            gt_edge = sobel(gt_arr_2d)
            p44_edge = sobel(p44_arr)
            edge_sim = compute_ssim(p44_edge, gt_edge)
            
            _, _, gt_high = decompose_frequencies(gt_arr_2d)
            _, _, p44_high = decompose_frequencies(p44_arr)
            hf_ratio = float(p44_high.var() / (gt_high.var() + 1e-8))
            
            # Gate & Residual stats
            g_arr = gate.squeeze(0).squeeze(0).cpu().numpy()
            r_arr = pred_res.squeeze(0).squeeze(0).cpu().numpy()
            
            records.append({
                "filename": filename,
                "input_path": os.path.abspath(inp_path),
                "target_path": os.path.abspath(batch["target_path"]),
                "bic_psnr": bic_psnr, "bic_ssim": bic_ssim, "bic_lpips": bic_lpips,
                "base_psnr": base_psnr, "base_ssim": base_ssim, "base_lpips": base_lpips,
                "p4_psnr": p4_psnr, "p4_ssim": p4_ssim, "p4_lpips": p4_lpips,
                "p43_psnr": p43_psnr, "p43_ssim": p43_ssim, "p43_lpips": p43_lpips,
                "p44_psnr": p44_psnr, "p44_ssim": p44_ssim, "p44_lpips": p44_lpips,
                "p44_l1": p44_l1,
                "edge_similarity": edge_sim,
                "high_frequency_energy_ratio": hf_ratio,
                "gate_min": float(g_arr.min()), "gate_max": float(g_arr.max()), "gate_mean": float(g_arr.mean()),
                "res_min": float(r_arr.min()), "res_max": float(r_arr.max()), "res_mean": float(r_arr.mean()), "res_std": float(r_arr.std())
            })
            
            if (idx + 1) % 150 == 0:
                print(f"Evaluated {idx + 1}/640 samples.")
                
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(phase44_dir, "phase44_metrics.csv"), index=False)
    
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
    p44_l1 = float(df["p44_l1"].mean())
    p44_edge_sim = float(df["edge_similarity"].mean())
    p44_hf_ratio = float(df["high_frequency_energy_ratio"].mean())
    
    avg_res_mag = float(df["res_mean"].abs().mean())
    
    # Strict acceptance logic
    if p44_psnr >= 28.2153 and p44_ssim >= 0.7611 and p44_lpips <= 0.2855:
        # Check that it actually beats Phase 4 in at least one metric
        if p44_lpips < p4_lpips or p44_ssim > p4_ssim or p44_psnr > p4_psnr:
            verdict = "ACCEPTED"
        else:
            verdict = "REJECTED"
    else:
        verdict = "REJECTED"
        
    # --- PLOT COMPARISON GRIDS (LR / GT / Bicubic / Baseline / Phase 4 / Phase 4.3 / Phase 4.4) ---
    print("\nGenerating Phase 4.4 7-panel comparative visualizations...")
    best_idx = df.sort_values(by="p44_psnr", ascending=False).iloc[0]
    worst_idx = df.sort_values(by="p44_psnr", ascending=True).iloc[0]
    worst_hf_idx = df.sort_values(by="high_frequency_energy_ratio", ascending=True).iloc[0]
    worst_edge_idx = df.sort_values(by="edge_similarity", ascending=True).iloc[0]
    worst_lpips_idx = df.sort_values(by="p44_lpips", ascending=False).iloc[0]
    
    vis_samples = [
        ("best_case", best_idx),
        ("worst_case", worst_idx),
        ("largest_hf_underestimation", worst_hf_idx),
        ("largest_edge_loss", worst_edge_idx),
        ("largest_lpips_error", worst_lpips_idx)
    ]
    
    for label, row in vis_samples:
        fn = row.filename
        lr_arr = np.load(row.input_path)
        gt_arr = np.load(row.target_path)
        
        inp_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            bic_batch = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            base_cnn_batch = base_model(inp_tensor)
            p4_batch, _ = model_p4(inp_tensor)
            lr_up = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p43_batch, _, _ = head_p43(lr_up, p4_batch, lr_edge)
            p44_batch, _, _ = head_p44(lr_up, p4_batch, lr_edge)
            
        bic_arr = np.clip(bic_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        base_cnn_arr = np.clip(base_cnn_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p4_arr = np.clip(p4_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p43_arr = np.clip(p43_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p44_arr = np.clip(p44_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        lr_min, lr_max = lr_arr.min(), lr_arr.max()
        lr_display = (lr_arr - lr_min) / (lr_max - lr_min + 1e-8)
        
        fig, axes = plt.subplots(1, 7, figsize=(21, 3.5))
        
        axes[0].imshow(gt_arr, cmap="gray")
        axes[0].set_title("1. Ground Truth")
        axes[0].axis("off")
        
        axes[1].imshow(lr_display, cmap="gray")
        axes[1].set_title("2. NoisyLR Input")
        axes[1].axis("off")
        
        axes[2].imshow(bic_arr, cmap="gray")
        axes[2].set_title(f"3. Bicubic\nPSNR: {row.bic_psnr:.2f}")
        axes[2].axis("off")
        
        axes[3].imshow(base_cnn_arr, cmap="gray")
        axes[3].set_title(f"4. Baseline CNN\nPSNR: {row.base_psnr:.2f}")
        axes[3].axis("off")
        
        axes[4].imshow(p4_arr, cmap="gray")
        axes[4].set_title(f"5. Phase 4\nPSNR: {row.p4_psnr:.2f} | LPIPS: {row.p4_lpips:.3f}")
        axes[4].axis("off")
        
        axes[5].imshow(p43_arr, cmap="gray")
        axes[5].set_title(f"6. Phase 4.3\nPSNR: {row.p43_psnr:.2f} | LPIPS: {row.p43_lpips:.3f}")
        axes[5].axis("off")
        
        axes[6].imshow(p44_arr, cmap="gray")
        axes[6].set_title(f"7. Phase 4.4\nPSNR: {row.p44_psnr:.2f} | LPIPS: {row.p44_lpips:.3f}")
        axes[6].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"phase44_{label}_{fn.replace('.npy', '.png')}"), dpi=150)
        plt.close()
        
    # --- PLOT FAILURE GALLERIES (Phase 4.4 worse than Phase 4) ---
    print("\nGenerating visual failure galleries where Phase 4.4 is worse than Phase 4...")
    df_worse = df[df["p44_psnr"] < df["p4_psnr"]].sort_values(by="p44_psnr", ascending=True)
    if len(df_worse) > 0:
        worst_worse = df_worse.iloc[0]
        fn = worst_worse.filename
        lr_arr = np.load(worst_worse.input_path)
        gt_arr = np.load(worst_worse.target_path)
        
        inp_tensor = torch.from_numpy(lr_arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p4_batch, _ = model_p4(inp_tensor)
            lr_up = torch.nn.functional.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p44_batch, _, _ = head_p44(lr_up, p4_batch, lr_edge)
            
        p4_arr = np.clip(p4_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        p44_arr = np.clip(p44_batch.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
        
        lr_min, lr_max = lr_arr.min(), lr_arr.max()
        lr_display = (lr_arr - lr_min) / (lr_max - lr_min + 1e-8)
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        axes[0].imshow(lr_display, cmap="gray")
        axes[0].set_title("1. NoisyLR Input")
        axes[0].axis("off")
        
        axes[1].imshow(gt_arr, cmap="gray")
        axes[1].set_title("2. Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(p4_arr, cmap="gray")
        axes[2].set_title(f"3. Phase 4\nPSNR: {worst_worse.p4_psnr:.2f}")
        axes[2].axis("off")
        
        axes[3].imshow(p44_arr, cmap="gray")
        axes[3].set_title(f"4. Phase 4.4\nPSNR: {worst_worse.p44_psnr:.2f}")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"phase44_worse_case_{fn.replace('.npy', '.png')}"), dpi=150)
        plt.close()
        
    # Get model info
    tot_params = sum(p.numel() for p in head_p44.parameters())
    frozen_params = sum(p.numel() for p in model_p4.parameters())
    
    # Write report
    report_md = f"""# Phase 4.4: Perceptual Evidence-Constrained HF Recovery Report

This report documents the training and evaluation of the perceptual evidence-constrained high-frequency recovery experiment on top of the frozen Phase 4 champion.

## 1. Hypothesis
Supervising the Lightweight HF Head with a differentiable LPIPS AlexNet loss during training forces the learned residual map to align with human visual perception, resolving the LPIPS degradation observed in Phase 4.3 while maintaining or improving PSNR/SSIM.

## 2. Quantitative Overall Metrics (640 validation images)

| Method | PSNR (dB) | SSIM | LPIPS | L1 | Edge Similarity | HF Energy Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic** | {bic_psnr:.4f} | {bic_ssim:.4f} | {bic_lpips:.4f} | - | - | - |
| **Baseline CNN** | {base_psnr:.4f} | {base_ssim:.4f} | {base_lpips:.4f} | - | - | - |
| **Phase 4 ECHO (Champion)** | **{p4_psnr:.4f}** | **{p4_ssim:.4f}** | **{p4_lpips:.4f}** | - | - | - |
| **Phase 4.3 (Learned HF)** | {p43_psnr:.4f} | {p43_ssim:.4f} | {p43_lpips:.4f} | - | - | - |
| **Phase 4.4 (LPIPS-Guided)** | **{p44_psnr:.4f}** | **{p44_ssim:.4f}** | **{p44_lpips:.4f}** | {p44_l1:.6f} | {p44_edge_sim:.4f} | {p44_hf_ratio:.4f} |

### Delta (Phase 4.4 vs. Phase 4):
- **PSNR Change:** **{p44_psnr - p4_psnr:+.4f}** dB
- **SSIM Change:** **{p44_ssim - p4_ssim:+.4f}**
- **LPIPS Change:** **{p44_lpips - p4_lpips:+.4f}**

---

## 3. Strict Acceptance Verification
- **LPIPS constraint (<= 0.2855):** {"PASS" if p44_lpips <= 0.2855 else "FAIL"}
- **SSIM constraint (>= 0.7611):** {"PASS" if p44_ssim >= 0.7611 else "FAIL"}
- **PSNR constraint (>= 28.2153):** {"PASS" if p44_psnr >= 28.2153 else "FAIL"}

---

## 4. Gate and Residual Statistics
- **Gate statistics:** Min: {df['gate_min'].mean():.4f} | Max: {df['gate_max'].mean():.4f} | Mean: {df['gate_mean'].mean():.4f}
- **Residual statistics:** Min: {df['res_min'].mean():.4f} | Max: {df['res_max'].mean():.4f} | Mean: {df['res_mean'].mean():.4f} | Std: {df['res_std'].mean():.4f}
- **Average Residual Magnitude:** {avg_res_mag:.6f}

---

## 5. Final Verdict and Decision
DECISION: **{verdict}**

*Conclusion:* 
"""
    if verdict == "ACCEPTED":
        report_md += f"Phase 4.4 is accepted as the new candidate champion! Supervising with LPIPS loss successfully resolved the perceptual degradation (LPIPS: **{p44_lpips:.4f}** vs Phase 4 **{p4_lpips:.4f}** | SSIM: **{p44_ssim:.4f}** vs **{p4_ssim:.4f}** | PSNR: **{p44_psnr:.4f}** vs **{p4_psnr:.4f}**)."
    else:
        report_md += "Phase 4.4 is rejected. While perceptual loss supervision helped constrain details, the configuration failed to beat Phase 4 across all three constraints simultaneously. Phase 4 remains the official champion."
        
    with open(os.path.join(phase44_dir, "phase44_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
        
    # Print terminal summary as requested
    print("\n" + "="*60)
    print("PHASE 4.4 RESULTS SUMMARY")
    print("="*25)
    print(f"{'Method':15s} | {'PSNR (dB)':9s} | {'SSIM':8s} | {'LPIPS':8s}")
    print("-"*60)
    print(f"{'Bicubic':15s} | {bic_psnr:9.4f} | {bic_ssim:8.4f} | {bic_lpips:8.4f}")
    print(f"{'Baseline CNN':15s} | {base_psnr:9.4f} | {base_ssim:8.4f} | {base_lpips:8.4f}")
    print(f"{'Phase 4 ECHO':15s} | {p4_psnr:9.4f} | {p4_ssim:8.4f} | {p4_lpips:8.4f}")
    print(f"{'Phase 4.3':15s} | {p43_psnr:9.4f} | {p43_ssim:8.4f} | {p43_lpips:8.4f}")
    print(f"{'Phase 4.4':15s} | {p44_psnr:9.4f} | {p44_ssim:8.4f} | {p44_lpips:8.4f}")
    print("-"*60)
    print(f"Phase 4.4 vs Phase 4:")
    print(f"PSNR delta: {p44_psnr - p4_psnr:+.4f} dB")
    print(f"SSIM delta: {p44_ssim - p4_ssim:+.4f}")
    print(f"LPIPS delta: {p44_lpips - p4_lpips:+.4f}")
    print("-"*60)
    print(f"Verdict: {verdict}")
    print("\nOfficial Champion:")
    print(f"{'Phase 4.4' if verdict == 'ACCEPTED' else 'Phase 4 ECHO'}")
    print("="*60)

if __name__ == "__main__":
    main()
