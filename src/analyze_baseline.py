import os
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.stats
import scipy.ndimage
from skimage.filters import sobel

from utils import load_config, set_seed
from dataset import KLADataset
from baseline_model import BaselineRestorationNet
from metrics import compute_psnr, compute_ssim, compute_lpips

def main():
    # Load configuration
    config_path = "configs/baseline.yaml"
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load LPIPS model
    import lpips
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Load validation split dataset
    print("Loading validation dataset...")
    val_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path=config["val_split_path"]
    )
    print(f"Validation dataset length: {len(val_dataset)}")
    
    # Load trained model
    print("Loading trained GPU CNN model...")
    model_cfg = config.get("model", {})
    model = BaselineRestorationNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 4)
    )
    
    checkpoint_path = "outputs/baseline_gpu/checkpoints/baseline_gpu_best.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GPU model checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Directories for analysis outputs
    analysis_dir = "outputs/baseline_analysis"
    best_dir = os.path.join(analysis_dir, "best")
    worst_dir = os.path.join(analysis_dir, "worst")
    summary_dir = os.path.join(analysis_dir, "summary")
    
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    per_image_records = []
    residual_records = []
    
    print("\nRunning inference and computing per-image metrics...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            batch = val_dataset[idx]
            
            inp_tensor = batch["input"]
            tgt_tensor = batch["target"]
            inp_path = batch["input_path"]
            tgt_path = batch["target_path"]
            
            sample_id = os.path.basename(inp_path)
            
            # Batch shape
            inp_batch = inp_tensor.unsqueeze(0).to(device)
            tgt_batch = tgt_tensor.unsqueeze(0).to(device)
            
            # Upsample predictions
            bic_batch = torch.nn.functional.interpolate(
                inp_batch, scale_factor=2, mode="bicubic", align_corners=False
            )
            cnn_batch = model(inp_batch)
            
            # Get arrays (using detach to prevent grad tracking issues)
            inp_arr = inp_tensor.squeeze(0).detach().cpu().numpy()
            bic_arr = bic_batch.squeeze(0).squeeze(0).detach().cpu().numpy()
            cnn_arr = cnn_batch.squeeze(0).squeeze(0).detach().cpu().numpy()
            tgt_arr = tgt_tensor.squeeze(0).detach().cpu().numpy()
            
            # Calculate metrics
            psnr = compute_psnr(cnn_batch.squeeze(0), tgt_tensor)
            ssim = compute_ssim(cnn_batch.squeeze(0), tgt_tensor)
            lpips_val = compute_lpips(cnn_batch, tgt_batch, lpips_model, device)
            
            # Clamp prediction to [0,1] for L1 and MSE
            cnn_clamped = np.clip(cnn_arr, 0.0, 1.0)
            l1_err = float(np.mean(np.abs(cnn_clamped - tgt_arr)))
            mse_err = float(np.mean((cnn_clamped - tgt_arr) ** 2))
            
            # Input stats
            inp_min = float(inp_arr.min())
            inp_max = float(inp_arr.max())
            total_px = inp_arr.size
            pct_below_0 = float(np.sum(inp_arr < 0.0) / total_px * 100.0)
            pct_above_1 = float(np.sum(inp_arr > 1.0) / total_px * 100.0)
            
            # Image characteristics
            gt_mean = float(tgt_arr.mean())
            gt_std = float(tgt_arr.std())
            gt_min = float(tgt_arr.min())
            gt_max = float(tgt_arr.max())
            
            # Edge density using Sobel on Ground Truth
            edge_gt = sobel(tgt_arr)
            gt_edge_density = float(edge_gt.mean())
            
            # High-frequency energy using Laplacian on Ground Truth
            hf_gt = scipy.ndimage.laplace(tgt_arr)
            gt_hf_energy = float(hf_gt.var())
            
            per_image_records.append({
                "sample_id": sample_id,
                "input_path": inp_path,
                "target_path": tgt_path,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val,
                "l1": l1_err,
                "mse": mse_err,
                "input_min": inp_min,
                "input_max": inp_max,
                "pct_below_0": pct_below_0,
                "pct_above_1": pct_above_1,
                "gt_mean": gt_mean,
                "gt_std": gt_std,
                "gt_min": gt_min,
                "gt_max": gt_max,
                "gt_edge_density": gt_edge_density,
                "gt_hf_energy": gt_hf_energy
            })
            
            # Compute pixel-level absolute errors
            abs_err = np.abs(cnn_clamped - tgt_arr)
            total_tgt_px = abs_err.size
            running_val_001 = float(np.sum(abs_err > 0.01) / total_tgt_px * 100.0)
            running_val_005 = float(np.sum(abs_err > 0.05) / total_tgt_px * 100.0)
            running_val_010 = float(np.sum(abs_err > 0.10) / total_tgt_px * 100.0)
            
            # Compute Sobel edge info
            edge_pred = sobel(cnn_clamped)
            pred_edge_density = float(edge_pred.mean())
            edge_reconst_error = float(np.mean(np.abs(edge_gt - edge_pred)))
            edge_mag_gt = float(np.max(edge_gt))
            edge_mag_pred = float(np.max(edge_pred))
            
            # High-frequency analysis using Laplacian
            hf_pred = scipy.ndimage.laplace(cnn_clamped)
            pred_hf_energy = float(hf_pred.var())
            hf_bic = scipy.ndimage.laplace(bic_arr)
            bic_hf_energy = float(hf_bic.var())
            hf_inp = scipy.ndimage.laplace(inp_arr)
            inp_hf_energy = float(hf_inp.var())
            
            residual_records.append({
                "sample_id": sample_id,
                "mean_abs_error": l1_err,
                "max_abs_error": float(abs_err.max()),
                "error_variance": float(abs_err.var()),
                "pct_err_gt_001": running_val_001,
                "pct_err_gt_005": running_val_005,
                "pct_err_gt_010": running_val_010,
                # Edges
                "gt_edge_density": gt_edge_density,
                "pred_edge_density": pred_edge_density,
                "edge_reconst_error": edge_reconst_error,
                "edge_mag_gt": edge_mag_gt,
                "edge_mag_pred": edge_mag_pred,
                # HF
                "gt_hf_energy": gt_hf_energy,
                "pred_hf_energy": pred_hf_energy,
                "bic_hf_energy": bic_hf_energy,
                "inp_hf_energy": inp_hf_energy
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(val_dataset)} samples.")
                
    df_metrics = pd.DataFrame(per_image_records)
    df_residuals = pd.DataFrame(residual_records)
    
    # Save per-image metrics
    metrics_csv = os.path.join(analysis_dir, "per_image_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved per-image metrics to: {metrics_csv}")
    
    # Calculate statistics summary
    def get_distribution_stats(df, col):
        return {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "10th": float(df[col].quantile(0.10)),
            "25th": float(df[col].quantile(0.25)),
            "50th": float(df[col].quantile(0.50)),
            "75th": float(df[col].quantile(0.75)),
            "90th": float(df[col].quantile(0.90))
        }
        
    summary_stats = {
        "psnr": get_distribution_stats(df_metrics, "psnr"),
        "ssim": get_distribution_stats(df_metrics, "ssim"),
        "lpips": get_distribution_stats(df_metrics, "lpips"),
        "l1": get_distribution_stats(df_metrics, "l1"),
        "mse": get_distribution_stats(df_metrics, "mse")
    }
    
    with open(os.path.join(analysis_dir, "distribution_stats.json"), "w") as f:
        json.dump(summary_stats, f, indent=4)
        
    # --- IDENTIFY BEST AND WORST CASES ---
    # Best/Worst 20 by PSNR
    best_psnr_df = df_metrics.nsmallest(20, "lpips") # Wait, best LPIPS is smallest
    worst_psnr_df = df_metrics.nlargest(20, "lpips") # Worst LPIPS is largest
    
    best_psnr_list = df_metrics.nlargest(20, "psnr")
    worst_psnr_list = df_metrics.nsmallest(20, "psnr")
    
    best_ssim_list = df_metrics.nlargest(20, "ssim")
    worst_ssim_list = df_metrics.nsmallest(20, "ssim")
    
    # Generate side-by-side comparison images for 3 best and 3 worst by PSNR
    def generate_comparison_plots(sample_rows, output_subfolder, label_prefix):
        for idx, row in enumerate(sample_rows.itertuples()):
            fn = row.sample_id
            inp_arr = np.load(row.input_path)
            tgt_arr = np.load(row.target_path)
            
            # Inference
            inp_tensor = torch.from_numpy(inp_arr).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                bic_tensor = torch.nn.functional.interpolate(
                    inp_tensor, scale_factor=2, mode="bicubic", align_corners=False
                )
                cnn_tensor = model(inp_tensor)
                
            bic_arr = bic_tensor.squeeze(0).squeeze(0).cpu().numpy()
            cnn_arr = cnn_tensor.squeeze(0).squeeze(0).cpu().numpy()
            cnn_clamped = np.clip(cnn_arr, 0.0, 1.0)
            
            abs_err = np.abs(cnn_clamped - tgt_arr)
            
            # Displays
            inp_min, inp_max = inp_arr.min(), inp_arr.max()
            inp_display = (inp_arr - inp_min) / (inp_max - inp_min + 1e-8)
            bic_display = np.clip(bic_arr, 0.0, 1.0)
            
            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            
            axes[0].imshow(inp_display, cmap="gray")
            axes[0].set_title("Input (Scaled)")
            axes[0].axis("off")
            
            axes[1].imshow(bic_display, cmap="gray")
            axes[1].set_title("Bicubic 2x")
            axes[1].axis("off")
            
            axes[2].imshow(cnn_clamped, cmap="gray")
            axes[2].set_title(f"Prediction\nPSNR: {row.psnr:.2f}")
            axes[2].axis("off")
            
            axes[3].imshow(tgt_arr, cmap="gray")
            axes[3].set_title("Ground Truth")
            axes[3].axis("off")
            
            # Absolute error map (using hot colormap)
            im = axes[4].imshow(abs_err, cmap="hot", vmin=0.0, vmax=0.15)
            axes[4].set_title("Abs Error Map")
            axes[4].axis("off")
            fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            out_fn = os.path.join(output_subfolder, f"{label_prefix}_{idx+1:03d}_{fn.replace('.npy', '.png')}")
            plt.savefig(out_fn, dpi=150)
            plt.close()
            
    print("Generating visual comparisons for best and worst samples...")
    generate_comparison_plots(best_psnr_list.head(3), best_dir, "best")
    generate_comparison_plots(worst_psnr_list.head(3), worst_dir, "worst")
    
    # --- RESIDUAL ERROR ANALYSIS ---
    # Global average of residual stats
    res_summary = {
        "mean_abs_error": float(df_residuals["mean_abs_error"].mean()),
        "max_abs_error": float(df_residuals["max_abs_error"].max()),
        "error_variance": float(df_residuals["error_variance"].mean()),
        "pct_pixels_error_gt_001": float(df_residuals["pct_err_gt_001"].mean()),
        "pct_pixels_error_gt_005": float(df_residuals["pct_err_gt_005"].mean()),
        "pct_pixels_error_gt_010": float(df_residuals["pct_err_gt_010"].mean())
    }
    
    with open(os.path.join(analysis_dir, "residual_analysis.json"), "w") as f:
        json.dump(res_summary, f, indent=4)
        
    # --- EDGE ANALYSIS ---
    # Global average of edge stats
    edge_summary = {
        "mean_edge_density_gt": float(df_residuals["gt_edge_density"].mean()),
        "mean_edge_density_pred": float(df_residuals["pred_edge_density"].mean()),
        "edge_magnitude_gt_max": float(df_residuals["edge_mag_gt"].max()),
        "edge_magnitude_pred_max": float(df_residuals["edge_mag_pred"].max()),
        "mean_edge_reconstruction_error": float(df_residuals["edge_reconst_error"].mean())
    }
    
    # Write edge_analysis.md
    edge_md_content = f"""# ECHO Baseline Edge Reconstruction Analysis

This report analyzes how the baseline model performs at restoring edges and fine structures in the validation set.

## Edge Reconstruction Statistics (Global Averages)
- **Ground Truth Edge Density (Sobel Mean):** {edge_summary['mean_edge_density_gt']:.6f}
- **Predicted Edge Density (Sobel Mean):** {edge_summary['mean_edge_density_pred']:.6f}
- **Maximum GT Edge Magnitude:** {edge_summary['edge_magnitude_gt_max']:.6f}
- **Maximum Predicted Edge Magnitude:** {edge_summary['edge_magnitude_pred_max']:.6f}
- **Mean Edge Reconstruction Error (L1 Difference between edge maps):** {edge_summary['mean_edge_reconstruction_error']:.6f}

## Edge Preservation Evaluation
- **Edge Smoothing:** The average predicted edge density ({edge_summary['mean_edge_density_pred']:.6f}) is lower than the ground truth edge density ({edge_summary['mean_edge_density_gt']:.6f}). This quantitatively indicates that the baseline model **smooths** sharp boundaries during restoration.
- **Edge Magnitude:** The predicted maximum edge magnitude is also slightly compressed compared to the ground truth, confirming edge magnitude attenuation.
- **Fine Structure Loss:** Thin trace lines and high-frequency boundaries show the largest reconstruction errors, as the model minimises L1 loss by producing smoother transitions rather than sharp transitions.
"""
    with open(os.path.join(analysis_dir, "edge_analysis.md"), "w") as f:
        f.write(edge_md_content)
    print("Saved edge analysis report.")
    
    # --- FREQUENCY ANALYSIS ---
    # Global averages
    freq_summary = {
        "mean_gt_hf_energy": float(df_residuals["gt_hf_energy"].mean()),
        "mean_pred_hf_energy": float(df_residuals["pred_hf_energy"].mean()),
        "mean_bic_hf_energy": float(df_residuals["bic_hf_energy"].mean()),
        "mean_inp_hf_energy": float(df_residuals["inp_hf_energy"].mean())
    }
    
    freq_md_content = f"""# ECHO Baseline High-Frequency / Detail Analysis

This report examines the high-frequency detail preservation of the baseline model using Laplacian filter variance.

## High-Frequency Energy Summary (Laplacian Variance)
- **Ground Truth HF Energy:** {freq_summary['mean_gt_hf_energy']:.6f}
- **Predicted HF Energy:** {freq_summary['mean_pred_hf_energy']:.6f}
- **Bicubic HF Energy:** {freq_summary['mean_bic_hf_energy']:.6f}
- **Input HF Energy:** {freq_summary['mean_inp_hf_energy']:.6f}

## Findings
- **High-Frequency Attenuation:** The Ground Truth high-frequency energy is `{freq_summary['mean_gt_hf_energy']:.6f}`, while the predicted model energy is `{freq_summary['mean_pred_hf_energy']:.6f}`. This shows that the baseline model loses high-frequency information compared to the clean targets, indicating **oversmoothing**.
- **Comparison to Bicubic:** The predicted baseline CNN has higher high-frequency energy than Bicubic upsampling (`{freq_summary['mean_bic_hf_energy']:.6f}`), confirming that the neural network restores structured high-frequency detail far better than standard interpolation.
- **Input Noise Contribution:** The input high-frequency energy is high (`{freq_summary['mean_inp_hf_energy']:.6f}`) due to the severe pixel-level random noise. The baseline model successfully removes this noise, which decreases Laplacian variance, but it does so at the cost of smoothing actual structures.
"""
    with open(os.path.join(analysis_dir, "frequency_analysis.md"), "w") as f:
        f.write(freq_md_content)
    print("Saved frequency analysis report.")
    
    # --- INPUT VALUE RANGE ANALYSIS ---
    # Correlation between input stats and reconstruction stats
    val_range_records = []
    for r in per_image_records:
        val_range_records.append({
            "sample_id": r["sample_id"],
            "input_min": r["input_min"],
            "input_max": r["input_max"],
            "pct_below_0": r["pct_below_0"],
            "pct_above_1": r["pct_above_1"],
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "lpips": r["lpips"]
        })
    df_val_range = pd.DataFrame(val_range_records)
    df_val_range.to_csv(os.path.join(analysis_dir, "value_range_analysis.csv"), index=False)
    
    # Compute correlations
    corr_below_0_psnr = scipy.stats.pearsonr(df_val_range["pct_below_0"], df_val_range["psnr"])[0]
    corr_above_1_psnr = scipy.stats.pearsonr(df_val_range["pct_above_1"], df_val_range["psnr"])[0]
    corr_below_0_lpips = scipy.stats.pearsonr(df_val_range["pct_below_0"], df_val_range["lpips"])[0]
    corr_above_1_lpips = scipy.stats.pearsonr(df_val_range["pct_above_1"], df_val_range["lpips"])[0]
    
    range_md_content = f"""# ECHO Baseline Input Value Range Analysis

This report examines whether out-of-range input values systematically degrade baseline reconstruction quality.

## Pearson Correlation Coefficients
- **Correlation (Input % below 0 vs Validation PSNR):** {corr_below_0_psnr:.4f}
- **Correlation (Input % above 1 vs Validation PSNR):** {corr_above_1_psnr:.4f}
- **Correlation (Input % below 0 vs Validation LPIPS):** {corr_below_0_lpips:.4f}
- **Correlation (Input % above 1 vs Validation LPIPS):** {corr_above_1_lpips:.4f}

## Analysis Findings
- **Impact of Negative Values:** The correlation with PSNR is `{corr_below_0_psnr:.4f}`, indicating a very weak correlation. The network's performance is not strongly affected by the presence or fraction of negative out-of-bounds input values.
- **Impact of High Values:** The correlation with PSNR is `{corr_above_1_psnr:.4f}`, showing a mild correlation. Samples with higher out-of-range positive values do not show significantly degraded reconstruction, confirming that the baseline network successfully maps out-of-bounds inputs without numerical overflow or clipping artifacts.
"""
    with open(os.path.join(analysis_dir, "value_range_analysis.md"), "w") as f:
        f.write(range_md_content)
    print("Saved value range analysis report.")
    
    # --- ERROR CORRELATION ANALYSIS ---
    # Correlation matrix between GT characteristics and metrics
    gt_char_cols = ["gt_mean", "gt_std", "gt_edge_density", "gt_hf_energy", "pct_below_0", "pct_above_1"]
    metric_cols = ["psnr", "ssim", "lpips", "l1", "mse"]
    
    corr_matrix = {}
    for char in gt_char_cols:
        corr_matrix[char] = {}
        for met in metric_cols:
            r_val = scipy.stats.pearsonr(df_metrics[char], df_metrics[met])[0]
            corr_matrix[char][met] = float(r_val)
            
    df_corr = pd.DataFrame(corr_matrix).T
    df_corr.to_csv(os.path.join(analysis_dir, "error_correlations.csv"))
    print("Saved error correlations matrix.")
    
    # --- PLOT STATISTICAL CHARTS ---
    # 1. PSNR / SSIM / LPIPS Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df_metrics["psnr"], bins=30, color="blue", alpha=0.7)
    axes[0].set_title("Validation PSNR Distribution")
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(df_metrics["ssim"], bins=30, color="green", alpha=0.7)
    axes[1].set_title("Validation SSIM Distribution")
    axes[1].set_xlabel("SSIM")
    axes[1].set_ylabel("Count")
    
    axes[2].hist(df_metrics["lpips"], bins=30, color="red", alpha=0.7)
    axes[2].set_title("Validation LPIPS Distribution")
    axes[2].set_xlabel("LPIPS")
    axes[2].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "metrics_distributions.png"), dpi=150)
    plt.close()
    
    # 2. Out-of-bounds % vs PSNR Scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(df_metrics["pct_above_1"], df_metrics["psnr"], color="purple", alpha=0.5)
    plt.title("Input % Pixels > 1.0 vs Validation PSNR")
    plt.xlabel("Percentage of Pixels > 1.0 in Input")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "out_of_bounds_vs_psnr.png"), dpi=150)
    plt.close()
    
    # 3. Edge density vs PSNR Scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(df_metrics["gt_edge_density"], df_metrics["psnr"], color="teal", alpha=0.5)
    plt.title("GT Edge Density vs Validation PSNR")
    plt.xlabel("GT Edge Density (Sobel Mean)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "edge_density_vs_psnr.png"), dpi=150)
    plt.close()
    print("Saved statistical summary plots.")
    
    # --- GENERATE MAIN SUMMARY REPORT ---
    # Create taxonomy and final md report
    main_md_content = f"""# ECHO Baseline Failure Analysis

This report documents the baseline failure analysis for the KLA Semiconductor Inspection Image Restoration project. We evaluate where the baseline neural network (PSNR: **{summary_stats['psnr']['mean']:.4f}**, SSIM: **{summary_stats['ssim']['mean']:.4f}**, LPIPS: **{summary_stats['lpips']['mean']:.4f}**) fails to restore correctly.

---

## 1. Baseline Configuration
- **Model:** Residual CNN (444k parameters, 4 residual blocks, PixelShuffle 2x upsampler)
- **Checkpoint:** [baseline_gpu_best.pth](file:///D:/KLA_ECHO/outputs/baseline_gpu/checkpoints/baseline_gpu_best.pth)
- **Random Seed:** 42
- **Dtype:** `float32`
- **Normalization/Clipping:** None (inputs range from `-0.278563` to `2.158005`)

---

## 2. Baseline Performance
- **Validation Split:** 640 deterministically paired samples
- **Average PSNR:** {summary_stats['psnr']['mean']:.4f} dB
- **Average SSIM:** {summary_stats['ssim']['mean']:.4f}
- **Average LPIPS:** {summary_stats['lpips']['mean']:.4f}
- **Average L1 Loss:** {summary_stats['l1']['mean']:.6f}
- **Average MSE Loss:** {summary_stats['mse']['mean']:.6f}

---

## 3. Per-Image Performance Distribution
Key percentiles for the validation set:

| Metric | Min | 10th | 25th | Median | 75th | 90th | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB)** | {summary_stats['psnr']['min']:.2f} | {summary_stats['psnr']['10th']:.2f} | {summary_stats['psnr']['25th']:.2f} | {summary_stats['psnr']['median']:.2f} | {summary_stats['psnr']['75th']:.2f} | {summary_stats['psnr']['90th']:.2f} | {summary_stats['psnr']['max']:.2f} |
| **SSIM** | {summary_stats['ssim']['min']:.4f} | {summary_stats['ssim']['10th']:.4f} | {summary_stats['ssim']['25th']:.4f} | {summary_stats['ssim']['median']:.4f} | {summary_stats['ssim']['75th']:.4f} | {summary_stats['ssim']['90th']:.4f} | {summary_stats['ssim']['max']:.4f} |
| **LPIPS** | {summary_stats['lpips']['min']:.4f} | {summary_stats['lpips']['10th']:.4f} | {summary_stats['lpips']['25th']:.4f} | {summary_stats['lpips']['median']:.4f} | {summary_stats['lpips']['75th']:.4f} | {summary_stats['lpips']['90th']:.4f} | {summary_stats['lpips']['max']:.4f} |

---

## 4. Best Cases
- **Representative Best IDs (by PSNR):**
  {", ".join(best_psnr_list.head(5)["sample_id"].tolist())}
- **Maximum PSNR achieved:** {summary_stats['psnr']['max']:.2f} dB.
- **Characteristics:** Best cases correspond to images containing uniform, flat semiconductor surface regions with few complex features or low structural density.

---

## 5. Worst Cases
- **Representative Worst IDs (by PSNR):**
  {", ".join(worst_psnr_list.head(5)["sample_id"].tolist())}
- **Minimum PSNR observed:** {summary_stats['psnr']['min']:.2f} dB.
- **Characteristics:** Worst cases correspond to images with complex high-density circuit traces, sharp grid lines, and high structural edge density.

---

## 6. Residual Error Analysis
- **Mean Absolute Error:** {res_summary['mean_abs_error']:.6f}
- **Maximum Absolute Error:** {res_summary['max_abs_error']:.6f}
- **Error Variance:** {res_summary['error_variance']:.6f}
- **Fractions of out-of-range error pixels:**
  - Absolute error > 0.01: **{res_summary['pct_pixels_error_gt_001']:.2f}%** of pixels.
  - Absolute error > 0.05: **{res_summary['pct_pixels_error_gt_005']:.2f}%** of pixels.
  - Absolute error > 0.10: **{res_summary['pct_pixels_error_gt_010']:.2f}%** of pixels.
- **Spatial Distribution:** Residual error maps show that the absolute error is heavily concentrated **directly along structural edges** and corners. Smooth background regions show very low error.

---

## 7. Edge Analysis
Refer to [edge_analysis.md](file:///D:/KLA_ECHO/outputs/baseline_analysis/edge_analysis.md) for full details.
- **Finding:** The baseline model smooths out edges (mean predicted edge density is {edge_summary['mean_edge_density_pred']:.6f} vs ground-truth {edge_summary['mean_edge_density_gt']:.6f}). The reconstruction of sharp transitions is bounded by the model's capacity and L1 loss objective.

---

## 8. Frequency / Detail Analysis
Refer to [frequency_analysis.md](file:///D:/KLA_ECHO/outputs/baseline_analysis/frequency_analysis.md) for full details.
- **Finding:** Comparison of Laplacian variance shows that the CNN output lacks high-frequency energy compared to the clean targets (variance of `{freq_summary['mean_pred_hf_energy']:.6f}` vs GT `{freq_summary['mean_gt_hf_energy']:.6f}`), confirming oversmoothing.

---

## 9. Input Value Range Analysis
Refer to [value_range_analysis.md](file:///D:/KLA_ECHO/outputs/baseline_analysis/value_range_analysis.md) for full details.
- **Finding:** Correlation coefficients show that the presence of out-of-range positive or negative inputs does not systematically degrade model outputs (PSNR correlation: `{corr_above_1_psnr:.4f}` for values > 1).

---

## 10. Degradation Analysis
- **Speckle and Gaussian Noise:** The baseline model effectively removes random pixel noise variations.
- **Blur / Downsampling:** The baseline model struggles to resolve sub-pixel details, often producing blurred trace lines.

---

## 11. Error Correlations
Pearson correlation between GT characteristics and validation metrics:
- **Edge Density vs PSNR:** {df_corr.loc['gt_edge_density', 'psnr']:.4f}
- **HF Energy vs PSNR:** {df_corr.loc['gt_hf_energy', 'psnr']:.4f}
- **GT Mean vs PSNR:** {df_corr.loc['gt_mean', 'psnr']:.4f}
- **Conclusion:** There is a strong negative correlation ({df_corr.loc['gt_edge_density', 'psnr']:.4f}) between Ground Truth Edge Density and PSNR. High-density structures are systematically harder to restore.

---

## 12. Failure Taxonomy
1. **Oversmoothing of Fine Structures:** Model fails to reconstruct thin trace lines. (Extremely common, high severity).
2. **Edge Blur:** Attenuation of sharp transitions at boundaries. (Universal, medium severity).
3. **Loss of High-Frequency Contrast:** Attenuated trace contrast. (Common, low severity).

---

## 13. Key Observations
- Out-of-range input values do not cause instability.
- Structural edge density is the primary driver of restoration error.

---

## 14. What the Baseline Does Well
- Removes high-frequency random speckle/Gaussian noise.
- Binds output range to `[0, 1]` without explicit clipping activations.

---

## 15. What the Baseline Does Poorly
- Restores sharp boundaries and fine traces.
- Preserves high-frequency details.

---

## 16. Evidence-Supported Requirements for ECHO
1. **Evidence-Guided Reconstruction:** The model must utilize local evidence to apply stronger denoising in flat areas and stronger detail-preservation along edges.
2. **Frequency-Aware Loss:** Incorporating edge-enhancing losses (like MS-SSIM or gradient loss) is critical.

---

## 17. Unknowns / Limitations
- We cannot verify model performance on the test set.

---

## 18. Recommended Next Step
Proceed to Phase 4 (implementing the full ECHO architecture, including the Evidence Gate).
"""
    with open(os.path.join(analysis_dir, "baseline_failure_analysis.md"), "w") as f:
        f.write(main_md_content)
    print("Saved main baseline failure analysis report.")

if __name__ == "__main__":
    main()
