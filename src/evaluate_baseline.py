import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import lpips

from utils import load_config, set_seed
from dataset import KLADataset
from baseline_model import BaselineRestorationNet, get_model_info
from metrics import compute_psnr, compute_ssim, compute_lpips

def main():
    # Load config
    config_path = "configs/baseline.yaml"
    config = load_config(config_path)
    
    # Set seed
    set_seed(config["seed"])
    
    # Device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device: {device}")
    
    # Initialize LPIPS model (AlexNet-based)
    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Load validation split dataset
    print("\nLoading validation dataset...")
    val_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path=config["val_split_path"]
    )
    print(f"Validation dataset length: {len(val_dataset)}")
    
    # Load trained model
    print("\nLoading trained CNN model...")
    model_cfg = config.get("model", {})
    model = BaselineRestorationNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 4)
    )
    
    checkpoint_path = config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 1. EVALUATION LOOPS
    bicubic_results = []
    cnn_results = []
    
    # For timing inference speed
    inference_times = []
    
    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            batch = val_dataset[idx]
            
            # Tensors shape (1, 128, 128) and (1, 256, 256)
            inp_tensor = batch["input"]
            tgt_tensor = batch["target"]
            inp_path = batch["input_path"]
            
            # Batch dimensions (1, 1, 128, 128) for evaluation
            inp_batch = inp_tensor.unsqueeze(0).to(device)
            tgt_batch = tgt_tensor.unsqueeze(0).to(device)
            
            # --- EVALUATE BICUBIC ---
            # PyTorch bicubic upsample
            bic_batch = torch.nn.functional.interpolate(
                inp_batch, scale_factor=2, mode="bicubic", align_corners=False
            )
            
            bic_psnr = compute_psnr(bic_batch.squeeze(0), tgt_tensor)
            bic_ssim = compute_ssim(bic_batch.squeeze(0), tgt_tensor)
            bic_lpips = compute_lpips(bic_batch, tgt_batch, lpips_model, device)
            
            bicubic_results.append({
                "psnr": bic_psnr,
                "ssim": bic_ssim,
                "lpips": bic_lpips
            })
            
            # --- EVALUATE CNN ---
            # Time inference for speed benchmark
            start_time = time.perf_counter()
            cnn_batch = model(inp_batch)
            # Ensure GPU execution finishes if CUDA
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            # Ignore the first few runs to warm up timing
            if idx >= 5:
                inference_times.append(elapsed)
                
            cnn_psnr = compute_psnr(cnn_batch.squeeze(0), tgt_tensor)
            cnn_ssim = compute_ssim(cnn_batch.squeeze(0), tgt_tensor)
            cnn_lpips = compute_lpips(cnn_batch, tgt_batch, lpips_model, device)
            
            cnn_results.append({
                "psnr": cnn_psnr,
                "ssim": cnn_ssim,
                "lpips": cnn_lpips
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(val_dataset)} validation samples.")
                
    # 2. CALCULATE AND SAVE METRICS
    os.makedirs("outputs/baseline/metrics", exist_ok=True)
    os.makedirs("outputs/baseline/samples", exist_ok=True)
    
    def process_stats(results):
        df_res = pd.DataFrame(results)
        return {
            "mean_psnr": float(df_res["psnr"].mean()),
            "median_psnr": float(df_res["psnr"].median()),
            "min_psnr": float(df_res["psnr"].min()),
            "max_psnr": float(df_res["psnr"].max()),
            "mean_ssim": float(df_res["ssim"].mean()),
            "median_ssim": float(df_res["ssim"].median()),
            "min_ssim": float(df_res["ssim"].min()),
            "max_ssim": float(df_res["ssim"].max()),
            "mean_lpips": float(df_res["lpips"].mean()),
            "median_lpips": float(df_res["lpips"].median()),
            "min_lpips": float(df_res["lpips"].min()),
            "max_lpips": float(df_res["lpips"].max())
        }
        
    bic_stats = process_stats(bicubic_results)
    cnn_stats = process_stats(cnn_results)
    
    # Save JSON files
    with open("outputs/baseline/metrics/bicubic_metrics.json", "w") as f:
        json.dump(bic_stats, f, indent=4)
    with open("outputs/baseline/metrics/baseline_metrics.json", "w") as f:
        json.dump(cnn_stats, f, indent=4)
        
    # Print metrics
    print("\n=============================================")
    print("EVALUATION RESULTS Summary")
    print("=============================================")
    print("Method       | PSNR Mean | SSIM Mean | LPIPS Mean")
    print("---------------------------------------------")
    print(f"Bicubic      | {bic_stats['mean_psnr']:.4f}    | {bic_stats['mean_ssim']:.4f}    | {bic_stats['mean_lpips']:.4f}")
    print(f"Baseline CNN | {cnn_stats['mean_psnr']:.4f}    | {cnn_stats['mean_ssim']:.4f}    | {cnn_stats['mean_lpips']:.4f}")
    
    # Save comparison CSV
    comparison_df = pd.DataFrame([
        {"Method": "Bicubic", "PSNR": bic_stats["mean_psnr"], "SSIM": bic_stats["mean_ssim"], "LPIPS": bic_stats["mean_lpips"]},
        {"Method": "Baseline CNN", "PSNR": cnn_stats["mean_psnr"], "SSIM": cnn_stats["mean_ssim"], "LPIPS": cnn_stats["mean_lpips"]}
    ])
    comparison_df.to_csv("outputs/baseline/metrics/comparison.csv", index=False)
    print("\nSaved comparison CSV to: outputs/baseline/metrics/comparison.csv")
    
    # 3. VISUAL COMPARISON PLOTS (First 3 validation samples)
    print("\nGenerating visual comparison plots...")
    for idx in range(3):
        batch = val_dataset[idx]
        inp_tensor = batch["input"]
        tgt_tensor = batch["target"]
        inp_path = batch["input_path"]
        fn = os.path.basename(inp_path)
        
        inp_batch = inp_tensor.unsqueeze(0).to(device)
        
        # Upsample predictions
        with torch.no_grad():
            bic_batch = torch.nn.functional.interpolate(
                inp_batch, scale_factor=2, mode="bicubic", align_corners=False
            )
            cnn_batch = model(inp_batch)
            
        # Move tensors to numpy for visual plotting
        inp_arr = inp_tensor.squeeze(0).detach().cpu().numpy()
        bic_arr = bic_batch.squeeze(0).squeeze(0).detach().cpu().numpy()
        cnn_arr = cnn_batch.squeeze(0).squeeze(0).detach().cpu().numpy()
        tgt_arr = tgt_tensor.squeeze(0).detach().cpu().numpy()
        
        # Scaling inputs for visualization display only (original values left untouched)
        inp_min, inp_max = inp_arr.min(), inp_arr.max()
        inp_display = (inp_arr - inp_min) / (inp_max - inp_min + 1e-8)
        
        # Since bicubic can go outside [0,1], also display scale
        bic_min, bic_max = bic_arr.min(), bic_arr.max()
        bic_display = np.clip(bic_arr, 0.0, 1.0) # clamped display copy
        
        cnn_min, cnn_max = cnn_arr.min(), cnn_arr.max()
        cnn_display = np.clip(cnn_arr, 0.0, 1.0) # clamped display copy
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(inp_display, cmap="gray")
        axes[0].set_title(f"Degraded Input\nRange: [{inp_min:.3f}, {inp_max:.3f}]")
        axes[0].axis("off")
        
        axes[1].imshow(bic_display, cmap="gray")
        axes[1].set_title(f"Bicubic 2x\nRange: [{bic_min:.3f}, {bic_max:.3f}]")
        axes[1].axis("off")
        
        axes[2].imshow(cnn_display, cmap="gray")
        axes[2].set_title(f"Baseline CNN\nRange: [{cnn_min:.3f}, {cnn_max:.3f}]")
        axes[2].axis("off")
        
        axes[3].imshow(tgt_arr, cmap="gray")
        axes[3].set_title(f"Ground Truth\nRange: [0.0, 1.0]")
        axes[3].axis("off")
        
        plt.tight_layout()
        out_plot_path = f"outputs/baseline/samples/bicubic_{idx+1:03d}.png"
        plt.savefig(out_plot_path, dpi=150)
        plt.close()
        print(f"Saved visualization: {out_plot_path}")
        
    # 4. TRAINING CURVES
    history_path = "outputs/baseline/loss_history.json"
    if os.path.exists(history_path):
        print("\nPlotting training curves...")
        with open(history_path, 'r') as f:
            hist = json.load(f)
            
        plt.figure(figsize=(8, 5))
        plt.plot(hist["train_loss"], label="Train Loss", color="blue")
        plt.plot(hist["val_loss"], label="Validation Loss", color="red")
        plt.title("Baseline Model Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (L1)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/baseline/loss_curve.png", dpi=150)
        plt.close()
        print("Saved training curves to: outputs/baseline/loss_curve.png")
        
    # 5. COMPUTATIONAL PERFORMANCE
    tot_params, train_params, model_size = get_model_info(model)
    avg_inference_time = np.mean(inference_times) if len(inference_times) > 0 else 0.0
    
    perf = {
        "parameter_count": tot_params,
        "trainable_parameter_count": train_params,
        "model_size_mb": float(model_size),
        "avg_inference_time_per_image_seconds": float(avg_inference_time),
        "device_used": str(device)
    }
    
    with open("outputs/baseline/metrics/performance.json", "w") as f:
        json.dump(perf, f, indent=4)
    print("Saved performance benchmark to: outputs/baseline/metrics/performance.json")

if __name__ == "__main__":
    main()
