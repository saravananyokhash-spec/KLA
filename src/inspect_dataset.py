import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_array(filepath, split, role):
    """
    Loads a numpy array and computes statistics.
    Does not modify the original data.
    """
    try:
        # Load the numpy array
        arr = np.load(filepath)
        
        # Verify shape
        shape = arr.shape
        ndim = arr.ndim
        
        if ndim == 2:
            height, width = shape
            channels = 1
        elif ndim == 3:
            # Check shape convention, e.g. (C, H, W) or (H, W, C)
            if shape[0] in [1, 3, 4]:
                channels, height, width = shape
            elif shape[2] in [1, 3, 4]:
                height, width, channels = shape
            else:
                height, width, channels = shape
        else:
            raise ValueError(f"Unexpected array dimension: {ndim} for file {filepath}")
            
        dtype = str(arr.dtype)
        total_elements = arr.size
        
        # Calculate statistics
        amin = float(arr.min())
        amax = float(arr.max())
        amean = float(arr.mean())
        astd = float(arr.std())
        
        # Calculate percentages
        below_zero = float(np.sum(arr < 0.0) / total_elements * 100.0)
        above_one = float(np.sum(arr > 1.0) / total_elements * 100.0)
        inside_range = float(np.sum((arr >= 0.0) & (arr <= 1.0)) / total_elements * 100.0)
        
        return {
            "path": os.path.abspath(filepath),
            "split": split,
            "role": role,
            "filename": os.path.basename(filepath),
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": dtype,
            "min": amin,
            "max": amax,
            "mean": amean,
            "std": astd,
            "values_below_zero_percent": below_zero,
            "values_above_one_percent": above_one,
            "values_inside_range_percent": inside_range
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    # Configure dataset root path via env variable or default to D:/kla
    dataset_root = os.environ.get("DATASET_ROOT", "D:/kla")
    print(f"Using DATASET_ROOT: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"CRITICAL ERROR: DATASET_ROOT '{dataset_root}' does not exist.")
        return

    # Define paths
    train_gt_dir = os.path.join(dataset_root, "train_set", "train", "GT")
    train_lr_dir = os.path.join(dataset_root, "train_set", "train", "NoisyLR")
    test_lr_dir = os.path.join(dataset_root, "test_set", "NoisyLR")
    
    print("\nVerifying paths:")
    print(f"Train GT: {train_gt_dir} (Exists: {os.path.exists(train_gt_dir)})")
    print(f"Train NoisyLR: {train_lr_dir} (Exists: {os.path.exists(train_lr_dir)})")
    print(f"Test NoisyLR: {test_lr_dir} (Exists: {os.path.exists(test_lr_dir)})")
    
    # Collect files
    train_gt_files = glob.glob(os.path.join(train_gt_dir, "*.npy"))
    train_lr_files = glob.glob(os.path.join(train_lr_dir, "*.npy"))
    test_lr_files = glob.glob(os.path.join(test_lr_dir, "*.npy"))
    
    print(f"\nDiscovered files:")
    print(f"Train GT files: {len(train_gt_files)}")
    print(f"Train NoisyLR files: {len(train_lr_files)}")
    print(f"Test NoisyLR files: {len(test_lr_files)}")
    
    records = []
    
    # Process Train GT files
    print("\nProcessing Train GT arrays...")
    for f in sorted(train_gt_files):
        res = analyze_array(f, "train", "target")
        if res:
            records.append(res)
            
    # Process Train NoisyLR files
    print("Processing Train NoisyLR arrays...")
    for f in sorted(train_lr_files):
        res = analyze_array(f, "train", "input")
        if res:
            records.append(res)
            
    # Process Test NoisyLR files
    print("Processing Test NoisyLR arrays...")
    for f in sorted(test_lr_files):
        res = analyze_array(f, "test", "input")
        if res:
            records.append(res)
            
    # Save the report
    df = pd.DataFrame(records)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    report_csv = os.path.join(output_dir, "dataset_report.csv")
    df.to_csv(report_csv, index=False)
    print(f"\nDataset report saved to: {report_csv}")
    
    # Print readable summaries
    print("\n==================================================")
    print("DATASET STATISTICAL SUMMARY")
    print("==================================================")
    
    for split in ["train", "test"]:
        for role in ["input", "target"]:
            sub = df[(df["split"] == split) & (df["role"] == role)]
            if len(sub) == 0:
                continue
            print(f"\nSplit: {split.upper()} | Role: {role.upper()}")
            print(f"  Count: {len(sub)}")
            print(f"  Resolution (Mean HxW): {sub['height'].mean():.1f} x {sub['width'].mean():.1f}")
            print(f"  Channels: {sub['channels'].iloc[0]} (dtype: {sub['dtype'].iloc[0]})")
            print(f"  Min Value (Overall range): {sub['min'].min():.6f} to {sub['max'].max():.6f}")
            print(f"  Mean Value: {sub['mean'].mean():.6f} | Std Dev: {sub['std'].mean():.6f}")
            print(f"  Out of range < 0: {sub['values_below_zero_percent'].mean():.4f}%")
            print(f"  Out of range > 1: {sub['values_above_one_percent'].mean():.4f}%")
            print(f"  Inside [0, 1] range: {sub['values_inside_range_percent'].mean():.4f}%")
            
    # Pairing Verification
    print("\n==================================================")
    print("PAIRING VERIFICATION")
    print("==================================================")
    train_input_fns = df[(df["split"] == "train") & (df["role"] == "input")]["filename"].tolist()
    train_target_fns = df[(df["split"] == "train") & (df["role"] == "target")]["filename"].tolist()
    
    paired_count = 0
    mismatches = []
    
    for fn in train_input_fns:
        if fn in train_target_fns:
            paired_count += 1
        else:
            mismatches.append(fn)
            
    print(f"Total Train Input files: {len(train_input_fns)}")
    print(f"Total Train Target files: {len(train_target_fns)}")
    print(f"Successfully paired train files: {paired_count}")
    if mismatches:
        print(f"WARNING: The following input files do not have matching target files: {mismatches[:10]}")
    else:
        print("Success: All train input files perfectly match target files by filename.")

    # Spatial Resolution Scale Analysis
    avg_gt_h = df[df["role"] == "target"]["height"].mean()
    avg_lr_h = df[(df["split"] == "train") & (df["role"] == "input")]["height"].mean()
    if not pd.isna(avg_gt_h) and not pd.isna(avg_lr_h):
        scale_factor = avg_gt_h / avg_lr_h
        print(f"GT height: {avg_gt_h} | LR height: {avg_lr_h} | Scale factor: {scale_factor:.1f}x")
    
    # Generate Visualizations for Samples
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Select 3 representative pairs for plotting
    sample_ids = ["000000.npy", "001000.npy", "002000.npy"]
    
    print("\nGenerating visual comparison samples...")
    for idx, fn in enumerate(sample_ids):
        lr_path = os.path.join(train_lr_dir, fn)
        gt_path = os.path.join(train_gt_dir, fn)
        
        if os.path.exists(lr_path) and os.path.exists(gt_path):
            lr_arr = np.load(lr_path)
            gt_arr = np.load(gt_path)
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            lr_min, lr_max = lr_arr.min(), lr_arr.max()
            gt_min, gt_max = gt_arr.min(), gt_arr.max()
            
            # Min-max scale display copy
            lr_display = (lr_arr - lr_min) / (lr_max - lr_min + 1e-8)
            gt_display = (gt_arr - gt_min) / (gt_max - gt_min + 1e-8)
            
            # Plot images
            axes[0, 0].imshow(lr_display, cmap="gray")
            axes[0, 0].set_title(f"Degraded Input ({fn})\nOriginal Range: [{lr_min:.4f}, {lr_max:.4f}]")
            axes[0, 0].axis("off")
            
            axes[0, 1].imshow(gt_display, cmap="gray")
            axes[0, 1].set_title(f"Ground Truth ({fn})\nOriginal Range: [{gt_min:.4f}, {gt_max:.4f}]")
            axes[0, 1].axis("off")
            
            # Plot histograms
            axes[1, 0].hist(lr_arr.ravel(), bins=50, color="orange", alpha=0.7)
            axes[1, 0].set_title("Input Histogram")
            axes[1, 0].set_xlabel("Value")
            axes[1, 0].set_ylabel("Count")
            
            axes[1, 1].hist(gt_arr.ravel(), bins=50, color="blue", alpha=0.7)
            axes[1, 1].set_title("Ground-Truth Histogram")
            axes[1, 1].set_xlabel("Value")
            axes[1, 1].set_ylabel("Count")
            
            plt.tight_layout()
            
            out_filename = os.path.join(samples_dir, f"sample_{idx+1:03d}_comparison.png")
            plt.savefig(out_filename, dpi=150)
            plt.close()
            print(f"Saved visual comparison: {out_filename}")
        else:
            print(f"Skipping visualization for {fn} (files not found).")
            
if __name__ == "__main__":
    main()
