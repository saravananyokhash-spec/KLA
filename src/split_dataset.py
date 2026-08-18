import os
import glob
import pandas as pd
import numpy as np

def main():
    # Setup dataset root
    dataset_root = os.environ.get("DATASET_ROOT", "D:/kla")
    print(f"Using DATASET_ROOT: {dataset_root}")
    
    train_lr_dir = os.path.join(dataset_root, "train_set", "train", "NoisyLR")
    train_gt_dir = os.path.join(dataset_root, "train_set", "train", "GT")
    
    if not os.path.exists(train_lr_dir) or not os.path.exists(train_gt_dir):
        raise FileNotFoundError(f"Train directory does not exist: {train_lr_dir} or {train_gt_dir}")
        
    lr_files = sorted(glob.glob(os.path.join(train_lr_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(train_gt_dir, "*.npy")))
    
    # Check counts
    if len(lr_files) != 3200 or len(gt_files) != 3200:
        print(f"Warning: Expected 3200 files, found LR={len(lr_files)}, GT={len(gt_files)}")
        
    # Match pair by filename
    lr_basenames = {os.path.basename(f): f for f in lr_files}
    gt_basenames = {os.path.basename(f): f for f in gt_files}
    
    pairs = []
    for fn in sorted(lr_basenames.keys()):
        if fn in gt_basenames:
            pairs.append((lr_basenames[fn], gt_basenames[fn]))
        else:
            raise ValueError(f"No GT file matching LR file: {fn}")
            
    # Set seed for deterministic shuffling
    np.random.seed(42)
    indices = np.arange(len(pairs))
    np.random.shuffle(indices)
    
    shuffled_pairs = [pairs[i] for i in indices]
    
    # Calculate split index
    split_idx = int(0.8 * len(shuffled_pairs)) # 80% Train, 20% Val
    train_pairs = shuffled_pairs[:split_idx]
    val_pairs = shuffled_pairs[split_idx:]
    
    print(f"Total paired samples: {len(pairs)}")
    print(f"Train split samples (80%): {len(train_pairs)}")
    print(f"Validation split samples (20%): {len(val_pairs)}")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs/baseline", exist_ok=True)
    
    # Convert to DataFrames and save
    train_df = pd.DataFrame(train_pairs, columns=["input_path", "target_path"])
    val_df = pd.DataFrame(val_pairs, columns=["input_path", "target_path"])
    
    # Convert to relative paths from workspace or keep absolute? Let's save as relative or absolute, but let's make them configurable. Absolute paths are safer because they are stored externally. Let's store absolute paths.
    train_csv = "outputs/baseline/train_split.csv"
    val_csv = "outputs/baseline/val_split.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Saved train split list to: {train_csv}")
    print(f"Saved validation split list to: {val_csv}")

if __name__ == "__main__":
    main()
