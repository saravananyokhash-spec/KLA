import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class KLADataset(Dataset):
    """
    PyTorch Dataset for KLA Semiconductor Inspection Image Restoration.
    Loads float32 NumPy arrays (.npy) representing degraded input images (128x128)
    and clean ground-truth target images (256x256) for training, and degraded
    input images (128x128) for test.
    """
    def __init__(self, dataset_root, split="train", transform=None, csv_path=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split.lower()
        self.transform = transform
        self.csv_path = csv_path
        
        if self.split not in ["train", "test", "val"]:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'test', or 'val'.")
            
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.dataset_root}")
            
        self.input_paths = []
        self.target_paths = []
        
        if self.csv_path is not None:
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV split file not found: {self.csv_path}")
            print(f"Loading dataset paths from CSV: {self.csv_path}")
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            self.input_paths = df["input_path"].tolist()
            # Load target paths if available (validation split has targets)
            if "target_path" in df.columns:
                self.target_paths = df["target_path"].tolist()
            else:
                self.target_paths = [None] * len(self.input_paths)
        else:
            if self.split == "train":
                self.input_dir = os.path.join(self.dataset_root, "train_set", "train", "NoisyLR")
                self.target_dir = os.path.join(self.dataset_root, "train_set", "train", "GT")
                
                if not os.path.exists(self.input_dir):
                    raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
                if not os.path.exists(self.target_dir):
                    raise FileNotFoundError(f"Target directory does not exist: {self.target_dir}")
                    
                input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.npy")))
                target_files = sorted(glob.glob(os.path.join(self.target_dir, "*.npy")))
                
                # Deterministic naming match pairing
                input_basenames = {os.path.basename(f): f for f in input_files}
                target_basenames = {os.path.basename(f): f for f in target_files}
                
                # Check for bidirectional mapping matches
                for fn, in_path in sorted(input_basenames.items()):
                    if fn in target_basenames:
                        self.input_paths.append(in_path)
                        self.target_paths.append(target_basenames[fn])
                    else:
                        raise ValueError(f"Mismatched input file: {fn} has no matching ground truth file.")
                        
                if len(self.input_paths) == 0:
                    raise ValueError(f"No valid paired samples found in {self.dataset_root} for train split.")
                    
                if len(self.input_paths) != len(input_files) or len(self.target_paths) != len(target_files):
                    raise ValueError(f"Count mismatch: Inputs found={len(input_files)}, Targets found={len(target_files)}, Paired={len(self.input_paths)}")
                    
            else: # test split
                self.input_dir = os.path.join(self.dataset_root, "test_set", "NoisyLR")
                
                if not os.path.exists(self.input_dir):
                    raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
                    
                self.input_paths = sorted(glob.glob(os.path.join(self.input_dir, "*.npy")))
                self.target_paths = [None] * len(self.input_paths)
                
                if len(self.input_paths) == 0:
                    raise ValueError(f"No files found in {self.input_dir} for test split.")
                
    def __len__(self):
        return len(self.input_paths)
        
    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        target_path = self.target_paths[idx]
        
        # Safe loading and missing file checks
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at index {idx}: {input_path}")
            
        input_arr = np.load(input_path)
        
        # Verify input dims
        if input_arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {input_arr.shape} at {input_path}")
            
        # Convert to float32 PyTorch tensor with channel dimension (1, H, W)
        input_tensor = torch.from_numpy(input_arr.astype(np.float32)).unsqueeze(0)
        
        target_tensor = None
        if self.split in ("train", "val"):
            if target_path is None or not os.path.exists(target_path):
                raise FileNotFoundError(f"Target file not found at index {idx}: {target_path}")
                
            target_arr = np.load(target_path)
            
            if target_arr.ndim != 2:
                raise ValueError(f"Expected 2D target array, got shape {target_arr.shape} at {target_path}")
                
            # Perform resolution scaling validation (2x scale factor check)
            in_h, in_w = input_arr.shape
            tg_h, tg_w = target_arr.shape
            
            if tg_h != 2 * in_h or tg_w != 2 * in_w:
                raise ValueError(
                    f"Shape mismatch at index {idx}: Input shape {input_arr.shape} does not map to "
                    f"Target shape {target_arr.shape} via 2x scaling."
                )
                
            target_tensor = torch.from_numpy(target_arr.astype(np.float32)).unsqueeze(0)
            
        # Compile dictionary item
        item = {
            "input": input_tensor,
            "target": target_tensor,
            "input_path": os.path.abspath(input_path),
            "target_path": os.path.abspath(target_path) if target_path else None
        }
        
        if self.transform:
            item = self.transform(item)
            
        return item
