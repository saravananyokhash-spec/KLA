import os
import torch
from dataset import KLADataset

def test_train_split(dataset_root):
    print("\n--- Testing TRAIN split ---")
    train_dataset = KLADataset(dataset_root=dataset_root, split="train")
    dataset_len = len(train_dataset)
    print(f"Train Dataset Length: {dataset_len}")
    
    # We must have 3200 samples
    if dataset_len != 3200:
        raise ValueError(f"Expected 3200 train samples, got {dataset_len}")
        
    # Load and test at least 3 samples
    num_samples_to_test = min(5, dataset_len)
    print(f"Testing {num_samples_to_test} samples...")
    
    for idx in range(num_samples_to_test):
        sample = train_dataset[idx]
        
        # Verify keys
        required_keys = ["input", "target", "input_path", "target_path"]
        for k in required_keys:
            if k not in sample:
                raise KeyError(f"Key '{k}' is missing from sample dict.")
                
        inp = sample["input"]
        tgt = sample["target"]
        inp_path = sample["input_path"]
        tgt_path = sample["target_path"]
        
        print(f"\nSample index {idx}:")
        print(f"  Input Path: {inp_path}")
        print(f"  Target Path: {tgt_path}")
        print(f"  Input Shape: {inp.shape} | Target Shape: {tgt.shape}")
        print(f"  Input Dtype: {inp.dtype} | Target Dtype: {tgt.dtype}")
        
        inp_min, inp_max = inp.min().item(), inp.max().item()
        tgt_min, tgt_max = tgt.min().item(), tgt.max().item()
        print(f"  Input Range: [{inp_min:.6f}, {inp_max:.6f}]")
        print(f"  Target Range: [{tgt_min:.6f}, {tgt_max:.6f}]")
        
        # Verify shapes
        if inp.shape != torch.Size([1, 128, 128]):
            raise ValueError(f"Expected input shape (1, 128, 128), got {inp.shape}")
        if tgt.shape != torch.Size([1, 256, 256]):
            raise ValueError(f"Expected target shape (1, 256, 256), got {tgt.shape}")
            
        # Verify dtypes are float32
        if inp.dtype != torch.float32 or tgt.dtype != torch.float32:
            raise TypeError(f"Expected float32 tensors, got input={inp.dtype}, target={tgt.dtype}")
            
        # Verify pairing matching by filenames
        inp_fn = os.path.basename(inp_path)
        tgt_fn = os.path.basename(tgt_path)
        if inp_fn != tgt_fn:
            raise ValueError(f"Pairing mismatch! Input filename '{inp_fn}' does not match Target filename '{tgt_fn}'")
            
        # Verification of dimension scaling
        c_in, h_in, w_in = inp.shape
        c_tg, h_tg, w_tg = tgt.shape
        if h_tg != 2 * h_in or w_tg != 2 * w_in:
            raise ValueError(f"Spatial dimension scaling is not 2x! Got Input={h_in}x{w_in}, Target={h_tg}x{w_tg}")
            
    print("Train split tests PASSED successfully.")

def test_test_split(dataset_root):
    print("\n--- Testing TEST split ---")
    test_dataset = KLADataset(dataset_root=dataset_root, split="test")
    dataset_len = len(test_dataset)
    print(f"Test Dataset Length: {dataset_len}")
    
    # We must have 400 samples
    if dataset_len != 400:
        raise ValueError(f"Expected 400 test samples, got {dataset_len}")
        
    num_samples_to_test = min(3, dataset_len)
    print(f"Testing {num_samples_to_test} samples...")
    
    for idx in range(num_samples_to_test):
        sample = test_dataset[idx]
        
        # Verify keys
        required_keys = ["input", "target", "input_path", "target_path"]
        for k in required_keys:
            if k not in sample:
                raise KeyError(f"Key '{k}' is missing from sample dict.")
                
        inp = sample["input"]
        tgt = sample["target"]
        inp_path = sample["input_path"]
        tgt_path = sample["target_path"]
        
        print(f"\nSample index {idx}:")
        print(f"  Input Path: {inp_path}")
        print(f"  Target Path: {tgt_path} (Expected: None)")
        print(f"  Input Shape: {inp.shape}")
        print(f"  Input Dtype: {inp.dtype}")
        
        inp_min, inp_max = inp.min().item(), inp.max().item()
        print(f"  Input Range: [{inp_min:.6f}, {inp_max:.6f}]")
        
        # Verify shapes
        if inp.shape != torch.Size([1, 128, 128]):
            raise ValueError(f"Expected input shape (1, 128, 128), got {inp.shape}")
            
        # Verify target is None
        if tgt is not None:
            raise ValueError(f"Expected target to be None for test split, got {type(tgt)}")
        if tgt_path is not None:
            raise ValueError(f"Expected target_path to be None for test split, got {tgt_path}")
            
    print("Test split tests PASSED successfully.")

def main():
    dataset_root = os.environ.get("DATASET_ROOT", "D:/kla")
    print(f"Using DATASET_ROOT: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"DATASET_ROOT '{dataset_root}' does not exist.")
        
    test_train_split(dataset_root)
    test_test_split(dataset_root)
    
    print("\n=============================================")
    print("ALL DATASET PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=============================================")

if __name__ == "__main__":
    main()
