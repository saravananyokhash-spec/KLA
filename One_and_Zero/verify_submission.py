import os
import shutil
import subprocess
import numpy as np
import sys

def main():
    print("=============================================================")
    print("ECHO SUBMISSION VALIDATION UTILITY")
    print("=============================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "verify_input")
    output_dir = os.path.join(base_dir, "verify_output")

    # Clean up old directories
    for path in (input_dir, output_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(input_dir, exist_ok=True)

    # 1. Create dummy input files
    print("Generating representative input .npy files...")
    np.random.seed(42)
    inputs = {
        "test_001.npy": np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32),
        "test_002.npy": np.random.uniform(-0.05, 1.05, (128, 128)).astype(np.float32), # contains slight out-of-range values
        "test_003.npy": np.random.uniform(0.1, 0.9, (128, 128, 1)).astype(np.float32), # contains channel dimension
    }

    for fn, arr in inputs.items():
        np.save(os.path.join(input_dir, fn), arr)
    print(f"Created {len(inputs)} dummy input files under: {input_dir}")

    # 2. Execute run.py
    run_script = os.path.join(base_dir, "run.py")
    cmd = [sys.executable, run_script, input_dir, output_dir]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Command output:")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing run.py:")
        print(e.stderr)
        sys.exit(1)

    # 3. Verify outputs
    print("Verifying outputs...")
    if not os.path.exists(output_dir):
        print(f"FAIL: Output directory was not created: {output_dir}")
        sys.exit(1)

    output_files = sorted(os.listdir(output_dir))
    expected_files = sorted(inputs.keys())

    if output_files != expected_files:
        print(f"FAIL: Output files mismatch. Expected: {expected_files}, got: {output_files}")
        sys.exit(1)

    for fn in expected_files:
        path = os.path.join(output_dir, fn)
        try:
            arr = np.load(path)
            
            # Check target resolution
            expected_shape = (256, 256)
            if arr.shape != expected_shape:
                print(f"FAIL: File {fn} has incorrect shape {arr.shape}, expected {expected_shape}")
                sys.exit(1)
                
            # Check finite values
            if not np.isfinite(arr).all():
                print(f"FAIL: File {fn} contains NaNs or Infs!")
                sys.exit(1)
                
            # Check values in [0, 1]
            if (arr < 0.0).any() or (arr > 1.0).any():
                print(f"FAIL: File {fn} has values outside [0.0, 1.0]!")
                sys.exit(1)
                
            print(f"  --> {fn} PASS (shape: {arr.shape}, range: [{arr.min():.4f}, {arr.max():.4f}])")
        except Exception as ex:
            print(f"FAIL: Error reading or validating {fn}: {ex}")
            sys.exit(1)

    # Clean up temporary test files
    for path in (input_dir, output_dir):
        if os.path.exists(path):
            shutil.rmtree(path)

    print("=============================================================")
    print("SUBMISSION VALIDATION PASS")
    print("=============================================================")

if __name__ == "__main__":
    main()
