import os
import shutil
import subprocess
import numpy as np
import sys

def main():
    print("=============================================================")
    print("ECHO STANDALONE SUBMISSION CREATION & VALIDATION UTILITY")
    print("=============================================================")

    # Target folder paths
    target_dir = r"D:\One_and_Zero"
    target_models_dir = os.path.join(target_dir, "models")
    
    # Source files inside KLA_ECHO
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # D:\KLA_ECHO
    src_run_py = os.path.join(base_dir, "One_and_Zero", "run.py")
    src_requirements_txt = os.path.join(base_dir, "One_and_Zero", "requirements.txt")
    src_readme_md = os.path.join(base_dir, "One_and_Zero", "README.md")
    
    src_p4_ckpt = os.path.join(base_dir, "outputs", "echo_phase4", "checkpoints", "echo_best.pth")
    src_p9_ckpt = os.path.join(base_dir, "outputs", "phase9_targeted", "checkpoints", "echo_phase9_best.pth")

    # 1. Create target folder structure
    print(f"Creating directories under {target_dir}...")
    os.makedirs(target_models_dir, exist_ok=True)

    # 2. Copy code files
    print("Copying submission package source files...")
    shutil.copy2(src_run_py, os.path.join(target_dir, "run.py"))
    shutil.copy2(src_requirements_txt, os.path.join(target_dir, "requirements.txt"))
    shutil.copy2(src_readme_md, os.path.join(target_dir, "README.md"))

    # 3. Copy model checkpoints
    print("Copying model checkpoints...")
    shutil.copy2(src_p4_ckpt, os.path.join(target_models_dir, "echo_best.pth"))
    shutil.copy2(src_p9_ckpt, os.path.join(target_models_dir, "phase9.pth"))
    print("Files successfully copied into standalone submission folder.")

    # 4. Perform local end-to-end test on the 400 KLA test images
    test_input_dir = r"D:\Test_input"
    test_output_dir = r"D:\Test_output"

    # Clear old generated files
    if os.path.exists(test_output_dir):
        print(f"Clearing old generated files from {test_output_dir}...")
        for fn in os.listdir(test_output_dir):
            if fn.endswith(".npy"):
                os.remove(os.path.join(test_output_dir, fn))
    else:
        os.makedirs(test_output_dir, exist_ok=True)

    print("\nRunning local end-to-end inference test...")
    target_run_py = os.path.join(target_dir, "run.py")
    cmd = [sys.executable, target_run_py, test_input_dir, test_output_dir]
    print(f"Executing: {' '.join(cmd)}")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Inference stdout:")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing submission run.py wrapper:")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit(1)

    # 5. Automatically verify outputs
    print("\nVerifying outputs...")
    if not os.path.exists(test_output_dir):
        print("FAIL: Test output directory was not created!")
        sys.exit(1)

    # Identify source files directly in D:\Test_input
    src_input_files = sorted([f for f in os.listdir(test_input_dir) if f.endswith(".npy")])
    output_files = sorted([f for f in os.listdir(test_output_dir) if f.endswith(".npy")])

    if len(output_files) != len(src_input_files):
        print(f"FAIL: File count mismatch. Inputs: {len(src_input_files)}, Outputs: {len(output_files)}")
        sys.exit(1)

    if output_files != src_input_files:
        print("FAIL: Output filenames do not match input filenames exactly!")
        sys.exit(1)

    print(f"Filenames match exactly. Checking all {len(output_files)} numpy arrays...")
    for idx, fn in enumerate(output_files):
        path = os.path.join(test_output_dir, fn)
        try:
            arr = np.load(path)
            
            # Grayscale check
            if arr.ndim != 2:
                print(f"FAIL: Output {fn} is not grayscale! Dims: {arr.ndim}")
                sys.exit(1)
                
            # Target resolution shape check (256, 256)
            expected_shape = (256, 256)
            if arr.shape != expected_shape:
                print(f"FAIL: Output {fn} has shape {arr.shape}, expected {expected_shape}")
                sys.exit(1)
                
            # Check finite values (No NaN, No Inf)
            if not np.isfinite(arr).all():
                print(f"FAIL: Output {fn} contains NaN or Inf!")
                sys.exit(1)
                
            # Check range values strictly within [0.0, 1.0]
            if (arr < 0.0).any() or (arr > 1.0).any():
                print(f"FAIL: Output {fn} has values outside [0.0, 1.0]! Range: [{arr.min():.4f}, {arr.max():.4f}]")
                sys.exit(1)
                
            # Detailed load and printing of first 3 samples
            if idx < 3:
                inp_arr = np.load(os.path.join(test_input_dir, fn))
                print(f"\n--- Output Sample Verification ({fn}) ---")
                print(f"  Input Shape   : {inp_arr.shape}")
                print(f"  Output Shape  : {arr.shape}")
                print(f"  Data Type     : {arr.dtype}")
                print(f"  Min Value     : {arr.min():.6f}")
                print(f"  Max Value     : {arr.max():.6f}")
                print(f"  Mean Value    : {arr.mean():.6f}")
                print(f"  Finite Status : {np.isfinite(arr).all()}")
                
                # Check that outputs are actually modified/restored and not just copied
                if inp_arr.shape == arr.shape and np.allclose(inp_arr, arr):
                    print("  FAIL: Output is simply a copy of the input!")
                    sys.exit(1)
                else:
                    print("  Fidelity Check: Output is successfully restored (not simply copied).")
                
        except Exception as ex:
            print(f"FAIL: Error reading or validating file {fn}: {ex}")
            sys.exit(1)

    print("\n=============================================================")
    print("STANDALONE SUBMISSION VALIDATION PASS")
    print("=============================================================")

if __name__ == "__main__":
    main()
