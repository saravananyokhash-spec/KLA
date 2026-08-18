# ECHO: Adaptive Dual-Model Restoration

ECHO is an evidence-constrained restoration and super-resolution pipeline designed for the KLA Semiconductor Inspection Image Restoration Problem Statement.

## Project Description
The project restores degraded, low-resolution grayscale semiconductor inspection images ($128 \times 128$ `.npy` arrays) into high-resolution, clean target inspection images ($256 \times 256$ `.npy` arrays). It resolves additive Gaussian noise, multiplicative speckle noise, and downsampling corruptions without hallucinating unsupportable detail.

## Model Architecture
The final submission uses a **Degradation-Aware Adaptive Dual-Model Routing** pipeline combining two specialized Spatial-Frequency expert models:
- **Phase 9 (Fidelity Expert)**: Optimized for clean/smooth images, preserving fine structures and structural PSNR/SSIM.
- **Phase 11 Detail-Preserving (Robustness Expert)**: Optimized for noisy/degraded images, providing superior perceptual quality (LPIPS) and noise suppression.

### Inference Flow
1. **Input Validation**: Verifies shapes and formats of input `.npy` arrays.
2. **Degradation Estimation**: Calculates a local high-frequency noise level metric using the deterministic Laplacian Median Absolute Deviation (MAD) operator.
3. **Adaptive Routing**: 
   - Clean/smooth inputs ($\sigma_n \le 0.015$) use 100% Phase 9 model predictions.
   - Noisy/degraded inputs ($\sigma_n \ge 0.060$) use 100% Phase 11 Detail-Preserving model predictions.
   - Intermediate inputs linearly blend predictions from both experts.
4. **Postprocessing Validation**: Ensures outputs are clamped strictly to `[0.0, 1.0]` and free from NaN/Inf values.

## Requirements
- Python $\ge$ 3.10
- `torch` $\ge$ 2.0.0
- `numpy` $\ge$ 1.20.0

## Directory Structure
```text
echo_submission/
├── run.py                 # Self-contained wrapper and router
├── requirements.txt       # Execution dependencies
├── README.md              # Project documentation
├── verify_submission.py   # Test suite utility
│
└── models/
    ├── echo_best.pth      # Phase 4 structural guidance model
    ├── phase9.pth         # Phase 9 fidelity expert checkpoint
    └── phase11_detail.pth # Phase 11 detail-preserving expert checkpoint
```

## Execution
Run the inference pipeline using:
```bash
python run.py <input-dir> <output-dir>
```

### Example
```bash
python run.py ./input ./output
```

## Input Format
- Grayscale numpy arrays saved as `.npy` files.
- Receptive field dimensions: 2D shape of `(H, W)` or `(H, W, 1)`.

## Output Format
- Grayscale numpy arrays saved as `.npy` files.
- Output filenames match input filenames exactly.
- Dimensions: shape `(2*H, 2*W)` (target resolution scale factor of 2x).
- Values are strictly finite (no NaN, no Inf) and clamped to the valid range `[0.0, 1.0]`.

## GPU Support
NVIDIA CUDA is automatically detected and utilized for accelerated execution when available, falling back to CPU execution otherwise.

## Offline Compatibility
Execution is completely self-contained and internet-independent. It does not require any external weights downloads, remote API keys, or Hugging Face repository connections.
