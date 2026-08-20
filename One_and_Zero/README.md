# ECHO: Evidence-Constrained Full Image Restoration

ECHO is an evidence-constrained restoration and super-resolution pipeline designed for the KLA Semiconductor Inspection Image Restoration Problem Statement.

## Project Description
The project restores degraded, low-resolution grayscale semiconductor inspection images ($128 \times 128$ `.npy` arrays) into high-resolution, clean target inspection images ($256 \times 256$ `.npy` arrays). It resolves additive Gaussian noise, multiplicative speckle noise, and downsampling corruptions without hallucinating unsupportable detail.

## Model Architecture
The final submission uses a **Spatial Frequency Restoration Network** (`SpatialFrequencyRestorationNet`) backed by deep structural guidance from a Phase 4 baseline model:
- **Spatial Multi-Scale Branch**: Extracts multi-scale visual details using multiple kernel fields.
- **Frequency Branch**: Decomposes the input into Low, Mid, and High radial bands using a differentiable Frequency Decomposition Module.
- **Gated Guidance Fusion**: Integrates spatial, frequency, and deep structural guidance from a frozen Phase 4 baseline.

## Requirements
- Python $\ge$ 3.10
- `torch` $\ge$ 2.0.0
- `numpy` $\ge$ 1.20.0

## Directory Structure
```text
echo_submission/
├── run.py                 # Self-contained inference wrapper
├── requirements.txt       # Pinned execution dependencies
├── README.md              # Project documentation
├── verify_submission.py   # Test suite utility
│
└── models/
    ├── echo_best.pth      # Phase 4 structural guidance model
    └── FINAL_MODEL.pth    # Phase 9 champion restoration model
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
