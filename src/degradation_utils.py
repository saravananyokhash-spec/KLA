"""Shared Gaussian blur / synthetic LR degradation utilities.

Used by Phase 6 evaluation and Phase 7 blur-aware training augmentation.
Images are never clipped, preserving the project's raw-input convention.
"""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(x, sigma):
    """Separable Gaussian blur for a BxCxHxW tensor; identity when sigma is 0."""
    if sigma <= 0:
        return x
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=channels)
    return F.conv2d(x, ky, padding=(radius, 0), groups=channels)


def synthetic_lr_from_target(target, noise_sigma, blur_sigma, sample_indices, device, seed=42):
    """Create deterministic LR inputs from GT: blur HR -> 2x bicubic downsample -> optional LR noise."""
    degraded_hr = gaussian_blur(target, blur_sigma)
    lr = F.interpolate(degraded_hr, scale_factor=0.5, mode="bicubic", align_corners=False)
    if noise_sigma:
        noise = torch.empty_like(lr)
        for i, sample_index in enumerate(sample_indices):
            generator = torch.Generator(device=device)
            generator.manual_seed(
                seed + sample_index * 1009 + int(noise_sigma * 1_000_000) + int(blur_sigma * 10_000)
            )
            noise[i] = torch.randn(lr[i].shape, generator=generator, device=device, dtype=lr.dtype)
        lr = lr + noise_sigma * noise
    return lr


def _sample_uniform(generator, low, high):
    return low + (high - low) * torch.rand((), generator=generator).item()


def _stable_sample_index(sample_key, seed):
    if isinstance(sample_key, str):
        return hash((sample_key, seed)) & 0x7FFFFFFF
    return int(sample_key)


def apply_training_degradation(original_lr, target, sample_keys, epoch, cfg, device):
    """Return batch LR inputs with controlled blur-aware augmentation.

    Preserves original disk inputs for ``preserve_original_probability`` fraction.
    Augmented samples use the same GT->blur->downsample->noise protocol as Phase 6.
    ``sample_keys`` should be stable identifiers (e.g. input paths) for reproducibility.
    """
    deg = cfg["degradation"]
    seed = cfg["training"]["seed"]
    preserve_p = float(deg["preserve_original_probability"])
    blur_sigma_min = float(deg["blur_sigma_min"])
    blur_sigma_max = float(deg["blur_sigma_max"])
    noise_sigma_min = float(deg.get("noise_sigma_min", 0.01))
    noise_sigma_max = float(deg.get("noise_sigma_max", 0.06))
    blur_only_fraction = float(deg.get("blur_only_fraction", 0.5))
    combined_enabled = bool(deg.get("combined_noise_blur", True))

    out = original_lr.clone()
    stats = {"original": 0, "blur_only": 0, "noise_blur": 0}

    for i, sample_key in enumerate(sample_keys):
        sample_index = _stable_sample_index(sample_key, seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 7919 + sample_index * 1009)

        if torch.rand((), generator=generator).item() < preserve_p:
            stats["original"] += 1
            continue

        blur_sigma = _sample_uniform(generator, blur_sigma_min, blur_sigma_max)
        noise_sigma = 0.0
        if combined_enabled and torch.rand((), generator=generator).item() >= blur_only_fraction:
            noise_sigma = _sample_uniform(generator, noise_sigma_min, noise_sigma_max)
            stats["noise_blur"] += 1
        else:
            stats["blur_only"] += 1

        out[i : i + 1] = synthetic_lr_from_target(
            target[i : i + 1],
            noise_sigma,
            blur_sigma,
            [sample_index],
            device,
            seed=seed + epoch * 17,
        )

    return out, stats


# Phase 6 evaluation protocol (unchanged).
PHASE6_CONDITIONS = {
    "gaussian_noise": {"low": (0.010, 0.0), "medium": (0.030, 0.0), "high": (0.060, 0.0)},
    "gaussian_blur": {"low": (0.0, 0.50), "medium": (0.0, 1.00), "high": (0.0, 1.50)},
    "noise_plus_blur": {"low": (0.010, 0.50), "medium": (0.030, 1.00), "high": (0.060, 1.50)},
}
