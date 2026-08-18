import numpy as np
import torch
import skimage.metrics

def compute_psnr(pred, gt):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR).
    Expects inputs as numpy arrays or torch tensors.
    Clamps prediction to [0.0, 1.0] before metric evaluation.
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
        
    # Clamp prediction to [0.0, 1.0] for metric evaluation
    pred_clamped = np.clip(pred, 0.0, 1.0)
    
    mse = np.mean((pred_clamped - gt) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(1.0 / np.sqrt(mse)))

def compute_ssim(pred, gt):
    """
    Computes Structural Similarity Index (SSIM).
    Expects inputs as numpy arrays or torch tensors.
    Clamps prediction to [0.0, 1.0] before metric evaluation.
    """
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
        
    # Clamp prediction to [0.0, 1.0]
    pred_clamped = np.clip(pred, 0.0, 1.0)
    
    # Squeeze channel dimensions if present (e.g. (1, H, W) -> (H, W))
    if pred_clamped.ndim == 3 and pred_clamped.shape[0] == 1:
        pred_clamped = np.squeeze(pred_clamped, axis=0)
    if gt.ndim == 3 and gt.shape[0] == 1:
        gt = np.squeeze(gt, axis=0)
        
    return float(skimage.metrics.structural_similarity(
        pred_clamped, gt, data_range=1.0
    ))

def compute_lpips(pred, gt, lpips_model, device):
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS).
    Expects inputs as torch tensors with shape (B, C, H, W) or (C, H, W).
    Clamps prediction to [0.0, 1.0], expands to 3 channels (RGB) if grayscale,
    and scales to [-1.0, 1.0] for the LPIPS network.
    """
    # Ensure batched format: (B, C, H, W)
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
        
    # Clamp prediction to [0.0, 1.0] for metric evaluation
    pred_clamped = torch.clamp(pred, 0.0, 1.0)
    
    # LPIPS model expects 3-channel inputs
    if pred_clamped.shape[1] == 1:
        pred_rgb = pred_clamped.repeat(1, 3, 1, 1)
        gt_rgb = gt.repeat(1, 3, 1, 1)
    else:
        pred_rgb = pred_clamped
        gt_rgb = gt
        
    # Scale from [0, 1] to [-1, 1]
    pred_scaled = 2.0 * pred_rgb - 1.0
    gt_scaled = 2.0 * gt_rgb - 1.0
    
    with torch.no_grad():
        val = lpips_model(pred_scaled.to(device), gt_scaled.to(device))
        
    return float(val.mean().item())
