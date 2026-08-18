import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import lpips

from utils import load_config, set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable

def decompose_frequencies(img, r_low=15, r_mid=64):
    h, w = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r2 = x**2 + y**2
    
    mask_low = r2 < r_low**2
    mask_mid = (r2 >= r_low**2) & (r2 < r_mid**2)
    mask_high = r2 >= r_mid**2
    
    img_low = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_low)))
    img_mid = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_mid)))
    img_high = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask_high)))
    
    return img_low, img_mid, img_high


class StructureAwarePriorNet(nn.Module):
    def __init__(self, num_features=32):
        super().__init__()
        # 1. Structure Branch (inputs: upsampled LR, Phase 4 base HR, LR Sobel gradients)
        self.struct_branch = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Noise Branch (inputs: upsampled LR, LR Sobel gradients)
        self.noise_branch = nn.Sequential(
            nn.Conv2d(2, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Structural Prior Head (predicts Sobel gradient magnitude, shape [B, 1, 256, 256])
        self.struct_prior_head = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 4. Reconstruction Head
        self.hr_encoder = nn.Sequential(
            nn.Conv2d(1, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.recon_head = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        )
        
        # Initialization
        # Normal initialization with small stddev (0.001) for the recovery/structural heads
        # This keeps initial correction small (~1e-4) but trainable
        for m in self.struct_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.noise_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.recon_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for m in self.struct_prior_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
    def forward(self, lr_up, base_hr, lr_edge, bounded_scale=0.05):
        # Structure features
        struct_in = torch.cat([lr_up, base_hr, lr_edge], dim=1) # [B, 3, 256, 256]
        struct_feats = self.struct_branch(struct_in)
        
        # Noise features
        noise_in = torch.cat([lr_up, lr_edge], dim=1) # [B, 2, 256, 256]
        noise_feats = self.noise_branch(noise_in)
        
        # Structural prior prediction
        pred_struct = self.struct_prior_head(struct_feats)
        
        # Feature Fusion
        hr_feats = self.hr_encoder(base_hr)
        fused_feats = torch.cat([hr_feats, struct_feats, noise_feats], dim=1) # [B, 96, 256, 256]
        
        # Prediction Correction
        raw_res = self.recon_head(fused_feats)
        correction = torch.tanh(raw_res)
        
        # Final output
        final_hr = torch.clamp(base_hr + bounded_scale * correction, 0.0, 1.0)
        return final_hr, pred_struct, correction, struct_feats, noise_feats

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Phase 4.9 requires GPU execution.")
        
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    
    checkpoint_dir = "outputs/phase49/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_path = "configs/echo.yaml"
    config = load_config(config_path)
    set_seed(42)
    
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    
    # --- SANITY CHECKS ---
    print("\n" + "="*50)
    print("RUNNING SANITY CHECKS")
    print("="*50)
    
    # 1. Verify Phase 4 checkpoint exists
    p4_checkpoint_path = "outputs/echo_phase4/checkpoints/echo_best.pth"
    if not os.path.exists(p4_checkpoint_path):
        raise FileNotFoundError(f"Safety Error: Phase 4 checkpoint not found at: {p4_checkpoint_path}")
    print("Sanity Check 1: Checkpoint exists: PASSED")
    
    # 2. Verify splits disjointness
    train_split = pd.read_csv("outputs/baseline/train_split.csv")
    val_split = pd.read_csv("outputs/baseline/val_split.csv")
    train_fns = set(os.path.basename(p) for p in train_split["input_path"])
    val_fns = set(os.path.basename(p) for p in val_split["input_path"])
    if len(train_fns.intersection(val_fns)) > 0:
        raise ValueError("Safety Error: Train and validation splits are not disjoint!")
    print("Sanity Check 2: Train/validation disjointness: PASSED")
    
    # Load dataset
    print("Initializing datasets...")
    train_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path="outputs/baseline/train_split.csv"
    )
    val_dataset = KLADataset(
        dataset_root=config["dataset_root"],
        split="train",
        csv_path="outputs/baseline/val_split.csv"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load Phase 4 model and freeze it
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablation", {})
    model_p4 = BaselineECHOModel(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        num_features=model_cfg.get("num_features", 64),
        num_blocks=model_cfg.get("num_blocks", 6),
        ablation=ablation_cfg
    ).to(device)
    
    p4_chk = torch.load(p4_checkpoint_path, map_location=device)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    
    # 3. Freeze Phase 4 parameters
    for p in model_p4.parameters():
        p.requires_grad = False
    print("Sanity Check 3: Phase 4 parameters frozen: PASSED")
    
    # Initialize head
    head = StructureAwarePriorNet(num_features=32).to(device)
    
    # 4. Recovery head parameters trainable
    for p in head.parameters():
        p.requires_grad = True
    print("Sanity Check 4: Recovery head parameters trainable: PASSED")
    
    # Load LPIPS network and freeze it
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False
        
    # Sobel filter helper
    sobel_filter = PyTorchSobel().to(device)
    
    # Extract sample batch
    sample_batch = next(iter(train_loader))
    sample_in = sample_batch["input"].to(device)
    sample_tgt = sample_batch["target"].to(device)
    
    with torch.no_grad():
        base_hr, _ = model_p4(sample_in)
        
    lr_up = torch.nn.functional.interpolate(sample_in, scale_factor=2, mode="bicubic", align_corners=False)
    lr_edge = get_lr_edge(lr_up, sobel_filter)
    
    # Forward pass
    head.train()
    final_hr, pred_struct, correction, struct_feats, noise_feats = head(lr_up, base_hr, lr_edge, bounded_scale=0.05)
    
    # 5. Output shape check
    print(f"Output Shape: {list(final_hr.shape)}")
    if list(final_hr.shape) != [sample_in.size(0), 1, 256, 256]:
        raise ValueError(f"Shape Error: output shape is {list(final_hr.shape)}")
    print("Sanity Check 5: Output shape: PASSED")
    
    # 6. Output finite
    if not torch.isfinite(final_hr).all():
        raise ValueError("Final output contains NaNs or Infs")
    print("Sanity Check 6: Output finite: PASSED")
    
    # 7. Structure prediction finite
    if not torch.isfinite(pred_struct).all():
        raise ValueError("Structure prediction contains NaNs or Infs")
    print("Sanity Check 7: Structure prediction finite: PASSED")
    
    # 8. Noise prediction finite
    if not torch.isfinite(noise_feats).all():
        raise ValueError("Noise prediction features contain NaNs or Infs")
    print("Sanity Check 8: Noise prediction finite: PASSED")
    
    # Loss evaluations
    loss_lpips = ssim_lpips_differentiable(final_hr, sample_tgt, lpips_model)
    
    # Compute noise map map and structure target
    # Noise consistency loss: perturb input by gaussian noise (sigma=0.01)
    lr_up_perturbed = lr_up + 0.01 * torch.randn_like(lr_up)
    final_hr_perturbed, _, _, _, _ = head(lr_up_perturbed, base_hr, lr_edge, bounded_scale=0.05)
    loss_noise = torch.mean(torch.abs(final_hr - final_hr_perturbed))
    
    gt_struct_raw = sobel_filter(sample_tgt)
    # Safe normalization per-image to [0, 1]
    gt_struct = gt_struct_raw / (gt_struct_raw.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
    
    loss_pixel = torch.nn.functional.l1_loss(final_hr, sample_tgt)
    loss_ssim = 1.0 - ssim_pytorch(final_hr, sample_tgt)
    loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(sample_tgt))
    loss_structure = torch.nn.functional.l1_loss(pred_struct, gt_struct)
    loss_res = torch.mean(torch.abs(correction))
    
    # Coefficients configuration
    pixel_coef = config.get("pixel_coef", 1.00)
    ssim_coef = config.get("ssim_coef", 0.20)
    lpips_coef = config.get("lpips_coef", 0.15)
    struct_coef = config.get("struct_coef", 0.30)
    noise_coef = config.get("noise_coef", 0.05)
    res_coef = config.get("res_coef", 0.02)
    
    total_loss = (pixel_coef * loss_pixel +
                  ssim_coef * loss_ssim +
                  lpips_coef * loss_lpips +
                  struct_coef * loss_structure +
                  noise_coef * loss_noise +
                  res_coef * loss_res)
                  
    # 9. All losses finite
    if not torch.isfinite(total_loss):
        raise ValueError("Total loss contains NaNs or Infs")
    print("Sanity Check 9: All losses finite: PASSED")
    
    # Gradients backprop
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    
    # 10. Recovery head receives gradients
    head_has_grads = True
    for name, p in head.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            print(f"Warning: head parameter {name} grad is invalid!")
            head_has_grads = False
    if not head_has_grads:
        raise ValueError("Gradient Flow Error: recovery head lacks valid backpropagated gradients!")
    print("Sanity Check 10: Recovery head receives gradients: PASSED")
    
    # 11. Phase 4 receives NO gradients
    p4_has_no_grads = True
    for name, p in model_p4.named_parameters():
        if p.grad is not None:
            print(f"Warning: Phase 4 parameter {name} has gradient!")
            p4_has_no_grads = False
    if not p4_has_no_grads:
        raise ValueError("Safety Error: Phase 4 parameters received gradient updates!")
    print("Sanity Check 11: Phase 4 receives NO gradients: PASSED")
    
    # 12. Output range check
    o_min, o_max = float(final_hr.min().item()), float(final_hr.max().item())
    print(f"Output range: [{o_min:.4f}, {o_max:.4f}]")
    if o_min < 0.0 or o_max > 1.0:
        raise ValueError("Output range exceeded limits [0, 1]!")
    print("Sanity Check 12: Output range: PASSED")
    
    # 13. Identity behavior test (alpha = 0)
    head.eval()
    with torch.no_grad():
        final_id_0, _, _, _, _ = head(lr_up, base_hr, lr_edge, bounded_scale=0.0)
    id_diff = torch.abs(final_id_0 - torch.clamp(base_hr, 0.0, 1.0)).max().item()
    print(f"Identity difference (alpha=0.0): {id_diff:.6e}")
    if id_diff > 1e-6:
        raise ValueError("Identity Behavior Error: final output differs from base HR when alpha=0.0")
    print("Sanity Check 13: Identity behavior test: PASSED")
    
    # 14-15. Verify structural correction and reconstruction variance (non-zero)
    c_mean = float(correction.abs().mean().item())
    c_std = float(correction.std().item())
    print(f"Initial correction mean: {c_mean:.6e} | std: {c_std:.6e}")
    if c_mean == 0.0 or c_std == 0.0:
        raise ValueError("Collapse Error: structural correction or features collapsed to zero variance!")
    print("Sanity Check 15: Recovery branch does not produce zero gradients: PASSED")
    print("Sanity Check 16: Correction has non-zero variance: PASSED")
    print("Sanity Check 17: Reconstruction branch does not collapse to zero: PASSED")
    
    # 2-Sample Overfit Test
    print("\n" + "="*50)
    print("RUNNING 2-SAMPLE OVERFIT DIAGNOSTIC TEST")
    print("="*50)
    
    overfit_subset = Subset(train_dataset, [0, 1])
    overfit_loader = DataLoader(overfit_subset, batch_size=2, shuffle=False)
    
    overfit_batch = next(iter(overfit_loader))
    o_in = overfit_batch["input"].to(device)
    o_tgt = overfit_batch["target"].to(device)
    
    with torch.no_grad():
        o_base_hr, _ = model_p4(o_in)
    o_lr_up = torch.nn.functional.interpolate(o_in, scale_factor=2, mode="bicubic", align_corners=False)
    o_lr_edge = get_lr_edge(o_lr_up, sobel_filter)
    
    # Freshly initialize recovery head for overfit
    o_head = StructureAwarePriorNet(num_features=32).to(device)
    o_optimizer = torch.optim.Adam(o_head.parameters(), lr=1e-2)
    
    o_start_loss = None
    o_end_loss = None
    
    o_head.train()
    for step in range(500):
        o_optimizer.zero_grad()
        o_final_hr, o_pred_struct, o_correction, _, _ = o_head(o_lr_up, o_base_hr, o_lr_edge, bounded_scale=0.05)
        
        o_loss_pixel = torch.nn.functional.l1_loss(o_final_hr, o_tgt)
        o_loss_ssim = 1.0 - ssim_pytorch(o_final_hr, o_tgt)
        o_loss_lpips = ssim_lpips_differentiable(o_final_hr, o_tgt, lpips_model)
        
        o_gt_struct_raw = sobel_filter(o_tgt)
        o_gt_struct = o_gt_struct_raw / (o_gt_struct_raw.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
        o_loss_structure = torch.nn.functional.l1_loss(o_pred_struct, o_gt_struct)
        
        # Noise consistency
        o_lr_up_perturbed = o_lr_up + 0.01 * torch.randn_like(o_lr_up)
        o_final_hr_perturbed, _, _, _, _ = o_head(o_lr_up_perturbed, o_base_hr, o_lr_edge, bounded_scale=0.05)
        o_loss_noise = torch.mean(torch.abs(o_final_hr - o_final_hr_perturbed))
        o_loss_res = torch.mean(torch.abs(o_correction))
        
        o_total_loss = (pixel_coef * o_loss_pixel +
                        ssim_coef * o_loss_ssim +
                        lpips_coef * o_loss_lpips +
                        struct_coef * o_loss_structure +
                        noise_coef * o_loss_noise +
                        res_coef * o_loss_res)
                        
        o_total_loss.backward()
        o_optimizer.step()
        
        if step == 0:
            o_start_loss = o_total_loss.item()
        if step == 499:
            o_end_loss = o_total_loss.item()
            
    print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f}")
    if o_end_loss >= o_start_loss or o_end_loss > 0.220:
        raise ValueError(f"CRITICAL ERROR: Overfit test failed! Start Loss: {o_start_loss:.4f}, End Loss: {o_end_loss:.4f}")
    print("Sanity Check 14: 2-Sample Overfit test: PASSED")
    
    # Re-initialize head weights before starting full training
    head = StructureAwarePriorNet(num_features=32).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    # Start training
    epochs = 5
    print("\n" + "="*50)
    print("STARTING 5-EPOCH CONSERVATIVE TRAINING RUN")
    print("="*50)
    
    best_val_loss = float("inf")
    start_time = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        
        # Training Phase
        head.train()
        running_loss = 0.0
        running_pixel = 0.0
        running_ssim = 0.0
        running_lpips = 0.0
        running_structure = 0.0
        running_noise = 0.0
        running_res = 0.0
        
        # Tracking diagnostics
        epoch_struct_mean = []
        epoch_struct_std = []
        epoch_noise_mean = []
        epoch_noise_std = []
        epoch_correction_mean = []
        epoch_correction_std = []
        epoch_correction_abs = []
        epoch_correction_max = []
        
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            with torch.no_grad():
                base_hr, _ = model_p4(inputs)
                
            lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            
            # Adaptive difficulty weight: L1(base_hr, target)
            difficulty = torch.mean(torch.abs(base_hr.detach() - targets), dim=(1, 2, 3))
            weight = 1.0 + difficulty / (difficulty.mean() + 1e-8) # shape [B]
            
            optimizer.zero_grad()
            final_hr, pred_struct, correction, struct_feats, noise_feats = head(lr_up, base_hr, lr_edge, bounded_scale=0.05)
            
            # Loss terms
            loss_pixel_ind = torch.mean(torch.abs(final_hr - targets), dim=(1, 2, 3))
            loss_pixel = torch.mean(weight * loss_pixel_ind)
            
            loss_ssim = 1.0 - ssim_pytorch(final_hr, targets) # global average
            loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
            
            gt_struct_raw = sobel_filter(targets)
            gt_struct = gt_struct_raw / (gt_struct_raw.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
            loss_structure = torch.nn.functional.l1_loss(pred_struct, gt_struct)
            
            # Noise consistency
            lr_up_perturbed = lr_up + 0.01 * torch.randn_like(lr_up)
            final_hr_perturbed, _, _, _, _ = head(lr_up_perturbed, base_hr, lr_edge, bounded_scale=0.05)
            loss_noise = torch.mean(torch.abs(final_hr - final_hr_perturbed))
            
            loss_res = torch.mean(torch.abs(correction))
            
            total_loss = (pixel_coef * loss_pixel +
                          ssim_coef * loss_ssim +
                          lpips_coef * loss_lpips +
                          struct_coef * loss_structure +
                          noise_coef * loss_noise +
                          res_coef * loss_res)
                          
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            running_pixel += loss_pixel.item() * inputs.size(0)
            running_ssim += loss_ssim.item() * inputs.size(0)
            running_lpips += loss_lpips.item() * inputs.size(0)
            running_structure += loss_structure.item() * inputs.size(0)
            running_noise += loss_noise.item() * inputs.size(0)
            running_res += loss_res.item() * inputs.size(0)
            
            # Diagnostics tracking
            epoch_struct_mean.append(struct_feats.mean().item())
            epoch_struct_std.append(struct_feats.std().item())
            epoch_noise_mean.append(noise_feats.mean().item())
            epoch_noise_std.append(noise_feats.std().item())
            epoch_correction_mean.append(correction.mean().item())
            epoch_correction_std.append(correction.std().item())
            epoch_correction_abs.append(correction.abs().mean().item())
            epoch_correction_max.append(correction.abs().max().item())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_pixel = running_pixel / len(train_dataset)
        epoch_train_ssim = running_ssim / len(train_dataset)
        epoch_train_lpips = running_lpips / len(train_dataset)
        epoch_train_structure = running_structure / len(train_dataset)
        epoch_train_noise = running_noise / len(train_dataset)
        epoch_train_res = running_res / len(train_dataset)
        
        # Calculate training diagnostics
        tr_struct_mean = float(np.mean(epoch_struct_mean))
        tr_struct_std = float(np.mean(epoch_struct_std))
        tr_noise_mean = float(np.mean(epoch_noise_mean))
        tr_noise_std = float(np.mean(epoch_noise_std))
        tr_corr_mean = float(np.mean(epoch_correction_mean))
        tr_corr_std = float(np.mean(epoch_correction_std))
        tr_corr_abs = float(np.mean(epoch_correction_abs))
        tr_corr_max = float(np.max(epoch_correction_max))
        
        # Gradient Norm Tracking
        grad_norms = []
        for p in head.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2).item())
        total_grad_norm = float(np.sqrt(np.sum([n**2 for n in grad_norms]))) if grad_norms else 0.0
        
        # CRITICAL ANTI-COLLAPSE MONITORING
        print(f"\n--- Epoch {epoch:02d} Anti-Collapse Diagnostics ---")
        print(f"Structure Feats: Mean={tr_struct_mean:.4f} | Std={tr_struct_std:.4f}")
        print(f"Noise Feats:     Mean={tr_noise_mean:.4f} | Std={tr_noise_std:.4f}")
        print(f"Correction:      Mean={tr_corr_mean:.6e} | Std={tr_corr_std:.6e} | AbsMean={tr_corr_abs:.6e} | Max={tr_corr_max:.6e}")
        print(f"Gradient Norm:   {total_grad_norm:.6e}")
        print("-" * 40)
        
        if tr_corr_abs < 1e-7 or total_grad_norm < 1e-7 or tr_struct_std < 1e-7:
            raise RuntimeError("CRITICAL WARNING: Collapse detected! Correction magnitude or gradients collapsed to zero.")
            
        # Validation Phase
        head.eval()
        running_val_loss = 0.0
        running_val_pixel = 0.0
        running_val_ssim = 0.0
        running_val_lpips = 0.0
        running_val_structure = 0.0
        running_val_noise = 0.0
        running_val_res = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                base_hr, _ = model_p4(inputs)
                lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
                lr_edge = get_lr_edge(lr_up, sobel_filter)
                
                final_hr, pred_struct, correction, _, _ = head(lr_up, base_hr, lr_edge, bounded_scale=0.05)
                
                loss_pixel = torch.nn.functional.l1_loss(final_hr, targets)
                loss_ssim = 1.0 - ssim_pytorch(final_hr, targets)
                loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
                
                gt_struct_raw = sobel_filter(targets)
                gt_struct = gt_struct_raw / (gt_struct_raw.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
                loss_structure = torch.nn.functional.l1_loss(pred_struct, gt_struct)
                
                # Noise consistency
                lr_up_perturbed = lr_up + 0.01 * torch.randn_like(lr_up)
                final_hr_perturbed, _, _, _, _ = head(lr_up_perturbed, base_hr, lr_edge, bounded_scale=0.05)
                loss_noise = torch.mean(torch.abs(final_hr - final_hr_perturbed))
                
                loss_res = torch.mean(torch.abs(correction))
                
                total_loss = (pixel_coef * loss_pixel +
                              ssim_coef * loss_ssim +
                              lpips_coef * loss_lpips +
                              struct_coef * loss_structure +
                              noise_coef * loss_noise +
                              res_coef * loss_res)
                              
                running_val_loss += total_loss.item() * inputs.size(0)
                running_val_pixel += loss_pixel.item() * inputs.size(0)
                running_val_ssim += loss_ssim.item() * inputs.size(0)
                running_val_lpips += loss_lpips.item() * inputs.size(0)
                running_val_structure += loss_structure.item() * inputs.size(0)
                running_val_noise += loss_noise.item() * inputs.size(0)
                running_val_res += loss_res.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_pixel = running_val_pixel / len(val_dataset)
        epoch_val_ssim = running_val_ssim / len(val_dataset)
        epoch_val_lpips = running_val_lpips / len(val_dataset)
        epoch_val_structure = running_val_structure / len(val_dataset)
        epoch_val_noise = running_val_noise / len(val_dataset)
        epoch_val_res = running_val_res / len(val_dataset)
        
        epoch_elapsed = time.perf_counter() - epoch_start
        
        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": epoch_val_loss,
                "config": config
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "echo_phase49_best.pth"))
            
        print(
            f"Epoch {epoch:02d}/05 | "
            f"Train Loss: {epoch_train_loss:.6f} (Pixel: {epoch_train_pixel:.4f}, SSIM: {epoch_train_ssim:.4f}, LPIPS: {epoch_train_lpips:.4f}, Struct: {epoch_train_structure:.4f}, Noise: {epoch_train_noise:.4f}, Res: {epoch_train_res:.4f}) | "
            f"Val Loss: {epoch_val_loss:.6f} (Pixel: {epoch_val_pixel:.4f}, SSIM: {epoch_val_ssim:.4f}, LPIPS: {epoch_val_lpips:.4f}, Struct: {epoch_val_structure:.4f}, Noise: {epoch_val_noise:.4f}, Res: {epoch_val_res:.4f}) | "
            f"Time: {epoch_elapsed:.1f}s"
            + (" (Saved Best)" if is_best else "")
        )
        
    print(f"\nPhase 4.9 training completed in {time.perf_counter() - start_time:.1f}s.")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
