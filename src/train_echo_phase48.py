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


class NoiseAwareGatedHead(nn.Module):
    def __init__(self, num_features=32):
        super().__init__()
        # 1. Structure Branch (inputs: upsampled LR, Phase 4 base HR, LR Sobel gradients)
        self.struct_branch = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Noise Branch (inputs: upsampled LR, LR Sobel gradients)
        self.noise_branch = nn.Sequential(
            nn.Conv2d(2, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Confidence Head
        self.confidence_head = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        )
        
        # 4. Residual Head
        self.residual_head = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        )
        
        # Initialization
        # Initialize final confidence head bias to logit(0.15) ≈ -1.7346
        nn.init.constant_(self.confidence_head[-1].bias, -1.7346)
        nn.init.normal_(self.confidence_head[-1].weight, std=0.01)
        
        # Initialize final residual head weights to be very small but non-zero
        nn.init.normal_(self.residual_head[-1].weight, std=1e-4)
        nn.init.constant_(self.residual_head[-1].bias, 0.0)
        
    def forward(self, lr_up, base_hr, lr_edge, alpha=0.10):
        # Structure Branch Features
        struct_in = torch.cat([lr_up, base_hr, lr_edge], dim=1) # [B, 3, 256, 256]
        struct_feats = self.struct_branch(struct_in)
        
        # Noise Branch Features
        noise_in = torch.cat([lr_up, lr_edge], dim=1) # [B, 2, 256, 256]
        noise_feats = self.noise_branch(noise_in)
        
        # Combine Features
        combined_feats = torch.cat([struct_feats, noise_feats], dim=1) # [B, 64, 256, 256]
        
        # Confidence map G in [0, 1]
        confidence = torch.sigmoid(self.confidence_head(combined_feats))
        
        # Bounded residual
        raw_res = self.residual_head(combined_feats)
        residual = 0.10 * torch.tanh(raw_res)
        
        # Conservative fusion
        final_hr = base_hr + alpha * confidence * residual
        return final_hr, confidence, residual

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Phase 4.8 requires GPU execution.")
        
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    
    checkpoint_dir = "outputs/echo_phase48/checkpoints"
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
    head = NoiseAwareGatedHead(num_features=32).to(device)
    
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
    final_hr, confidence, residual = head(lr_up, base_hr, lr_edge, alpha=0.10)
    
    # 5. Output shape check
    print(f"Output Shape: {list(final_hr.shape)}")
    if list(final_hr.shape) != [sample_in.size(0), 1, 256, 256]:
        raise ValueError(f"Shape Error: output shape is {list(final_hr.shape)}")
    print("Sanity Check 5: Output shape: PASSED")
    
    # 6. Confidence range check
    g_min, g_max = float(confidence.min().item()), float(confidence.max().item())
    print(f"Confidence range: [{g_min:.4f}, {g_max:.4f}]")
    if g_min < 0.0 or g_max > 1.0:
        raise ValueError("Confidence map out of range [0, 1]!")
    print("Sanity Check 6: Confidence range: PASSED")
    
    # 7. Finite confidence values
    if not torch.isfinite(confidence).all():
        raise ValueError("Confidence contains NaNs or Infs")
    print("Sanity Check 7: Finite confidence values: PASSED")
    
    # 8. Finite residual values
    if not torch.isfinite(residual).all():
        raise ValueError("Residual contains NaNs or Infs")
    print("Sanity Check 8: Finite residual values: PASSED")
    
    # 9. Finite output
    if not torch.isfinite(final_hr).all():
        raise ValueError("Final output contains NaNs or Infs")
    print("Sanity Check 9: Finite output: PASSED")
    
    # Loss evaluations
    loss_lpips = ssim_lpips_differentiable(final_hr, sample_tgt, lpips_model)
    # 10. LPIPS loss finite
    if not torch.isfinite(loss_lpips):
        raise ValueError("LPIPS loss contains NaNs or Infs")
    print("Sanity Check 10: LPIPS loss finite: PASSED")
    
    # Compute noise map and structure weights
    # noise_map = abs(HF(lr_up)) * (1.0 - normalize(Sobel(lr_up)))
    lr_up_flat = lr_up.squeeze(1).cpu().numpy()
    lr_hf_list = []
    for img in lr_up_flat:
        _, _, hf = decompose_frequencies(img)
        lr_hf_list.append(hf)
    lr_hf_map = torch.from_numpy(np.stack(lr_hf_list)).unsqueeze(1).to(device).abs()
    
    noise_map = lr_hf_map * (1.0 - lr_edge)
    
    gt_edge_map = sobel_filter(sample_tgt)
    gt_weight = gt_edge_map / (gt_edge_map.mean(dim=(2, 3), keepdim=True) + 1e-8)
    
    # 11. Total loss finite
    loss_pixel = torch.nn.functional.l1_loss(final_hr, sample_tgt)
    loss_ssim = 1.0 - ssim_pytorch(final_hr, sample_tgt)
    loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(sample_tgt))
    
    gt_residual = sample_tgt - base_hr.detach()
    loss_struct_res = torch.mean(gt_weight * torch.abs(residual - gt_residual))
    loss_res_reg = torch.mean(torch.abs(residual))
    loss_gate_reg = torch.mean(torch.abs(confidence))
    loss_noise_cons = torch.mean(confidence * noise_map)
    
    # Coefficients configuration
    pixel_coef = config.get("pixel_coef", 1.0)
    ssim_coef = config.get("ssim_coef", 0.25)
    edge_coef = config.get("edge_coef", 0.20)
    lpips_coef = config.get("lpips_coef", 0.10)
    struct_res_coef = config.get("struct_res_coef", 0.15)
    res_reg_coef = config.get("res_reg_coef", 0.10)
    gate_reg_coef = config.get("gate_reg_coef", 0.05)
    noise_cons_coef = config.get("noise_cons_coef", 0.05)
    
    total_loss = (pixel_coef * loss_pixel +
                  ssim_coef * loss_ssim +
                  edge_coef * loss_edge +
                  lpips_coef * loss_lpips +
                  struct_res_coef * loss_struct_res +
                  res_reg_coef * loss_res_reg +
                  gate_reg_coef * loss_gate_reg +
                  noise_cons_coef * loss_noise_cons)
                  
    if not torch.isfinite(total_loss):
        raise ValueError("Total loss contains NaNs or Infs")
    print("Sanity Check 11: Total loss finite: PASSED")
    
    # Gradients backprop
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total_loss.backward()
    
    # 12. Recovery head receives gradients
    head_has_grads = True
    for name, p in head.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            print(f"Warning: head parameter {name} grad is invalid!")
            head_has_grads = False
    if not head_has_grads:
        raise ValueError("Gradient Flow Error: recovery head lacks valid backpropagated gradients!")
    print("Sanity Check 12: Recovery head receives gradients: PASSED")
    
    # 13. Phase 4 receives NO gradients
    p4_has_no_grads = True
    for name, p in model_p4.named_parameters():
        if p.grad is not None:
            print(f"Warning: Phase 4 parameter {name} has gradient!")
            p4_has_no_grads = False
    if not p4_has_no_grads:
        raise ValueError("Safety Error: Phase 4 parameters received gradient updates!")
    print("Sanity Check 13: Phase 4 receives NO gradients: PASSED")
    
    # 14. Identity behavior test (alpha = 0)
    head.eval()
    with torch.no_grad():
        final_id_0, _, _ = head(lr_up, base_hr, lr_edge, alpha=0.0)
    id_diff = torch.abs(final_id_0 - base_hr).max().item()
    print(f"Identity difference (alpha=0.0): {id_diff:.6e}")
    if id_diff > 1e-6:
        raise ValueError("Identity Behavior Error: final output differs from base HR when alpha=0.0")
    print("Sanity Check 14: Identity behavior test: PASSED")
    
    # 15. Confidence does NOT collapse to exactly zero at initialization
    head.train()
    with torch.no_grad():
        _, init_conf, init_res = head(lr_up, base_hr, lr_edge, alpha=0.10)
    c_mean = float(init_conf.mean().item())
    print(f"Initial confidence mean: {c_mean:.4f}")
    if c_mean < 0.05 or c_mean > 0.30:
        raise ValueError(f"Confidence Initialization Error: mean confidence is {c_mean:.4f}, expected around 0.1–0.2")
    print("Sanity Check 15: Confidence does not collapse to zero: PASSED")
    
    # 16. Residual magnitude is small but non-zero
    r_mean = float(init_res.abs().mean().item())
    print(f"Initial residual absolute mean: {r_mean:.6e}")
    if r_mean == 0.0 or r_mean > 1e-3:
        raise ValueError(f"Residual Initialization Error: mean residual is {r_mean:.6e}, expected small but non-zero")
    print("Sanity Check 16: Residual magnitude is small but non-zero: PASSED")
    
    # 17. Final output differs only conservatively from Phase 4 at initialization
    diff_init = torch.abs(final_hr - base_hr).max().item()
    print(f"Max difference from base at init: {diff_init:.6e}")
    if diff_init == 0.0 or diff_init > 1e-4:
        raise ValueError(f"Output Initialization Error: max difference is {diff_init:.6e}, expected conservative change")
    print("Sanity Check 17: Final output differs conservatively at init: PASSED")
    
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
    
    # Compute noise map and structure weights for overfit
    o_lr_up_flat = o_lr_up.squeeze(1).cpu().numpy()
    o_lr_hf_list = []
    for img in o_lr_up_flat:
        _, _, hf = decompose_frequencies(img)
        o_lr_hf_list.append(hf)
    o_lr_hf_map = torch.from_numpy(np.stack(o_lr_hf_list)).unsqueeze(1).to(device).abs()
    o_noise_map = o_lr_hf_map * (1.0 - o_lr_edge)
    
    o_gt_edge_map = sobel_filter(o_tgt)
    o_gt_weight = o_gt_edge_map / (o_gt_edge_map.mean(dim=(2, 3), keepdim=True) + 1e-8)
    
    # Freshly initialize recovery head for overfit
    o_head = NoiseAwareGatedHead(num_features=32).to(device)
    o_optimizer = torch.optim.Adam(o_head.parameters(), lr=1e-2)
    
    o_start_loss = None
    o_end_loss = None
    
    o_head.train()
    for step in range(500):
        o_optimizer.zero_grad()
        o_final_hr, o_confidence, o_residual = o_head(o_lr_up, o_base_hr, o_lr_edge, alpha=0.10)
        
        o_loss_pixel = torch.nn.functional.l1_loss(o_final_hr, o_tgt)
        o_loss_ssim = 1.0 - ssim_pytorch(o_final_hr, o_tgt)
        o_loss_edge = torch.nn.functional.l1_loss(sobel_filter(o_final_hr), sobel_filter(o_tgt))
        o_loss_lpips = ssim_lpips_differentiable(o_final_hr, o_tgt, lpips_model)
        
        o_gt_residual = o_tgt - o_base_hr.detach()
        o_loss_struct_res = torch.mean(o_gt_weight * torch.abs(o_residual - o_gt_residual))
        o_loss_res_reg = torch.mean(torch.abs(o_residual))
        o_loss_gate_reg = torch.mean(torch.abs(o_confidence))
        o_loss_noise_cons = torch.mean(o_confidence * o_noise_map)
        
        o_total_loss = (pixel_coef * o_loss_pixel +
                        ssim_coef * o_loss_ssim +
                        edge_coef * o_loss_edge +
                        lpips_coef * o_loss_lpips +
                        struct_res_coef * o_loss_struct_res +
                        res_reg_coef * o_loss_res_reg +
                        gate_reg_coef * o_loss_gate_reg +
                        noise_cons_coef * o_loss_noise_cons)
                        
        o_total_loss.backward()
        o_optimizer.step()
        
        if step == 0:
            o_start_loss = o_total_loss.item()
        if step == 499:
            o_end_loss = o_total_loss.item()
            
    print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f}")
    if o_end_loss >= o_start_loss or o_end_loss > 0.220:
        raise ValueError(f"CRITICAL ERROR: Overfit test failed! Start Loss: {o_start_loss:.4f}, End Loss: {o_end_loss:.4f}")
    print("2-Sample Overfit test: PASSED")
    
    # Re-initialize head weights before starting full training
    head = NoiseAwareGatedHead(num_features=32).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    # Start training
    epochs = 10
    print("\n" + "="*50)
    print("STARTING 10-EPOCH CONSERVATIVE TRAINING RUN")
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
        running_edge = 0.0
        running_lpips = 0.0
        running_struct_res = 0.0
        running_res_reg = 0.0
        running_gate_reg = 0.0
        running_noise_cons = 0.0
        
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            with torch.no_grad():
                base_hr, _ = model_p4(inputs)
                
            lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            
            # Compute batch noise map and structure weights
            inputs_flat = lr_up.squeeze(1).cpu().numpy()
            lr_hf_list = []
            for img in inputs_flat:
                _, _, hf = decompose_frequencies(img)
                lr_hf_list.append(hf)
            lr_hf_map = torch.from_numpy(np.stack(lr_hf_list)).unsqueeze(1).to(device).abs()
            noise_map = lr_hf_map * (1.0 - lr_edge)
            
            gt_edge_map = sobel_filter(targets)
            gt_weight = gt_edge_map / (gt_edge_map.mean(dim=(2, 3), keepdim=True) + 1e-8)
            
            optimizer.zero_grad()
            final_hr, confidence, residual = head(lr_up, base_hr, lr_edge, alpha=0.10)
            
            loss_pixel = torch.nn.functional.l1_loss(final_hr, targets)
            loss_ssim = 1.0 - ssim_pytorch(final_hr, targets)
            loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(targets))
            loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
            
            gt_residual = targets - base_hr.detach()
            loss_struct_res = torch.mean(gt_weight * torch.abs(residual - gt_residual))
            loss_res_reg = torch.mean(torch.abs(residual))
            loss_gate_reg = torch.mean(torch.abs(confidence))
            loss_noise_cons = torch.mean(confidence * noise_map)
            
            total_loss = (pixel_coef * loss_pixel +
                          ssim_coef * loss_ssim +
                          edge_coef * loss_edge +
                          lpips_coef * loss_lpips +
                          struct_res_coef * loss_struct_res +
                          res_reg_coef * loss_res_reg +
                          gate_reg_coef * loss_gate_reg +
                          noise_cons_coef * loss_noise_cons)
                          
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            running_pixel += loss_pixel.item() * inputs.size(0)
            running_ssim += loss_ssim.item() * inputs.size(0)
            running_edge += loss_edge.item() * inputs.size(0)
            running_lpips += loss_lpips.item() * inputs.size(0)
            running_struct_res += loss_struct_res.item() * inputs.size(0)
            running_res_reg += loss_res_reg.item() * inputs.size(0)
            running_gate_reg += loss_gate_reg.item() * inputs.size(0)
            running_noise_cons += loss_noise_cons.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_pixel = running_pixel / len(train_dataset)
        epoch_train_ssim = running_ssim / len(train_dataset)
        epoch_train_edge = running_edge / len(train_dataset)
        epoch_train_lpips = running_lpips / len(train_dataset)
        epoch_train_struct_res = running_struct_res / len(train_dataset)
        epoch_train_res_reg = running_res_reg / len(train_dataset)
        epoch_train_gate_reg = running_gate_reg / len(train_dataset)
        epoch_train_noise_cons = running_noise_cons / len(train_dataset)
        
        # Validation Phase
        head.eval()
        running_val_loss = 0.0
        running_val_pixel = 0.0
        running_val_ssim = 0.0
        running_val_edge = 0.0
        running_val_lpips = 0.0
        running_val_struct_res = 0.0
        running_val_res_reg = 0.0
        running_val_gate_reg = 0.0
        running_val_noise_cons = 0.0
        
        val_gate_means = []
        val_gate_maxes = []
        val_res_mags = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                base_hr, _ = model_p4(inputs)
                lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
                lr_edge = get_lr_edge(lr_up, sobel_filter)
                
                # Compute batch noise map and structure weights
                inputs_flat = lr_up.squeeze(1).cpu().numpy()
                lr_hf_list = []
                for img in inputs_flat:
                    _, _, hf = decompose_frequencies(img)
                    lr_hf_list.append(hf)
                lr_hf_map = torch.from_numpy(np.stack(lr_hf_list)).unsqueeze(1).to(device).abs()
                noise_map = lr_hf_map * (1.0 - lr_edge)
                
                gt_edge_map = sobel_filter(targets)
                gt_weight = gt_edge_map / (gt_edge_map.mean(dim=(2, 3), keepdim=True) + 1e-8)
                
                final_hr, confidence, residual = head(lr_up, base_hr, lr_edge, alpha=0.10)
                
                loss_pixel = torch.nn.functional.l1_loss(final_hr, targets)
                loss_ssim = 1.0 - ssim_pytorch(final_hr, targets)
                loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(targets))
                loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
                
                gt_residual = targets - base_hr.detach()
                loss_struct_res = torch.mean(gt_weight * torch.abs(residual - gt_residual))
                loss_res_reg = torch.mean(torch.abs(residual))
                loss_gate_reg = torch.mean(torch.abs(confidence))
                loss_noise_cons = torch.mean(confidence * noise_map)
                
                total_loss = (pixel_coef * loss_pixel +
                              ssim_coef * loss_ssim +
                              edge_coef * loss_edge +
                              lpips_coef * loss_lpips +
                              struct_res_coef * loss_struct_res +
                              res_reg_coef * loss_res_reg +
                              gate_reg_coef * loss_gate_reg +
                              noise_cons_coef * loss_noise_cons)
                              
                running_val_loss += total_loss.item() * inputs.size(0)
                running_val_pixel += loss_pixel.item() * inputs.size(0)
                running_val_ssim += loss_ssim.item() * inputs.size(0)
                running_val_edge += loss_edge.item() * inputs.size(0)
                running_val_lpips += loss_lpips.item() * inputs.size(0)
                running_val_struct_res += loss_struct_res.item() * inputs.size(0)
                running_val_res_reg += loss_res_reg.item() * inputs.size(0)
                running_val_gate_reg += loss_gate_reg.item() * inputs.size(0)
                running_val_noise_cons += loss_noise_cons.item() * inputs.size(0)
                
                val_gate_means.append(confidence.mean().item())
                val_gate_maxes.append(confidence.max().item())
                val_res_mags.append(residual.abs().mean().item())
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_pixel = running_val_pixel / len(val_dataset)
        epoch_val_ssim = running_val_ssim / len(val_dataset)
        epoch_val_edge = running_val_edge / len(val_dataset)
        epoch_val_lpips = running_val_lpips / len(val_dataset)
        epoch_val_struct_res = running_val_struct_res / len(val_dataset)
        epoch_val_res_reg = running_val_res_reg / len(val_dataset)
        epoch_val_gate_reg = running_val_gate_reg / len(val_dataset)
        epoch_val_noise_cons = running_val_noise_cons / len(val_dataset)
        
        epoch_gate_mean = float(np.mean(val_gate_means))
        epoch_gate_max = float(np.max(val_gate_maxes))
        epoch_res_mag = float(np.mean(val_res_mags))
        
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
            torch.save(checkpoint, os.path.join(checkpoint_dir, "echo_phase48_best.pth"))
            
        print(
            f"Epoch {epoch:02d}/10 | "
            f"Train Loss: {epoch_train_loss:.6f} (Pixel: {epoch_train_pixel:.4f}, SSIM: {epoch_train_ssim:.4f}, Edge: {epoch_train_edge:.4f}, LPIPS: {epoch_train_lpips:.4f}, ResLoss: {epoch_train_struct_res:.4f}, GateLoss: {epoch_train_gate_reg:.4f}, NoiseLoss: {epoch_train_noise_cons:.4f}) | "
            f"Val Loss: {epoch_val_loss:.6f} (Pixel: {epoch_val_pixel:.4f}, SSIM: {epoch_val_ssim:.4f}, Edge: {epoch_val_edge:.4f}, LPIPS: {epoch_val_lpips:.4f}, ResLoss: {epoch_val_struct_res:.4f}, GateLoss: {epoch_val_gate_reg:.4f}, NoiseLoss: {epoch_val_noise_cons:.4f}) | "
            f"GateMean: {epoch_gate_mean:.4f} | GateMax: {epoch_gate_max:.4f} | ResMag: {epoch_res_mag:.6f} | "
            f"Time: {epoch_elapsed:.1f}s"
            + (" (Saved Best)" if is_best else "")
        )
        
    # Save last checkpoint
    checkpoint_last = {
        "epoch": epochs,
        "model_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": epoch_val_loss,
        "config": config
    }
    torch.save(checkpoint_last, os.path.join(checkpoint_dir, "echo_phase48_last.pth"))
    print(f"\nPhase 4.8 diagnostic training completed in {time.perf_counter() - start_time:.1f}s.")

if __name__ == "__main__":
    main()
