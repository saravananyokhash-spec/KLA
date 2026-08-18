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

class EvidenceGatedPriorNet(nn.Module):
    def __init__(self, num_features=32):
        super().__init__()
        # 1. Evidence Encoder (lightweight encoder)
        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Prior Encoder (trainable CNN recovery module)
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1)
        )
        
        # 3. Confidence / Evidence Gate Network
        self.gate_network = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize prior_encoder output weight and bias to exactly zero
        # This guarantees safe identity behaviour at initialization
        nn.init.constant_(self.prior_encoder[-1].weight, 0.0)
        nn.init.constant_(self.prior_encoder[-1].bias, 0.0)
        
    def forward(self, lr_up, base_hr, lr_edge, alpha=0.10):
        # Concatenate inputs: LR upsampled, Phase 4 base HR, LR Sobel gradients
        x = torch.cat([lr_up, base_hr, lr_edge], dim=1) # [B, 3, 256, 256]
        
        # Evidence features extraction
        ev_feats = self.evidence_encoder(x)
        
        # Confidence Gate G
        gate = self.gate_network(ev_feats)
        
        # Learned Prior residual R_prior with tanh constraint
        raw_prior = self.prior_encoder(x)
        r_prior = 0.10 * torch.tanh(raw_prior)
        
        # Conservative fusion
        final_hr = base_hr + alpha * gate * r_prior
        return final_hr, gate, r_prior

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is not available! Phase 4.7 requires GPU execution.")
        
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    
    checkpoint_dir = "outputs/echo_phase47/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config_path = "configs/echo.yaml"
    config = load_config(config_path)
    set_seed(42)
    
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    
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
    print("Sanity Check 3: Phase 4 frozen: PASSED")
    
    # Initialize Evidence Gated Prior Net
    head = EvidenceGatedPriorNet(num_features=32).to(device)
    
    # 4. Recovery head parameters trainable
    for p in head.parameters():
        p.requires_grad = True
    print("Sanity Check 4: Recovery head trainable: PASSED")
    
    # Load LPIPS network and freeze it
    print("Loading LPIPS network...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False
        
    # Sobel filter helper
    sobel_filter = PyTorchSobel().to(device)
    
    # Sanity checks
    print("\nRunning model sanity checks...")
    sample_batch = next(iter(train_loader))
    sample_in = sample_batch["input"].to(device)
    sample_tgt = sample_batch["target"].to(device)
    
    with torch.no_grad():
        base_hr, _ = model_p4(sample_in)
        
    lr_up = torch.nn.functional.interpolate(sample_in, scale_factor=2, mode="bicubic", align_corners=False)
    lr_edge = get_lr_edge(lr_up, sobel_filter)
    
    # Forward pass
    head.train()
    final_hr, gate, r_prior = head(lr_up, base_hr, lr_edge, alpha=0.10)
    
    # 5. Output shape check
    print(f"Output Shape: {list(final_hr.shape)}")
    if list(final_hr.shape) != [sample_in.size(0), 1, 256, 256]:
        raise ValueError(f"Shape Error: final HR shape is {list(final_hr.shape)}")
    print("Sanity Check 5: Output shape verification: PASSED")
    
    # 6. Gate range check
    g_min, g_max = float(gate.min().item()), float(gate.max().item())
    print(f"Gate Range: [{g_min:.4f}, {g_max:.4f}]")
    if g_min < 0.0 or g_max > 1.0:
        raise ValueError("Gate value error: gate exceeds range [0, 1]")
    print("Sanity Check 6: Gate range check: PASSED")
    
    # 7-8. Finite outputs check
    if not torch.isfinite(r_prior).all() or not torch.isfinite(final_hr).all():
        raise ValueError("Finite Error: predicted prior or final HR contains NaNs or Infs")
    print("Sanity Checks 7-8: Finite prior & output checks: PASSED")
    
    # 9-10. Differentiable loss and gradient checks
    loss_lpips = ssim_lpips_differentiable(final_hr, sample_tgt, lpips_model)
    if not torch.isfinite(loss_lpips):
        raise ValueError("LPIPS Loss Error: LPIPS loss contains NaNs or Infs")
    print("Sanity Check 9: LPIPS loss is finite: PASSED")
    
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    loss_pixel = torch.nn.functional.l1_loss(final_hr, sample_tgt)
    loss_ssim = 1.0 - ssim_pytorch(final_hr, sample_tgt)
    loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(sample_tgt))
    loss_res = torch.mean(torch.abs(r_prior))
    loss_gate = torch.mean(torch.abs(gate))
    
    # Loss formulation weights
    lambda_res = config.get("lambda_res", 0.01)
    lambda_gate = config.get("lambda_gate", 0.01)
    
    total_loss = (1.0 * loss_pixel +
                  0.2 * loss_ssim +
                  0.1 * loss_edge +
                  0.05 * loss_lpips +
                  lambda_res * loss_res +
                  lambda_gate * loss_gate)
                  
    if not torch.isfinite(total_loss):
        raise ValueError("Total Loss Error: total loss contains NaNs or Infs")
    print("Sanity Check 10: Total loss is finite: PASSED")
    
    optimizer.zero_grad()
    total_loss.backward()
    
    head_has_grads = True
    for name, p in head.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            print(f"Warning: head parameter {name} grad is invalid!")
            head_has_grads = False
    if not head_has_grads:
        raise ValueError("Gradient Flow Error: recovery head lacks valid backpropagated gradients!")
    print("Sanity Check 11: Gradient exists in recovery head: PASSED")
    
    p4_has_no_grads = True
    for name, p in model_p4.named_parameters():
        if p.grad is not None:
            print(f"Warning: Phase 4 parameter {name} has gradient!")
            p4_has_no_grads = False
    if not p4_has_no_grads:
        raise ValueError("Safety Error: Phase 4 parameters received gradient updates!")
    print("Sanity Check 12: No gradient exists in Phase 4: PASSED")
    
    # 13. Identity Behaviour Test
    print("\nRunning Identity behavior test...")
    head.eval()
    with torch.no_grad():
        final_id_0, _, _ = head(lr_up, base_hr, lr_edge, alpha=0.0)
    id_diff = torch.abs(final_id_0 - base_hr).max().item()
    print(f"Identity difference: {id_diff:.6e}")
    if id_diff > 1e-6:
        raise ValueError("Identity Behavior Error: final output differs from base HR when alpha=0.0")
    print("Sanity Check 13: Identity behaviour test: PASSED")
    
    # 2-Sample Overfit Test
    print("\nRunning 2-Sample Overfit test with LPIPS loss...")
    overfit_subset = Subset(train_dataset, [0, 1])
    overfit_loader = DataLoader(overfit_subset, batch_size=2, shuffle=False)
    
    overfit_batch = next(iter(overfit_loader))
    o_in = overfit_batch["input"].to(device)
    o_tgt = overfit_batch["target"].to(device)
    
    with torch.no_grad():
        o_base_hr, _ = model_p4(o_in)
    o_lr_up = torch.nn.functional.interpolate(o_in, scale_factor=2, mode="bicubic", align_corners=False)
    o_lr_edge = get_lr_edge(o_lr_up, sobel_filter)
    
    head.train()
    o_start_loss = None
    o_end_loss = None
    
    o_optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    
    for step in range(500):
        o_optimizer.zero_grad()
        o_final_hr, o_gate, o_r_prior = head(o_lr_up, o_base_hr, o_lr_edge, alpha=0.10)
        
        o_loss_pixel = torch.nn.functional.l1_loss(o_final_hr, o_tgt)
        o_loss_ssim = 1.0 - ssim_pytorch(o_final_hr, o_tgt)
        o_loss_edge = torch.nn.functional.l1_loss(sobel_filter(o_final_hr), sobel_filter(o_tgt))
        o_loss_lpips = ssim_lpips_differentiable(o_final_hr, o_tgt, lpips_model)
        o_loss_res = torch.mean(torch.abs(o_r_prior))
        o_loss_gate = torch.mean(torch.abs(o_gate))
        
        o_total_loss = (1.0 * o_loss_pixel +
                        0.2 * o_loss_ssim +
                        0.1 * o_loss_edge +
                        0.05 * o_loss_lpips +
                        lambda_res * o_loss_res +
                        lambda_gate * o_loss_gate)
                        
        o_total_loss.backward()
        o_optimizer.step()
        
        if step == 0:
            o_start_loss = o_total_loss.item()
        if step == 499:
            o_end_loss = o_total_loss.item()
            
    print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f}")
    if o_end_loss >= o_start_loss or o_end_loss > 0.155:
        raise ValueError(f"CRITICAL ERROR: Overfit test failed! Start Loss: {o_start_loss:.4f}, End Loss: {o_end_loss:.4f}")
    print("Sanity Check 14: 2-sample overfit diagnostic test: PASSED")
    
    # Re-initialize head weights before starting full training
    head = EvidenceGatedPriorNet(num_features=32).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    # Start training
    epochs = 5
    print("\n" + "="*50)
    print("STARTING 5-EPOCH EVIDENCE-GATED PRIOR TRAINING RUN")
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
        running_res = 0.0
        running_gate = 0.0
        
        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            
            with torch.no_grad():
                base_hr, _ = model_p4(inputs)
                
            lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            
            optimizer.zero_grad()
            final_hr, gate, r_prior = head(lr_up, base_hr, lr_edge, alpha=0.10)
            
            loss_pixel = torch.nn.functional.l1_loss(final_hr, targets)
            loss_ssim = 1.0 - ssim_pytorch(final_hr, targets)
            loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(targets))
            loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
            loss_res = torch.mean(torch.abs(r_prior))
            loss_gate = torch.mean(torch.abs(gate))
            
            total_loss = (1.0 * loss_pixel +
                          0.2 * loss_ssim +
                          0.1 * loss_edge +
                          0.05 * loss_lpips +
                          lambda_res * loss_res +
                          lambda_gate * loss_gate)
                          
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            running_pixel += loss_pixel.item() * inputs.size(0)
            running_ssim += loss_ssim.item() * inputs.size(0)
            running_edge += loss_edge.item() * inputs.size(0)
            running_lpips += loss_lpips.item() * inputs.size(0)
            running_res += loss_res.item() * inputs.size(0)
            running_gate += loss_gate.item() * inputs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_pixel = running_pixel / len(train_dataset)
        epoch_train_ssim = running_ssim / len(train_dataset)
        epoch_train_edge = running_edge / len(train_dataset)
        epoch_train_lpips = running_lpips / len(train_dataset)
        epoch_train_res = running_res / len(train_dataset)
        epoch_train_gate = running_gate / len(train_dataset)
        
        # Validation Phase
        head.eval()
        running_val_loss = 0.0
        running_val_pixel = 0.0
        running_val_ssim = 0.0
        running_val_edge = 0.0
        running_val_lpips = 0.0
        running_val_res = 0.0
        running_val_gate = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                
                base_hr, _ = model_p4(inputs)
                lr_up = torch.nn.functional.interpolate(inputs, scale_factor=2, mode="bicubic", align_corners=False)
                lr_edge = get_lr_edge(lr_up, sobel_filter)
                
                final_hr, gate, r_prior = head(lr_up, base_hr, lr_edge, alpha=0.10)
                
                loss_pixel = torch.nn.functional.l1_loss(final_hr, targets)
                loss_ssim = 1.0 - ssim_pytorch(final_hr, targets)
                loss_edge = torch.nn.functional.l1_loss(sobel_filter(final_hr), sobel_filter(targets))
                loss_lpips = ssim_lpips_differentiable(final_hr, targets, lpips_model)
                loss_res = torch.mean(torch.abs(r_prior))
                loss_gate = torch.mean(torch.abs(gate))
                
                total_loss = (1.0 * loss_pixel +
                              0.2 * loss_ssim +
                              0.1 * loss_edge +
                              0.05 * loss_lpips +
                              lambda_res * loss_res +
                              lambda_gate * loss_gate)
                              
                running_val_loss += total_loss.item() * inputs.size(0)
                running_val_pixel += loss_pixel.item() * inputs.size(0)
                running_val_ssim += loss_ssim.item() * inputs.size(0)
                running_val_edge += loss_edge.item() * inputs.size(0)
                running_val_lpips += loss_lpips.item() * inputs.size(0)
                running_val_res += loss_res.item() * inputs.size(0)
                running_val_gate += loss_gate.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_pixel = running_val_pixel / len(val_dataset)
        epoch_val_ssim = running_val_ssim / len(val_dataset)
        epoch_val_edge = running_val_edge / len(val_dataset)
        epoch_val_lpips = running_val_lpips / len(val_dataset)
        epoch_val_res = running_val_res / len(val_dataset)
        epoch_val_gate = running_val_gate / len(val_dataset)
        
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
            torch.save(checkpoint, os.path.join(checkpoint_dir, "echo_phase47_best.pth"))
            
        print(
            f"Epoch {epoch:02d}/05 | "
            f"Train Loss: {epoch_train_loss:.6f} (Pixel: {epoch_train_pixel:.4f}, SSIM: {epoch_train_ssim:.4f}, Edge: {epoch_train_edge:.4f}, LPIPS: {epoch_train_lpips:.4f}, Res: {epoch_train_res:.4f}, Gate: {epoch_train_gate:.4f}) | "
            f"Val Loss: {epoch_val_loss:.6f} (Pixel: {epoch_val_pixel:.4f}, SSIM: {epoch_val_ssim:.4f}, Edge: {epoch_val_edge:.4f}, LPIPS: {epoch_val_lpips:.4f}, Res: {epoch_val_res:.4f}, Gate: {epoch_val_gate:.4f}) | "
            f"Time: {epoch_elapsed:.1f}s"
            + (" (Saved Best)" if is_best else "")
        )
        
    print(f"\nPhase 4.7 training completed in {time.perf_counter() - start_time:.1f}s.")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
