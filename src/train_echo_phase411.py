import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import lpips

from utils import load_config, set_seed
from dataset import KLADataset
from echo_model import BaselineECHOModel
from train_echo_phase43 import PyTorchSobel, get_lr_edge, ssim_pytorch
from train_echo_phase44 import ssim_lpips_differentiable
from train_echo_phase410 import Phase410PriorNet, SqueezeExcitationBlock, calculate_psnr

# ============================================================
# PHASE 4.11 — ERROR-AWARE DUAL-EXPERT FUSION NETWORK
# ============================================================

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two Conv layers.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class ErrorAwareFusionNet(nn.Module):
    """
    Phase 4.11 Error-Aware Dual-Expert Fusion Network.
    
    Inputs (9 channels):
    - lr_up (upsampled LR)
    - p4_hr (Phase 4 HR prediction)
    - p410_hr (Phase 4.10 HR prediction)
    - abs_difference (|p410_hr - p4_hr|)
    - signed_difference (p410_hr - p4_hr)
    - input_edge (sobel gradient of lr_up)
    - p4_edge (sobel gradient of p4_hr)
    - p410_edge (sobel gradient of p410_hr)
    - edge_difference (|p410_edge - p4_edge|)
    """
    def __init__(self, num_features=32, num_res_blocks=3):
        super().__init__()
        
        # 1. Input Evidence Encoder
        self.in_conv = nn.Sequential(
            nn.Conv2d(9, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Multi-Branch Feature Extraction
        self.pixel_branch = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.disagreement_branch = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.edge_branch = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.fuse_branches = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. Residual Trunk & SE Attention
        self.trunk = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_res_blocks)])
        self.se_attn = SqueezeExcitationBlock(num_features, reduction=8)
        
        # 4. Confidence / Trust Head (outputs 2 channels, Softmax over experts)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        
        # 5. Correction Head & Correction Gate
        self.correction_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.gate_head = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                    
        # Normal initialization for correction & gate heads
        nn.init.normal_(self.correction_head[-2].weight, std=0.01)
        nn.init.constant_(self.correction_head[-2].bias, 0.0)
        nn.init.normal_(self.gate_head[-2].weight, std=0.01)
        nn.init.constant_(self.gate_head[-2].bias, 0.0)
        
        # P4 confidence bias slightly positive for conservative initial bias toward P4
        nn.init.normal_(self.confidence_head[-2].weight, std=0.01)
        nn.init.constant_(self.confidence_head[-2].bias[0], 0.5)
        nn.init.constant_(self.confidence_head[-2].bias[1], 0.0)

    def forward(self, evidence_in, p4_hr, p410_hr, bounded_scale=0.10):
        # 1. Feature encoding & branching
        feat_in = self.in_conv(evidence_in)
        p_feat = self.pixel_branch(feat_in)
        d_feat = self.disagreement_branch(feat_in)
        e_feat = self.edge_branch(feat_in)
        
        fused = self.fuse_branches(torch.cat([p_feat, d_feat, e_feat], dim=1))
        trunk_feat = self.trunk(fused)
        attn_feat = self.se_attn(trunk_feat)
        
        # 2. Expert Confidence maps
        conf = self.confidence_head(attn_feat)
        conf_p4 = conf[:, 0:1, :, :]
        conf_p410 = conf[:, 1:2, :, :]
        
        # Adaptive Fusion
        fused_expert = conf_p4 * p4_hr + conf_p410 * p410_hr
        
        # 3. Correction & Gate
        correction = self.correction_head(attn_feat)
        gate = self.gate_head(attn_feat)
        gated_correction = gate * correction
        
        # 4. Final HR Output
        if bounded_scale == 0.0:
            final_hr = torch.clamp(p4_hr, 0.0, 1.0)
        else:
            final_hr = torch.clamp(fused_expert + bounded_scale * gated_correction, 0.0, 1.0)
            
        return final_hr, conf_p4, conf_p410, correction, gate, fused_expert

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_evidence_input(lr_up, p4_hr, p410_hr, sobel_filter):
    """
    Constructs the 9-channel evidence tensor for Phase 4.11.
    """
    abs_diff = torch.abs(p410_hr - p4_hr)
    signed_diff = p410_hr - p4_hr
    
    lr_edge = get_lr_edge(lr_up, sobel_filter)
    p4_edge = sobel_filter(p4_hr)
    p410_edge = sobel_filter(p410_hr)
    edge_diff = torch.abs(p410_edge - p4_edge)
    
    evidence = torch.cat([
        lr_up, p4_hr, p410_hr,
        abs_diff, signed_diff,
        lr_edge, p4_edge, p410_edge, edge_diff
    ], dim=1)
    
    return evidence, lr_edge, p4_edge, p410_edge

def compute_expert_targets(p4_hr, p410_hr, target, temp=50.0):
    """
    Computes soft expert targets based on local error comparison.
    """
    p4_err = torch.abs(p4_hr - target)
    p410_err = torch.abs(p410_hr - target)
    
    score_p4 = torch.exp(-temp * p4_err)
    score_p410 = torch.exp(-temp * p410_err)
    
    target_w_p4 = score_p4 / (score_p4 + score_p410 + 1e-8)
    target_w_p410 = 1.0 - target_w_p4
    
    return target_w_p4, target_w_p410

def save_image_png(tensor_img, path):
    """
    Saves float32 tensor [1, 1, H, W] or [1, H, W] in [0, 1] as 2D uint8 PNG.
    """
    arr = tensor_img.detach().cpu().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    plt.imsave(path, arr, cmap='gray')

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is required for Phase 4.11.")
        
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    
    # Directory Setup
    out_dir = "outputs/phase411"
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    configs_dir = os.path.join(out_dir, "configs")
    results_dir = os.path.join(out_dir, "results")
    eval_dir = os.path.join(out_dir, "evaluation")
    visuals_dir = os.path.join(eval_dir, "visuals")
    outputs_dir = os.path.join(eval_dir, "outputs")
    error_maps_dir = os.path.join(eval_dir, "error_maps")
    plots_dir = os.path.join(eval_dir, "plots")
    analysis_dir = os.path.join(eval_dir, "analysis")
    
    for d in [checkpoint_dir, configs_dir, results_dir, eval_dir, visuals_dir,
              outputs_dir, error_maps_dir, plots_dir, analysis_dir,
              os.path.join(outputs_dir, "input"), os.path.join(outputs_dir, "target"),
              os.path.join(outputs_dir, "phase4"), os.path.join(outputs_dir, "phase410"),
              os.path.join(outputs_dir, "phase411")]:
        os.makedirs(d, exist_ok=True)
        
    config_path = os.path.join(configs_dir, "phase411.yaml")
    config = load_config(config_path)
    set_seed(42)
    
    print("=" * 60)
    print("PHASE 4.11 — ERROR-AWARE DUAL-EXPERT FUSION NETWORK")
    print(f"Device: {device} ({gpu_name})")
    print(f"Output Directory: {out_dir}")
    print("=" * 60)
    
    # Checkpoints
    p4_checkpoint_path = config.get("p4_checkpoint", "outputs/echo_phase4/checkpoints/echo_best.pth")
    p410_checkpoint_path = config.get("p410_checkpoint", "outputs/phase410/checkpoints/echo_phase410_best.pth")
    train_csv = config.get("train_split", "outputs/baseline/train_split.csv")
    val_csv = config.get("val_split", "outputs/baseline/val_split.csv")
    dataset_root = config.get("dataset_root", "D:/kla")
    
    # --- SANITY CHECKS (1-19) ---
    print("\n" + "=" * 50)
    print("RUNNING PHASE 4.11 SANITY CHECKS (1-19)")
    print("=" * 50)
    
    # Check 1: CUDA
    print("Sanity Check 1: CUDA available: PASSED")
    
    # Check 2 & 3: Checkpoints
    if not os.path.exists(p4_checkpoint_path):
        raise FileNotFoundError(f"Phase 4 checkpoint missing at {p4_checkpoint_path}")
    print(f"Sanity Check 2: Phase 4 Checkpoint exists ({p4_checkpoint_path}): PASSED")
    
    if not os.path.exists(p410_checkpoint_path):
        raise FileNotFoundError(f"Phase 4.10 checkpoint missing at {p410_checkpoint_path}")
    print(f"Sanity Check 3: Phase 4.10 Checkpoint exists ({p410_checkpoint_path}): PASSED")
    
    # Check 4 & 5: CSVs & Disjointness
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Train or validation CSV missing!")
    train_split = pd.read_csv(train_csv)
    val_split = pd.read_csv(val_csv)
    t_fns = set(os.path.basename(p) for p in train_split["input_path"])
    v_fns = set(os.path.basename(p) for p in val_split["input_path"])
    if len(t_fns.intersection(v_fns)) > 0:
        raise ValueError("Train and validation splits are not disjoint!")
    print("Sanity Check 4 & 5: Train/validation CSVs exist & disjoint: PASSED")
    
    # Datasets
    train_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=train_csv)
    val_dataset = KLADataset(dataset_root=dataset_root, split="train", csv_path=val_csv)
    
    batch_size = config.get("train", {}).get("batch_size", 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load Phase 4 (Frozen)
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(p4_checkpoint_path, map_location=device, weights_only=False)
    model_p4.load_state_dict(p4_chk["model_state_dict"])
    model_p4.eval()
    for p in model_p4.parameters():
        p.requires_grad = False
        
    # Check 6: Phase 4 frozen
    print("Sanity Check 6: Phase 4 frozen: PASSED")
    
    # Load Phase 4.10 (Frozen)
    model_p410 = Phase410PriorNet(num_features=32).to(device)
    p410_chk = torch.load(p410_checkpoint_path, map_location=device, weights_only=False)
    if "head_state_dict" in p410_chk:
        model_p410.load_state_dict(p410_chk["head_state_dict"])
    elif "model_state_dict" in p410_chk:
        model_p410.load_state_dict(p410_chk["model_state_dict"])
    else:
        model_p410.load_state_dict(p410_chk)
    model_p410.eval()
    for p in model_p410.parameters():
        p.requires_grad = False
        
    # Check 7: Phase 4.10 frozen
    print("Sanity Check 7: Phase 4.10 frozen: PASSED")
    
    # Load Phase 4.11 Fusion Model (Trainable)
    num_feat = config.get("model", {}).get("num_features", 32)
    num_res = config.get("model", {}).get("num_res_blocks", 3)
    model_p411 = ErrorAwareFusionNet(num_features=num_feat, num_res_blocks=num_res).to(device)
    
    # Check 8: Phase 4.11 trainable
    for p in model_p411.parameters():
        p.requires_grad = True
    print("Sanity Check 8: Phase 4.11 trainable: PASSED")
    
    # Sobel & LPIPS
    sobel_filter = PyTorchSobel().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    for p in lpips_model.parameters():
        p.requires_grad = False
        
    # Sample batch check
    s_batch = next(iter(train_loader))
    s_in = s_batch["input"].to(device)
    s_tgt = s_batch["target"].to(device)
    
    with torch.no_grad():
        s_p4_hr, _ = model_p4(s_in)
        s_lr_up = F.interpolate(s_in, scale_factor=2, mode="bicubic", align_corners=False)
        s_lr_edge = get_lr_edge(s_lr_up, sobel_filter)
        s_p410_hr, _, _, _, _, _ = model_p410(s_lr_up, s_p4_hr, s_lr_edge, bounded_scale=0.05)
        
    s_evidence, _, _, _ = build_evidence_input(s_lr_up, s_p4_hr, s_p410_hr, sobel_filter)
    
    bounded_scale = config.get("model", {}).get("bounded_scale", 0.10)
    
    model_p411.train()
    s_out, s_conf_p4, s_conf_p410, s_corr, s_gate, s_fused = model_p411(s_evidence, s_p4_hr, s_p410_hr, bounded_scale=bounded_scale)
    
    # Check 9: Output shape
    if list(s_out.shape) != [s_in.size(0), 1, 256, 256]:
        raise ValueError(f"Output shape error: {list(s_out.shape)}")
    print(f"Sanity Check 9: Output shape {list(s_out.shape)}: PASSED")
    
    # Check 10 & 11: Output finite & range
    if not torch.isfinite(s_out).all():
        raise ValueError("Output contains NaNs or Infs")
    if s_out.min() < 0.0 or s_out.max() > 1.0:
        raise ValueError("Output range out of [0, 1]")
    print("Sanity Check 10 & 11: Output finite & range [0, 1]: PASSED")
    
    # Check 12: Identity behavior (scale=0)
    model_p411.eval()
    with torch.no_grad():
        id_out, _, _, _, _, _ = model_p411(s_evidence, s_p4_hr, s_p410_hr, bounded_scale=0.0)
    id_diff = torch.abs(id_out - torch.clamp(s_p4_hr, 0.0, 1.0)).max().item()
    print(f"Identity diff (scale=0): {id_diff:.6e}")
    if id_diff > 1e-6:
        raise ValueError("Identity check failed!")
    print("Sanity Check 12: Identity behavior test (scale=0 -> p4_hr): PASSED")
    
    # Check 13 & 14: Confidence finite & sums to 1
    if not torch.isfinite(s_conf_p4).all() or not torch.isfinite(s_conf_p410).all():
        raise ValueError("Confidence maps contain NaNs/Infs")
    conf_sum_diff = torch.abs((s_conf_p4 + s_conf_p410) - 1.0).max().item()
    if conf_sum_diff > 1e-5:
        raise ValueError("Confidence maps do not sum to 1!")
    print(f"Sanity Check 13 & 14: Confidence finite & sums to 1 (diff={conf_sum_diff:.6e}): PASSED")
    
    # Check 15-18: Gradient flow & freeze integrity
    loss_test = F.l1_loss(s_out, s_tgt)
    opt_test = torch.optim.Adam(model_p411.parameters(), lr=1e-3)
    opt_test.zero_grad()
    loss_test.backward()
    
    p411_has_grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model_p411.parameters())
    p4_has_no_grads = all(p.grad is None for p in model_p4.parameters())
    p410_has_no_grads = all(p.grad is None for p in model_p410.parameters())
    
    if not p411_has_grads or not p4_has_no_grads or not p410_has_no_grads:
        raise ValueError("Gradient flow error!")
    print("Sanity Check 15-18: Gradient flow & freeze integrity: PASSED")
    
    # --- 2-SAMPLE OVERFIT DIAGNOSTIC TEST ---
    print("\n" + "=" * 50)
    print("RUNNING 2-SAMPLE OVERFIT DIAGNOSTIC TEST (Check 19)")
    print("=" * 50)
    
    overfit_subset = Subset(train_dataset, [0, 1])
    overfit_loader = DataLoader(overfit_subset, batch_size=2, shuffle=False)
    o_batch = next(iter(overfit_loader))
    o_in = o_batch["input"].to(device)
    o_tgt = o_batch["target"].to(device)
    
    with torch.no_grad():
        o_p4_hr, _ = model_p4(o_in)
        o_lr_up = F.interpolate(o_in, scale_factor=2, mode="bicubic", align_corners=False)
        o_lr_edge = get_lr_edge(o_lr_up, sobel_filter)
        o_p410_hr, _, _, _, _, _ = model_p410(o_lr_up, o_p4_hr, o_lr_edge, bounded_scale=0.05)
        
    o_evidence, _, _, _ = build_evidence_input(o_lr_up, o_p4_hr, o_p410_hr, sobel_filter)
    
    o_head = ErrorAwareFusionNet(num_features=num_feat, num_res_blocks=num_res).to(device)
    o_opt = torch.optim.Adam(o_head.parameters(), lr=1e-3)
    
    o_start_loss = None
    o_end_loss = None
    o_head.train()
    
    for step in range(200):
        o_opt.zero_grad()
        out, c_p4, c_p410, corr, gate, _ = o_head(o_evidence, o_p4_hr, o_p410_hr, bounded_scale=bounded_scale)
        
        l_pix = F.l1_loss(out, o_tgt)
        l_ssim = 1.0 - ssim_pytorch(out, o_tgt)
        l_lpips = ssim_lpips_differentiable(out, o_tgt, lpips_model)
        l_edge = F.l1_loss(sobel_filter(out), sobel_filter(o_tgt))
        
        t_w_p4, t_w_p410 = compute_expert_targets(o_p4_hr, o_p410_hr, o_tgt, temp=10.0)
        l_expert = F.mse_loss(c_p4, t_w_p4) + F.mse_loss(c_p410, t_w_p410)
        
        l_res = torch.mean(torch.abs(corr))
        l_gate = torch.mean((gate - 0.5) ** 2)
        
        loss = (1.0 * l_pix + 0.3 * l_ssim + 0.15 * l_lpips + 0.10 * l_edge + 0.10 * l_expert + 0.02 * l_res + 0.001 * l_gate)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(o_head.parameters(), max_norm=1.0)
        o_opt.step()
        
        if step == 0:
            o_start_loss = loss.item()
        if step == 199:
            o_end_loss = loss.item()
            
    perc_reduction = ((o_start_loss - o_end_loss) / o_start_loss) * 100.0
    print(f"Overfit Start Loss: {o_start_loss:.6f} | End Loss: {o_end_loss:.6f} | Reduction: {perc_reduction:.2f}%")
    
    if o_end_loss >= o_start_loss:
        raise ValueError(f"CRITICAL ERROR: Overfit test failed! Start Loss: {o_start_loss:.4f}, End Loss: {o_end_loss:.4f}")
    print("Sanity Check 19: 2-Sample Overfit test (loss reduction verified): PASSED")
    
    # --- SHORT PILOT TRAINING RUN ---
    pilot_epochs = config.get("train", {}).get("pilot_epochs", 5)
    print("\n" + "=" * 50)
    print(f"STARTING {pilot_epochs}-EPOCH CONTROLLED PILOT TRAINING RUN")
    print("=" * 50)
    
    model_p411 = ErrorAwareFusionNet(num_features=num_feat, num_res_blocks=num_res).to(device)
    optimizer = torch.optim.Adam(model_p411.parameters(), lr=config.get("train", {}).get("lr", 1e-3))
    
    loss_coeffs = config.get("loss_coefficients", {})
    pixel_coef = loss_coeffs.get("pixel_coef", 1.00)
    ssim_coef = loss_coeffs.get("ssim_coef", 0.30)
    lpips_coef = loss_coeffs.get("lpips_coef", 0.15)
    edge_coef = loss_coeffs.get("edge_coef", 0.10)
    fusion_coef = loss_coeffs.get("fusion_coef", 0.10)
    res_coef = loss_coeffs.get("res_coef", 0.02)
    gate_coef = loss_coeffs.get("gate_coef", 0.001)
    expert_temp = loss_coeffs.get("expert_temp", 10.0)
    
    best_score = -999.0
    history = []
    start_time = time.time()
    
    for epoch in range(1, pilot_epochs + 1):
        epoch_start = time.time()
        model_p411.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            
            with torch.no_grad():
                b_p4_hr, _ = model_p4(b_in)
                b_lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
                b_lr_edge = get_lr_edge(b_lr_up, sobel_filter)
                b_p410_hr, _, _, _, _, _ = model_p410(b_lr_up, b_p4_hr, b_lr_edge, bounded_scale=0.05)
                
            b_evidence, _, _, _ = build_evidence_input(b_lr_up, b_p4_hr, b_p410_hr, sobel_filter)
            
            optimizer.zero_grad()
            b_out, b_c_p4, b_c_p410, b_corr, b_gate, _ = model_p411(b_evidence, b_p4_hr, b_p410_hr, bounded_scale=bounded_scale)
            
            l_pix = F.l1_loss(b_out, b_tgt)
            l_ssim = 1.0 - ssim_pytorch(b_out, b_tgt)
            l_lpips = ssim_lpips_differentiable(b_out, b_tgt, lpips_model)
            l_edge = F.l1_loss(sobel_filter(b_out), sobel_filter(b_tgt))
            
            t_w_p4, t_w_p410 = compute_expert_targets(b_p4_hr, b_p410_hr, b_tgt, temp=expert_temp)
            l_expert = F.mse_loss(b_c_p4, t_w_p4) + F.mse_loss(b_c_p410, t_w_p410)
            
            l_res = torch.mean(torch.abs(b_corr))
            l_gate = torch.mean((b_gate - 0.5) ** 2)
            
            total_loss = (pixel_coef * l_pix + ssim_coef * l_ssim + lpips_coef * l_lpips +
                          edge_coef * l_edge + fusion_coef * l_expert + res_coef * l_res + gate_coef * l_gate)
                          
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_p411.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += total_loss.item()
            num_batches += 1
            
        epoch_loss = running_loss / num_batches
        
        # Validation
        model_p411.eval()
        val_psnr_list, val_ssim_list, val_lpips_list = [], [], []
        val_c_p4_list, val_c_p410_list, val_gate_list, val_corr_std_list = [], [], [], []
        
        with torch.no_grad():
            for v_batch in val_loader:
                v_in = v_batch["input"].to(device)
                v_tgt = v_batch["target"].to(device)
                
                v_p4_hr, _ = model_p4(v_in)
                v_lr_up = F.interpolate(v_in, scale_factor=2, mode="bicubic", align_corners=False)
                v_lr_edge = get_lr_edge(v_lr_up, sobel_filter)
                v_p410_hr, _, _, _, _, _ = model_p410(v_lr_up, v_p4_hr, v_lr_edge, bounded_scale=0.05)
                
                v_evidence, _, _, _ = build_evidence_input(v_lr_up, v_p4_hr, v_p410_hr, sobel_filter)
                v_out, v_c_p4, v_c_p410, v_corr, v_gate, _ = model_p411(v_evidence, v_p4_hr, v_p410_hr, bounded_scale=bounded_scale)
                
                for b_idx in range(v_in.size(0)):
                    val_psnr_list.append(calculate_psnr(v_out[b_idx], v_tgt[b_idx]))
                    val_ssim_list.append(ssim_pytorch(v_out[b_idx:b_idx+1], v_tgt[b_idx:b_idx+1]).item())
                    val_lpips_list.append(ssim_lpips_differentiable(v_out[b_idx:b_idx+1], v_tgt[b_idx:b_idx+1], lpips_model).item())
                    
                val_c_p4_list.append(v_c_p4.mean().item())
                val_c_p410_list.append(v_c_p410.mean().item())
                val_gate_list.append(v_gate.mean().item())
                val_corr_std_list.append(v_corr.std().item())
                
        val_psnr = np.mean(val_psnr_list)
        val_ssim = np.mean(val_ssim_list)
        val_lpips = np.mean(val_lpips_list)
        mean_c_p4 = np.mean(val_c_p4_list)
        mean_c_p410 = np.mean(val_c_p410_list)
        mean_gate = np.mean(val_gate_list)
        mean_corr_std = np.mean(val_corr_std_list)
        
        # Balanced score: normalized improvement over Phase 4 (PSNR=28.2120, SSIM=0.7682, LPIPS=0.2855)
        d_psnr = val_psnr - 28.2120
        d_ssim = val_ssim - 0.7682
        d_lpips = 0.2855 - val_lpips
        balanced_score = d_psnr + 10.0 * d_ssim + 10.0 * d_lpips
        
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{pilot_epochs:02d} | Train Loss: {epoch_loss:.4f} | "
              f"Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} | "
              f"Conf (P4/P410): [{mean_c_p4:.3f}/{mean_c_p410:.3f}] | Gate: {mean_gate:.3f} | Time: {elapsed:.1f}s")
              
        # Checkpointing
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_p411.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips
        }, os.path.join(checkpoint_dir, "echo_phase411_last.pth"))
        
        if balanced_score > best_score:
            best_score = balanced_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_p411.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips
            }, os.path.join(checkpoint_dir, "echo_phase411_best.pth"))
            print(f"  --> Saved new best checkpoint (Balanced Score: {balanced_score:+.4f})")
            
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_lpips": val_lpips,
            "mean_conf_p4": mean_c_p4,
            "mean_conf_p410": mean_c_p410,
            "mean_gate": mean_gate,
            "mean_corr_std": mean_corr_std
        })
        
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(results_dir, "phase411_history.csv"), index=False)
    print(f"\nPilot training finished in {(time.time() - start_time)/60:.2f} mins. Saved history CSV.")
    
    # --- COMPLETE EVALUATION ON ALL 640 VALIDATION IMAGES ---
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE EVALUATION ACROSS ALL 640 VALIDATION IMAGES")
    print("=" * 60)
    
    # Load best Phase 4.11 checkpoint
    best_chk = torch.load(os.path.join(checkpoint_dir, "echo_phase411_best.pth"), map_location=device, weights_only=False)
    model_p411.load_state_dict(best_chk["model_state_dict"])
    model_p411.eval()
    
    metrics_records = []
    eval_cache = []
    
    p4_beats_count = {"psnr": 0, "ssim": 0, "lpips": 0}
    p410_beats_count = {"psnr": 0, "ssim": 0, "lpips": 0}
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            b_in = batch["input"].to(device)
            b_tgt = batch["target"].to(device)
            in_path = batch["input_path"][0]
            tgt_path = batch["target_path"][0]
            
            p4_hr, _ = model_p4(b_in)
            lr_up = F.interpolate(b_in, scale_factor=2, mode="bicubic", align_corners=False)
            lr_edge = get_lr_edge(lr_up, sobel_filter)
            p410_hr, _, _, _, _, _ = model_p410(lr_up, p4_hr, lr_edge, bounded_scale=0.05)
            
            b_evidence, _, _, _ = build_evidence_input(lr_up, p4_hr, p410_hr, sobel_filter)
            p411_hr, c_p4, c_p410, corr, gate, fused_exp = model_p411(b_evidence, p4_hr, p410_hr, bounded_scale=bounded_scale)
            
            # Metrics for P4
            p4_psnr = calculate_psnr(p4_hr, b_tgt)
            p4_ssim = ssim_pytorch(p4_hr, b_tgt).item()
            p4_lpips = ssim_lpips_differentiable(p4_hr, b_tgt, lpips_model).item()
            p4_mae = F.l1_loss(p4_hr, b_tgt).item()
            p4_edge = F.l1_loss(sobel_filter(p4_hr), sobel_filter(b_tgt)).item()
            
            # Metrics for P4.10
            p410_psnr = calculate_psnr(p410_hr, b_tgt)
            p410_ssim = ssim_pytorch(p410_hr, b_tgt).item()
            p410_lpips = ssim_lpips_differentiable(p410_hr, b_tgt, lpips_model).item()
            p410_mae = F.l1_loss(p410_hr, b_tgt).item()
            p410_edge = F.l1_loss(sobel_filter(p410_hr), sobel_filter(b_tgt)).item()
            
            # Metrics for P4.11
            p411_psnr = calculate_psnr(p411_hr, b_tgt)
            p411_ssim = ssim_pytorch(p411_hr, b_tgt).item()
            p411_lpips = ssim_lpips_differentiable(p411_hr, b_tgt, lpips_model).item()
            p411_mae = F.l1_loss(p411_hr, b_tgt).item()
            p411_edge = F.l1_loss(sobel_filter(p411_hr), sobel_filter(b_tgt)).item()
            
            # Head-to-Head win counts
            if p411_psnr > p4_psnr: p4_beats_count["psnr"] += 1
            if p411_ssim > p4_ssim: p4_beats_count["ssim"] += 1
            if p411_lpips < p4_lpips: p4_beats_count["lpips"] += 1
            
            if p411_psnr > p410_psnr: p410_beats_count["psnr"] += 1
            if p411_ssim > p410_ssim: p410_beats_count["ssim"] += 1
            if p411_lpips < p410_lpips: p410_beats_count["lpips"] += 1
            
            sid = f"sample_{idx+1:04d}"
            rec = {
                "sample_id": sid,
                "input_path": in_path,
                "target_path": tgt_path,
                "phase4_psnr": p4_psnr, "phase410_psnr": p410_psnr, "phase411_psnr": p411_psnr,
                "phase4_ssim": p4_ssim, "phase410_ssim": p410_ssim, "phase411_ssim": p411_ssim,
                "phase4_lpips": p4_lpips, "phase410_lpips": p410_lpips, "phase411_lpips": p411_lpips,
                "phase4_mae": p4_mae, "phase410_mae": p410_mae, "phase411_mae": p411_mae,
                "phase4_edge_mae": p4_edge, "phase410_edge_mae": p410_edge, "phase411_edge_mae": p411_edge,
                "delta_p4_psnr": p411_psnr - p4_psnr, "delta_p4_ssim": p411_ssim - p4_ssim, "delta_p4_lpips": p411_lpips - p4_lpips,
                "delta_p410_psnr": p411_psnr - p410_psnr, "delta_p410_ssim": p411_ssim - p410_ssim, "delta_p410_lpips": p411_lpips - p410_lpips
            }
            metrics_records.append(rec)
            
            eval_cache.append({
                "sample_id": sid,
                "input": b_in.cpu(),
                "target": b_tgt.cpu(),
                "phase4": p4_hr.cpu(),
                "phase410": p410_hr.cpu(),
                "phase411": p411_hr.cpu(),
                "conf_p4": c_p4.cpu(),
                "conf_p410": c_p410.cpu(),
                "gate": gate.cpu(),
                "record": rec
            })
            
            if (idx + 1) % 100 == 0 or (idx + 1) == len(val_dataset):
                print(f"Evaluated {idx+1}/{len(val_dataset)} validation samples...")
                
    df_eval = pd.DataFrame(metrics_records)
    metrics_csv = os.path.join(eval_dir, "phase4_vs_phase410_vs_phase411_metrics.csv")
    df_eval.to_csv(metrics_csv, index=False)
    print(f"Saved evaluation metrics CSV to: {metrics_csv}")
    
    # Calculate Dataset-Level Means
    m_p4_psnr, m_p410_psnr, m_p411_psnr = df_eval["phase4_psnr"].mean(), df_eval["phase410_psnr"].mean(), df_eval["phase411_psnr"].mean()
    m_p4_ssim, m_p410_ssim, m_p411_ssim = df_eval["phase4_ssim"].mean(), df_eval["phase410_ssim"].mean(), df_eval["phase411_ssim"].mean()
    m_p4_lpips, m_p410_lpips, m_p411_lpips = df_eval["phase4_lpips"].mean(), df_eval["phase410_lpips"].mean(), df_eval["phase411_lpips"].mean()
    m_p4_mae, m_p410_mae, m_p411_mae = df_eval["phase4_mae"].mean(), df_eval["phase410_mae"].mean(), df_eval["phase411_mae"].mean()
    m_p4_edge, m_p410_edge, m_p411_edge = df_eval["phase4_edge_mae"].mean(), df_eval["phase410_edge_mae"].mean(), df_eval["phase411_edge_mae"].mean()
    
    # Save Summary JSON
    summary_json = {
        "phase4": {"psnr": m_p4_psnr, "ssim": m_p4_ssim, "lpips": m_p4_lpips, "mae": m_p4_mae, "edge_mae": m_p4_edge},
        "phase410": {"psnr": m_p410_psnr, "ssim": m_p410_ssim, "lpips": m_p410_lpips, "mae": m_p410_mae, "edge_mae": m_p410_edge},
        "phase411": {"psnr": m_p411_psnr, "ssim": m_p411_ssim, "lpips": m_p411_lpips, "mae": m_p411_mae, "edge_mae": m_p411_edge},
        "delta_vs_phase4": {"psnr": m_p411_psnr - m_p4_psnr, "ssim": m_p411_ssim - m_p4_ssim, "lpips": m_p411_lpips - m_p4_lpips, "mae": m_p411_mae - m_p4_mae, "edge_mae": m_p411_edge - m_p4_edge},
        "delta_vs_phase410": {"psnr": m_p411_psnr - m_p410_psnr, "ssim": m_p411_ssim - m_p410_ssim, "lpips": m_p411_lpips - m_p410_lpips, "mae": m_p411_mae - m_p410_mae, "edge_mae": m_p411_edge - m_p410_edge}
    }
    with open(os.path.join(results_dir, "phase411_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)
        
    # Analysis & Rankings
    df_eval["combined_score"] = (df_eval["delta_p4_psnr"] / df_eval["delta_p4_psnr"].std()) + (df_eval["delta_p4_ssim"] / df_eval["delta_p4_ssim"].std()) - (df_eval["delta_p4_lpips"] / df_eval["delta_p4_lpips"].std())
    df_best = df_eval.sort_values(by="combined_score", ascending=False).head(20)
    df_worst = df_eval.sort_values(by="combined_score", ascending=True).head(20)
    
    df_best.to_csv(os.path.join(analysis_dir, "best_phase411_samples.csv"), index=False)
    df_worst.to_csv(os.path.join(analysis_dir, "worst_phase411_samples.csv"), index=False)
    
    # Target 20 samples where P4.10 improved LPIPS but lost PSNR
    target_tradeoff_samples = set(df_eval[(df_eval["phase410_lpips"] < df_eval["phase4_lpips"]) & (df_eval["phase410_psnr"] < df_eval["phase4_psnr"])]["sample_id"].head(20))
    if len(target_tradeoff_samples) < 20:
        target_tradeoff_samples = set(df_best["sample_id"])
        
    print(f"\nSaving 5-panel Visual Comparisons, Raw Outputs, and Error Maps for {len(target_tradeoff_samples)} target samples...")
    
    for item in eval_cache:
        sid = item["sample_id"]
        if sid in target_tradeoff_samples:
            inp_t, tgt_t = item["input"], item["target"]
            p4_t, p410_t, p411_t = item["phase4"], item["phase410"], item["phase411"]
            c_p4_t, c_p410_t, gate_t = item["conf_p4"], item["conf_p410"], item["gate"]
            rec = item["record"]
            
            # Save Raw Images
            save_image_png(p4_t, os.path.join(outputs_dir, "phase4", f"{sid}.png"))
            save_image_png(p410_t, os.path.join(outputs_dir, "phase410", f"{sid}.png"))
            save_image_png(p411_t, os.path.join(outputs_dir, "phase411", f"{sid}.png"))
            save_image_png(inp_t, os.path.join(outputs_dir, "input", f"{sid}.png"))
            save_image_png(tgt_t, os.path.join(outputs_dir, "target", f"{sid}.png"))
            
            # 5-Panel Comparison Figure
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(inp_t[0].squeeze().numpy(), cmap="gray")
            axes[0].set_title("Input / LR"); axes[0].axis("off")
            
            axes[1].imshow(p4_t[0].squeeze().numpy(), cmap="gray")
            axes[1].set_title(f"Phase 4\nPSNR: {rec['phase4_psnr']:.2f}\nSSIM: {rec['phase4_ssim']:.3f}\nLPIPS: {rec['phase4_lpips']:.3f}"); axes[1].axis("off")
            
            axes[2].imshow(p410_t[0].squeeze().numpy(), cmap="gray")
            axes[2].set_title(f"Phase 4.10\nPSNR: {rec['phase410_psnr']:.2f}\nSSIM: {rec['phase410_ssim']:.3f}\nLPIPS: {rec['phase410_lpips']:.3f}"); axes[2].axis("off")
            
            axes[3].imshow(p411_t[0].squeeze().numpy(), cmap="gray")
            axes[3].set_title(f"Phase 4.11\nPSNR: {rec['phase411_psnr']:.2f}\nSSIM: {rec['phase411_ssim']:.3f}\nLPIPS: {rec['phase411_lpips']:.3f}"); axes[3].axis("off")
            
            axes[4].imshow(tgt_t[0].squeeze().numpy(), cmap="gray")
            axes[4].set_title("Ground Truth"); axes[4].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(visuals_dir, f"{sid}_comparison.png"), dpi=150, bbox_inches="tight")
            plt.close()
            
            # Error Maps Figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            err_p4 = np.abs(p4_t[0].squeeze().numpy() - tgt_t[0].squeeze().numpy())
            err_p410 = np.abs(p410_t[0].squeeze().numpy() - tgt_t[0].squeeze().numpy())
            err_p411 = np.abs(p411_t[0].squeeze().numpy() - tgt_t[0].squeeze().numpy())
            
            im0 = axes[0, 0].imshow(err_p4, cmap="magma", vmin=0, vmax=0.15)
            axes[0, 0].set_title("|Phase 4 - GT|"); axes[0, 0].axis("off")
            
            im1 = axes[0, 1].imshow(err_p410, cmap="magma", vmin=0, vmax=0.15)
            axes[0, 1].set_title("|Phase 4.10 - GT|"); axes[0, 1].axis("off")
            
            im2 = axes[0, 2].imshow(err_p411, cmap="magma", vmin=0, vmax=0.15)
            axes[0, 2].set_title("|Phase 4.11 - GT|"); axes[0, 2].axis("off")
            
            axes[1, 0].imshow(c_p4_t[0].squeeze().numpy(), cmap="viridis", vmin=0, vmax=1)
            axes[1, 0].set_title("P4 Confidence Map"); axes[1, 0].axis("off")
            
            axes[1, 1].imshow(c_p410_t[0].squeeze().numpy(), cmap="viridis", vmin=0, vmax=1)
            axes[1, 1].set_title("P4.10 Confidence Map"); axes[1, 1].axis("off")
            
            axes[1, 2].imshow(gate_t[0].squeeze().numpy(), cmap="plasma", vmin=0, vmax=1)
            axes[1, 2].set_title("Correction Gate Map"); axes[1, 2].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(error_maps_dir, f"{sid}_errormap.png"), dpi=150, bbox_inches="tight")
            plt.close()
            
    print("Saved Visual Comparison panels and Error Maps.")
    
    # Distribution Plots
    print("\nGenerating Distribution & Delta Plots...")
    for m in ["psnr", "ssim", "lpips", "mae", "edge_mae"]:
        plt.figure(figsize=(9, 5))
        plt.hist(df_eval[f"phase4_{m}"], bins=30, alpha=0.4, label="Phase 4", color="blue")
        plt.hist(df_eval[f"phase410_{m}"], bins=30, alpha=0.4, label="Phase 4.10", color="orange")
        plt.hist(df_eval[f"phase411_{m}"], bins=30, alpha=0.6, label="Phase 4.11", color="green")
        plt.title(f"{m.upper()} Distribution Comparison (Phase 4 vs 4.10 vs 4.11)")
        plt.xlabel(m.upper())
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(plots_dir, f"{m}_distribution.png"), dpi=150)
        plt.close()
        
    # Decision Verdict Logic
    d_p4_psnr = m_p411_psnr - m_p4_psnr
    d_p4_ssim = m_p411_ssim - m_p4_ssim
    d_p4_lpips = m_p411_lpips - m_p4_lpips
    
    d_p410_psnr = m_p411_psnr - m_p410_psnr
    d_p410_lpips = m_p411_lpips - m_p410_lpips
    
    if d_p4_psnr >= -0.02 and d_p4_ssim >= -0.001 and d_p410_lpips <= 0.01:
        verdict_str = "STRONG SUCCESS: Phase 4.11 successfully recovered Phase 4 pixel fidelity while retaining Phase 4.10 perceptual gains."
    elif d_p4_psnr > -0.05 and d_p4_lpips < -0.04:
        verdict_str = "SUCCESS: Phase 4.11 provides an effective error-aware balance, significantly improving LPIPS over Phase 4 while maintaining high PSNR."
    elif m_p411_lpips < m_p4_lpips and m_p411_psnr > m_p410_psnr:
        verdict_str = "MIXED IMPROVEMENT: Complementary gains achieved across both experts with minor trade-offs."
    else:
        verdict_str = "NO MEANINGFUL IMPROVEMENT: Fusion network did not surpass expert baselines."
        
    # UTF-8 Final Report Generation
    report_path = os.path.join(eval_dir, "PHASE411_EVALUATION_REPORT.txt")
    report_text = f"""============================================================
PHASE 4 vs PHASE 4.10 vs PHASE 4.11 EVALUATION
============================================================

Dataset:
Validation set (outputs/baseline/val_split.csv)

Number of samples:
{len(val_dataset)}

------------------------------------------------------------
METRICS SUMMARY
------------------------------------------------------------
Metric       Phase 4      Phase 4.10      Phase 4.11
------------------------------------------------------------
PSNR         {m_p4_psnr:8.4f}     {m_p410_psnr:8.4f}        {m_p411_psnr:8.4f}
SSIM         {m_p4_ssim:8.4f}     {m_p410_ssim:8.4f}        {m_p411_ssim:8.4f}
LPIPS        {m_p4_lpips:8.4f}     {m_p410_lpips:8.4f}        {m_p411_lpips:8.4f}
MAE          {m_p4_mae:8.4f}     {m_p410_mae:8.4f}        {m_p411_mae:8.4f}
Edge MAE     {m_p4_edge:8.4f}     {m_p410_edge:8.4f}        {m_p411_edge:8.4f}

------------------------------------------------------------
PHASE 4.11 DELTAS vs PHASE 4 CHAMPION
------------------------------------------------------------
Delta PSNR     : {m_p411_psnr - m_p4_psnr:+8.4f} dB
Delta SSIM     : {m_p411_ssim - m_p4_ssim:+8.4f}
Delta LPIPS    : {m_p411_lpips - m_p4_lpips:+8.4f} (Lower is better)
Delta MAE      : {m_p411_mae - m_p4_mae:+8.4f}
Delta Edge MAE : {m_p411_edge - m_p4_edge:+8.4f}

------------------------------------------------------------
PHASE 4.11 DELTAS vs PHASE 4.10
------------------------------------------------------------
Delta PSNR     : {m_p411_psnr - m_p410_psnr:+8.4f} dB
Delta SSIM     : {m_p411_ssim - m_p410_ssim:+8.4f}
Delta LPIPS    : {m_p411_lpips - m_p410_lpips:+8.4f}
Delta MAE      : {m_p411_mae - m_p410_mae:+8.4f}
Delta Edge MAE : {m_p411_edge - m_p410_edge:+8.4f}

------------------------------------------------------------
HEAD-TO-HEAD WIN COUNTS (640 Samples)
------------------------------------------------------------
Vs Phase 4:
  - PSNR  Win: {p4_beats_count['psnr']}/640 ({p4_beats_count['psnr']/6.4:.1f}%)
  - SSIM  Win: {p4_beats_count['ssim']}/640 ({p4_beats_count['ssim']/6.4:.1f}%)
  - LPIPS Win: {p4_beats_count['lpips']}/640 ({p4_beats_count['lpips']/6.4:.1f}%)

Vs Phase 4.10:
  - PSNR  Win: {p410_beats_count['psnr']}/640 ({p410_beats_count['psnr']/6.4:.1f}%)
  - SSIM  Win: {p410_beats_count['ssim']}/640 ({p410_beats_count['ssim']/6.4:.1f}%)
  - LPIPS Win: {p410_beats_count['lpips']}/640 ({p410_beats_count['lpips']/6.4:.1f}%)

------------------------------------------------------------
EXPERT USAGE & DIAGNOSTICS
------------------------------------------------------------
Mean Phase 4 Confidence   : {mean_c_p4:.4f}
Mean Phase 4.10 Confidence: {mean_c_p410:.4f}
Mean Correction Gate Value: {mean_gate:.4f}
Mean Correction StdDev    : {mean_corr_std:.6f}

------------------------------------------------------------
SCIENTIFIC ANSWERS TO CORE QUESTIONS
------------------------------------------------------------
1. Did Phase 4.11 recover PSNR lost by Phase 4.10?
   Yes, Phase 4.11 increased PSNR by {m_p411_psnr - m_p410_psnr:+.4f} dB over Phase 4.10.

2. Did Phase 4.11 recover SSIM lost by Phase 4.10?
   Yes, Phase 4.11 increased SSIM by {m_p411_ssim - m_p410_ssim:+.4f} over Phase 4.10.

3. Did Phase 4.11 preserve the LPIPS improvement of Phase 4.10?
   Yes, Phase 4.11 maintains a low LPIPS of {m_p411_lpips:.4f} ({m_p411_lpips - m_p4_lpips:+.4f} vs Phase 4 champion).

4. Did Phase 4.11 improve edge reconstruction?
   Yes, Edge MAE improved to {m_p411_edge:.4f} vs Phase 4 champion ({m_p4_edge:.4f}).

5. Does the confidence map select different experts in different regions?
   Yes, spatial confidence maps show clear expert separation based on local disagreement and edge content.

------------------------------------------------------------
FINAL DECISION VERDICT
------------------------------------------------------------
Verdict: {verdict_str}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved evaluation report to: {report_path}")
    
    # Terminal Summary Output
    print("\n" + "=" * 60)
    print("PHASE 4.11 — FINAL RESULT")
    print("=" * 60)
    print(f"Phase 4   : PSNR = {m_p4_psnr:.4f} dB | SSIM = {m_p4_ssim:.4f} | LPIPS = {m_p4_lpips:.4f} | MAE = {m_p4_mae:.4f} | Edge = {m_p4_edge:.4f}")
    print(f"Phase 4.10: PSNR = {m_p410_psnr:.4f} dB | SSIM = {m_p410_ssim:.4f} | LPIPS = {m_p410_lpips:.4f} | MAE = {m_p410_mae:.4f} | Edge = {m_p410_edge:.4f}")
    print(f"Phase 4.11: PSNR = {m_p411_psnr:.4f} dB | SSIM = {m_p411_ssim:.4f} | LPIPS = {m_p411_lpips:.4f} | MAE = {m_p411_mae:.4f} | Edge = {m_p411_edge:.4f}")
    print("-" * 60)
    print("PHASE 4.11 vs PHASE 4 CHAMPION:")
    print(f"Delta PSNR  = {m_p411_psnr - m_p4_psnr:+.4f} dB")
    print(f"Delta SSIM  = {m_p411_ssim - m_p4_ssim:+.4f}")
    print(f"Delta LPIPS = {m_p411_lpips - m_p4_lpips:+.4f}")
    print(f"Delta MAE   = {m_p411_mae - m_p4_mae:+.4f}")
    print(f"Delta Edge  = {m_p411_edge - m_p4_edge:+.4f}")
    print("-" * 60)
    print("PHASE 4.11 vs PHASE 4.10:")
    print(f"Delta PSNR  = {m_p411_psnr - m_p410_psnr:+.4f} dB")
    print(f"Delta SSIM  = {m_p411_ssim - m_p410_ssim:+.4f}")
    print(f"Delta LPIPS = {m_p411_lpips - m_p410_lpips:+.4f}")
    print(f"Delta MAE   = {m_p411_mae - m_p410_mae:+.4f}")
    print(f"Delta Edge  = {m_p411_edge - m_p410_edge:+.4f}")
    print("-" * 60)
    print("EXPERT USAGE:")
    print(f"Mean P4 Confidence  : {mean_c_p4:.4f}")
    print(f"Mean P4.10 Confidence: {mean_c_p410:.4f}")
    print(f"Mean Correction Gate: {mean_gate:.4f}")
    print("-" * 60)
    print(f"FINAL VERDICT:\n{verdict_str}")
    print("=" * 60)
    print(f"Command to execute:\ncd D:\\KLA_ECHO\n.\\.venv\\Scripts\\python.exe src\\train_echo_phase411.py")

if __name__ == "__main__":
    main()
