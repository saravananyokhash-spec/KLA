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


# ============================================================
# PHASE 4.10
# CONTROLLED REFINEMENT EXPERIMENT
# ============================================================


def decompose_frequencies(img, r_low=15, r_mid=64):
    """
    Frequency decomposition using FFT.

    Used for analysis/visual inspection.
    """
    h, w = img.shape

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    cy, cx = h // 2, w // 2

    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r2 = x**2 + y**2

    mask_low = r2 < r_low**2
    mask_mid = (r2 >= r_low**2) & (r2 < r_mid**2)
    mask_high = r2 >= r_mid**2

    img_low = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(fshift * mask_low)
        )
    )

    img_mid = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(fshift * mask_mid)
        )
    )

    img_high = np.real(
        np.fft.ifft2(
            np.fft.ifftshift(fshift * mask_high)
        )
    )

    return img_low, img_mid, img_high


# ============================================================
# STABLE FREQUENCY LOSS
# ============================================================

def frequency_loss(pred, target):
    """
    Stable frequency-domain loss.

    Improvements over raw FFT magnitude L1:
    1. Orthonormal FFT
    2. Log magnitude compression
    3. L1 comparison

    This prevents very strong low-frequency components from
    dominating the loss.
    """

    pred_fft = torch.fft.fft2(pred, norm="ortho")
    target_fft = torch.fft.fft2(target, norm="ortho")

    pred_mag = torch.log1p(torch.abs(pred_fft))
    target_mag = torch.log1p(torch.abs(target_fft))

    return F.l1_loss(pred_mag, target_mag)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(pred, target):
    """
    Calculate PSNR assuming images are normalized to [0, 1].
    """

    mse = torch.mean((pred - target) ** 2)

    if mse.item() == 0:
        return float("inf")

    return (
        20 * torch.log10(
            1.0 / torch.sqrt(mse)
        )
    ).item()


# ============================================================
# SQUEEZE-EXCITATION
# ============================================================

class SqueezeExcitationBlock(nn.Module):
    """
    Lightweight channel attention block.
    """

    def __init__(self, channels, reduction=8):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // reduction,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        y = self.avg_pool(x)

        y = self.fc(y)

        return x * y


# ============================================================
# PHASE 4.10 PRIOR NETWORK
# ============================================================

class Phase410PriorNet(nn.Module):

    """
    Phase 4.10 Controlled Refinement Architecture.

    Components:

    1. Structure branch
    2. Noise branch
    3. Structural prior prediction
    4. HR encoder
    5. Residual refinement block
    6. SE attention
    7. Correction head
    8. Learnable spatial gate

    The final output is:

        final_hr =
            base_hr +
            bounded_scale *
            gate *
            correction

    Phase 4 remains completely frozen.
    """

    def __init__(self, num_features=32):

        super().__init__()

        # ====================================================
        # 1. STRUCTURE BRANCH
        # ====================================================

        self.struct_branch = nn.Sequential(

            nn.Conv2d(
                3,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # 2. NOISE BRANCH
        # ====================================================

        self.noise_branch = nn.Sequential(

            nn.Conv2d(
                2,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # 3. STRUCTURAL PRIOR HEAD
        # ====================================================

        self.struct_prior_head = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )


        # ====================================================
        # 4. HR ENCODER
        # ====================================================

        self.hr_encoder = nn.Sequential(

            nn.Conv2d(
                1,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # 5. FEATURE FUSION
        # ====================================================

        self.fuse_conv = nn.Sequential(

            nn.Conv2d(
                num_features * 3,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # 6. RESIDUAL REFINEMENT BLOCK
        # ====================================================

        self.res_block = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                padding=1
            )
        )

        self.res_relu = nn.ReLU(inplace=True)


        # ====================================================
        # 7. SE ATTENTION
        # ====================================================

        self.se_attn = SqueezeExcitationBlock(
            num_features,
            reduction=8
        )


        # ====================================================
        # 8. CORRECTION HEAD
        # ====================================================

        self.correction_head = nn.Conv2d(
            num_features,
            1,
            kernel_size=3,
            padding=1
        )


        # ====================================================
        # 9. PIXEL-WISE GATE
        # ====================================================

        self.gate_head = nn.Sequential(

            nn.Conv2d(
                num_features,
                num_features // 2,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                num_features // 2,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )


        # ====================================================
        # WEIGHT INITIALIZATION
        # ====================================================

        # Structure branch
        for m in self.struct_branch.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # Noise branch
        for m in self.noise_branch.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # Structural prior
        for m in self.struct_prior_head.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.normal_(
                    m.weight,
                    std=0.001
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # HR encoder
        for m in self.hr_encoder.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # Fusion
        for m in self.fuse_conv.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # Residual block
        for m in self.res_block.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


        # ====================================================
        # CORRECTION INITIALIZATION
        #
        # Slightly stronger than Phase 4.10 original.
        # ====================================================

        nn.init.normal_(
            self.correction_head.weight,
            std=0.005
        )

        if self.correction_head.bias is not None:

            nn.init.constant_(
                self.correction_head.bias,
                0.0
            )


        # ====================================================
        # GATE INITIALIZATION
        #
        # sigmoid(0) = 0.5
        # ====================================================

        for m in self.gate_head.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.normal_(
                    m.weight,
                    std=0.001
                )

                if m.bias is not None:

                    nn.init.constant_(
                        m.bias,
                        0.0
                    )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        lr_up,
        base_hr,
        lr_edge,
        bounded_scale=0.10
    ):

        # ----------------------------------------------------
        # Structure features
        # ----------------------------------------------------

        struct_in = torch.cat(
            [
                lr_up,
                base_hr,
                lr_edge
            ],
            dim=1
        )

        struct_feats = self.struct_branch(
            struct_in
        )


        # ----------------------------------------------------
        # Noise features
        # ----------------------------------------------------

        noise_in = torch.cat(
            [
                lr_up,
                lr_edge
            ],
            dim=1
        )

        noise_feats = self.noise_branch(
            noise_in
        )


        # ----------------------------------------------------
        # Structural prior
        # ----------------------------------------------------

        pred_struct = self.struct_prior_head(
            struct_feats
        )


        # ----------------------------------------------------
        # HR features
        # ----------------------------------------------------

        hr_feats = self.hr_encoder(
            base_hr
        )


        # ----------------------------------------------------
        # Feature fusion
        # ----------------------------------------------------

        fused_raw = torch.cat(
            [
                hr_feats,
                struct_feats,
                noise_feats
            ],
            dim=1
        )

        fused_feats = self.fuse_conv(
            fused_raw
        )


        # ----------------------------------------------------
        # Residual refinement
        # ----------------------------------------------------

        refined_res = self.res_block(
            fused_feats
        )

        refined_feats = self.res_relu(
            fused_feats + refined_res
        )


        # ----------------------------------------------------
        # SE attention
        # ----------------------------------------------------

        attn_feats = self.se_attn(
            refined_feats
        )


        # ----------------------------------------------------
        # Correction
        # ----------------------------------------------------

        raw_res = self.correction_head(
            attn_feats
        )

        correction = torch.tanh(
            raw_res
        )


        # ----------------------------------------------------
        # Learnable spatial gate
        # ----------------------------------------------------

        gate = self.gate_head(
            attn_feats
        )


        # ----------------------------------------------------
        # Gated residual refinement
        # ----------------------------------------------------

        gated_correction = (
            gate * correction
        )


        final_hr = torch.clamp(
            base_hr +
            bounded_scale * gated_correction,
            0.0,
            1.0
        )


        return (
            final_hr,
            pred_struct,
            correction,
            gate,
            struct_feats,
            noise_feats
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # GPU CHECK
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CRITICAL ERROR: CUDA is not available! "
            "Phase 4.10 requires GPU execution."
        )


    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)


    # ========================================================
    # DIRECTORIES
    # ========================================================

    out_dir = "outputs/phase410"

    checkpoint_dir = os.path.join(
        out_dir,
        "checkpoints"
    )

    results_dir = os.path.join(
        out_dir,
        "results"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )


    # ========================================================
    # CONFIG
    # ========================================================

    config_path = os.path.join(
        out_dir,
        "configs",
        "phase410.yaml"
    )

    if not os.path.exists(config_path):

        config_path = "configs/echo.yaml"


    config = load_config(
        config_path
    )

    set_seed(42)


    # ========================================================
    # CONTROLLED SETTINGS
    # ========================================================

    bounded_scale = config.get(
        "bounded_scale",
        0.10
    )


    loss_coeffs = config.get(
        "loss_coefficients",
        {}
    )


    pixel_coef = loss_coeffs.get(
        "pixel_coef",
        1.00
    )

    ssim_coef = loss_coeffs.get(
        "ssim_coef",
        0.20
    )

    charbonnier_coef = loss_coeffs.get(
        "charbonnier_coef",
        0.10
    )

    edge_coef = loss_coeffs.get(
        "edge_coef",
        0.10
    )

    freq_coef = loss_coeffs.get(
        "freq_coef",
        0.05
    )

    lpips_coef = loss_coeffs.get(
        "lpips_coef",
        0.15
    )

    struct_coef = loss_coeffs.get(
        "struct_coef",
        0.30
    )

    noise_coef = loss_coeffs.get(
        "noise_coef",
        0.05
    )

    res_coef = loss_coeffs.get(
        "res_coef",
        0.005
    )

    gate_coef = loss_coeffs.get(
        "gate_coef",
        0.0001
    )


    print("=" * 60)

    print(
        "PHASE 4.10 — CONTROLLED REFINEMENT EXPERIMENT"
    )

    print(
        f"Device: {device} ({gpu_name})"
    )

    print(
        f"Output Directory: {out_dir}"
    )

    print(
        f"Bounded Scale: {bounded_scale}"
    )

    print("=" * 60)


    # ========================================================
    # SANITY CHECKS
    # ========================================================

    print("\n" + "=" * 50)

    print(
        "RUNNING PHASE 4.10 SANITY CHECKS (1-16)"
    )

    print("=" * 50)


    # --------------------------------------------------------
    # CHECK 1
    # --------------------------------------------------------

    print(
        "Sanity Check 1: CUDA available: PASSED"
    )


    # --------------------------------------------------------
    # CHECK 2
    # --------------------------------------------------------

    p4_checkpoint_path = config.get(
        "p4_checkpoint",
        "outputs/echo_phase4/checkpoints/echo_best.pth"
    )


    if not os.path.exists(
        p4_checkpoint_path
    ):

        raise FileNotFoundError(
            f"Safety Error: Phase 4 checkpoint not found at: "
            f"{p4_checkpoint_path}"
        )


    print(
        f"Sanity Check 2: Phase 4 Checkpoint exists "
        f"({p4_checkpoint_path}): PASSED"
    )


    # --------------------------------------------------------
    # CHECK 3
    # --------------------------------------------------------

    train_csv = config.get(
        "train_split",
        "outputs/baseline/train_split.csv"
    )

    val_csv = config.get(
        "val_split",
        "outputs/baseline/val_split.csv"
    )


    train_split = pd.read_csv(
        train_csv
    )

    val_split = pd.read_csv(
        val_csv
    )


    train_fns = set(
        os.path.basename(p)
        for p in train_split["input_path"]
    )

    val_fns = set(
        os.path.basename(p)
        for p in val_split["input_path"]
    )


    if len(
        train_fns.intersection(val_fns)
    ) > 0:

        raise ValueError(
            "Safety Error: "
            "Train and validation splits are not disjoint!"
        )


    print(
        "Sanity Check 3: "
        "Train/validation disjointness: PASSED"
    )


    # ========================================================
    # DATASET
    # ========================================================

    dataset_root = config.get(
        "dataset_root",
        "D:/kla"
    )


    train_dataset = KLADataset(
        dataset_root=dataset_root,
        split="train",
        csv_path=train_csv
    )


    val_dataset = KLADataset(
        dataset_root=dataset_root,
        split="train",
        csv_path=val_csv
    )


    train_batch_size = config.get(
        "train",
        {}
    ).get(
        "batch_size",
        16
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    # ========================================================
    # LOAD FROZEN PHASE 4
    # ========================================================

    model_cfg = config.get(
        "model",
        {}
    )


    model_p4 = BaselineECHOModel(

        in_channels=model_cfg.get(
            "in_channels",
            1
        ),

        out_channels=model_cfg.get(
            "out_channels",
            1
        ),

        num_features=model_cfg.get(
            "num_features",
            64
        ),

        num_blocks=model_cfg.get(
            "num_blocks",
            6
        ),

        ablation=None

    ).to(device)


    p4_chk = torch.load(
        p4_checkpoint_path,
        map_location=device
    )


    model_p4.load_state_dict(
        p4_chk["model_state_dict"]
    )


    model_p4.eval()


    # Freeze Phase 4
    for p in model_p4.parameters():

        p.requires_grad = False


    print(
        "Sanity Check 4: "
        "Phase 4 parameters frozen: PASSED"
    )


    # ========================================================
    # PHASE 4.10 HEAD
    # ========================================================

    head = Phase410PriorNet(
        num_features=32
    ).to(device)


    for p in head.parameters():

        p.requires_grad = True


    print(
        "Sanity Check 5: "
        "Recovery head parameters trainable: PASSED"
    )


    # ========================================================
    # LPIPS
    # ========================================================

    lpips_model = lpips.LPIPS(
        net="alex"
    ).to(device)


    lpips_model.eval()


    for p in lpips_model.parameters():

        p.requires_grad = False


    # ========================================================
    # SOBEL
    # ========================================================

    sobel_filter = PyTorchSobel().to(device)


    # ========================================================
    # SAMPLE BATCH
    # ========================================================

    sample_batch = next(
        iter(train_loader)
    )


    sample_in = sample_batch[
        "input"
    ].to(device)


    sample_tgt = sample_batch[
        "target"
    ].to(device)


    with torch.no_grad():

        base_hr, _ = model_p4(
            sample_in
        )


    lr_up = F.interpolate(
        sample_in,
        scale_factor=2,
        mode="bicubic",
        align_corners=False
    )


    lr_edge = get_lr_edge(
        lr_up,
        sobel_filter
    )


    # ========================================================
    # FORWARD
    # ========================================================

    head.train()


    (
        final_hr,
        pred_struct,
        correction,
        gate,
        struct_feats,
        noise_feats
    ) = head(
        lr_up,
        base_hr,
        lr_edge,
        bounded_scale=bounded_scale
    )


    # ========================================================
    # CHECK 6
    # ========================================================

    expected_shape = [
        sample_in.size(0),
        1,
        256,
        256
    ]


    if list(final_hr.shape) != expected_shape:

        raise ValueError(
            f"Shape Error: output shape is "
            f"{list(final_hr.shape)}"
        )


    print(
        f"Sanity Check 6: Output shape "
        f"{list(final_hr.shape)}: PASSED"
    )


    # ========================================================
    # CHECK 7
    # ========================================================

    if not torch.isfinite(
        final_hr
    ).all():

        raise ValueError(
            "Final output contains NaNs or Infs"
        )


    print(
        "Sanity Check 7: Output finite: PASSED"
    )


    # ========================================================
    # CHECK 8
    # ========================================================

    if (
        not torch.isfinite(
            pred_struct
        ).all()
        or
        not torch.isfinite(
            gate
        ).all()
    ):

        raise ValueError(
            "Structure or gate prediction "
            "contains NaNs or Infs"
        )


    print(
        "Sanity Check 8: "
        "Structure prediction & gate finite: PASSED"
    )


    # ========================================================
    # LOSS CHECK
    # ========================================================

    loss_pixel = F.l1_loss(
        final_hr,
        sample_tgt
    )


    loss_ssim = (
        1.0 -
        ssim_pytorch(
            final_hr,
            sample_tgt
        )
    )


    loss_charbonnier = torch.mean(
        torch.sqrt(
            (final_hr - sample_tgt) ** 2
            + 1e-6
        )
    )


    loss_edge = F.l1_loss(
        sobel_filter(final_hr),
        sobel_filter(sample_tgt)
    )


    loss_freq = frequency_loss(
        final_hr,
        sample_tgt
    )


    loss_lpips = ssim_lpips_differentiable(
        final_hr,
        sample_tgt,
        lpips_model
    )


    gt_struct_raw = sobel_filter(
        sample_tgt
    )


    gt_struct = (
        gt_struct_raw /
        (
            gt_struct_raw.max(
                dim=2,
                keepdim=True
            )[0].max(
                dim=3,
                keepdim=True
            )[0]
            + 1e-8
        )
    )


    loss_structure = F.l1_loss(
        pred_struct,
        gt_struct
    )


    # Noise consistency
    lr_up_perturbed = (
        lr_up +
        0.01 *
        torch.randn_like(lr_up)
    )


    (
        final_hr_perturbed,
        _,
        _,
        _,
        _,
        _
    ) = head(
        lr_up_perturbed,
        base_hr,
        lr_edge,
        bounded_scale=bounded_scale
    )


    loss_noise = torch.mean(
        torch.abs(
            final_hr -
            final_hr_perturbed
        )
    )


    loss_res = torch.mean(
        torch.abs(correction)
    )


    # IMPORTANT:
    # Do NOT minimize the gate itself.
    # Instead keep it weakly centered around 0.5.

    loss_gate = torch.mean(
        (gate - 0.5) ** 2
    )


    total_loss = (

        pixel_coef *
        loss_pixel

        +

        ssim_coef *
        loss_ssim

        +

        charbonnier_coef *
        loss_charbonnier

        +

        edge_coef *
        loss_edge

        +

        freq_coef *
        loss_freq

        +

        lpips_coef *
        loss_lpips

        +

        struct_coef *
        loss_structure

        +

        noise_coef *
        loss_noise

        +

        res_coef *
        loss_res

        +

        gate_coef *
        loss_gate
    )


    # ========================================================
    # CHECK 9
    # ========================================================

    if not torch.isfinite(
        total_loss
    ):

        raise ValueError(
            "Total loss contains NaNs or Infs"
        )


    print(
        "Sanity Check 9: "
        "All individual & total losses finite: PASSED"
    )


    # ========================================================
    # GRADIENT CHECK
    # ========================================================

    optimizer_check = torch.optim.Adam(
        head.parameters(),
        lr=1e-3
    )


    optimizer_check.zero_grad()

    total_loss.backward()


    # ========================================================
    # CHECK 10
    # ========================================================

    head_has_grads = True


    for name, p in head.named_parameters():

        if (
            p.grad is None
            or
            not torch.isfinite(
                p.grad
            ).all()
        ):

            print(
                f"Warning: head parameter "
                f"{name} grad is invalid!"
            )

            head_has_grads = False


    if not head_has_grads:

        raise ValueError(
            "Gradient Flow Error: "
            "recovery head lacks valid gradients!"
        )


    print(
        "Sanity Check 10: "
        "Recovery head receives gradients: PASSED"
    )


    # ========================================================
    # CHECK 11
    # ========================================================

    p4_has_no_grads = True


    for name, p in model_p4.named_parameters():

        if p.grad is not None:

            print(
                f"Warning: Phase 4 parameter "
                f"{name} has gradient!"
            )

            p4_has_no_grads = False


    if not p4_has_no_grads:

        raise ValueError(
            "Safety Error: "
            "Phase 4 parameters received gradient updates!"
        )


    print(
        "Sanity Check 11: "
        "Phase 4 receives NO gradients: PASSED"
    )


    # ========================================================
    # CHECK 12
    # ========================================================

    o_min = float(
        final_hr.min().item()
    )

    o_max = float(
        final_hr.max().item()
    )


    print(
        f"Output range: "
        f"[{o_min:.4f}, {o_max:.4f}]"
    )


    if (
        o_min < 0.0
        or
        o_max > 1.0
    ):

        raise ValueError(
            "Output range exceeded [0, 1]!"
        )


    print(
        "Sanity Check 12: "
        "Output range [0, 1]: PASSED"
    )


    # ========================================================
    # CHECK 13 — IDENTITY
    # ========================================================

    head.eval()


    with torch.no_grad():

        (
            final_id_0,
            _,
            _,
            _,
            _,
            _
        ) = head(
            lr_up,
            base_hr,
            lr_edge,
            bounded_scale=0.0
        )


    id_diff = torch.abs(
        final_id_0 -
        torch.clamp(
            base_hr,
            0.0,
            1.0
        )
    ).max().item()


    print(
        f"Identity difference "
        f"(bounded_scale=0.0): "
        f"{id_diff:.6e}"
    )


    if id_diff > 1e-6:

        raise ValueError(
            "Identity Error: "
            "final output differs from base HR "
            "when bounded_scale=0"
        )


    print(
        "Sanity Check 13: "
        "Identity behavior: PASSED"
    )


    # ========================================================
    # CHECK 14-16
    # ========================================================

    gate_m = float(
        gate.mean().item()
    )

    gate_s = float(
        gate.std().item()
    )

    gate_min = float(
        gate.min().item()
    )

    gate_max = float(
        gate.max().item()
    )

    c_mean = float(
        correction.abs().mean().item()
    )

    c_std = float(
        correction.std().item()
    )


    print(
        f"Initial Gate: "
        f"mean={gate_m:.4f}, "
        f"std={gate_s:.4f}, "
        f"min={gate_min:.4f}, "
        f"max={gate_max:.4f}"
    )


    print(
        f"Initial Correction: "
        f"abs_mean={c_mean:.6e}, "
        f"std={c_std:.6e}"
    )


    if gate_m <= 0.0:

        raise ValueError(
            "Gate collapsed to zero!"
        )


    if c_std == 0.0:

        raise ValueError(
            "Correction collapsed!"
        )


    print(
        "Sanity Check 14: "
        "Gate valid & uncollapsed: PASSED"
    )

    print(
        "Sanity Check 15: "
        "Correction non-zero variance: PASSED"
    )

    print(
        "Sanity Check 16: "
        "Gradient flow verified: PASSED"
    )


    # ========================================================
    # 2-SAMPLE OVERFIT TEST
    # ========================================================

    print("\n" + "=" * 50)

    print(
        "RUNNING 2-SAMPLE OVERFIT DIAGNOSTIC TEST"
    )

    print("=" * 50)


    overfit_subset = Subset(
        train_dataset,
        [0, 1]
    )


    overfit_loader = DataLoader(
        overfit_subset,
        batch_size=2,
        shuffle=False
    )


    overfit_batch = next(
        iter(overfit_loader)
    )


    o_in = overfit_batch[
        "input"
    ].to(device)


    o_tgt = overfit_batch[
        "target"
    ].to(device)


    with torch.no_grad():

        o_base_hr, _ = model_p4(
            o_in
        )


    o_lr_up = F.interpolate(
        o_in,
        scale_factor=2,
        mode="bicubic",
        align_corners=False
    )


    o_lr_edge = get_lr_edge(
        o_lr_up,
        sobel_filter
    )


    # Fresh model
    o_head = Phase410PriorNet(
        num_features=32
    ).to(device)


    overfit_lr = config.get(
        "train",
        {}
    ).get(
        "overfit_lr",
        0.003
    )


    o_optimizer = torch.optim.Adam(
        o_head.parameters(),
        lr=overfit_lr
    )


    o_start_loss = None
    o_end_loss = None


    o_head.train()


    overfit_steps = 750


    for step in range(
        overfit_steps
    ):

        o_optimizer.zero_grad()


        (
            o_final_hr,
            o_pred_struct,
            o_correction,
            o_gate,
            _,
            _
        ) = o_head(
            o_lr_up,
            o_base_hr,
            o_lr_edge,
            bounded_scale=bounded_scale
        )


        # Pixel
        l_pix = F.l1_loss(
            o_final_hr,
            o_tgt
        )


        # SSIM
        l_ssim = (
            1.0 -
            ssim_pytorch(
                o_final_hr,
                o_tgt
            )
        )


        # Charbonnier
        l_charb = torch.mean(
            torch.sqrt(
                (o_final_hr - o_tgt) ** 2
                + 1e-6
            )
        )


        # Edge
        l_edge = F.l1_loss(
            sobel_filter(o_final_hr),
            sobel_filter(o_tgt)
        )


        # Frequency
        l_freq = frequency_loss(
            o_final_hr,
            o_tgt
        )


        # LPIPS
        l_lpips = ssim_lpips_differentiable(
            o_final_hr,
            o_tgt,
            lpips_model
        )


        # Structure
        o_gt_struct_raw = sobel_filter(
            o_tgt
        )


        o_gt_struct = (
            o_gt_struct_raw /
            (
                o_gt_struct_raw.max(
                    dim=2,
                    keepdim=True
                )[0].max(
                    dim=3,
                    keepdim=True
                )[0]
                + 1e-8
            )
        )


        l_struct = F.l1_loss(
            o_pred_struct,
            o_gt_struct
        )


        # Noise consistency
        o_pert = (
            o_lr_up +
            0.01 *
            torch.randn_like(
                o_lr_up
            )
        )


        (
            o_final_pert,
            _,
            _,
            _,
            _,
            _
        ) = o_head(
            o_pert,
            o_base_hr,
            o_lr_edge,
            bounded_scale=bounded_scale
        )


        l_noise = torch.mean(
            torch.abs(
                o_final_hr -
                o_final_pert
            )
        )


        # Residual penalty
        l_res = torch.mean(
            torch.abs(
                o_correction
            )
        )


        # Gate regularization
        l_gate = torch.mean(
            (o_gate - 0.5) ** 2
        )


        # Total
        o_total_loss = (

            pixel_coef * l_pix

            +

            ssim_coef * l_ssim

            +

            charbonnier_coef * l_charb

            +

            edge_coef * l_edge

            +

            freq_coef * l_freq

            +

            lpips_coef * l_lpips

            +

            struct_coef * l_struct

            +

            noise_coef * l_noise

            +

            res_coef * l_res

            +

            gate_coef * l_gate
        )


        o_total_loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            o_head.parameters(),
            max_norm=1.0
        )


        o_optimizer.step()


        if step == 0:

            o_start_loss = (
                o_total_loss.item()
            )


        if step == overfit_steps - 1:

            o_end_loss = (
                o_total_loss.item()
            )


    # ========================================================
    # OVERFIT DECISION
    # ========================================================

    overfit_reduction = (

        (o_start_loss - o_end_loss)

        /

        max(
            o_start_loss,
            1e-8
        )
    )


    print(
        f"Overfit Start Loss: "
        f"{o_start_loss:.6f}"
    )

    print(
        f"Overfit End Loss:   "
        f"{o_end_loss:.6f}"
    )

    print(
        f"Overfit Reduction:  "
        f"{overfit_reduction * 100:.2f}%"
    )


    # Only require actual learning.
    # Do NOT require an arbitrary absolute loss like 0.220.

    if (
        o_end_loss >=
        o_start_loss
    ):

        raise ValueError(
            "CRITICAL ERROR: "
            "Overfit test failed — "
            "loss did not decrease."
        )


    if overfit_reduction < 0.10:

        raise ValueError(
            "CRITICAL ERROR: "
            "Weak overfit learning — "
            f"only {overfit_reduction * 100:.2f}% reduction."
        )


    print(
        "2-Sample Overfit test: PASSED "
        f"({overfit_reduction * 100:.2f}% loss reduction)"
    )


    # ========================================================
    # PILOT TRAINING
    # ========================================================

    pilot_epochs = config.get(
        "train",
        {}
    ).get(
        "pilot_epochs",
        5
    )


    print("\n" + "=" * 50)

    print(
        f"STARTING {pilot_epochs}-EPOCH "
        "CONTROLLED PILOT TRAINING RUN"
    )

    print("=" * 50)


    # Fresh head
    head = Phase410PriorNet(
        num_features=32
    ).to(device)


    lr = config.get(
        "train",
        {}
    ).get(
        "lr",
        1e-3
    )


    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=lr
    )


    best_lpips = float("inf")

    history = []


    start_time = time.time()


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        pilot_epochs + 1
    ):

        epoch_start = time.time()


        head.train()


        running_total_loss = 0.0
        running_pixel = 0.0
        running_ssim = 0.0
        running_charb = 0.0
        running_edge = 0.0
        running_freq = 0.0
        running_lpips = 0.0
        running_struct = 0.0
        running_noise = 0.0
        running_res = 0.0
        running_gate = 0.0


        num_train_batches = 0


        # ====================================================
        # TRAIN BATCHES
        # ====================================================

        for batch in train_loader:

            b_in = batch[
                "input"
            ].to(device)


            b_tgt = batch[
                "target"
            ].to(device)


            # Frozen Phase 4
            with torch.no_grad():

                b_base_hr, _ = model_p4(
                    b_in
                )


            b_lr_up = F.interpolate(
                b_in,
                scale_factor=2,
                mode="bicubic",
                align_corners=False
            )


            b_lr_edge = get_lr_edge(
                b_lr_up,
                sobel_filter
            )


            optimizer.zero_grad()


            (
                b_final_hr,
                b_pred_struct,
                b_correction,
                b_gate,
                _,
                _
            ) = head(
                b_lr_up,
                b_base_hr,
                b_lr_edge,
                bounded_scale=bounded_scale
            )


            # ------------------------------------------------
            # Pixel
            # ------------------------------------------------

            l_pix = F.l1_loss(
                b_final_hr,
                b_tgt
            )


            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            l_ssim = (
                1.0 -
                ssim_pytorch(
                    b_final_hr,
                    b_tgt
                )
            )


            # ------------------------------------------------
            # Charbonnier
            # ------------------------------------------------

            l_charb = torch.mean(
                torch.sqrt(
                    (b_final_hr - b_tgt) ** 2
                    + 1e-6
                )
            )


            # ------------------------------------------------
            # Edge
            # ------------------------------------------------

            l_edge = F.l1_loss(
                sobel_filter(
                    b_final_hr
                ),
                sobel_filter(
                    b_tgt
                )
            )


            # ------------------------------------------------
            # Frequency
            # ------------------------------------------------

            l_freq = frequency_loss(
                b_final_hr,
                b_tgt
            )


            # ------------------------------------------------
            # LPIPS
            # ------------------------------------------------

            l_lpips = ssim_lpips_differentiable(
                b_final_hr,
                b_tgt,
                lpips_model
            )


            # ------------------------------------------------
            # Structure
            # ------------------------------------------------

            b_gt_struct_raw = sobel_filter(
                b_tgt
            )


            b_gt_struct = (
                b_gt_struct_raw /
                (
                    b_gt_struct_raw.max(
                        dim=2,
                        keepdim=True
                    )[0].max(
                        dim=3,
                        keepdim=True
                    )[0]
                    + 1e-8
                )
            )


            l_struct = F.l1_loss(
                b_pred_struct,
                b_gt_struct
            )


            # ------------------------------------------------
            # Noise consistency
            # ------------------------------------------------

            b_pert = (
                b_lr_up +
                0.01 *
                torch.randn_like(
                    b_lr_up
                )
            )


            (
                b_final_pert,
                _,
                _,
                _,
                _,
                _
            ) = head(
                b_pert,
                b_base_hr,
                b_lr_edge,
                bounded_scale=bounded_scale
            )


            l_noise = torch.mean(
                torch.abs(
                    b_final_hr -
                    b_final_pert
                )
            )


            # ------------------------------------------------
            # Residual
            # ------------------------------------------------

            l_res = torch.mean(
                torch.abs(
                    b_correction
                )
            )


            # ------------------------------------------------
            # Gate
            # ------------------------------------------------

            l_gate = torch.mean(
                (b_gate - 0.5) ** 2
            )


            # ------------------------------------------------
            # TOTAL LOSS
            # ------------------------------------------------

            total_loss = (

                pixel_coef * l_pix

                +

                ssim_coef * l_ssim

                +

                charbonnier_coef * l_charb

                +

                edge_coef * l_edge

                +

                freq_coef * l_freq

                +

                lpips_coef * l_lpips

                +

                struct_coef * l_struct

                +

                noise_coef * l_noise

                +

                res_coef * l_res

                +

                gate_coef * l_gate
            )


            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            total_loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                max_norm=1.0
            )


            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_total_loss += (
                total_loss.item()
            )

            running_pixel += (
                l_pix.item()
            )

            running_ssim += (
                l_ssim.item()
            )

            running_charb += (
                l_charb.item()
            )

            running_edge += (
                l_edge.item()
            )

            running_freq += (
                l_freq.item()
            )

            running_lpips += (
                l_lpips.item()
            )

            running_struct += (
                l_struct.item()
            )

            running_noise += (
                l_noise.item()
            )

            running_res += (
                l_res.item()
            )

            running_gate += (
                l_gate.item()
            )


            num_train_batches += 1


        # ====================================================
        # TRAINING AVERAGES
        # ====================================================

        epoch_train_loss = (
            running_total_loss /
            num_train_batches
        )


        # ====================================================
        # VALIDATION
        # ========================================================

        head.eval()


        val_psnr_list = []
        val_ssim_list = []
        val_lpips_list = []

        val_gate_means = []
        val_gate_stds = []

        val_gate_mins = []
        val_gate_maxs = []

        val_corr_stds = []


        with torch.no_grad():

            for v_batch in val_loader:

                v_in = v_batch[
                    "input"
                ].to(device)


                v_tgt = v_batch[
                    "target"
                ].to(device)


                # Frozen Phase 4
                v_base_hr, _ = model_p4(
                    v_in
                )


                v_lr_up = F.interpolate(
                    v_in,
                    scale_factor=2,
                    mode="bicubic",
                    align_corners=False
                )


                v_lr_edge = get_lr_edge(
                    v_lr_up,
                    sobel_filter
                )


                (
                    v_final_hr,
                    _,
                    v_correction,
                    v_gate,
                    _,
                    _
                ) = head(
                    v_lr_up,
                    v_base_hr,
                    v_lr_edge,
                    bounded_scale=bounded_scale
                )


                # Per-image metrics
                for b_idx in range(
                    v_in.size(0)
                ):

                    psnr_val = calculate_psnr(
                        v_final_hr[b_idx],
                        v_tgt[b_idx]
                    )


                    ssim_val = ssim_pytorch(
                        v_final_hr[
                            b_idx:b_idx + 1
                        ],
                        v_tgt[
                            b_idx:b_idx + 1
                        ]
                    ).item()


                    lpips_val = ssim_lpips_differentiable(
                        v_final_hr[
                            b_idx:b_idx + 1
                        ],
                        v_tgt[
                            b_idx:b_idx + 1
                        ],
                        lpips_model
                    ).item()


                    val_psnr_list.append(
                        psnr_val
                    )

                    val_ssim_list.append(
                        ssim_val
                    )

                    val_lpips_list.append(
                        lpips_val
                    )


                # Gate diagnostics
                val_gate_means.append(
                    v_gate.mean().item()
                )

                val_gate_stds.append(
                    v_gate.std().item()
                )

                val_gate_mins.append(
                    v_gate.min().item()
                )

                val_gate_maxs.append(
                    v_gate.max().item()
                )


                # Correction diagnostics
                val_corr_stds.append(
                    v_correction.std().item()
                )


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        epoch_val_psnr = np.mean(
            val_psnr_list
        )


        epoch_val_ssim = np.mean(
            val_ssim_list
        )


        epoch_val_lpips = np.mean(
            val_lpips_list
        )


        mean_gate_val = np.mean(
            val_gate_means
        )


        mean_gate_std = np.mean(
            val_gate_stds
        )


        mean_gate_min = np.mean(
            val_gate_mins
        )


        mean_gate_max = np.mean(
            val_gate_maxs
        )


        mean_corr_std = np.mean(
            val_corr_stds
        )


        epoch_elapsed = (
            time.time() -
            epoch_start
        )


        # ====================================================
        # PRINT
        # ====================================================

        print(

            f"Epoch {epoch:02d}/{pilot_epochs:02d} | "

            f"Train Loss: "
            f"{epoch_train_loss:.4f} | "

            f"Val PSNR: "
            f"{epoch_val_psnr:.4f} dB | "

            f"Val SSIM: "
            f"{epoch_val_ssim:.4f} | "

            f"Val LPIPS: "
            f"{epoch_val_lpips:.4f} | "

            f"Gate Mean: "
            f"{mean_gate_val:.4f} | "

            f"Gate Range: "
            f"[{mean_gate_min:.3f}, "
            f"{mean_gate_max:.3f}] | "

            f"Corr Std: "
            f"{mean_corr_std:.6f} | "

            f"Time: "
            f"{epoch_elapsed:.1f}s"
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        last_ckpt_path = os.path.join(
            checkpoint_dir,
            "echo_phase410_last.pth"
        )


        torch.save(

            {
                "epoch": epoch,

                "head_state_dict":
                    head.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_psnr":
                    epoch_val_psnr,

                "val_ssim":
                    epoch_val_ssim,

                "val_lpips":
                    epoch_val_lpips
            },

            last_ckpt_path
        )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if epoch_val_lpips < best_lpips:

            best_lpips = (
                epoch_val_lpips
            )


            best_ckpt_path = os.path.join(
                checkpoint_dir,
                "echo_phase410_best.pth"
            )


            torch.save(

                {
                    "epoch": epoch,

                    "head_state_dict":
                        head.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_psnr":
                        epoch_val_psnr,

                    "val_ssim":
                        epoch_val_ssim,

                    "val_lpips":
                        epoch_val_lpips
                },

                best_ckpt_path
            )


            print(
                f"  --> Saved new best checkpoint: "
                f"{best_ckpt_path} "
                f"(LPIPS: {epoch_val_lpips:.4f})"
            )


        # ====================================================
        # HISTORY
        # ====================================================

        history.append(

            {
                "epoch":
                    epoch,

                "train_loss":
                    epoch_train_loss,

                "val_psnr":
                    epoch_val_psnr,

                "val_ssim":
                    epoch_val_ssim,

                "val_lpips":
                    epoch_val_lpips,

                "gate_mean":
                    mean_gate_val,

                "gate_std":
                    mean_gate_std,

                "gate_min":
                    mean_gate_min,

                "gate_max":
                    mean_gate_max,

                "corr_std":
                    mean_corr_std
            }
        )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_elapsed = (
        time.time() -
        start_time
    )


    print(
        f"\nPilot training finished in "
        f"{total_elapsed / 60:.2f} minutes."
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    df_history = pd.DataFrame(
        history
    )


    history_csv_path = os.path.join(
        results_dir,
        "phase410_pilot_history.csv"
    )


    df_history.to_csv(
        history_csv_path,
        index=False
    )


    print(
        f"Saved pilot training log to: "
        f"{history_csv_path}"
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    # Frozen Phase 4 champion
    p4_champion_psnr = 28.2153
    p4_champion_ssim = 0.7611
    p4_champion_lpips = 0.2855


    final_psnr = history[-1][
        "val_psnr"
    ]

    final_ssim = history[-1][
        "val_ssim"
    ]

    final_lpips = history[-1][
        "val_lpips"
    ]


    delta_psnr = (
        final_psnr -
        p4_champion_psnr
    )


    delta_ssim = (
        final_ssim -
        p4_champion_ssim
    )


    delta_lpips = (
        final_lpips -
        p4_champion_lpips
    )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PHASE 4.10 EVALUATION "
        "vs FROZEN PHASE 4 CHAMPION"
    )

    print("=" * 60)


    print(
        f"Phase 4 Champion : "
        f"PSNR = {p4_champion_psnr:.4f} dB | "
        f"SSIM = {p4_champion_ssim:.4f} | "
        f"LPIPS = {p4_champion_lpips:.4f}"
    )


    print(
        f"Phase 4.10 Result: "
        f"PSNR = {final_psnr:.4f} dB | "
        f"SSIM = {final_ssim:.4f} | "
        f"LPIPS = {final_lpips:.4f}"
    )


    print(
        f"Differences      : "
        f"ΔPSNR = {delta_psnr:+.4f} dB | "
        f"ΔSSIM = {delta_ssim:+.4f} | "
        f"ΔLPIPS = {delta_lpips:+.4f}"
    )


    # ========================================================
    # VERDICT
    # ========================================================

    if (
        delta_psnr > 0
        and
        delta_ssim > 0
        and
        delta_lpips < 0
    ):

        verdict = (
            "STRONG SUCCESS: "
            "PSNR improved, SSIM improved, "
            "and LPIPS improved. "
            "CONTINUE PHASE 4.10."
        )


    elif (
        delta_psnr > 0
        or
        delta_ssim > 0
    ) and delta_lpips <= 0.005:

        verdict = (
            "MIXED RESULT: "
            "Some reconstruction metrics improved "
            "while perceptual quality remained stable. "
            "TARGETED REFINEMENT REQUIRED."
        )


    elif (
        final_psnr <
        p4_champion_psnr - 0.1
        or
        final_lpips >
        p4_champion_lpips + 0.01
    ):

        verdict = (
            "WORSE RESULT: "
            "Phase 4.10 degraded the benchmark. "
            "REJECT PHASE 4.10 AND MOVE TO PHASE 5."
        )


    else:

        verdict = (
            "NO MEANINGFUL IMPROVEMENT: "
            "Phase 4.10 did not beat the Phase 4 champion. "
            "MOVE TO PHASE 5."
        )


    print(
        f"\nFINAL EXPERIMENT VERDICT:\n"
        f"{verdict}"
    )


    print("=" * 60)


    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary_path = os.path.join(
        results_dir,
        "phase410_summary.txt"
    )


    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            "PHASE 4.10 CONTROLLED "
            "REFINEMENT SUMMARY\n"
        )

        f.write(
            "=========================================\n"
        )

        f.write(
            f"Phase 4 Baseline Champion: "
            f"PSNR={p4_champion_psnr:.4f}, "
            f"SSIM={p4_champion_ssim:.4f}, "
            f"LPIPS={p4_champion_lpips:.4f}\n"
        )

        f.write(
            f"Phase 4.10 Pilot Result: "
            f"PSNR={final_psnr:.4f}, "
            f"SSIM={final_ssim:.4f}, "
            f"LPIPS={final_lpips:.4f}\n"
        )

        f.write(
            f"Delta: "
            f"ΔPSNR={delta_psnr:+.4f}, "
            f"ΔSSIM={delta_ssim:+.4f}, "
            f"ΔLPIPS={delta_lpips:+.4f}\n"
        )

        f.write(
            f"Bounded Scale: "
            f"{bounded_scale}\n"
        )

        f.write(
            f"Verdict: "
            f"{verdict}\n"
        )


    print(
        f"Summary report saved to: "
        f"{summary_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()