import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. MODEL ARCHITECTURE CODE (PHASE 4 BASELINE & PHASE 9 RESTORATION NET)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class EvidenceGate(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate_conv(x)

class BaselineECHOModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=6):
        super().__init__()
        # Shallow Feature Extraction
        self.shallow_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.shallow_relu = nn.ReLU(inplace=True)
        
        # Shared Trunk
        trunk_blocks = max(1, num_blocks - 2)
        self.trunk = nn.Sequential(*[ResidualBlock(num_features) for _ in range(trunk_blocks)])
        
        # Main Feature Branch
        self.main_branch = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # Structure / Edge Detail Branch
        self.detail_branch = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # Evidence Gate
        self.evidence_gate = EvidenceGate(num_features * 2)
        
        # Reconstruction Trunk
        self.reconstruction_trunk = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # PixelShuffle Upsampler (2x)
        self.upsample_conv = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Residual Reconstruction
        self.reconstruct_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Global residual shortcut path
        identity = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        
        # Shallow features
        feat = self.shallow_relu(self.shallow_conv(x))
        
        # Shared trunk
        feat_trunk = self.trunk(feat)
        
        # Main branch
        feat_main = self.main_branch(feat_trunk)
        
        # Detail branch
        feat_detail = self.detail_branch(feat_trunk)
        
        # Evidence Gate E
        feat_concat = torch.cat([feat_main, feat_detail], dim=1)
        E = self.evidence_gate(feat_concat)
        
        # Spatially-varying evidence-weighted feature fusion
        feat_fused = (1.0 - E) * feat_main + E * feat_detail
        
        # Reconstruction trunk
        feat_recon = self.reconstruction_trunk(feat_fused)
        
        # Upsampling
        feat_up = self.pixel_shuffle(self.upsample_conv(feat_recon))
        
        # Final output
        out = self.reconstruct_conv(feat_up) + identity
        return out, E


class FrequencyDecompositionModule(nn.Module):
    def __init__(self, cutoff_low=0.15, cutoff_high=0.40):
        super(FrequencyDecompositionModule, self).__init__()
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high

    def _get_radial_masks(self, H, W, device):
        u = torch.fft.fftfreq(H, device=device).view(-1, 1)
        v = torch.fft.rfftfreq(W, device=device).view(1, -1)
        radius = torch.sqrt(u**2 + v**2)

        mask_lf = torch.exp(- (radius**2) / (2 * (self.cutoff_low**2)))
        mask_hf = 1.0 - torch.exp(- (radius**2) / (2 * (self.cutoff_high**2)))
        mask_mf = 1.0 - mask_lf - mask_hf
        
        mask_lf = torch.clamp(mask_lf, 0.0, 1.0)
        mask_hf = torch.clamp(mask_hf, 0.0, 1.0)
        mask_mf = torch.clamp(mask_mf, 0.0, 1.0)
        
        total = mask_lf + mask_mf + mask_hf + 1e-8
        mask_lf = mask_lf / total
        mask_mf = mask_mf / total
        mask_hf = mask_hf / total

        return mask_lf.unsqueeze(0).unsqueeze(0), mask_mf.unsqueeze(0).unsqueeze(0), mask_hf.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm="ortho")
        mask_lf, mask_mf, mask_hf = self._get_radial_masks(H, W, x.device)

        fft_lf = fft_x * mask_lf
        fft_mf = fft_x * mask_mf
        fft_hf = fft_x * mask_hf

        x_lf = torch.fft.irfft2(fft_lf, s=(H, W), norm="ortho")
        x_mf = torch.fft.irfft2(fft_mf, s=(H, W), norm="ortho")
        x_hf = torch.fft.irfft2(fft_hf, s=(H, W), norm="ortho")

        return x_lf, x_mf, x_hf


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res


class SpatialMultiScaleBranch(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(SpatialMultiScaleBranch, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3)

        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.res1 = ResBlock(out_channels)
        self.res2 = ResBlock(out_channels)

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        cat = torch.cat([f3, f5, f7], dim=1)
        out = F.prelu(self.fuse(cat), torch.tensor(0.2, device=x.device))
        out = self.res1(out)
        out = self.res2(out)
        return out


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, cutoff_low=0.15, cutoff_high=0.40):
        super(FrequencyBranch, self).__init__()
        self.decomp = FrequencyDecompositionModule(cutoff_low=cutoff_low, cutoff_high=cutoff_high)

        self.lf_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )

        self.mf_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )

        self.hf_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.fuse_freq = nn.Sequential(
            nn.Conv2d(48, out_channels, kernel_size=1),
            nn.PReLU(out_channels),
            ResBlock(out_channels)
        )

    def forward(self, x):
        x_lf, x_mf, x_hf = self.decomp(x)
        f_lf = self.lf_net(x_lf)
        f_mf = self.mf_net(x_mf)
        f_hf = self.hf_net(x_hf)

        cat_freq = torch.cat([f_lf, f_mf, f_hf], dim=1)
        out_freq = self.fuse_freq(cat_freq)
        return out_freq, x_lf, x_mf, x_hf


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialFrequencyFusionModule(nn.Module):
    def __init__(self, spatial_ch=32, freq_ch=32, guidance_ch=32, out_ch=64):
        super(SpatialFrequencyFusionModule, self).__init__()
        total_in = spatial_ch + freq_ch + guidance_ch
        self.guidance_conv = nn.Conv2d(1, guidance_ch, kernel_size=3, padding=1)

        self.reduce = nn.Conv2d(total_in, out_ch, kernel_size=1)
        self.attn = ChannelAttention(out_ch)
        self.res1 = ResBlock(out_ch)
        self.res2 = ResBlock(out_ch)

        self.reconstruct = nn.Sequential(
            nn.Conv2d(out_ch, 32, kernel_size=3, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, f_spatial, f_freq, p4_img):
        f_p4 = F.prelu(self.guidance_conv(p4_img), torch.tensor(0.2, device=p4_img.device))
        cat_all = torch.cat([f_spatial, f_freq, f_p4], dim=1)

        feat = F.prelu(self.reduce(cat_all), torch.tensor(0.2, device=p4_img.device))
        feat = self.attn(feat)
        feat = self.res1(feat)
        feat = self.res2(feat)

        out_raw = self.reconstruct(feat)
        out_hr = torch.clamp(out_raw, 0.0, 1.0)
        return out_hr, feat


class SpatialFrequencyRestorationNet(nn.Module):
    def __init__(self, spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40):
        super(SpatialFrequencyRestorationNet, self).__init__()
        self.spatial_branch = SpatialMultiScaleBranch(in_channels=1, out_channels=spatial_channels)
        self.freq_branch = FrequencyBranch(in_channels=1, out_channels=freq_channels, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
        self.fusion = SpatialFrequencyFusionModule(spatial_ch=spatial_channels, freq_ch=freq_channels, guidance_ch=32, out_ch=fusion_channels)

    def forward(self, lr_up, p4_guidance):
        f_spatial = self.spatial_branch(lr_up)
        f_freq, x_lf, x_mf, x_hf = self.freq_branch(lr_up)
        final_hr, feat_fused = self.fusion(f_spatial, f_freq, p4_guidance)

        return final_hr, x_lf, x_mf, x_hf, feat_fused

# ==============================================================================
# 2. INFERENCE RUN PIPELINE
# ==============================================================================

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check local checkpoint paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_p4_path = os.path.join(base_dir, "models", "echo_best.pth")
    model_p5_path = os.path.join(base_dir, "models", "phase9.pth")

    if not os.path.exists(model_p4_path):
        print(f"Error: Phase 4 guidance model checkpoint not found at: {model_p4_path}")
        sys.exit(1)
    if not os.path.exists(model_p5_path):
        print(f"Error: Phase 9 champion model checkpoint not found at: {model_p5_path}")
        sys.exit(1)

    # 1. Load Phase 4 Guidance Model
    print("Loading required Phase 4 guidance model for Phase 9 inference...")
    model_p4 = BaselineECHOModel(in_channels=1, out_channels=1, num_features=64, num_blocks=6).to(device)
    p4_chk = torch.load(model_p4_path, map_location=device, weights_only=False)
    p4_state = p4_chk["model_state_dict"] if isinstance(p4_chk, dict) and "model_state_dict" in p4_chk else p4_chk
    model_p4.load_state_dict(p4_state, strict=True)
    model_p4.eval()
    for p in model_p4.parameters():
        p.requires_grad = False
    print("Phase 4 checkpoint loaded successfully.")

    # 2. Load Final Spatial-Frequency Restoration Model (Phase 9 Champion)
    print("Loading Phase 9 champion model...")
    student = SpatialFrequencyRestorationNet(
        spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40
    ).to(device)
    p5_chk = torch.load(model_p5_path, map_location=device, weights_only=False)
    p5_state = p5_chk["model_state_dict"] if isinstance(p5_chk, dict) and "model_state_dict" in p5_chk else p5_chk
    student.load_state_dict(p5_state, strict=True)
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    p9_params = sum(p.numel() for p in student.parameters())
    print(f"Phase 9 checkpoint:\n{os.path.abspath(model_p5_path)}")
    print(f"\nModel parameters:\n{p9_params}")
    print("Phase 9 checkpoint loaded successfully.")
    print("Starting batch inference...")

    # Load input files sorted deterministically (supporting NoisyLR fallback)
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_files:
        fallback_dir = os.path.join(input_dir, "NoisyLR")
        if os.path.exists(fallback_dir):
            input_files = sorted(glob.glob(os.path.join(fallback_dir, "*.npy")))

    if not input_files:
        print(f"Warning: No .npy files found in {input_dir}")
        print("ECHO INFERENCE COMPLETE")
        print("Input files: 0")
        print("Successful outputs: 0")
        print("Failed outputs: 0")
        print(f"Output directory: {output_dir}")
        return

    success_count = 0
    fail_count = 0

    for file_path in input_files:
        fn = os.path.basename(file_path)
        try:
            # Load numpy array
            arr = np.load(file_path)
            
            # Dimensions preprocessing normalization check
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            elif arr.ndim != 2:
                raise ValueError(f"Unsupported array shape: {arr.shape}. Expected 2D grayscale.")
                
            in_h, in_w = arr.shape
            
            # Convert to float32 tensor of shape [1, 1, H, W]
            inp_tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 2x upsampling grid construction
                lr_up = F.interpolate(inp_tensor, scale_factor=2, mode="bicubic", align_corners=False)
                
                # Phase 4 guidance prediction
                p4_raw, _ = model_p4(inp_tensor)
                p4_guidance = torch.clamp(p4_raw, 0.0, 1.0)
                
                # Final restoration
                pred_hr, _, _, _, _ = student(lr_up, p4_guidance)
                pred_hr = torch.clamp(pred_hr, 0.0, 1.0)
                
            # Convert back to numpy
            output = pred_hr.squeeze(0).squeeze(0).cpu().numpy()
            
            # Postprocessing safety checks
            output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
            output = np.clip(output, 0.0, 1.0)
            
            # Verify target resolution (2x scaling check)
            out_h, out_w = output.shape
            if out_h != 2 * in_h or out_w != 2 * in_w:
                raise ValueError(f"Output shape mismatch: got {output.shape}, expected {(2*in_h, 2*in_w)} via 2x scaling.")
                
            # Save array
            np.save(os.path.join(output_dir, fn), output)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing file {fn}: {e}")
            fail_count += 1

    print("\nECHO INFERENCE COMPLETE")
    print(f"Input files: {len(input_files)}")
    print(f"Successful outputs: {success_count}")
    print(f"Failed outputs: {fail_count}")
    print(f"Output directory: {output_dir}")

    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
