import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDecompositionModule(nn.Module):
    """
    Differentiable 2D Frequency Decomposition Module using Real FFT (rfft2 / irfft2).
    Decomposes a spatial tensor into Low (LF), Mid (MF), and High (HF) frequency components
    such that LF + MF + HF = Input exactly.
    """
    def __init__(self, cutoff_low=0.15, cutoff_high=0.40):
        super(FrequencyDecompositionModule, self).__init__()
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high

    def _get_radial_masks(self, H, W, device):
        # Create normalized frequency coordinate grid [0, 0.5]
        u = torch.fft.fftfreq(H, device=device).view(-1, 1)
        v = torch.fft.rfftfreq(W, device=device).view(1, -1)
        radius = torch.sqrt(u**2 + v**2) # Radius in range [0, ~0.707]

        # Smooth Gaussian-like frequency masks
        mask_lf = torch.exp(- (radius**2) / (2 * (self.cutoff_low**2)))
        mask_hf = 1.0 - torch.exp(- (radius**2) / (2 * (self.cutoff_high**2)))
        mask_mf = 1.0 - mask_lf - mask_hf
        
        # Ensure exact partition of unity
        mask_lf = torch.clamp(mask_lf, 0.0, 1.0)
        mask_hf = torch.clamp(mask_hf, 0.0, 1.0)
        mask_mf = torch.clamp(mask_mf, 0.0, 1.0)
        
        total = mask_lf + mask_mf + mask_hf + 1e-8
        mask_lf = mask_lf / total
        mask_mf = mask_mf / total
        mask_hf = mask_hf / total

        return mask_lf.unsqueeze(0).unsqueeze(0), mask_mf.unsqueeze(0).unsqueeze(0), mask_hf.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """
        Input: x of shape [B, C, H, W]
        Returns: (x_lf, x_mf, x_hf) each of shape [B, C, H, W]
        """
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
    """
    Multi-Scale Spatial Feature Extractor using multiple kernel receptive fields (3x3, 5x5, 7x7).
    """
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
    """
    Dedicated Frequency Processing Branch for Low, Mid, and High frequency bands.
    """
    def __init__(self, in_channels=1, out_channels=32, cutoff_low=0.15, cutoff_high=0.40):
        super(FrequencyBranch, self).__init__()
        self.decomp = FrequencyDecompositionModule(cutoff_low=cutoff_low, cutoff_high=cutoff_high)

        # Low Frequency Sub-network
        self.lf_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )

        # Mid Frequency Sub-network
        self.mf_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1)
        )

        # High Frequency Sub-network (with Tanh constraint to prevent hallucination)
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
    """
    Fuses Multi-Scale Spatial Features, Frequency Features, and Phase 4 Guidance Features.
    """
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
    """
    Phase 5 — Multi-Scale Spatial-Frequency Restoration Network.
    """
    def __init__(self, spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40):
        super(SpatialFrequencyRestorationNet, self).__init__()
        self.spatial_branch = SpatialMultiScaleBranch(in_channels=1, out_channels=spatial_channels)
        self.freq_branch = FrequencyBranch(in_channels=1, out_channels=freq_channels, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
        self.fusion = SpatialFrequencyFusionModule(spatial_ch=spatial_channels, freq_ch=freq_channels, guidance_ch=32, out_ch=fusion_channels)

    def forward(self, lr_up, p4_guidance):
        """
        Inputs:
          lr_up: Bicubic upsampled LR image [B, 1, 256, 256]
          p4_guidance: Frozen Phase 4 predicted HR image [B, 1, 256, 256]
        Outputs:
          final_hr: Phase 5 restored HR image [B, 1, 256, 256]
          x_lf, x_mf, x_hf: Spatial frequency component images [B, 1, 256, 256]
          feat_fused: Fused feature map [B, 64, 256, 256]
        """
        f_spatial = self.spatial_branch(lr_up)
        f_freq, x_lf, x_mf, x_hf = self.freq_branch(lr_up)
        final_hr, feat_fused = self.fusion(f_spatial, f_freq, p4_guidance)

        return final_hr, x_lf, x_mf, x_hf, feat_fused
