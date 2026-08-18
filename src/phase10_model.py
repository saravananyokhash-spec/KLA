import torch
import torch.nn as nn
import torch.nn.functional as F
from phase5_model import SpatialFrequencyRestorationNet

class DegradationEstimator(nn.Module):
    """
    Lightweight CNN estimator operating on 128x128 input image to predict degradation class probabilities:
    - 0: clean
    - 1: noise only
    - 2: blur only
    - 3: noise+blur combined
    """
    def __init__(self, in_channels=1, num_classes=4):
        super(DegradationEstimator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        return logits


class ConditionedSpatialFrequencyRestorationNet(SpatialFrequencyRestorationNet):
    """
    Degradation-Aware conditioned Spatial Frequency Restoration Network.
    Warm-starts from Phase 9 backbone, applying feature modulation (FiLM) based on the input's estimated degradation.
    """
    def __init__(self, spatial_channels=32, freq_channels=32, fusion_channels=64, cutoff_low=0.15, cutoff_high=0.40, use_conditioning=True):
        super(ConditionedSpatialFrequencyRestorationNet, self).__init__(
            spatial_channels=spatial_channels,
            freq_channels=freq_channels,
            fusion_channels=fusion_channels,
            cutoff_low=cutoff_low,
            cutoff_high=cutoff_high
        )
        self.use_conditioning = use_conditioning
        if self.use_conditioning:
            self.estimator = DegradationEstimator(in_channels=1, num_classes=4)
            # Fully connected layers to predict scale and bias parameter vectors from predicted 4D degradation vector
            self.film_spatial = nn.Linear(4, spatial_channels * 2)
            self.film_freq = nn.Linear(4, freq_channels * 2)
            
    def forward(self, lr_up, p4_guidance, lr_input=None):
        """
        Inputs:
          lr_up: Bicubic upsampled LR image [B, 1, 256, 256]
          p4_guidance: Frozen Phase 4 predicted HR image [B, 1, 256, 256]
          lr_input: Degraded low-resolution input image [B, 1, 128, 128]
        Outputs:
          final_hr: Phase 10 restored HR image [B, 1, 256, 256]
          x_lf, x_mf, x_hf: Spatial frequency component images [B, 1, 256, 256]
          feat_fused: Fused feature map [B, fusion_channels, 256, 256]
          logits: degradation prediction logits (if conditioning is active)
        """
        if self.use_conditioning and lr_input is not None:
            logits = self.estimator(lr_input)
            cond = F.softmax(logits, dim=1)  # shape [B, 4]
            
            # Predict FiLM parameters for spatial branch features
            s_film = self.film_spatial(cond)
            s_gamma, s_beta = torch.split(s_film, self.film_spatial.out_features // 2, dim=1)
            s_gamma = s_gamma.view(s_gamma.size(0), -1, 1, 1)
            s_beta = s_beta.view(s_beta.size(0), -1, 1, 1)
            
            # Predict FiLM parameters for frequency branch features
            f_film = self.film_freq(cond)
            f_gamma, f_beta = torch.split(f_film, self.film_freq.out_features // 2, dim=1)
            f_gamma = f_gamma.view(f_gamma.size(0), -1, 1, 1)
            f_beta = f_beta.view(f_beta.size(0), -1, 1, 1)
        else:
            logits = None
            s_gamma, s_beta = None, None
            f_gamma, f_beta = None, None
            
        # Feature extraction from branches
        f_spatial = self.spatial_branch(lr_up)
        if s_gamma is not None:
            f_spatial = (1.0 + s_gamma) * f_spatial + s_beta
            
        f_freq, x_lf, x_mf, x_hf = self.freq_branch(lr_up)
        if f_gamma is not None:
            f_freq = (1.0 + f_gamma) * f_freq + f_beta
            
        final_hr, feat_fused = self.fusion(f_spatial, f_freq, p4_guidance)
        
        return final_hr, x_lf, x_mf, x_hf, feat_fused, logits
