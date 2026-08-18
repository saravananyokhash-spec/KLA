import torch
import torch.nn as nn

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
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=6, model_version="v1", ablation=None):
        super().__init__()
        self.ablation = ablation if ablation is not None else {}
        self.model_version = model_version
        
        if self.model_version == "v3":
            self.hf_residual_conv = nn.Sequential(
                nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features // 2, out_channels, kernel_size=3, padding=1)
            )
            self.hf_gate_conv = nn.Sequential(
                nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features // 2, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            
        # 1. Shallow Feature Extraction
        self.shallow_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.shallow_relu = nn.ReLU(inplace=True)
        
        # 2. Shared Residual Trunk (e.g. 4 blocks)
        trunk_blocks = max(1, num_blocks - 2)
        self.trunk = nn.Sequential(*[ResidualBlock(num_features) for _ in range(trunk_blocks)])
        
        # 3. Main Feature Branch
        self.main_branch = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # 4. Structure / Edge Detail Branch
        self.detail_branch = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # 5. Evidence Gate
        # Gate operates on concatenated features from main and detail branches
        self.evidence_gate = EvidenceGate(num_features * 2)
        
        # 6. Reconstruction Trunk
        self.reconstruction_trunk = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )
        
        # 7. PixelShuffle Upsampler (2x)
        self.upsample_conv = nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # 8. Residual Reconstruction
        self.reconstruct_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Global residual shortcut path (Bicubic upsampling of input to target scale)
        identity = nn.functional.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        
        # Shallow features
        feat = self.shallow_relu(self.shallow_conv(x))
        
        # Shared trunk
        feat_trunk = self.trunk(feat)
        
        # Main branch
        feat_main = self.main_branch(feat_trunk)
        
        # Detail branch
        feat_detail = self.detail_branch(feat_trunk)
        
        # Calculate Evidence Gate map E
        # E is of shape [batch, 1, H, W]
        feat_concat = torch.cat([feat_main, feat_detail], dim=1)
        E = self.evidence_gate(feat_concat)
        
        # Apply ablation flags if configured
        if self.ablation.get("disable_gate", False):
            # Evidence gate is disabled: treat all regions equally with static mix
            E_fused = torch.full_like(E, 0.5)
        else:
            E_fused = E
            
        if self.ablation.get("disable_detail", False):
            # Detail branch is disabled: set edge features weight to 0
            E_fused = torch.zeros_like(E)
            
        # Spatially-varying evidence-weighted feature fusion
        if self.model_version == "v2":
            # Phase 5: Gated frequency/detail residual fusion
            feat_fused = feat_main + E_fused * feat_detail
        else:
            # Phase 4: Gated pathway selection
            feat_fused = (1.0 - E_fused) * feat_main + E_fused * feat_detail
        
        # Reconstruction trunk
        feat_recon = self.reconstruction_trunk(feat_fused)
        
        # Upsampling
        feat_up = self.pixel_shuffle(self.upsample_conv(feat_recon))
        
        # Final prediction (restored residual + global shortcut)
        out = self.reconstruct_conv(feat_up) + identity
        
        if self.model_version == "v3":
            gate = self.hf_gate_conv(feat_up)
            feat_hf_res = self.hf_residual_conv(feat_up)
            out_final = out + gate * feat_hf_res
            return out_final, gate, out, feat_hf_res
        else:
            return out, E_fused

def get_model_info(model):
    """Returns parameter count and size in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return total_params, trainable_params, model_size_mb
