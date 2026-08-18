import torch
import torch.nn as nn
from echo_model import BaselineECHOModel

class GatedResidualRefinement(nn.Module):
    """Refinement module that takes a frozen baseline model and adds a
    learnable gated residual. The base model is loaded from the Phase 4
    checkpoint and its parameters are frozen.
    """
    def __init__(self, base_checkpoint_path, num_features=64, num_res_blocks=4, gate_init=0.0):
        super().__init__()
        # Load frozen baseline model
        self.base = BaselineECHOModel(num_features=num_features)
        self.base.load_state_dict(torch.load(base_checkpoint_path, map_location='cpu'))
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

        # Residual correction network
        layers = []
        for _ in range(num_res_blocks):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))  # output single-channel correction
        self.correction_net = nn.Sequential(*layers)

        # Gating network (sigmoid output per‑pixel)
        self.gate = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # Initialise gate to a low value (conservative start)
        nn.init.constant_(self.gate[-2].weight, 0.0)
        nn.init.constant_(self.gate[-2].bias, gate_init)

    def forward(self, x):
        with torch.no_grad():
            base_out = self.base(x)
            # BaselineECHOModel returns (out, E_fused)
            if isinstance(base_out, tuple):
                base_out = base_out[0]
        # Extract shallow features for gating and correction
        shallow_feat = self.base.shallow_relu(self.base.shallow_conv(x))
        gate_map = self.gate(shallow_feat)
        correction = self.correction_net(shallow_feat)
        refined = base_out + gate_map * correction
        return refined, gate_map, correction
