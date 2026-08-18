import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two Conv layers and a shortcut connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class BaselineRestorationNet(nn.Module):
    """
    Lightweight Baseline Restoration Network for 2x super-resolution and denoising.
    Input shape: (B, 1, 128, 128)
    Output shape: (B, 1, 256, 256)
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=4):
        super().__init__()
        
        # 1. Feature Extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # 2. Residual Blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        
        # 3. 2x Upsampling (using PixelShuffle)
        # PixelShuffle(2) requires input channels to be out_channels * (scale^2) = out_channels * 4.
        # We will map num_features to num_features * 4, PixelShuffle(2) will upscale spatially and map back to num_features channels.
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # 4. Reconstruction
        self.reconstruct = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        feat = self.feat_extract(x)
        res = self.res_blocks(feat)
        upscaled = self.upsample(res)
        out = self.reconstruct(upscaled)
        return out

def get_model_info(model):
    """
    Returns:
      - total_params: Total number of parameters
      - trainable_params: Number of trainable parameters
      - size_mb: Approximate model size in megabytes (float32 = 4 bytes)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 1 parameter = 4 bytes in float32. 1 MB = 1024 * 1024 bytes
    size_mb = (total_params * 4) / (1024 * 1024)
    return total_params, trainable_params, size_mb

if __name__ == "__main__":
    # If run directly, display model summary
    model = BaselineRestorationNet()
    tot, train, size = get_model_info(model)
    print("Baseline Neural Model Info:")
    print(f"  Total Parameters: {tot:,}")
    print(f"  Trainable Parameters: {train:,}")
    print(f"  Approximate Model Size: {size:.3f} MB")
    
    # Test forward pass shape
    dummy_input = torch.randn(1, 1, 128, 128)
    dummy_output = model(dummy_input)
    print(f"  Input Shape: {dummy_input.shape}")
    print(f"  Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (1, 1, 256, 256), "Output shape should be (1, 1, 256, 256)!"
    print("  Shape test successful!")
