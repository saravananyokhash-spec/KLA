import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientLoss(nn.Module):
    """Computes differentiable L1 difference on image gradients (horizontal and vertical)"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Horizontal gradients
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Vertical gradients
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 difference
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y

class SSIMLoss(nn.Module):
    """Computes differentiable SSIM loss using a uniform box filter"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Pad inputs slightly to preserve shape after average pooling
        pad = self.window_size // 2
        
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=pad)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Local variances and covariance
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=pad) - mu1_mu2
        
        # Prevent small negative values in variance calculations due to float precision
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
        
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        return 1.0 - ssim_map.mean()

class LaplacianLoss(nn.Module):
    """Computes differentiable L1 difference on Laplacian filtered outputs"""
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ]).unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        kernel = self.kernel.to(device=pred.device, dtype=pred.dtype)
        pred_lap = F.conv2d(pred, kernel, padding=1)
        target_lap = F.conv2d(target, kernel, padding=1)
        return F.l1_loss(pred_lap, target_lap)

class ECHOLoss(nn.Module):
    """Composite loss function: L1 + Gradient Loss + SSIM Loss + Laplacian Loss"""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights if weights is not None else {"pixel": 1.0, "edge": 0.1, "ssim": 0.1, "hf": 0.05}
        self.l1_loss = nn.L1Loss()
        self.gradient_loss = GradientLoss()
        self.ssim_loss = SSIMLoss()
        self.laplacian_loss = LaplacianLoss()
        
    def forward(self, pred, target, hf_res=None):
        loss_pixel = self.l1_loss(pred, target)
        loss_edge = self.gradient_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_hf = self.laplacian_loss(pred, target)
        
        loss_hf_reg = 0.0
        if hf_res is not None:
            loss_hf_reg = torch.mean(torch.abs(hf_res))
            
        total_loss = (
            self.weights.get("pixel", 1.0) * loss_pixel +
            self.weights.get("edge", 0.0) * loss_edge +
            self.weights.get("ssim", 0.0) * loss_ssim +
            self.weights.get("hf", 0.0) * loss_hf +
            self.weights.get("hf_reg", 0.0) * loss_hf_reg
        )
        
        return total_loss, {
            "loss": total_loss.item(),
            "loss_pixel": loss_pixel.item(),
            "loss_edge": loss_edge.item(),
            "loss_ssim": loss_ssim.item(),
            "loss_hf": loss_hf.item(),
            "loss_hf_reg": loss_hf_reg if isinstance(loss_hf_reg, float) else loss_hf_reg.item()
        }
