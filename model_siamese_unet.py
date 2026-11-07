"""
================================================================================
SIAMESE U-NET / FC-Siam-diff: Fully Convolutional Siamese Networks for Change Detection
================================================================================

Reference Publication:
    Daudt, R. C., Le Saux, B., & Boulch, A. (2018).
    "Fully Convolutional Siamese Networks for Change Detection."
    In 2018 25th IEEE International Conference on Image Processing (ICIP), pp. 4063-4067.
    DOI: 10.1109/ICIP.2018.8451652

Official GitHub Repository:
    https://github.com/rcdaudt/fully_convolutional_change_detection (siamunet_diff.py)

Architectural Overview:
    - Shared Dual-Branch Convolutional Encoder: Processes pre-fire (t1) and post-fire (t2)
      images with identical shared weights.
    - Feature Differencing at Skip Connections: At each spatial resolution level k:
          d_k = |f_k(t1) - f_k(t2)|
    - Transposed-Convolution Decoder: Reconstructs high-resolution change probability maps
      by concatenating upsampled features with the absolute feature differences d_k.
    - Configurable Input Channels: Supports Landsat 8 multispectral bands (e.g. 6 bands: B2-B7),
      RGB (3 bands), or single-channel NBR.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic 2x Conv2d + BatchNorm + ReLU block with Dropout."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = nn.Dropout2d(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do2 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.do1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.do2(F.relu(self.bn2(self.conv2(x)), inplace=True))
        return x


class DecoderBlock(nn.Module):
    """Upsampling Transposed Conv + Concat with Skip Difference + Conv refinement."""
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.2):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # Concatenation of upsampled features and skip difference
        total_in = in_channels + skip_channels
        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.do1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do2 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip_diff):
        x = self.upconv(x)
        # Ensure spatial dimension match if padding disparity exists
        if x.shape[2:] != skip_diff.shape[2:]:
            x = F.interpolate(x, size=skip_diff.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_diff], dim=1)
        x = self.do1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        x = self.do2(F.relu(self.bn2(self.conv2(x)), inplace=True))
        return x


class SiameseUNetDiff(nn.Module):
    """
    FC-Siam-diff (Fully Convolutional Siamese Difference) Architecture.
    
    Args:
        in_channels (int): Number of spectral channels per image (default: 6 for Landsat B2-B7).
        num_classes (int): Number of target classes (default: 1 for binary burn scar detection).
        base_channels (int): Number of filters in the initial layer (default: 16 as in Daudt et al.).
        dropout_rate (float): Dropout probability for regularization (default: 0.2).
    """
    def __init__(self, in_channels=6, num_classes=1, base_channels=16, dropout_rate=0.2):
        super(SiameseUNetDiff, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        c1 = base_channels       # 16
        c2 = base_channels * 2   # 32
        c3 = base_channels * 4   # 64
        c4 = base_channels * 8   # 128

        # --- Weight-Shared Encoder Stages ---
        self.enc1 = ConvBlock(in_channels, c1, dropout_rate)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(c1, c2, dropout_rate)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # --- Bridge / Center ---
        self.center = ConvBlock(c4, c4, dropout_rate)

        # --- Decoder Stages with Difference Skip Connections ---
        self.dec4 = DecoderBlock(c4, c4, c3, dropout_rate)
        self.dec3 = DecoderBlock(c3, c3, c2, dropout_rate)
        self.dec2 = DecoderBlock(c2, c2, c1, dropout_rate)
        self.dec1 = DecoderBlock(c1, c1, c1, dropout_rate)

        # Final 1x1 classification head
        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward_encoder(self, x):
        """Pass one temporal image through the shared encoder."""
        s1 = self.enc1(x)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        return s1, s2, s3, s4, p4

    def forward(self, x1, x2):
        """
        Forward bitemporal image pair.
        Args:
            x1 (Tensor): Pre-change image [Batch, in_channels, H, W]
            x2 (Tensor): Post-change image [Batch, in_channels, H, W]
        Returns:
            Tensor: Change prediction logits [Batch, num_classes, H, W]
        """
        # 1. Feature extraction through shared encoder
        s1_t1, s2_t1, s3_t1, s4_t1, p4_t1 = self.forward_encoder(x1)
        s1_t2, s2_t2, s3_t2, s4_t2, p4_t2 = self.forward_encoder(x2)

        # 2. Multi-scale feature differencing for skip connections: d_k = |s_k(t1) - s_k(t2)|
        diff1 = torch.abs(s1_t1 - s1_t2)
        diff2 = torch.abs(s2_t1 - s2_t2)
        diff3 = torch.abs(s3_t1 - s3_t2)
        diff4 = torch.abs(s4_t1 - s4_t2)

        # 3. Center differencing & processing
        diff_center = torch.abs(p4_t1 - p4_t2)
        center_out = self.center(diff_center)

        # 4. Decoder with difference skip connections
        d4 = self.dec4(center_out, diff4)
        d3 = self.dec3(d4, diff3)
        d2 = self.dec2(d3, diff2)
        d1 = self.dec1(d2, diff1)

        # 5. Final projection
        logits = self.final_conv(d1)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme foreground-background class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets: [B, 1, H, W] or [B, H, W], float
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


if __name__ == "__main__":
    print("=" * 70)
    print("Testing SiameseUNetDiff (FC-Siam-diff) from Daudt et al. (ICIP 2018)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseUNetDiff(in_channels=6, num_classes=1, base_channels=16).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Create dummy bitemporal tiles matching the Final_with_India dataset
    # Shape: [Batch Size = 2, Channels = 6, Height = 256, Width = 256]
    pre_img = torch.randn(2, 6, 256, 256, device=device)
    post_img = torch.randn(2, 6, 256, 256, device=device)
    gt_mask = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    # Forward pass
    logits = model(pre_img, post_img)
    probs = torch.sigmoid(logits)
    pred_binary = (probs > 0.5).int()

    print(f"Input Pre-Image Shape  : {pre_img.shape}")
    print(f"Input Post-Image Shape : {post_img.shape}")
    print(f"Output Logits Shape    : {logits.shape}")
    print(f"Output Probs Range     : [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Predicted Binary Shape : {pred_binary.shape}")

    # Loss computation check
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, gt_mask)
    print(f"Focal Loss Output      : {loss.item():.4f}")
    loss.backward()
    print("Backward pass verified successfully.")
    print("=" * 70)
