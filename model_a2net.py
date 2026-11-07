"""
================================================================================
A2NET: LIGHTWEIGHT REMOTE SENSING CHANGE DETECTION WITH PROGRESSIVE AGGREGATION
AND SUPERVISED ATTENTION
================================================================================

Reference Publication:
    Guan, Y., Ma, A., Zhong, Y., Zheng, Z., & Zhang, L. (2023).
    "Lightweight Remote Sensing Change Detection with Progressive Aggregation
    and Supervised Attention."
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 61, pp. 1-13, 2023.
    DOI: 10.1109/TGRS.2023.3281488

Official GitHub Repository:
    https://github.com/guanyuezhen/A2Net (models/model.py, models/MobileNetV2.py)

Architectural Overview:
    1. Weight-Shared Lightweight Backbone:
       - Uses Inverted Residual blocks (MobileNetV2 architecture) with linear bottlenecks
         to extract multi-scale bitemporal representations across stages C2, C3, C4, and C5.
    2. Neighbor Feature Aggregation Module (NAM):
       - Aggregates contextual features across adjacent encoder scales to bridge semantic
         gaps before temporal difference calculation.
    3. Progressive Change Identifying Module (PCIM):
       - Computes temporal feature differences and extracts change features progressively:
             Delta_S_k = |S_k(t1) - S_k(t2)|
    4. Supervised Attention Module (SAM):
       - Generates attention maps from deeper semantic features to guide and filter out
         spectral noise in shallow high-resolution change features.
    5. Progressive Decoder:
       - Fuses guided multi-scale change representations into full-resolution change predictions.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. MobileNetV2 Backbone Components
# -----------------------------------------------------------------------------

class InvertedResidual(nn.Module):
    """Inverted Residual Block: Expand -> Depthwise Conv -> Linear Bottleneck."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # Pointwise 1x1 conv expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        # Depthwise 3x3 conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise 1x1 conv linear projection
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 feature extractor outputting 4 multi-scale stages."""
    def __init__(self, in_channels=6):
        super(MobileNetV2Backbone, self).__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 16, stride=1, expand_ratio=1)
        )
        # Stage C2: 1/4 resolution (out_channels = 24)
        self.stage_c2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6)
        )
        # Stage C3: 1/8 resolution (out_channels = 32)
        self.stage_c3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6)
        )
        # Stage C4: 1/16 resolution (out_channels = 96)
        self.stage_c4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
        # Stage C5: 1/32 resolution (out_channels = 160)
        self.stage_c5 = nn.Sequential(
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6)
        )

    def forward(self, x):
        x = self.stem(x)       # 1/2
        c2 = self.stage_c2(x)  # 1/4, 24 channels
        c3 = self.stage_c3(c2) # 1/8, 32 channels
        c4 = self.stage_c4(c3) # 1/16, 96 channels
        c5 = self.stage_c5(c4) # 1/32, 160 channels
        return c2, c3, c4, c5


# -----------------------------------------------------------------------------
# 2. Neighbor Feature Aggregation Module (NAM)
# -----------------------------------------------------------------------------

class FeatureFusionModule(nn.Module):
    """Combines neighbor-fused representations with identity feature."""
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NeighborFeatureAggregation(nn.Module):
    """Aggregates features across adjacent temporal encoder stages."""
    def __init__(self, in_channels=[24, 32, 96, 160], out_dim=64):
        super(NeighborFeatureAggregation, self).__init__()
        c2_ch, c3_ch, c4_ch, c5_ch = in_channels
        mid_dim = out_dim // 2

        # Scale 2: combines c2 and upsampled c3
        self.conv_s2_c2 = nn.Sequential(nn.Conv2d(c2_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s2_c3 = nn.Sequential(nn.Conv2d(c3_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_s2 = FeatureFusionModule(mid_dim * 2, out_dim)

        # Scale 3: combines pooled c2, c3, and upsampled c4
        self.conv_s3_c2 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(c2_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s3_c3 = nn.Sequential(nn.Conv2d(c3_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s3_c4 = nn.Sequential(nn.Conv2d(c4_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_s3 = FeatureFusionModule(mid_dim * 3, out_dim)

        # Scale 4: combines pooled c3, c4, and upsampled c5
        self.conv_s4_c3 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(c3_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s4_c4 = nn.Sequential(nn.Conv2d(c4_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s4_c5 = nn.Sequential(nn.Conv2d(c5_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_s4 = FeatureFusionModule(mid_dim * 3, out_dim)

        # Scale 5: combines pooled c4 and c5
        self.conv_s5_c4 = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(c4_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.conv_s5_c5 = nn.Sequential(nn.Conv2d(c5_ch, mid_dim, 3, padding=1), nn.BatchNorm2d(mid_dim), nn.ReLU(True))
        self.fuse_s5 = FeatureFusionModule(mid_dim * 2, out_dim)

    def forward(self, c2, c3, c4, c5):
        # Scale 2 (1/4)
        c2_s2 = self.conv_s2_c2(c2)
        c3_s2 = F.interpolate(self.conv_s2_c3(c3), size=c2.shape[2:], mode='bilinear', align_corners=False)
        s2 = self.fuse_s2(torch.cat([c2_s2, c3_s2], dim=1))

        # Scale 3 (1/8)
        c2_s3 = self.conv_s3_c2(c2)
        c3_s3 = self.conv_s3_c3(c3)
        c4_s3 = F.interpolate(self.conv_s3_c4(c4), size=c3.shape[2:], mode='bilinear', align_corners=False)
        s3 = self.fuse_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1))

        # Scale 4 (1/16)
        c3_s4 = self.conv_s4_c3(c3)
        c4_s4 = self.conv_s4_c4(c4)
        c5_s4 = F.interpolate(self.conv_s4_c5(c5), size=c4.shape[2:], mode='bilinear', align_corners=False)
        s4 = self.fuse_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1))

        # Scale 5 (1/32)
        c4_s5 = self.conv_s5_c4(c4)
        c5_s5 = self.conv_s5_c5(c5)
        s5 = self.fuse_s5(torch.cat([c4_s5, c5_s5], dim=1))

        return s2, s3, s4, s5


# -----------------------------------------------------------------------------
# 3. Supervised Attention Module (SAM) & Decoder
# -----------------------------------------------------------------------------

class SupervisedAttentionModule(nn.Module):
    """Uses deep change representations to generate attention masks guiding shallow levels."""
    def __init__(self, in_channels):
        super(SupervisedAttentionModule, self).__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_fine, feat_coarse):
        # Generate attention gate from coarse features
        mask = torch.sigmoid(self.conv_mask(feat_coarse))
        mask_up = F.interpolate(mask, size=feat_fine.shape[2:], mode='bilinear', align_corners=False)
        # Attention gating: suppress unchanged background clutter
        gated = feat_fine * mask_up + feat_fine
        out = self.conv_refine(gated)
        return out, mask


# -----------------------------------------------------------------------------
# 4. Full A2Net Architecture
# -----------------------------------------------------------------------------

class A2Net(nn.Module):
    """
    A2Net: Lightweight Remote Sensing Change Detection Network with Progressive
    Aggregation and Supervised Attention (Guan et al., IEEE TGRS 2023).

    Args:
        in_channels (int): Input spectral band count (default: 6 for Landsat 8 B2-B7).
        num_classes (int): Prediction class count (default: 1 for binary change).
        feat_dim (int): Common aggregated feature dimension (default: 64).
    """
    def __init__(self, in_channels=6, num_classes=1, feat_dim=64):
        super(A2Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Weight-shared MobileNetV2 temporal backbone
        self.backbone = MobileNetV2Backbone(in_channels=in_channels)

        # 2. Neighbor Feature Aggregation Modules (NAM)
        self.nam = NeighborFeatureAggregation(in_channels=[24, 32, 96, 160], out_dim=feat_dim)

        # 3. Temporal Difference Projections: Delta_k = |S_k(t1) - S_k(t2)|
        self.diff_conv2 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True))
        self.diff_conv3 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True))
        self.diff_conv4 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True))
        self.diff_conv5 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True))

        # 4. Supervised Attention Modules (SAM)
        self.sam4 = SupervisedAttentionModule(feat_dim)
        self.sam3 = SupervisedAttentionModule(feat_dim)
        self.sam2 = SupervisedAttentionModule(feat_dim)

        # 5. Progressive Decoder & Fusion
        self.decoder_fuse = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False), # 1/1 resolution
            nn.Conv2d(feat_dim // 2, num_classes, kernel_size=1)
        )

    def forward(self, x1, x2):
        """
        Bitemporal change detection forward pass.
        Args:
            x1 (Tensor): Pre-change image [B, in_channels, H, W]
            x2 (Tensor): Post-change image [B, in_channels, H, W]
        Returns:
            Tensor: Final change logits [B, num_classes, H, W]
        """
        # Step 1: Multi-scale feature extraction through shared backbone
        c2_1, c3_1, c4_1, c5_1 = self.backbone(x1)
        c2_2, c3_2, c4_2, c5_2 = self.backbone(x2)

        # Step 2: Neighbor Feature Aggregation (NAM)
        s2_1, s3_1, s4_1, s5_1 = self.nam(c2_1, c3_1, c4_1, c5_1)
        s2_2, s3_2, s4_2, s5_2 = self.nam(c2_2, c3_2, c4_2, c5_2)

        # Step 3: Progressive Change Information (PCIM) differencing
        d2 = self.diff_conv2(torch.abs(s2_1 - s2_2))
        d3 = self.diff_conv3(torch.abs(s3_1 - s3_2))
        d4 = self.diff_conv4(torch.abs(s4_1 - s4_2))
        d5 = self.diff_conv5(torch.abs(s5_1 - s5_2))

        # Step 4: Supervised Attention Guidance (coarse to fine)
        d4_guided, _ = self.sam4(d4, d5)
        d3_guided, _ = self.sam3(d3, d4_guided)
        d2_guided, _ = self.sam2(d2, d3_guided)

        # Step 5: Final Decoding & Upsampling to full resolution
        logits = self.decoder_fuse(d2_guided)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
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
    print("Testing A2Net from Guan et al. (IEEE TGRS 2023)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = A2Net(in_channels=6, num_classes=1, feat_dim=64).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Dummy Landsat bitemporal tiles: [B=2, C=6, H=256, W=256]
    pre_img = torch.randn(2, 6, 256, 256, device=device)
    post_img = torch.randn(2, 6, 256, 256, device=device)
    gt_mask = torch.randint(0, 2, (2, 1, 256, 256), dtype=torch.float32, device=device)

    logits = model(pre_img, post_img)
    probs = torch.sigmoid(logits)
    pred_binary = (probs > 0.5).int()

    print(f"Input Pre-Image Shape  : {pre_img.shape}")
    print(f"Input Post-Image Shape : {post_img.shape}")
    print(f"Output Logits Shape    : {logits.shape}")
    print(f"Output Probs Range     : [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Predicted Binary Shape : {pred_binary.shape}")

    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, gt_mask)
    print(f"Focal Loss Output      : {loss.item():.4f}")
    loss.backward()
    print("Backward pass verified successfully.")
    print("=" * 70)
