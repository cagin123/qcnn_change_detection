"""
================================================================================
CHANGEFORMER: A TRANSFORMER-BASED SIAMESE NETWORK FOR CHANGE DETECTION
================================================================================

Reference Publication:
    Bandara, W. G. C., & Patel, V. M. (2022).
    "A Transformer-Based Siamese Network for Change Detection."
    In IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2022), pp. 207-210.
    DOI: 10.1109/IGARSS46834.2022.9884242
    Extended Paper: "ChangeFormer: A Transformer-Based Dual-Branch Network for
    Remote Sensing Image Change Detection." Remote Sensing 14(10), 2415 (2022).

Official GitHub Repository:
    https://github.com/wgcban/ChangeFormer (models/ChangeFormer.py)

Architectural Overview:
    1. Siamese Hierarchical Transformer Encoder (MiT-style):
       - 4 Overlapped Patch Embedding stages generating feature maps at 1/4, 1/8, 1/16, and 1/32 resolutions.
       - Efficient Self-Attention with Spatial Reduction (SR) ratios to reduce token length.
       - Mix-FFN: Incorporates 3x3 depthwise convolutions within feed-forward layers to encode
         positional information dynamically without static positional embeddings.
    2. Multi-Scale Feature Differencing:
       - At each stage i in {1, 2, 3, 4}, computes absolute difference:
             D_i = |F_{1, i} - F_{2, i}|
    3. Lightweight MLP Decoder:
       - Linearly projects all difference features to a uniform channel dimension.
       - Bilinearly upsamples all stages to 1/4 resolution and fuses them via channel concatenation.
       - Fused features are upsampled 4x to original resolution for final pixel-wise change classification.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Overlapped Patch Embedding
# -----------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    """Image / Feature Map to Overlapping Patch Embedding."""
    def __init__(self, in_channels, embed_dim, patch_size=7, stride=4, padding=3):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# -----------------------------------------------------------------------------
# 2. Efficient Self-Attention with Spatial Reduction Ratio
# -----------------------------------------------------------------------------

class EfficientAttention(nn.Module):
    """
    Self-Attention with Spatial Reduction (SR) ratio to reduce complexity on high-res maps.
    """
    def __init__(self, dim, num_heads=4, sr_ratio=1, dropout=0.0):
        super(EfficientAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_reduced = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_reduced = self.norm(x_reduced)
            kv = self.kv(x_reduced).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # [B, heads, N_kv, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


# -----------------------------------------------------------------------------
# 3. Mix-FFN (Convolutional Positional Encoding)
# -----------------------------------------------------------------------------

class MixFFN(nn.Module):
    """Mix-FFN module incorporating 3x3 depthwise convolution."""
    def __init__(self, in_features, hidden_features=None, dropout=0.0):
        super(MixFFN, self).__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        # Apply depthwise convolution in spatial domain
        x_spatial = x.transpose(1, 2).view(B, -1, H, W)
        x_conv = self.dwconv(x_spatial)
        x = x_conv.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single MiT Transformer Block with Efficient Attention and Mix-FFN."""
    def __init__(self, dim, num_heads, sr_ratio=1, mlp_ratio=4.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


# -----------------------------------------------------------------------------
# 4. Full ChangeFormer Architecture
# -----------------------------------------------------------------------------

class ChangeFormer(nn.Module):
    """
    ChangeFormer: A Transformer-Based Siamese Network for Remote Sensing Change Detection.

    Args:
        in_channels (int): Number of spectral input bands (default: 6 for Landsat 8 B2-B7).
        num_classes (int): Output class count (default: 1 for binary burn detection).
        embed_dims (list): Channel dimensions for the 4 transformer stages.
        num_heads (list): Number of attention heads for each stage.
        sr_ratios (list): Spatial reduction ratios for each stage.
        depths (list): Number of transformer blocks per stage.
        decoder_dim (int): Common channel dimension for the MLP decoder.
    """
    def __init__(self, in_channels=6, num_classes=1,
                 embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8],
                 sr_ratios=[8, 4, 2, 1],
                 depths=[2, 2, 2, 2],
                 decoder_dim=128):
        super(ChangeFormer, self).__init__()
        self.embed_dims = embed_dims

        # --- Stage 1: 1/4 resolution ---
        self.patch_embed1 = OverlapPatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4, padding=3)
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], sr_ratio=sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # --- Stage 2: 1/8 resolution ---
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1)
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], sr_ratio=sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # --- Stage 3: 1/16 resolution ---
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1)
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], sr_ratio=sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # --- Stage 4: 1/32 resolution ---
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1)
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], sr_ratio=sr_ratios[3])
            for _ in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # --- MLP Difference Decoder ---
        self.dec_proj1 = nn.Sequential(nn.Conv2d(embed_dims[0], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(True))
        self.dec_proj2 = nn.Sequential(nn.Conv2d(embed_dims[1], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(True))
        self.dec_proj3 = nn.Sequential(nn.Conv2d(embed_dims[2], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(True))
        self.dec_proj4 = nn.Sequential(nn.Conv2d(embed_dims[3], decoder_dim, 1), nn.BatchNorm2d(decoder_dim), nn.ReLU(True))

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # 4x upsampling to match input resolution
        self.classifier = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1)
        )

    def forward_encoder(self, x):
        """Pass single image through 4-stage hierarchical Transformer."""
        B = x.shape[0]

        # Stage 1
        out1, H1, W1 = self.patch_embed1(x)
        for blk in self.block1:
            out1 = blk(out1, H1, W1)
        out1 = self.norm1(out1)
        f1 = out1.transpose(1, 2).view(B, self.embed_dims[0], H1, W1)

        # Stage 2
        out2, H2, W2 = self.patch_embed2(f1)
        for blk in self.block2:
            out2 = blk(out2, H2, W2)
        out2 = self.norm2(out2)
        f2 = out2.transpose(1, 2).view(B, self.embed_dims[1], H2, W2)

        # Stage 3
        out3, H3, W3 = self.patch_embed3(f2)
        for blk in self.block3:
            out3 = blk(out3, H3, W3)
        out3 = self.norm3(out3)
        f3 = out3.transpose(1, 2).view(B, self.embed_dims[2], H3, W3)

        # Stage 4
        out4, H4, W4 = self.patch_embed4(f3)
        for blk in self.block4:
            out4 = blk(out4, H4, W4)
        out4 = self.norm4(out4)
        f4 = out4.transpose(1, 2).view(B, self.embed_dims[3], H4, W4)

        return [f1, f2, f3, f4]

    def forward(self, x1, x2):
        """
        Bitemporal change detection forward pass.
        Args:
            x1 (Tensor): Pre-change image [B, in_channels, H, W]
            x2 (Tensor): Post-change image [B, in_channels, H, W]
        Returns:
            Tensor: Change prediction logits [B, num_classes, H, W]
        """
        # Step 1: Siamese multi-scale feature extraction
        feats1 = self.forward_encoder(x1)
        feats2 = self.forward_encoder(x2)

        # Step 2: Multi-scale feature differencing: D_i = |F_{1, i} - F_{2, i}|
        diffs = [torch.abs(f1 - f2) for f1, f2 in zip(feats1, feats2)]

        # Step 3: Project and upsample all difference features to 1/4 scale (Stage 1 resolution)
        target_size = diffs[0].shape[2:]
        p1 = self.dec_proj1(diffs[0])
        p2 = F.interpolate(self.dec_proj2(diffs[1]), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.dec_proj3(diffs[2]), size=target_size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.dec_proj4(diffs[3]), size=target_size, mode='bilinear', align_corners=False)

        # Step 4: Concatenation and fusion
        fused = torch.cat([p1, p2, p3, p4], dim=1)
        fused = self.fuse_conv(fused)

        # Step 5: Full-resolution classification
        logits = self.classifier(fused)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme foreground-background class imbalance."""
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
    print("Testing ChangeFormer from Bandara & Patel (IGARSS 2022 / Remote Sensing 2022)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChangeFormer(
        in_channels=6, num_classes=1,
        embed_dims=[32, 64, 128, 256],
        num_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        depths=[2, 2, 2, 2],
        decoder_dim=128
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Bitemporal Landsat tiles: [B=2, C=6, H=256, W=256]
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
