"""
================================================================================
BITEMPORAL IMAGE TRANSFORMER (BIT) FOR REMOTE SENSING CHANGE DETECTION
================================================================================

Reference Publication:
    Chen, H., Zou, Z., & Shi, Z. (2021).
    "Remote Sensing Image Change Detection With Transformers."
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), vol. 60, pp. 1-14, 2022.
    DOI: 10.1109/TGRS.2021.3095166

Official GitHub Repository:
    https://github.com/justchenhao/BIT_CD (models/networks.py, models/help_funcs.py)

Architectural Overview:
    1. Siamese CNN Backbone: Extracts compact, multi-scale feature maps from bitemporal
       images (t1 and t2) at 1/4 or 1/8 resolution.
    2. Spatial Semantic Tokenizer: Condenses spatial feature maps into compact semantic
       tokens via spatial attention pooling:
           T_i = Softmax(W_s * F_i) @ F_i^T  in R^{B x L x C}
    3. Temporal Transformer Encoder: Concatenates tokens [T_1, T_2] and applies multi-head
       self-attention to model long-range bitemporal contextual dependencies across time.
    4. Transformer Decoder: Re-projects contextualized tokens back onto spatial pixel
       embeddings via cross-attention.
    5. Feature Differencing & Prediction Head: Computes absolute feature differences
       and upsamples to original resolution to predict change masks.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Siamese CNN Backbone (ResNet-style)
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard ResNet Residual Block with 2 convolutions and skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """Lightweight 4-stage ResNet feature extractor for BIT."""
    def __init__(self, in_channels=6, base_channels=32):
        super(ResNetBackbone, self).__init__()
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/4 resolution
        )
        # Stage 1: 1/4 resolution
        self.layer1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, stride=1),
            ResidualBlock(base_channels, base_channels, stride=1)
        )
        # Stage 2: 1/8 resolution
        self.layer2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2, stride=1)
        )
        # 1x1 projection to uniform token embedding dimension
        self.proj = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)     # [B, C, H/4, W/4]
        x = self.layer1(x)   # [B, C, H/4, W/4]
        x = self.layer2(x)   # [B, 2C, H/8, W/8]
        x = self.proj(x)     # [B, C, H/8, W/8]
        return x


# -----------------------------------------------------------------------------
# 2. Transformer Components (Self-Attention & Cross-Attention)
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-Head Self/Cross Attention in pure PyTorch."""
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, Nq, C = q.shape
        _, Nk, _ = k.shape

        q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, Nq, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm and MLP."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block: Pixel queries attend to contextual semantic tokens."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super(TransformerDecoderBlock, self).__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm_out = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, pixel_queries, token_keys_values):
        # Cross attention between pixel spatial features and compact tokens
        q = self.norm_q(pixel_queries)
        kv = self.norm_kv(token_keys_values)
        pixel_queries = pixel_queries + self.cross_attn(q, kv, kv)
        pixel_queries = pixel_queries + self.mlp(self.norm_out(pixel_queries))
        return pixel_queries


# -----------------------------------------------------------------------------
# 3. Full Bitemporal Image Transformer (BIT)
# -----------------------------------------------------------------------------

class BitemporalImageTransformer(nn.Module):
    """
    BIT: Bitemporal Image Transformer architecture for Remote Sensing Change Detection.

    Args:
        in_channels (int): Input spectral bands (default: 6 for Landsat 8 B2-B7).
        num_classes (int): Number of target classes (default: 1 for binary change).
        token_len (int): Number of semantic tokens extracted per image (default: 4).
        dim (int): Channel dimension of the transformer representations (default: 32).
        enc_depth (int): Number of Transformer Encoder blocks (default: 2).
        dec_depth (int): Number of Transformer Decoder blocks (default: 2).
        num_heads (int): Attention heads (default: 4).
    """
    def __init__(self, in_channels=6, num_classes=1, token_len=4,
                 dim=32, enc_depth=2, dec_depth=2, num_heads=4):
        super(BitemporalImageTransformer, self).__init__()
        self.token_len = token_len
        self.dim = dim

        # 1. Shared Siamese CNN Backbone
        self.backbone = ResNetBackbone(in_channels=in_channels, base_channels=dim)

        # 2. Semantic Tokenizer: Computes spatial attention maps to aggregate tokens
        self.token_conv = nn.Conv2d(dim, token_len, kernel_size=1, bias=False)

        # 3. Learned Positional Embeddings for bitemporal tokens [T1, T2]
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, dim))

        # 4. Temporal Transformer Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=2.0)
            for _ in range(enc_depth)
        ])

        # 5. Transformer Decoder: Re-projects tokens back onto pixel maps
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=2.0)
            for _ in range(dec_depth)
        ])

        # 6. Prediction Head & Upsampling (8x to match input resolution)
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 1/4
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 1/2
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 1/1
            nn.Conv2d(dim // 4, num_classes, kernel_size=1)
        )

    def extract_semantic_tokens(self, feat_map):
        """
        Condenses spatial feature map into semantic tokens.
        feat_map: [B, C, H, W]
        Returns: [B, token_len, C]
        """
        B, C, H, W = feat_map.shape
        attn = self.token_conv(feat_map)                       # [B, L, H, W]
        attn = attn.view(B, self.token_len, H * W)             # [B, L, N]
        attn = F.softmax(attn, dim=-1)

        feat_flat = feat_map.view(B, C, H * W)                # [B, C, N]
        tokens = torch.bmm(attn, feat_flat.transpose(1, 2))   # [B, L, C]
        return tokens

    def forward(self, x1, x2):
        """
        Args:
            x1 (Tensor): Pre-change image [B, in_channels, H, W]
            x2 (Tensor): Post-change image [B, in_channels, H, W]
        Returns:
            Tensor: Change prediction logits [B, num_classes, H, W]
        """
        B, _, H, W = x1.shape

        # Step 1: Extract spatial feature maps using shared backbone
        f1 = self.backbone(x1)  # [B, dim, H/8, W/8]
        f2 = self.backbone(x2)  # [B, dim, H/8, W/8]
        _, C, h_feat, w_feat = f1.shape

        # Step 2: Tokenization
        t1 = self.extract_semantic_tokens(f1)  # [B, L, dim]
        t2 = self.extract_semantic_tokens(f2)  # [B, L, dim]

        # Step 3: Concatenate bitemporal tokens & apply Temporal Transformer Encoder
        tokens = torch.cat([t1, t2], dim=1) + self.pos_embedding  # [B, 2L, dim]
        for blk in self.encoder:
            tokens = blk(tokens)

        # Split back into contextualized tokens for each timepoint
        t1_ctx, t2_ctx = tokens.chunk(2, dim=1)  # [B, L, dim] each

        # Step 4: Re-project contextual tokens back into spatial feature space
        f1_flat = f1.permute(0, 2, 3, 1).view(B, h_feat * w_feat, C)
        f2_flat = f2.permute(0, 2, 3, 1).view(B, h_feat * w_feat, C)

        for dec_blk in self.decoder:
            f1_flat = dec_blk(f1_flat, t1_ctx)
            f2_flat = dec_blk(f2_flat, t2_ctx)

        f1_refined = f1_flat.view(B, h_feat, w_feat, C).permute(0, 3, 1, 2)
        f2_refined = f2_flat.view(B, h_feat, w_feat, C).permute(0, 3, 1, 2)

        # Step 5: Feature differencing & Upsampling classifier
        diff = torch.abs(f1_refined - f2_refined)  # [B, dim, H/8, W/8]
        logits = self.classifier(diff)              # [B, num_classes, H, W]

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
    print("Testing Bitemporal Image Transformer (BIT) from Chen et al. (IEEE TGRS 2021)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BitemporalImageTransformer(
        in_channels=6, num_classes=1, token_len=4, dim=32, enc_depth=2, dec_depth=2
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # Bitemporal input tiles: [B=2, C=6, H=256, W=256]
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
