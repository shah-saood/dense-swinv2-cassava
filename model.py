"""
Dense-SwinV2: A Dual-Branch Hybrid Architecture for Fine-Grained Image Classification.

Branch 1 (local):  DenseNet-201       -> captures texture and local patterns
Branch 2 (global): Swin Transformer V2 -> captures long-range context
Fusion:            Multi-Scale Attention Module (MSAM) + bidirectional feature reprojection

Reference:
    Shah Saood, S. H. Khan, R. Iqbal. "Hybrid Dense-SwinV2: A Novel Dual-Branch
    Architecture for Cassava Disease Classification." Spectrum of Engineering
    Sciences, Vol. 4, No. 3, pp. 1-15, 2026.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import densenet201, DenseNet201_Weights


# ---------------------------------------------------------------------------
# Multi-Scale Attention Module
# ---------------------------------------------------------------------------
class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention Module (MSAM).

    Applies parallel depthwise convolutions at multiple kernel sizes,
    aggregates them, and produces a channel-wise attention map.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2,
                      groups=channels, bias=False)
            for k in (3, 5, 7)
        ])
        self.project = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi = torch.cat([branch(x) for branch in self.branches], dim=1)
        multi = self.bn(self.project(multi))

        b, c, _, _ = multi.shape
        w = self.avg_pool(multi).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w


# ---------------------------------------------------------------------------
# Bidirectional feature reprojection
# ---------------------------------------------------------------------------
class FeatureReprojection(nn.Module):
    """
    Projects two feature maps (dense, swin) into a shared channel dimension
    and back, so cross-branch information can flow both ways.
    """

    def __init__(self, in_dense: int, in_swin: int, hidden: int):
        super().__init__()
        self.dense_to_hidden = nn.Conv2d(in_dense, hidden, 1, bias=False)
        self.swin_to_hidden = nn.Conv2d(in_swin, hidden, 1, bias=False)
        self.hidden_to_dense = nn.Conv2d(hidden, in_dense, 1, bias=False)
        self.hidden_to_swin = nn.Conv2d(hidden, in_swin, 1, bias=False)
        self.bn_dense = nn.BatchNorm2d(in_dense)
        self.bn_swin = nn.BatchNorm2d(in_swin)

    def forward(self, dense: torch.Tensor, swin: torch.Tensor):
        d_h = self.dense_to_hidden(dense)
        s_h = self.swin_to_hidden(swin)
        fused = d_h + s_h

        dense_out = F.relu(self.bn_dense(self.hidden_to_dense(fused)) + dense)
        swin_out = F.relu(self.bn_swin(self.hidden_to_swin(fused)) + swin)
        return dense_out, swin_out


# ---------------------------------------------------------------------------
# Dual-branch model
# ---------------------------------------------------------------------------
class DenseSwinV2(nn.Module):
    """
    Dense-SwinV2 dual-branch classifier.

    Args:
        num_classes:  Number of output classes (default 5 for cassava).
        pretrained:   Load ImageNet-pretrained weights (default True).
        dropout:      Dropout rate before the final classifier.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # --- Branch 1: DenseNet-201 (local features) -------------------
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = densenet201(weights=weights)
        self.dense_features = densenet.features          # -> (B, 1920, H/32, W/32)
        dense_out_channels = 1920

        # --- Branch 2: Swin Transformer V2 (global context) ------------
        # timm provides 'swinv2_tiny_window8_256' or similar variants.
        self.swin = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        swin_out_channels = self.swin.num_features       # 768 for the tiny variant

        # --- Fusion modules --------------------------------------------
        fusion_dim = 512
        self.reprojection = FeatureReprojection(
            in_dense=dense_out_channels,
            in_swin=swin_out_channels,
            hidden=fusion_dim,
        )
        self.attention = MultiScaleAttention(channels=fusion_dim)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # --- Classifier -------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: dense local features
        dense = self.dense_features(x)                   # (B, 1920, h, w)

        # Branch 2: swin global features (reshape tokens -> 2D map)
        swin = self.swin.forward_features(x)             # (B, H', W', C)
        swin = swin.permute(0, 3, 1, 2).contiguous()     # (B, C, H', W')

        # Resize both to a common spatial size
        target_size = dense.shape[-2:]
        if swin.shape[-2:] != target_size:
            swin = F.interpolate(swin, size=target_size,
                                 mode="bilinear", align_corners=False)

        # Bidirectional reprojection into a shared space
        dense_p, swin_p = self.reprojection(dense, swin)

        # Fuse: sum the two reprojected branches
        fused = self.fusion_conv(
            self.reprojection.dense_to_hidden(dense_p) +
            self.reprojection.swin_to_hidden(swin_p)
        )

        # Multi-Scale Attention refinement
        fused = self.attention(fused)

        # Global average pool -> classifier
        pooled = self.gap(fused).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = DenseSwinV2(num_classes=5, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")   # expect (2, 5)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
