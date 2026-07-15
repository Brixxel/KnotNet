"""CNN Image Encoder für KnotGraphNet V5."""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """CNN: (B, 4, H, H) → (B, D, H/16, H/16)"""

    def __init__(self, in_ch: int = 4, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 24, 7, stride=2, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, img: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:  (B, 3, H, W)
            skel: (B, 1, H, W)
        Returns:
            features: (B, D, H/16, W/16)
        """
        return self.net(torch.cat([img, skel], dim=1))