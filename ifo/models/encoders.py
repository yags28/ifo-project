"""Frozen image encoders mapping (B, 3, H, W) uint8/float images -> (B, D) latents. Owner: Yagnesh.

Cache encoded latents to Drive (`ifo.data`) so predictor training never re-runs the encoder.
"""

from __future__ import annotations

import torch
from torch import nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 (via timm), returns the CLS embedding (D=384). Needs `pip install -e '.[models]'`."""

    def __init__(self, model_name: str = "vit_small_patch14_dinov2.lvd142m", image_size: int = 224):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=image_size)
        self.backbone.eval().requires_grad_(False)
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def latent_dim(self) -> int:
        return self.backbone.num_features

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = nn.functional.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        return self.backbone((x - self.mean) / self.std)


class JEPAEncoder(nn.Module):
    """Week 2: encoder pretrained from scratch on robot interaction data (I-JEPA/V-JEPA style). TODO(Yagnesh)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3 (Nov 6-12): see roadmap")
