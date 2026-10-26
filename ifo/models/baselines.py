"""Baselines. Owner: Vivek.

- BC: image -> action policy trained WITH demo actions (oracle upper bound).
- TCN: time-contrastive triplet encoder across robot / recolored views (Sermanet et al., 2018).
- XIRL: temporal cycle-consistency encoder (Zakka et al., 2021).

TCN and XIRL expose the same `forward(images) -> (B, D)` interface as `ifo.models.encoders`, so a
`LatentPredictor` can be trained on top and they plug straight into the shared CEM planner.
"""

from __future__ import annotations

from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO(Vivek): Phase 1 (by Nov 1)")


class TCNEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO(Vivek): Phase 1-2 (by Nov 5)")


class XIRLEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TODO(Vivek): Phase 3 (by Nov 12)")
