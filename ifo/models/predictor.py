"""Action-conditioned latent predictor (the world model). Owner: Yagnesh.

Every encoder (DINOv2, JEPA-scratch, TCN, XIRL) gets one of these trained on top of its frozen
latents, so the CEM planner sees the same interface for all methods.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentPredictor(nn.Module):
    """Residual MLP: z_{t+1} = z_t + f([z_t, a_t]). v1 baseline; swap for a transformer over patch tokens later."""

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        dims = [latent_dim + action_dim] + [hidden] * layers
        blocks: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU()]
        self.net = nn.Sequential(*blocks, nn.Linear(hidden, latent_dim))

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """(B, D), (B, H, A) -> (B, H, D). This is the `rollout_fn` the CEM planner expects."""
        z, out = z0, []
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            out.append(z)
        return torch.stack(out, dim=1)


def prediction_loss(model: LatentPredictor, z: torch.Tensor, a: torch.Tensor, rollout_steps: int) -> torch.Tensor:
    """Teacher-forced 1-step L1 loss + multi-step open-loop rollout L1 loss.

    z: (B, T, D) target latents from the frozen encoder, a: (B, T-1, A).
    """
    one_step = (model(z[:, :-1], a) - z[:, 1:]).abs().mean()
    k = min(rollout_steps, a.shape[1])
    multi = (model.rollout(z[:, 0], a[:, :k]) - z[:, 1 : k + 1]).abs().mean()
    return one_step + multi
