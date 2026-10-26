"""Cross-entropy-method MPC shared by every method. Owner: Yagnesh.

The planner is encoder-agnostic: it only needs
  - `rollout_fn(z0, actions) -> z_pred`  (world-model rollout in latent space)
  - `cost_fn(z_pred) -> cost`            (see `ifo.planning.costs`)
so swapping goal-only vs trajectory conditioning, or DINOv2 vs TCN/XIRL latents,
changes nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

RolloutFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (B,D), (B,H,A) -> (B,H,D)
CostFn = Callable[[torch.Tensor], torch.Tensor]  # (B,H,D) -> (B,)


@dataclass
class CEMConfig:
    horizon: int = 8
    samples: int = 256
    elites: int = 32
    iters: int = 4
    init_std: float = 0.5
    min_std: float = 0.05


@torch.no_grad()
def cem_plan(
    z0: torch.Tensor,
    rollout_fn: RolloutFn,
    cost_fn: CostFn,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    cfg: CEMConfig = CEMConfig(),
    init_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the optimized action sequence (H, A) from a single start latent z0 (D,).

    Pass the shifted previous plan as `init_mean` for warm-started receding-horizon MPC.
    """
    act_dim = action_low.shape[-1]
    mean = init_mean.clone() if init_mean is not None else torch.zeros(cfg.horizon, act_dim, device=z0.device)
    std = torch.full_like(mean, cfg.init_std)
    z0_batch = z0.unsqueeze(0).expand(cfg.samples, -1)

    for _ in range(cfg.iters):
        noise = torch.randn(cfg.samples, cfg.horizon, act_dim, device=z0.device)
        actions = (mean + std * noise).clamp(action_low, action_high)
        costs = cost_fn(rollout_fn(z0_batch, actions))
        elite_idx = costs.topk(cfg.elites, largest=False).indices
        elites = actions[elite_idx]
        mean = elites.mean(0)
        std = elites.std(0).clamp_min(cfg.min_std)

    return mean


def shift_plan(plan: torch.Tensor) -> torch.Tensor:
    """Warm start for the next MPC step: drop the executed action, repeat the last one."""
    return torch.cat([plan[1:], plan[-1:]], dim=0)
