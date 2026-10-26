"""Planning costs: the only thing that differs between goal-only and trajectory conditioning. Owner: Yagnesh.

Both take the predicted latent rollout (B, H, D) and return a cost per sample (B,).
"""

from __future__ import annotations

import torch


def goal_cost(z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """V-JEPA-2-AC-style baseline: L1 distance of the final predicted latent to the demo's final latent."""
    return (z_pred[:, -1] - z_goal).abs().mean(-1)


def trajectory_cost(z_pred: torch.Tensor, z_demo_window: torch.Tensor, terminal_weight: float = 1.0) -> torch.Tensor:
    """Ours: per-step L1 distance between the predicted rollout and the time-aligned demo window (H, D).

    `z_demo_window` is the slice of the demo latent trajectory the planner should follow next;
    choosing it is the job of a `DemoTracker` (progress-index alignment). Soft-DTW alignment is a
    planned ablation.
    """
    per_step = (z_pred - z_demo_window.unsqueeze(0)).abs().mean(-1)  # (B, H)
    weights = torch.ones(per_step.shape[1], device=per_step.device)
    weights[-1] += terminal_weight
    return (per_step * weights).sum(-1) / weights.sum()


class DemoTracker:
    """Tracks progress along the demo latent trajectory and hands out the next H-step target window.

    Progress = index of the demo latent nearest to the current latent, constrained to move forward
    by at most `max_jump` steps per call (stops it skipping ahead to the goal).
    """

    def __init__(self, z_demo: torch.Tensor, horizon: int, max_jump: int = 3):
        self.z_demo = z_demo  # (T, D)
        self.horizon = horizon
        self.max_jump = max_jump
        self.progress = 0

    def update(self, z_now: torch.Tensor) -> int:
        lo = self.progress
        hi = min(self.progress + self.max_jump + 1, len(self.z_demo))
        dists = (self.z_demo[lo:hi] - z_now).abs().mean(-1)
        self.progress = lo + int(dists.argmin())
        return self.progress

    def window(self) -> torch.Tensor:
        idx = torch.arange(self.progress + 1, self.progress + 1 + self.horizon).clamp_max(len(self.z_demo) - 1)
        return self.z_demo[idx]
