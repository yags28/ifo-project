"""CPU smoke tests for the shared pieces. Run: pytest -q"""

import numpy as np
import torch

from ifo.eval import dtw_distance, success_rate
from ifo.models import LatentPredictor, prediction_loss
from ifo.planning import CEMConfig, DemoTracker, cem_plan, goal_cost, trajectory_cost

D, A, H = 16, 9, 6


def test_predictor_shapes_and_loss():
    model = LatentPredictor(D, A, hidden=64, layers=2)
    z, a = torch.randn(4, 10, D), torch.randn(4, 9, A)
    assert model.rollout(z[:, 0], a[:, :H]).shape == (4, H, D)
    loss = prediction_loss(model, z, a, rollout_steps=H)
    loss.backward()
    assert torch.isfinite(loss)


def test_cem_reaches_goal_in_linear_latent():
    # Toy world model z' = z + 0.1 * W a: CEM should drive z toward the goal.
    torch.manual_seed(0)
    W = torch.randn(A, D)
    rollout = lambda z0, acts: z0.unsqueeze(1) + 0.1 * torch.cumsum(acts @ W, dim=1)
    z0 = torch.zeros(D)
    z_goal = rollout(z0.unsqueeze(0), torch.full((1, H, A), 0.5))[0, -1]
    low, high = -torch.ones(A), torch.ones(A)
    plan = cem_plan(z0, rollout, lambda zp: goal_cost(zp, z_goal), low, high, CEMConfig(horizon=H, iters=8))
    assert plan.shape == (H, A)
    final_err = goal_cost(rollout(z0.unsqueeze(0), plan.unsqueeze(0)), z_goal).item()
    start_err = (z0 - z_goal).abs().mean().item()
    assert final_err < 0.5 * start_err


def test_trajectory_cost_and_tracker():
    z_demo = torch.cumsum(torch.ones(20, D), dim=0)
    tracker = DemoTracker(z_demo, horizon=H)
    assert tracker.update(z_demo[2]) == 2
    assert tracker.update(z_demo[15]) == 5  # can't jump more than max_jump=3 steps
    window = tracker.window()
    assert window.shape == (H, D)
    on_track = trajectory_cost(window.unsqueeze(0), window)
    off_track = trajectory_cost(window.unsqueeze(0) + 1.0, window)
    assert on_track.item() == 0.0 and off_track.item() > 0.0


def test_metrics():
    t = np.linspace(0, 1, 30)[:, None]
    assert dtw_distance(t, t) == 0.0
    assert dtw_distance(t, t[::2]) < dtw_distance(t, t + 0.5)  # time warping is cheap, offsets are not
    mean, se = success_rate([True, False, True, True])
    assert mean == 0.75 and se > 0
