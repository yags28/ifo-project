"""Shared experiment config and seeding. Every script takes a YAML file that maps onto `ExperimentConfig`."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

# The three Franka Kitchen tasks in scope (names match FrankaKitchen-v1 `tasks_to_complete`).
TASKS = ("microwave", "slide cabinet", "kettle")
SEEDS = (0, 1, 2, 3, 4)
DATA_FRACTIONS = (0.25, 0.5, 1.0)

# Root for datasets/checkpoints. On Colab, point this at the shared Drive folder.
DATA_ROOT = Path(os.environ.get("IFO_DATA_ROOT", "data"))
CKPT_ROOT = Path(os.environ.get("IFO_CKPT_ROOT", "checkpoints"))


@dataclass
class ExperimentConfig:
    name: str = "debug"
    method: str = "ours_traj"  # ours_traj | goal_only | tcn | xirl | bc
    task: str = "microwave"
    encoder: str = "dinov2_vits14"  # dinov2_vits14 | jepa_scratch | tcn | xirl
    seed: int = 0
    data_fraction: float = 1.0
    image_size: int = 96
    # world model / training
    batch_size: int = 64
    lr: float = 3e-4
    train_steps: int = 20_000
    rollout_horizon: int = 8
    # planner
    plan_horizon: int = 8
    cem_samples: int = 256
    cem_elites: int = 32
    cem_iters: int = 4
    # eval
    eval_episodes: int = 50
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
