"""Franka Kitchen wrapper. Owner: Manu.

FrankaKitchen-v1 (gymnasium-robotics 1.4.x): 9-D action in [-1, 1] (joint velocities + gripper),
59-D state observation, 280 steps per episode, success via `info["episode_task_completions"]`.
Requires mujoco<3.4 (3.17 breaks the env's joint-type assert). Headless: set MUJOCO_GL=egl on Colab.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

ACTION_DIM = 9
STATE_DIM = 59


def make_kitchen_env(tasks: str | list[str] = "microwave", image_size: int = 96, **kwargs) -> gym.Env:
    """Franka Kitchen with `env.render()` returning (image_size, image_size, 3) uint8 frames."""
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    return gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=tasks,
        render_mode="rgb_array",
        width=image_size,
        height=image_size,
        **kwargs,
    )


def task_success(info: dict, tasks: str | list[str]) -> bool:
    tasks = [tasks] if isinstance(tasks, str) else tasks
    return all(t in info["episode_task_completions"] for t in tasks)


def rollout(env: gym.Env, policy, seed: int | None = None, max_steps: int | None = None) -> dict:
    """Run one episode; `policy(obs_dict, frame) -> action`. Returns frames, states, actions, success."""
    obs, info = env.reset(seed=seed)
    frame = env.render()
    frames, states, actions = [frame], [obs["observation"].copy()], []
    max_steps = max_steps or env.spec.max_episode_steps
    terminated = truncated = False
    for _ in range(max_steps):
        action = np.asarray(policy(obs, frame), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        states.append(obs["observation"].copy())
        actions.append(action)
        if terminated or truncated:
            break
    tasks = env.unwrapped.goal.keys()
    return {
        "frames": np.stack(frames),
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions) if actions else np.zeros((0, ACTION_DIM), np.float32),
        "success": task_success(info, list(tasks)),
    }
