"""Franka Kitchen render + embodiment-style smoke tests (CPU, ~5 s)."""

import numpy as np
import pytest

from ifo.envs import ACTION_DIM, STATE_DIM, STYLES, EmbodimentStyler, make_kitchen_env, rollout


@pytest.fixture(scope="module")
def env():
    e = make_kitchen_env("microwave", image_size=64)
    yield e
    e.close()


def test_rollout_shapes(env):
    out = rollout(env, lambda obs, frame: np.zeros(ACTION_DIM), seed=0, max_steps=5)
    assert out["frames"].shape == (6, 64, 64, 3) and out["frames"].dtype == np.uint8
    assert out["states"].shape == (6, STATE_DIM)
    assert out["actions"].shape == (5, ACTION_DIM)
    assert out["success"] is False


def test_styles_change_pixels_and_restore(env):
    env.reset(seed=0)
    styler = EmbodimentStyler(env)
    assert len(styler.ids) > 0
    frames = {}
    for style in STYLES:
        styler.set_style(style)
        frames[style] = env.render().astype(np.int16)
    styler.set_style("robot")
    assert np.abs(frames["recolor"] - frames["robot"]).mean() > 1.0
    assert np.abs(frames["masked"] - frames["robot"]).mean() > 1.0
    assert np.array_equal(env.render().astype(np.int16), frames["robot"])
