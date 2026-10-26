"""Cross-embodiment rendering: the stand-in for "human" demo videos. Owner: Manu.

Re-render the same trajectory with the demonstrator's appearance changed, so the video carries the
object/contact dynamics but not the robot's look. Styles:
  robot   - unchanged (the imitator's own view)
  recolor - arm geoms recolored (default: skin-like tone), materials stripped
  masked  - arm geoms invisible (objects move "by themselves")
TODO(Manu): alternate camera viewpoint style; floating-hand style.
"""

from __future__ import annotations

import mujoco
import numpy as np

ARM_BODY_PREFIX = "panda0"
STYLES = ("robot", "recolor", "masked")


def arm_geom_ids(model: mujoco.MjModel) -> np.ndarray:
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    arm_bodies = {i for i, name in enumerate(body_names) if name.startswith(ARM_BODY_PREFIX)}
    return np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bodies], dtype=np.int64)


class EmbodimentStyler:
    """Switches the arm's appearance in place on a live env's MjModel; `set_style("robot")` restores it."""

    def __init__(self, env, recolor_rgba=(0.87, 0.67, 0.55, 1.0)):
        self.model = env.unwrapped.model
        self.ids = arm_geom_ids(self.model)
        self.recolor_rgba = np.asarray(recolor_rgba, dtype=np.float32)
        self._orig_rgba = self.model.geom_rgba[self.ids].copy()
        self._orig_matid = self.model.geom_matid[self.ids].copy()
        self.style = "robot"

    def set_style(self, style: str) -> None:
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}")
        m = self.model
        m.geom_rgba[self.ids] = self._orig_rgba
        m.geom_matid[self.ids] = self._orig_matid
        if style == "recolor":
            m.geom_matid[self.ids] = -1  # materials override geom_rgba, so drop them
            m.geom_rgba[self.ids] = self.recolor_rgba
        elif style == "masked":
            m.geom_rgba[self.ids, 3] = 0.0
            m.geom_matid[self.ids] = -1
        self.style = style
