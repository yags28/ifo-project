"""Day-1 check: render one Franka Kitchen frame in every embodiment style and save a side-by-side PNG.

    python scripts/render_check.py --out outputs/render_check.png
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from ifo.envs import STYLES, EmbodimentStyler, make_kitchen_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/render_check.png")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    env = make_kitchen_env("microwave", image_size=args.size)
    env.reset(seed=0)
    styler = EmbodimentStyler(env)
    frames = []
    for style in STYLES:
        styler.set_style(style)
        frames.append(env.render())
    styler.set_style("robot")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, np.concatenate(frames, axis=1))
    print(f"saved {out} ({' | '.join(STYLES)})")


if __name__ == "__main__":
    main()
