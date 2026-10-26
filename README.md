# IfO via Physics-Aware Latent Trajectory Matching (CS 593)

Team: Manu Vazhunnavar, Vivek Agarwal, Yagnesh Dawankar

Teach a Franka arm a Franka Kitchen task from a **video-only** demo of a visually different
demonstrator, by matching the demo's trajectory in a predictive (JEPA-style) latent space with
CEM/MPC, and test whether trajectory conditioning beats goal-image-only conditioning.

**Milestones:** midterm update Thu Nov 5 · all methods running Nov 12 · experiments done Nov 19 · final Nov 25.

## Layout and owners

| Path | What | Owner |
|---|---|---|
| `ifo/envs/kitchen.py` | FrankaKitchen-v1 wrapper, `rollout()`, success check | Manu |
| `ifo/envs/recolor.py` | Cross-embodiment styles (`robot`/`recolor`/`masked`) = stand-in "human" demos | Manu |
| `ifo/data/` | Dataset building (Minari replay + rendering) and loaders | Manu |
| `ifo/eval/` | Success rate, DTW path fidelity, batched eval, figures | Manu |
| `ifo/models/encoders.py` | Frozen DINOv2 encoder; JEPA-from-scratch (week 2) | Yagnesh |
| `ifo/models/predictor.py` | Action-conditioned latent predictor + loss | Yagnesh |
| `ifo/planning/` | Shared CEM planner; `goal_cost` vs `trajectory_cost` + `DemoTracker` | Yagnesh |
| `ifo/models/baselines.py` | BC (oracle), TCN, XIRL | Vivek |
| `configs/` | One YAML per experiment (maps onto `ifo.config.ExperimentConfig`) | all |

Contract between streams: every encoder is `images (B,3,H,W) -> (B,D)`; every world model exposes
`rollout(z0 (B,D), actions (B,H,9)) -> (B,H,D)`; the planner only sees those two plus a cost.

## Setup

**Colab (main compute):** open `notebooks/colab_setup.ipynb`, set `REPO_URL`, run all. Large files
(datasets, latents, checkpoints) go in the shared Drive folder `ifo-shared/`, never in git.
Training scripts must resume from checkpoints because Colab sessions die.

**Local (CPU, for development):**
```bash
python -m venv .venv            # or: conda create -p ./.venv python=3.11
.venv/bin/pip install -e '.[dev]'           # add ,models for DINOv2
export MINARI_DATASETS_PATH=$PWD/data/minari
.venv/bin/minari download D4RL/kitchen/complete-v2   # also mixed-v2, partial-v2
.venv/bin/python -m pytest -q
.venv/bin/python scripts/render_check.py   # -> outputs/render_check.png
```

## Gotchas found during setup
- **Pin `mujoco<3.4`.** MuJoCo 3.17 breaks FrankaKitchen-v1 (`AssertionError` in `get_joint_qpos`). Tested with 3.3.7 + gymnasium-robotics 1.4.2.
- Minari needs the `[hf,hdf5]` extras to download and read datasets.
- `D4RL/kitchen/complete-v2` has only **19 episodes**, each doing microwave → kettle → light switch → slide cabinet in sequence. Per-task demos have to be cut at task-completion steps; use `mixed-v2`/`partial-v2` for more interaction data.
- Action replay of Minari episodes is faithful: `env = ds.recover_environment(render_mode="rgb_array")`, `env.reset(seed=ep.id)`, then step `ep.actions`. Max state error stays under 0.04 and all tasks complete. So we can render any resolution or style by replaying.
- If pytest crashes importing `launch`/`lark` (ROS on the machine), run `env -u PYTHONPATH python -m pytest`.

## Conventions
- Seeds `ifo.config.SEEDS`; data fractions `DATA_FRACTIONS`; tasks `TASKS`.
- One branch per person (`yagnesh/...`, `manu/...`, `vivek/...`), PRs into `main`, keep `pytest` green.
- Log runs to the shared wandb project; results CSVs to `ifo-shared/results/`.
