# Automata League — Sumo

Two humanoid robots inside a raised circular ring. A robot wins by putting its
opponent out of the ring or down on the ground, by any physical means. Built on
MuJoCo, trained with TorchRL PPO and GPU-parallel MuJoCo-Warp using self-play.

The second environment in the Automata League, after
[automataleague-parkour](https://github.com/AutomataLeague/automataleague-parkour).

## Setup

```bash
uv sync                              # core: scene, task logic, CPU duels, rendering
uv sync --extra train --extra gpu    # training + MuJoCo-Warp (GPU box)
```

## Quick start

```python
from automataleague_sumo import make_env, list_environments

print([s.env_id for s in list_environments()])      # ['sumo-1']

env = make_env("sumo-1", robot="g1", level=0, backend="cpu")
obs_a, obs_b = env.reset(seed=0)
(obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(act_a, act_b)
```

Render preview stills of the arena:

```bash
MUJOCO_GL=egl uv run python tools/render_scene.py
```

## The environment: `sumo-1`

A cylindrical dohyo of radius 1.5 m raised 0.3 m above the floor. Both robots
spawn diametrically opposite at 60% of the ring radius, facing each other, with
per-episode noise on position, heading and joint angles.

A side loses by leaving the ring radius, by dropping below the platform surface,
or by going down (base too low, or torso tilted past 50 degrees). A 15 second
timeout is a draw, as is a simultaneous loss.

Observations are `3 * n_joints + 23` wide, derived from the robot, and are
expressed entirely in the robot's own base frame. A rigid rotation of the whole
arena leaves them unchanged, which is what makes a single shared policy valid for
both sides in self-play.

### Curriculum

| Level | Opponent | Goal |
| --- | --- | --- |
| 0 | passive, zero action | balance and hold the centre |
| 1 | passive, standing | push the dummy out of the ring |
| 2 | frozen level 1 snapshot | beat a real but static opponent |
| 3 | current policy on both sides | naive self-play |
| 4 | sampled checkpoint pool | league play |

The shaping terms of the reward decay across levels, from full weight at level 0
to 0.2 at level 4, so the final policy optimizes the actual win condition rather
than proxies like hugging the centre. Only the terminal win term is zero sum.

## Robots

Robots are pluggable through `RobotSpec`. Observation and action widths derive
from joint count, so adding a robot mostly means writing a new `RobotSpec`. The
CPU env currently assumes both sides share a robot (`sumo_cpu.py` reads
`scene.a.robot` for `action_scale`, `observation_dim` and `action_dim`), so a
second robot with a different action scale or joint count needs that handled
too before a cross-robot matchup works.

| Robot | Joints | Notes |
| --- | --- | --- |
| `g1` | 29 | Unitree G1, vendored from MuJoCo Menagerie |

We load `g1_mjx.xml`, the primitive-collider variant, not `g1.xml`. Two humanoids
in sustained contact make mesh colliders prohibitive under MuJoCo-Warp, and both
backends must share one model so CPU evaluation cannot disagree with GPU training.

## Tests

```bash
MUJOCO_GL=egl uv run pytest          # CPU suite
```

Phase C will add a `gpu`-marked suite (requires CUDA + mujoco-warp, run with
`uv run pytest -m gpu`); no test carries that marker yet, so there is nothing to
run against it today.

## Status

Phases A and B are complete: arena, task logic, CPU duel backend, registry.
Phase C (batched MuJoCo-Warp backend, PPO, levels 0 and 1) and Phase D
(self-play, Elo leaderboard) are next.

## Licence

Vendored robot models under `assets/` retain their upstream licences.
