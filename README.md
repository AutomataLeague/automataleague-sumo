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

env = make_env("sumo-1", robot="g1", backend="cpu")
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

### No difficulty levels

There is no curriculum ladder, on purpose. In a competitive game the difficulty
**is** the opponent, and under self-play the opponent improves exactly as fast as
the policy does. An authored schedule of environment levels would be a second
difficulty knob fighting the first.

`opponent` selects who drives side B, which is a statement about the game rather
than about how hard it is:

| `opponent` | who is on the other side |
| --- | --- |
| `self` | the current policy, driving both robots. The real game, and the default. |
| `pool` | sampled from a growing set of past snapshots plus the current policy |
| `zero` | a passive dummy. Only for bootstrapping standing on a fresh robot. |

The pool is what makes "stronger and growing" monotone. Naive self-play against
the current policy alone can cycle: it beats its present self by exploiting a
weakness, then trains that weakness away and forgets the counter.

`shaping_scale` multiplies every shaping term and never the terminal win, so
lowering it moves the policy closer to optimising the actual win condition.

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

## Training

The GPU backend steps `num_envs` parallel **duels**, each duel being one world
holding both robots. Under self-play the policy batch is twice that, because both
contestants are ordinary policy rows.

```bash
# validate and benchmark the backend before spending GPU hours
python tools/warp_smoke.py --num-envs 2048

# check the reward actually pays for the behaviour being asked for
python tools/reward_balance.py

# find out what "doing nothing" scores, so a training curve can be judged
python tools/baselines.py

# can the policy take a hit? a standing pose is not balance
python tools/push_test.py checkpoints/standing/ppo_best.pt

# watch what a checkpoint actually does
python tools/render_policy.py checkpoints/standing/ppo_best.pt -o duel.mp4

# bootstrap standing on a fresh robot — see the override recipe below,
# the shipped reward weights do NOT work for this
python examples/ppo_sumo.py run_name=standing env.num_envs=2048 \
    env.arena.opponent=zero env.arena.opponent_loses_by=none \
    env.reward_weights.alive=0.3 env.reward_weights.center=0.1 \
    env.reward_weights.push=0 env.reward_weights.engage=0

# the real game, warm-started from the standing policy
python examples/ppo_sumo.py env.num_envs=2048 \
    init_checkpoint=checkpoints/standing/ppo_best.pt
```

### The standing reward, and why the defaults are wrong for it

Run `tools/reward_balance.py` before training. With the shipped
`RewardConfig`, the net per-step reward for simply staying alive at the 0.9 m
spawn radius is **-0.130** guaranteed, **-0.080** even under the most generous
reading of the `engage` term. Every extra step the robot survives at spawn costs
it reward, and the break-even radius is 0.474 m, well inside where it starts.

That is not a theoretical concern. Measured over the first 9M frames of standing:
the policy crept inward from 0.98 m to 0.34 m while its episode length sat at 60
steps, and only began extending episodes (60 → 66) once it had crossed inside
0.474 m. It optimized exactly what it was asked to.

To bootstrap standing, make survival unconditionally positive and drop the two
terms that describe an opponent which is only scenery:

```
env.reward_weights.alive=0.3     # was 0.05
env.reward_weights.center=0.1    # was 0.5  -> break-even moves to 2.6 m, outside the ring
env.reward_weights.push=0        # the dummy's radius is noise the learner cannot control
env.reward_weights.engage=0      # likewise for facing a robot that is lying down
```

### Judge standing against doing nothing, not against zero

The G1 spawns in a stance it cannot passively hold, so it survives a while and
falls over regardless of what drives it. `tools/baselines.py` measures the bar
(sumo-1 against a passive dummy, 12 seeds):

| policy | mean episode length |
| --- | --- |
| zero action | 73.6 steps |
| small random, U(-0.2, 0.2) | 75.9 steps |
| random, U(-1, 1) | 62.1 steps |

**A policy has learned nothing about balance until it beats ~76 steps**,
and a full episode is 750, so success is a ten-fold improvement rather than a
marginal one. The first run's curve rose the whole way to 66 steps and still sat
*below* the do-nothing bar; without this table it read as steady progress.

### Standing result

With the override recipe above, 2048 duels, 40M frames, 33 minutes on a DGX
Spark (GB10):

| | do nothing | default reward, 10M | survival reward, 40M |
| --- | --- | --- | --- |
| episode length (eval) | 73.6 | 50.4 | **731.9** / 750 |
| duels reaching the timeout | — | 0% | **96.9%** |

Learning is a sharp transition between 15M and 22M frames, not a gradual climb:
episode length sits near 100 until then and reaches the cap within 7M frames.
Nothing before 15M distinguishes the successful run from the failed one on
episode length alone, which is worth knowing before killing a run early.

That policy also transfers to side B unchanged, which is the rotation-invariant
observation doing its job: `tools/render_policy.py --both-sides` drives both
robots from it and both stand for the full episode.

### Surviving a full episode is not balance

`tools/push_test.py` shoves the base partway through an episode and reports
whether the robot recovers. The policy above, which never lost a duel:

| shove | survived |
| --- | --- |
| 0.0 m/s | 6/6 |
| 0.5 m/s | **0/6** (fell within 57 steps) |

It had learned one fixed stance, which is the cheapest solution available when
the opponent never makes contact and the only variation inside an episode comes
from the reset. Nothing in the standing metrics could show this, because they all
measure the situation the policy trained in.

`SumoConfig.push_speed` / `push_interval_steps` add unobserved random horizontal
impulses during training, set by the registry (1.0 m/s every 75 steps). They are absent from the observation on purpose: a
disturbance the policy can see coming is a control input, and lets it pre-brace
rather than learn to recover.

Retraining with them, warm-started from the unperturbed policy, 60M frames:

| shove | no perturbation | trained with pushes |
| --- | --- | --- |
| 0.0 m/s | 6/6 | 6/6 |
| 0.5 m/s | 0/6 | **6/6** |
| 1.0 m/s | 0/6 | **6/6** |
| 1.5 m/s | 0/6 | 3/6 |
| 2.0 m/s | 0/6 | 0/6 |

Robust through the 1.0 m/s it trained against and halfway to 1.5, with the wall
at 2.0. Clean episode length went from 731.9 to a full 750 out of 750, with 100%
of duels reaching the timeout.

The warm start is itself the sharpest measurement of the old policy's
brittleness: loaded in, its first recorded episode length under 1.0 m/s shoves
was **83.6 steps**, and the first push lands at step 75. It survived exactly
until something touched it, then relearned balance from scratch over the same
~20M frames the cold run needed.

`tools/warp_smoke.py` checks the four things that are expensive to discover late:
that the solver has not diverged, that contacts fit inside `nconmax` (MuJoCo-Warp
silently *drops* contacts past the cap rather than raising), that the two sides
are wired symmetrically, and what the throughput actually is.

## Tests

```bash
MUJOCO_GL=egl uv run pytest              # CPU suite
MUJOCO_GL=egl uv run pytest -m gpu       # + CUDA and mujoco-warp (spark/jetson)
```

## Status

Phases A and B are complete: arena, task logic, CPU duel backend, registry.
Phase C has the batched MuJoCo-Warp backend, the PPO stack, and **standing
solved**: 750 of 750 steps, and robust to a 1.0 m/s shove in 6 of 6 seeds. That
policy drives both robots unchanged.

Still to come: `opponent="pool"` (currently raises `NotImplementedError`), which
is the only difficulty mechanism the task needs; a rebalanced fighting reward,
since the shaping terms currently integrate to more than the win over a full
episode; a measured `action_scale` to replace the provisional 0.5; and the Elo
leaderboard.

## Licence

Vendored robot models under `assets/` retain their upstream licences.
