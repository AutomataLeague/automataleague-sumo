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
import numpy as np
from automataleague_sumo import make_env, list_environments

print([s.env_id for s in list_environments()])      # ['sumo-1']

env = make_env("sumo-1", robot="g1", backend="cpu")  # one renderable duel
obs_a, obs_b = env.reset(seed=0)

# Both sides are ordinary policy rows; you drive each one yourself.
n = env.scene.a.robot.action_dim                     # 29 for the G1
for _ in range(100):
    act_a, act_b = np.zeros(n), np.zeros(n)          # your policy goes here
    (obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(act_a, act_b)
    if terminated or truncated:
        print(info["outcome"])                       # 1 = A won, 2 = B won, 3 = draw
        break
```

`backend="warp"` gives the batched GPU version instead, stepping `num_envs`
duels at once. That is what training uses.

Render preview stills of the arena:

```bash
MUJOCO_GL=egl uv run python tools/render_scene.py
```

## The environment: `sumo-1`

A cylindrical dohyo of radius 1.5 m raised 0.3 m above the floor. Both robots
spawn diametrically opposite at 25% of the ring radius, facing each other, with
per-episode noise on position, heading and joint angles.

That is 0.75 m apart, inside each other's 0.59 m arm reach, which is deliberate
and is what real sumo does: wrestlers face off at the shikiri lines about 0.7 m
apart, not across the ring. It is also the only thing giving the reward an
approach gradient. `push` needs contact, `win` needs a ring-out, and `alive` and
`centre` are both maximised by standing still in the middle. Spawn them a ring
apart and two competent standing policies never touch, every episode is a draw,
and there is nothing to climb out of.

A side loses by leaving the ring radius, by dropping below the platform surface,
or by going down (base too low, or torso tilted past 50 degrees). A 15 second
timeout is a draw, as is a simultaneous loss.

Observations are `3 * n_joints + 23` wide, derived from the robot, and are
expressed entirely in the robot's own base frame. A rigid rotation of the whole
arena leaves them unchanged, which is what makes a single shared policy valid for
both sides in self-play.

### No difficulty levels

There is no curriculum ladder, on purpose. In a competitive game the difficulty
**is** the opponent, and under self-play it tracks the policy's own strength. An
authored schedule of environment levels would be a second difficulty knob
fighting the first.

There is also no opponent pool. Sampling from past checkpoints exists to stop
self-play cycling between exploitable strategies, and cycling needs a discrete
strategy space to cycle in. Whole-body balance and shoving does not obviously
have one. If naive self-play does destabilise, it shows up directly as win rate
oscillating against held-out old checkpoints, and a pool can be added then.

`opponent` therefore has exactly two values:

| `opponent` | who is on the other side |
| --- | --- |
| `self` | the current policy, driving both robots. The real game, and the default. |
| `zero` | a passive dummy that cannot lose. Only for bootstrapping standing on a fresh robot. |

The dummy cannot lose because a zero-action humanoid collapses on its own in
about 1.2 s, which would hand the learner a free win roughly 60 steps into every
episode. That is derived from the opponent mode, not configured separately: a
handicap with its own switch is a handicap that can be left on by accident.

### The reward

Three things, in the order they matter, plus two regularizers:

| term | what it pays for | weight |
| --- | --- | --- |
| `win` | putting the opponent out or down. Terminal, zero sum. | 10.0 |
| `push` | driving them from the centre to the rim | 3.0 |
| `alive` | a whole episode spent standing | 2.0 |
| `centre` | a whole episode spent pinned against the rim (a penalty) | 1.0 |
| `action`, `joint_vel` | regularizers | 0.5 each |

**Every weight is a whole-episode value**, so they are directly comparable to
each other and to `win`. That is the property whose absence caused the only
reward bug this project has had: with per-step weights, a facing bonus worth 0.3
a step quietly outscored a terminal +10 by thirteen times over a 750-step
episode, and nothing in the numbers said so. `RewardConfig` now refuses to
construct if the whole shaping budget can outscore a win.

`push` is a delta on the opponent's radius, so it telescopes to the total change
across the episode. It cannot be farmed by shoving them out and letting them back
in, and it is the only term not divided by the episode horizon.

Run `tools/reward_balance.py` before training. It prints every term against
`win` and exits non-zero if surviving at the rim scores worse than giving up.

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

You need a CUDA GPU with MuJoCo-Warp. On a DGX Spark (GB10) at 2048 duels this
runs about 39k policy rows/s, so **1B frames takes roughly 7.5 hours**. 4096
duels buys nothing. Every command below is prefixed with `MUJOCO_GL=egl` because
evaluation renders headlessly.

Each run writes to `checkpoints/<run_name>/`: a `ppo_eval_<frames>.pt` snapshot
per evaluation, a `ppo_best.pt`, and `metrics.jsonl` with every batch. Keep the
jsonl. Weights and Biases is on by default (`logger.backend=""` disables it), but
the local jsonl is what survived the one run where the wandb step counter was
wrong and a whole 1B-frame run logged as a single row.

### The recipe

Four stages. Each has a gate you should actually check before spending the next
block of GPU time, because every one of them has silently failed here at least
once.

**0. Preflight (about two minutes, no training).**

```bash
# does the reward pay for the behaviour you are asking for? exits non-zero if not
MUJOCO_GL=egl uv run python tools/reward_balance.py

# what does DOING NOTHING score? a policy below this has learned nothing
MUJOCO_GL=egl uv run python tools/baselines.py

# contact headroom and A/B symmetry on the GPU backend, at your real batch size
MUJOCO_GL=egl uv run python tools/warp_smoke.py --num-envs 2048
```

**1. Bootstrap standing against a dummy (~50M frames, ~20 min).** Two fresh
robots both collapse in 1.5 s, which scores as a draw and gives the win term
nothing to work with, so the robot learns to stand first.

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=standing \
    env.num_envs=2048 collector.total_frames=50_000_000 \
    env.arena.opponent=zero env.reward_weights.push=0
```

*Gate:* episode length must beat the `baselines.py` number (~76 steps for the
G1), and the policy must survive a shove. A held pose is not balance:

```bash
MUJOCO_GL=egl uv run python tools/push_test.py checkpoints/standing/ppo_best.pt
```

**2. Self-play, warm-started from it (1B frames, ~7.5 h).**

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=sumo1 \
    env.num_envs=2048 collector.total_frames=1_000_000_000 \
    init_checkpoint=checkpoints/standing/ppo_best.pt
```

*Gate:* nothing in the training curves will tell you whether this worked. See
stage 3. What you can watch for is `train/saturated_fraction` staying near zero;
a rising value means the policy is diverging and the run aborts itself.

**3. Find out which checkpoint is actually best.**

```bash
MUJOCO_GL=egl uv run python tools/round_robin.py checkpoints/sumo1/ppo_eval_*.pt
MUJOCO_GL=egl uv run python tools/render_progression.py checkpoints/sumo1/ppo_eval_*.pt
```

> **Do not assume the last checkpoint, or `ppo_best.pt`, is the best one.** In
> this project the final checkpoint was beaten by an earlier one **four separate
> times**, once by 58.3% over a thousand duels. `ppo_best.pt` is chosen by a
> heuristic eval score; the round robin is the measurement. Warm-start the next
> run from whatever the tournament ranks first, not from the newest file.

**Optional, to push further.** There is headroom past 1B, but a plain warm start
restarts the learning-rate anneal at full strength on an already-converged policy
and knocks it down to 28.6% against the field for ~200M frames. Continue at a
constant, reduced rate instead:

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=sumo1_continue \
    env.num_envs=2048 collector.total_frames=300_000_000 \
    init_checkpoint=checkpoints/sumo1/<best from the round robin>.pt \
    loss.anneal_lr=false loss.anneal_clip_epsilon=false optim.lr=1.0e-4
```

Check the checkpoint you are continuing from is numerically fit to continue:

```bash
MUJOCO_GL=egl uv run python tools/policy_saturation.py checkpoints/sumo1/<best>.pt
```

### Configuration

Everything is Hydra. `examples/config_ppo.yaml` is the full list with a comment
on each; override any of it as `key=value` on the command line. A `null` under
`env.arena`, `env.reward_weights` or `env.termination` means "keep the registry
default", so you can change one knob without restating the rest.

| knob | default | when you would change it |
| --- | --- | --- |
| `env.num_envs` | 2048 | parallel duels. Lower it if you run out of GPU memory. |
| `env.arena.opponent` | `self` | `zero` only for the standing bootstrap. |
| `env.arena.action_scale` | 0.5 | the pose window. **Measure it**, see below. |
| `env.arena.push_speed` | 1.0 | unobserved shoves. 0 disables (also set the interval to 0). |
| `env.reward_weights.*` | see table above | whole-episode values. |
| `collector.total_frames` | 20M | the run length. |
| `optim.lr` | 3e-4 | reduce for a continuation. |
| `loss.anneal_lr` | true | set false when continuing a converged policy. |
| `network.max_loc` | 5.0 | soft bound on the policy mean. Only raise it if a checkpoint you are continuing already exceeds it. |
| `init_checkpoint` | null | warm start. `init_critic=false` if the reward changed scale. |

### The rest of the toolbox

```bash
# is the action range large enough for the task to be possible at all?
MUJOCO_GL=egl uv run python tools/measure_reach.py

# across a chain of warm starts, label by hand: each run restarts its counter
MUJOCO_GL=egl uv run python tools/round_robin.py \
    1000M=checkpoints/v6/ppo_eval_1000013824.pt \
    1290M=checkpoints/v7/ppo_eval_290062336.pt

# watch one checkpoint, or put two against each other
MUJOCO_GL=egl uv run python tools/render_policy.py checkpoints/sumo1/ppo_best.pt -o duel.mp4
MUJOCO_GL=egl uv run python tools/render_versus.py old.pt new.pt -o versus.mp4
```

### Bootstrapping standing

Two robots that both collapse in 1.5 s produce nothing but simultaneous losses,
which score as draws and give the win term nothing to work with. So a fresh robot
learns to stand first, against a dummy, and self-play warm-starts from that.
Stages 1 and 2 of the recipe above.

`push` is zeroed for the bootstrap because the dummy's radius is decided by how
it happens to topple, which the learner cannot influence. Paying for it is pure
variance in the advantage estimate.

### `action_scale`, measured

`q_target = home + action_scale * action`, action in [-1, 1]. So it is the
maximum radians any joint can be commanded away from the home stance: not a gain
and not a speed limit, but a hard geometric cap on the poses the policy can ask
for. `tools/measure_reach.py` reports the kinematic envelope it buys (widest
reachable foot spreads, so an upper bound on gait rather than a step length):

| `action_scale` | stance width | stride | max crouch | reach | clipped by joint limits |
| --- | --- | --- | --- | --- | --- |
| 0.2 (11°) | 0.52 m | 0.34 m | 0.05 m | 0.32 m | 0% |
| 0.3 (17°) | 0.66 m | 0.50 m | 0.11 m | 0.36 m | 1% |
| 0.4 (23°) | 0.80 m | 0.69 m | 0.19 m | 0.39 m | 3% |
| **0.5 (29°)** | 0.89 m | 0.78 m | 0.29 m | 0.42 m | 4% |
| 0.7 (40°) | 1.11 m | 1.01 m | 0.48 m | 0.49 m | 8% |
| 1.0 (57°) | 1.41 m | 1.17 m | 0.77 m | 0.56 m | 15% |

`action_scale` is a **symmetric window around the home pose**, which is why one
number cannot serve the whole robot. The G1's home pose is a relaxed carry with
the elbows bent 1.28 rad, so at a uniform 0.5 the elbow can only reach 0.78 rad:
the arm can never get within 45 degrees of straight and simply hangs. It is not a
policy choice, it is a hard cap, and it looks exactly like the policy declining
to use its arms.

`RobotSpec.joint_scale` gives per-joint multipliers on top. The G1 keeps 0.5 on
the legs (see the crouch finding below) and takes 2.5x on the arms, which opens
the elbow window to `[0.03, 2.53]` and raises measured arm reach from 0.42 m to
0.59 m.

Two things to read off it.

**The curve never flattens.** The mechanical joint limits are not the binding
constraint anywhere in this range; `action_scale` is. Every increase buys real
reach, so choosing it is a trade against controllability rather than against the
robot's geometry.

**0.5 is the largest scale that cannot crouch into a loss.** The base starts at
0.784 m and the down-rule fires below 0.431 m, so a drop past 0.353 m loses the
duel outright. At 0.5 the deepest commandable crouch is 0.294 m, just inside
that. From 0.7 the action space contains poses that lose immediately.

To start tight and loosen, warm-start across runs rather than scheduling inside
one, since changing the scale changes what the same action numbers mean:

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=tight \
    env.arena.action_scale=0.3
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=loose \
    env.arena.action_scale=0.6 init_checkpoint=checkpoints/tight/ppo_best.pt
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

## Self-play result

450M frames, 2048 duels, 3h24m on a DGX Spark, warm-started from the standing
policy through a 120M-frame intermediate.

| | 120M, before the stepping-out rule | 450M, after |
| --- | --- | --- |
| eval draws | 86% | **0.4%** |
| eval episode length | 682 steps | **184 steps** (~3.7 s) |
| opponent driven to | 0.53 m | **1.21 m** (rim is 1.50 m) |

The robots close from the tachiai, clinch, and drive each other out. Duels are
short and decisive, which is what a sumo bout looks like: real bouts average
around five seconds.

**`win_rate` is pinned at 0.5 and carries no information.** Every duel produces
one winner and one loser and both rows are in the same batch, so it is a
structural identity, not a measurement. If it ever drifts off 0.5 that is a bug
in the row-outcome bookkeeping. The metrics that move are the final radii and the
draw rate.

**Checkpoint scoring is task-dependent, and getting it backwards is silent.** The
score was `episode_length + 100*win_rate + 10*opp_radius`, written when the task
was survival. Against a real opponent a long episode is a *stalemate*: that score
selected a 140M checkpoint drawing 65% of its duels over the 450M one drawing 0%
and driving its opponent twice as far. It now scores episode length only against
a dummy, and against a real opponent scores how far the loser is driven and how
decisive the duels are.

## Measuring progress: the round robin

Self-play metrics cannot say whether a policy improved. `tools/round_robin.py`
plays every checkpoint against every other, both orderings so any side advantage
cancels, counting only each world's **first** conclusion because worlds auto-reset
and fast pairings would otherwise be over-weighted. It costs about four minutes
against a seven-hour run, so it should gate a run rather than follow one.

It has caught, in this project alone: a training ceiling twice at the same place,
which turned "needs more frames" into a specific hypothesis about the environment;
that `ppo_best.pt` was a stalemating checkpoint from a third of the way in; and
that the final checkpoint is not the best, four times.

**A field win rate is only meaningful relative to its field.** The average across
participants is 50% by construction, so adding stronger checkpoints moves
everyone. The same 1000M checkpoint scored 76.5% against one field and 67.4%
against another, with identical weights. Only a **head to head** between two named
policies is invariant, which is what any claim of improvement should quote.

**Transitivity** is reported alongside: the count of ordered triples where A beats
B, B beats C and C beats A. Measured 0 of 336 and 0 of 504 across separate runs.
No cycling means improvement is a strict ordering, so training against the current
policy alone suffices and an opponent pool would buy nothing. That is the evidence
for the design decision rather than an argument for it.

## Results

**The vendored collision model was missing 56% of contact.** Menagerie's
`g1_mjx.xml` is stripped for *locomotion*, where only the feet touch anything. For
wrestling that is exactly backwards. 12 of 30 bodies had no collision geom at all,
including both shoulders and both forearms, and the head sphere sat 45 mm high
covering only the top 57%. Replaying one policy through both models: 1877 → 4225
robot-to-robot contacts. This **capped training**: two separate 1B-frame runs
plateaued at ~58% win rate. After grafting five primitives per robot in
`RobotSpec.extra_colliders` (mass 0, `assets/` untouched), the curve climbs
monotonically to 76.5% and the new policy beats the old champion 910 to 106.

**Audit a vendored asset against your task, not the one it shipped for.**

**Past 1B frames there is headroom, but a warm restart costs most of it.** A
lineage of 1000M → 1290M → 1590M cumulative frames, scored by round robin:

| | head to head | duels | significance |
| --- | --- | --- | --- |
| best of the continuation (1490M) vs 1000M | **57.4%** | 1013 | +4.7σ |
| final checkpoint (1590M) vs 1000M | 47.9% | 1016 | −1.3σ |
| 1490M vs the final 1590M | 58.3% | 1007 | +5.3σ |

So 590M further frames bought a real but modest gain, an order of magnitude
smaller than the collision fix, and the run's *last* checkpoint was no better than
its start. Each warm restart also knocked the policy down hard, to 28.6% against
the field, taking ~200M frames to recover. `init_checkpoint` restarts the
learning-rate anneal at full strength on an already-converged policy.

## Numerical stability

Three runs died on non-finite numbers, and every one traced to a single unbounded
quantity: the **pre-squash mean** of the TanhNormal policy.

Its log-prob carries a `-log(1 - a²)` jacobian that grows without limit as an
action approaches the bound. Nothing bounded `loc`, so a rare state could drive
the mean tens of units past saturation, where it commands no additional motion
but inflates log-prob into the thousands. The importance ratio is
`exp(new_log_prob − old_log_prob)`, and float32 `exp` overflows at 88. A
diagnostic dump caught a stored log-prob of **3427** with the network otherwise
healthy: entropy 8.39, σ 0.43, explained variance 0.80.

`BoundedLocNormalScale` applies `max_loc * tanh(loc / max_loc)`. It is absent by
default, so checkpoints saved before it replay bit-identically.

- **TorchRL's `ClipPPOLoss` bounds the ratio on only one branch.** `gain1` uses an
  unclamped `log_weight.exp()`, and PPO's pessimistic `min` selects it exactly
  when the advantage is negative. An overflowing ratio bypasses the clip entirely.
- **Do not fix this by clamping.** `torch.clamp` has zero gradient outside its
  range, so it deletes the corrective force on the samples that most need pulling
  back, and it is self-reinforcing. Clamping the importance ratio destroyed a run
  far more thoroughly than the overflow it prevented: within 13M frames every
  sample was clamped, the policy gradient was identically zero, the entropy bonus
  ran unopposed and σ collapsed 0.36 → 0.0027 over 40M frames. Use a smooth squash
  or skip the update.
- **A non-finite loss skips the minibatch**; a non-finite *gradient norm* skips
  too, because `clip_grad_norm_` returns the norm before clipping and would
  otherwise scale every parameter by an infinity. Either one aborts after 25
  consecutive.
- **`train/saturated_fraction`** reports ratios beyond e²⁰, and 2 consecutive
  batches over 25% aborts. A policy that disowns the data it just collected has
  diverged.

**None of this was visible in the training curves.** During the worst collapse
`train/reward` *rose* from −1.46 to −0.54, because episodes ending in 15 steps
instead of 96 accrue less of the per-episode shaping cost, and `skipped_updates`
stayed at 0.

## Status

Arena, task logic, both backends, the PPO stack and the tournament are complete.
Standing is solved (750 of 750 steps, robust to a 1.0 m/s shove in 6 of 6 seeds)
and self-play produces real grappling, with the robots using their arms to control
the opponent. `action_scale` is measured, per joint.

Still to come: **cross-robot matchups** (both backends raise `NotImplementedError`
when the two sides differ, which is the largest gap against the original brief),
the Elo leaderboard, and opponent posture in the observation — the policy currently
sees only 10 numbers about its opponent and nothing about its joint state or lean.

## Licence

Vendored robot models under `assets/` retain their upstream licences.
