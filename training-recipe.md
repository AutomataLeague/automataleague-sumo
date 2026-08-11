# Training a sumo policy

How the shipped policy was trained, what we checked at each stage, and the
things that went wrong. The short version lives in the [README](README.md); this
is the long one.

Frame counts are the reproducible quantity here; wall-clock depends entirely on
your GPU. `env.num_envs` is the number of parallel **duels**, each holding both
robots, so the policy batch under self-play is twice that. Lower it if you run
out of memory; it changes throughput, not the recipe.

All commands assume `MUJOCO_GL=egl`, because evaluation renders headlessly.

---

## What we learned

Seven findings from this project specifically. Each one cost GPU time, each one
surprised us, and the rest of this document is the evidence behind them.

**1. The training curves here are blind, and not by accident.** Self-play win
rate is pinned at exactly 0.5 by construction: every duel makes one winner and
one loser, and under a shared policy both rows land in the same batch. Reward is
near-flat for a related reason, and sat at about −1.5 in *every* run we did,
including the healthy ones and the one that had silently gone NaN 500M frames
earlier. Progress can only be measured by playing checkpoints against frozen
past checkpoints. Build that tournament before you spend the GPU time, not after.

**2. A 1B-frame plateau turned out to be the collision model, not the learning.**
Two separate runs flattened at the same 58% win rate and stayed there for 600M
frames, which reads exactly like "needs more compute". The vendored MJX model was
stripped for locomotion: **12 of 30 bodies had no collision geom at all**,
including both shoulders and both forearms, and **56% of all robot-to-robot
contact was missing**. Both policies had been exploiting the same holes
symmetrically, so the stalemate looked stable and earned.

The method is cheap and worth copying: replay one trained policy's exact actions
through both the stock and the corrected model and count contacts. Same motion,
two models, so the difference is contact that was *missing* rather than contact
caused by the change. Ours went 1,877 → 4,225. Audit a vendored asset against
*your* task, not the one it shipped for.

**3. A policy that never lost a duel had no balance at all.** It held a stance
for the full 750-step episode, and a gentle 0.5 m/s shove put it down in **6 of 6
seeds**. Standing still and staying upright are different skills, and only the
first one shows up in an episode-length metric. The fix is perturbations the
policy *cannot see* in its observation: a disturbance you can anticipate is a
control input, not a disturbance.

**4. A hard capability cap is indistinguishable from a policy choice.** The G1's
arms hung by its sides for an entire run, which reads as the policy deciding not
to use them. In fact `action_scale` is a symmetric window around the **home**
pose, and the G1's home pose already bends the elbow 1.28 rad, so at a uniform
0.5 the elbow could never get within 45° of straight. The policy was not
declining to reach; reaching was not in the action space. Per-joint scaling
(`RobotSpec.joint_scale`) raised measured arm reach from 0.42 m to 0.59 m.

The generalizable check: compare your action window against the **home pose**,
not against the joint limits. Limits never bound anywhere in our usable range, so
they told us nothing, while the home pose silently cost us half the arm.

**5. Twice, what looked like a failing policy was a policy correctly obeying a
reward we had mis-specified.** Surviving at the spawn radius scored **−0.130 per
step**, so creeping inward really was optimal. Separately, with per-step weights,
a shaping term worth 0.3 a step outscored a terminal +10 by **thirteen times**
over an episode, and no number anywhere revealed it. Denominating every weight as
a whole-episode value makes the comparison readable at a glance, and the config
now refuses to construct if the shaping budget can outscore a win.

**6. Bounding a runaway with `clamp()` destroyed a run far worse than the
overflow it prevented.** `torch.clamp` has zero gradient outside its range, so
clamping the PPO importance ratio deleted the policy gradient on exactly the
samples that had run away, and it is self-reinforcing: within 13M frames every
sample was clamped, the entropy bonus ran unopposed, and σ collapsed from 0.36 to
0.0027 over 40M frames. Worst of all it was silent, because `train/reward` *rose*
from −1.46 to −0.54 throughout the collapse: shorter episodes accrue less of the
per-episode shaping cost. Use a smooth squash or skip the update, and bound the
source rather than the symptom.

**7. More frames is not reliably more skill, and the newest checkpoint is often
not the best one.** An earlier checkpoint beat the final one **four separate
times**, once by 58.3% over a thousand duels. And 590M frames beyond 1B bought a
57.4% head-to-head edge over the policy they started from, an order of magnitude
less than a single collision fix delivered. Every warm start also dips before it
recovers: 100M frames in, the continuation scored **37.6%** against the very
checkpoint it started from.

We assumed that dip was damage done by restarting the learning-rate anneal at
full strength on a converged policy, and ran the control: same start, same seed,
same 300M budget, constant `lr=1e-4` with no annealing. **The hypothesis was
wrong, twice over.**

| vs the shared starting checkpoint | +100M | +200M | +300M |
| --- | --- | --- | --- |
| default schedule (3e-4, annealed) | 37.6% | **83.2%** | 74.2% |
| constant reduced rate (1e-4) | 33.9% | 59.9% | 53.0% |

The reduced rate did not prevent the dip; it was marginally deeper. And the
default schedule beat it at every matched point head to head (67.1%, 63.5%,
63.7%, all past 6 sigma). So the dip is not damage from the restart, it is the
policy moving off the optimum it had settled into, and the run that moves further
comes back higher. Lowering the learning rate just learns less.

**The common thread:** in nearly every case above, the instrument was silent. A
test that passed whatever the code did, a win rate that was a constant by
construction, a reward curve identical whether the run was healthy or wrecked.
When something is wrong and nothing is complaining, suspect the measurement
before the model.

---

## The recipe

Four stages, each with a gate to check before spending the next block of GPU
time. Every one of these gates exists because that stage silently failed here at
least once, and none of the failures were visible in a training curve.

### 0. Preflight (no training)

```bash
MUJOCO_GL=egl uv run python tools/reward_balance.py     # exits non-zero if the reward is wrong
MUJOCO_GL=egl uv run python tools/baselines.py          # what does DOING NOTHING score?
MUJOCO_GL=egl uv run python tools/warp_smoke.py --num-envs 2048
```

**Why.** The shipped reward once made surviving at the spawn radius worth
**−0.130 per step**, so the optimal policy was to creep inward rather than
stand, and every training curve looked like progress. `reward_balance.py` prints
each term against `win` and exits non-zero if survival does not pay.

`baselines.py` answers a question a training curve cannot: the G1 spawns in a
stance it cannot passively hold, so it survives a while and falls over no matter
what drives it. Zero actions last **73.6 steps**, small random noise 75.9. The
first "successful" standing policy scored **66**, meaning it had learned
something actively worse than doing nothing.

`warp_smoke.py` checks contact headroom. **MuJoCo-Warp silently drops contacts
past `nconmax` instead of raising**, which reads as two robots passing through
each other rather than as an error.

### 1. Bootstrap standing against a dummy (~50M frames)

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=standing \
    env.num_envs=2048 collector.total_frames=50_000_000 \
    env.arena.opponent=zero env.reward_weights.push=0
```

Two fresh robots both collapse in about 1.5 s. That scores as a simultaneous
loss, which is a draw, so the win term has nothing to work with. The robot
learns to stand first.

`push` is zeroed because the dummy's radius is decided by how it happens to
topple, which the learner cannot influence. Paying for it is pure variance in
the advantage estimate.

The dummy **cannot lose**, derived from the opponent mode rather than configured
separately. A zero-action G1 sags ~0.45 m in ~1.2 s against a 0.431 m fall
threshold, so under the ordinary rules the learner would collect a free +10
about 60 steps into every episode.

> **Gate.** Episode length must beat the `baselines.py` number, and the policy
> must survive a shove:
>
> ```bash
> MUJOCO_GL=egl uv run python tools/push_test.py checkpoints/standing/ppo_best.pt
> ```
>
> **A held pose is not balance.** The first standing policy survived a full
> 750-step episode, never lost a duel, and fell to a 0.5 m/s shove in **6 of 6
> seeds**. Unobserved push perturbations fixed it: 6 of 6 at 1.0 m/s. They are
> deliberately absent from the observation, because a disturbance the policy can
> see coming is a control input, not a disturbance.

### 2. Self-play (1B frames)

```bash
MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=sumo1 \
    env.num_envs=2048 collector.total_frames=1_000_000_000 \
    init_checkpoint=checkpoints/standing/ppo_best.pt
```

One policy drives both robots. That is only sound because the observation is
written entirely in each robot's own base frame, so it carries no absolute side
identity; `tests/sumo/test_observation.py` asserts it by rotating the whole
world.

> **Gate.** Nothing in the training curves will tell you whether this worked,
> which is what stage 3 is for. The one thing worth watching live is
> `train/saturated_fraction`; a rising value means the policy is diverging, and
> the run aborts itself if it stays high.

### 3. Find out which checkpoint is actually best

```bash
MUJOCO_GL=egl uv run python tools/round_robin.py checkpoints/sumo1/ppo_eval_*.pt
MUJOCO_GL=egl uv run python tools/render_progression.py checkpoints/sumo1/ppo_eval_*.pt
```

> **Do not assume the last checkpoint, or `ppo_best.pt`, is the best one.** An
> earlier checkpoint beat the final one **four separate times** here, once by
> 58.3% over a thousand duels. Warm-start the next run from whatever the
> tournament ranks first.

### Optional: continue past 1B

Use the **default schedule**. It is tempting to lower the learning rate when
continuing an already-converged policy, and we measured that against the default
with everything else held fixed: it is worse at every point (a 63-67% head-to-head
loss) and it does not avoid the dip it was meant to avoid.

Expect the continuation to score *below* its own starting checkpoint around 100M
frames in, and to pass it by 200M. That dip is the policy leaving the optimum it
had settled into, not damage, so do not kill the run when you see it.

```bash
MUJOCO_GL=egl uv run python tools/policy_saturation.py checkpoints/sumo1/<best>.pt

MUJOCO_GL=egl uv run python examples/ppo_sumo.py run_name=sumo1_continue \
    env.num_envs=2048 collector.total_frames=300_000_000 \
    init_checkpoint=checkpoints/sumo1/<best>.pt
```

---

## Measuring progress

**Self-play win rate is pinned at exactly 0.5 by construction.** Every duel makes
one winner and one loser, and under a shared policy both rows sit in the same
batch. It is a structural identity, not a measurement; if it drifts off 0.5 that
is a bug in the outcome bookkeeping. Training reward is near-flat for a related
reason: only the terminal term is zero sum, so as both sides improve their
returns cancel. It sat at about −1.5 in **every** run, including the ones that
worked and the one that had silently gone NaN.

So progress is measured by `tools/round_robin.py`, which plays every checkpoint
against every other, both orderings so any side advantage cancels, counting only
each world's **first** conclusion because worlds auto-reset and fast pairings
would otherwise be over-weighted. It costs a tiny fraction of what the run
itself cost, so it should gate the next run rather than merely follow the last.

**A field win rate is only meaningful relative to its field.** The average across
participants is 50% by construction, so adding stronger checkpoints moves
everyone's number. The same 1000M checkpoint scored 76.5% against one field and
67.4% against another, with identical weights. Quote a **head to head** between
two named policies instead; that is invariant.

Across a chain of warm starts, label by hand, because each run restarts its own
frame counter and a 1000M checkpoint would otherwise sort after the 300M run
that continues from it:

```bash
MUJOCO_GL=egl uv run python tools/round_robin.py \
    1000M=checkpoints/v6/ppo_eval_1000013824.pt \
    1290M=checkpoints/v7/ppo_eval_290062336.pt
```

**Transitivity** is reported alongside: ordered triples where A beats B, B beats
C and C beats A. Measured **0 of 336** and **0 of 504** on separate runs. No
cycling means improvement is a strict ordering, so training against the current
policy alone suffices and an opponent pool would buy nothing. That is the
evidence behind the design decision rather than an argument for it.

Judge videos on **five duels, not one**. The same checkpoint produced a 46-step
duel and a 160-step one back to back, and a single clip supports whichever story
you already believe.

---

## Results

### The vendored collision model was missing 56% of contact

Menagerie's `g1_mjx.xml` is stripped for *locomotion*, where the only thing that
touches anything is the feet. For wrestling that is exactly backwards: the upper
body is where the whole sport happens.

**12 of 30 bodies had no collision geom at all**, including both shoulders and
both forearms, and the head sphere sat 45 mm high covering only the top 57%. An
arm swept at face height passed straight through.

Replaying one trained policy through both models, so the difference is contact
that was *missing* rather than contact caused by the change:

| | robot-to-robot contacts |
| --- | --- |
| stripped model | 1,877 |
| corrected model | **4,225** |
| missing | **2,348, or 56%** |

The upper arm alone, where a grip is taken, accounted for 936 contacts and
previously had exactly zero.

This **capped training**. Two separate 1B-frame runs plateaued at ~58% win rate
and went flat for 600M frames; we had assumed we needed more compute. After
grafting five primitives per robot via `RobotSpec.extra_colliders` (mass 0, so
inertia is unchanged at 66.6823 kg, and `assets/` stays byte-identical):

| training frames | win rate against the field |
| --- | --- |
| 20M | 11.6% |
| 300M | 44.5% |
| 680M | 61.2% |
| **1000M** | **76.5%** |

Improvement at every step, and the new policy beats the old champion **910 to
106** over 1,024 duels.

**Audit a vendored asset against your task, not the one it shipped for.**

### Past 1B there is headroom, but a warm restart costs most of it

A lineage of 1000M → 1290M → 1590M cumulative frames:

| head to head | result | duels | significance |
| --- | --- | --- | --- |
| best continuation (1490M) vs 1000M | **57.4%** | 1013 | +4.7σ |
| final checkpoint (1590M) vs 1000M | 47.9% | 1016 | −1.3σ |
| 1490M vs the final 1590M | 58.3% | 1007 | +5.3σ |

590M further frames bought a real but modest gain, an order of magnitude smaller
than the collision fix, and the run's *last* checkpoint was no better than its
start.

### `action_scale`, measured rather than guessed

`q_target = home + action_scale * action`, action in [-1, 1]. It is a hard
geometric cap on the poses the policy can ask for, not a gain and not a speed
limit. `tools/measure_reach.py` reports the envelope it buys:

| `action_scale` | stance width | stride | max crouch | reach | clipped by joint limits |
| --- | --- | --- | --- | --- | --- |
| 0.2 (11°) | 0.52 m | 0.34 m | 0.05 m | 0.32 m | 0% |
| 0.3 (17°) | 0.66 m | 0.50 m | 0.11 m | 0.36 m | 1% |
| **0.5 (29°)** | 0.89 m | 0.78 m | 0.29 m | 0.42 m | 4% |
| 0.7 (40°) | 1.11 m | 1.01 m | 0.48 m | 0.49 m | 8% |
| 1.0 (57°) | 1.41 m | 1.17 m | 0.77 m | 0.56 m | 15% |

Joint limits never bind over this range, so the choice trades against
controllability rather than against the robot's geometry.

**0.5 is the largest uniform scale that cannot crouch into an immediate loss.**
The base starts at 0.784 m and the down-rule fires below 0.431 m, so a drop past
0.353 m loses outright. At 0.5 the deepest commandable crouch is 0.294 m. From
0.7 the action space contains poses that lose the duel instantly.

**One number cannot serve the whole robot**, because the scale is a symmetric
window around the *home* pose. The G1's home pose bends the elbows 1.28 rad, so
at a uniform 0.5 the elbow can only reach 0.78 rad: the arms can never get within
45° of straight and simply hang. That is a hard cap, and it looks exactly like a
policy declining to use its arms. `RobotSpec.joint_scale` gives per-joint
multipliers; the G1 keeps 0.5 on the legs and takes 2.5x on the arms, raising
measured arm reach from 0.42 m to 0.59 m.

Change it **between** runs with a warm start, never on a schedule inside one: it
changes what the same action numbers mean.

---

## Numerical stability

Three runs died on non-finite numbers. Every one traced to a single unbounded
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
`tools/policy_saturation.py` measures a checkpoint before you warm-start from it;
a healthy 290M policy reads median |loc| 0.36, 99th percentile 2.07, max 3.62.

Three more things worth knowing:

**TorchRL's `ClipPPOLoss` bounds the ratio on only one branch.** `gain1` uses an
unclamped `log_weight.exp()`, and PPO's pessimistic `min` selects it exactly when
the advantage is negative, so an overflowing ratio bypasses the clip entirely.

**Do not fix this by clamping.** `torch.clamp` has zero gradient outside its
range, so it deletes the corrective force on the samples that most need pulling
back, and it is self-reinforcing. Clamping the importance ratio destroyed a run
far more thoroughly than the overflow it prevented: within 13M frames every
sample was clamped, the policy gradient was identically zero, the entropy bonus
ran unopposed, and σ collapsed 0.36 → 0.0027 over 40M frames. Use a smooth squash
or skip the update, and bound the source.

**None of it was visible in the training curves.** During that collapse
`train/reward` *rose* from −1.46 to −0.54, because episodes ending in 15 steps
instead of 96 accrue less of the per-episode shaping cost, and
`train/skipped_updates` stayed at 0.

The guards that exist now: a non-finite loss skips the minibatch; a non-finite
gradient *norm* skips too, because `clip_grad_norm_` returns the norm before
clipping and would otherwise scale every parameter by an infinity; either aborts
after 25 consecutive. `train/saturated_fraction` reports ratios beyond e²⁰, and
two consecutive batches over 25% aborts the run.

---

## Other things that cost real time

- **A foot outside the ring did not lose.** Before the rule was tightened to
  "beyond the rim **and** below the surface", a robot stood on the floor outside
  the ring for 7.1% of steps with the duel continuing.
- **Checkpoint scoring is task-dependent, and getting it backwards is silent.**
  The eval score was `episode_length + 100*win_rate + 10*opp_radius`, written
  when the task was survival. Against a real opponent `win_rate` is a constant
  0.5 and a long episode is a *stalemate*, so it selected a checkpoint drawing
  65% of its duels over one drawing 0% and driving its opponent twice as far.
- **TorchRL's `WandbLogger` never advanced wandb's step counter**, so a whole 1B
  run logged as a single row. The local `checkpoints/<run>/metrics.jsonl` is what
  saved the analysis. Keep it.
- **Observations need clipping.** Contact drives joint velocities to 49.5 rad/s
  into an unnormalised network; `OBS_CLIP` is 25.
- **A/B asymmetry below ~1e-4 on the first step is expected**, not a bug. The
  solver is capped at 5 iterations and stops before convergence, so the two
  robots' different constraint orderings break ties differently (measured
  4.2e-6). A genuinely crossed index measures 4.4e-3; reproduce it with
  `tools/warp_smoke.py --inject-crossed-index`.
