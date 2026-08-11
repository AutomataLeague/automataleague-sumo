"""How far outside the tanh range does the policy's mean actually go?

    python tools/policy_saturation.py checkpoints/contact_v7/ppo_eval_290062336.pt

The policy is a TanhNormal: it emits a pre-squash mean `loc`, samples around it
with std `scale`, and squashes with tanh. Everything past |loc| ~ 3 is the same
action, because tanh(3) = 0.995 already. But nothing bounds `loc`, and its
log-prob carries a -log(1 - a^2) jacobian term that grows without limit as the
action approaches the bound.

That term is why three runs of this project have died on non-finite numbers. The
importance ratio is exp(new_log_prob - old_log_prob), and once log-probs are in
the thousands their differences overflow float32, which happens at 88. A
diagnostic dump caught log_prob_absmax at 3427 with every other quantity in the
network healthy.

This measures the thing the training curves cannot show: the distribution of
|loc| over real rollouts, how much of it is past the point where tanh saturates,
and the resulting log-prob magnitude. Run it on a checkpoint BEFORE warm-starting
from it, the way baselines.py is run before believing an episode length.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.env import configs_from_cfg
from automataleague_sumo.training.models import build_actor

# tanh(3) = 0.995, so a mean past this commands an action that is already at the
# bound; more magnitude buys no additional motion and only inflates the jacobian.
SATURATED = 3.0
# exp() overflows float32 above this, so a log-prob DIFFERENCE beyond it makes the
# importance ratio infinite. Log-probs of this size are the fuel for that.
OVERFLOW = 88.0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    actor = build_actor(cfg, get_robot(cfg.env.robot), torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc)

    locs, log_probs, actions = [], [], []
    # Stochastic, not deterministic: the log-probs that break training are the ones
    # of SAMPLED actions during collection, and the deterministic mean never
    # produces the tail this exists to find.
    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        for ep in range(args.episodes):
            obs_a, obs_b = env.reset(seed=args.seed + ep)
            for _ in range(tc.max_episode_steps):
                out = []
                for obs in (obs_a, obs_b):
                    td = actor(TensorDict(
                        {"observation": torch.as_tensor(obs, dtype=torch.float32)[None]},
                        batch_size=[1]))
                    locs.append(td["loc"].numpy()[0])
                    actions.append(td["action"].numpy()[0])
                    key = ("action_log_prob" if "action_log_prob" in td.keys()
                           else "sample_log_prob")
                    log_probs.append(float(td[key]))
                    out.append(td["action"].numpy()[0])
                (obs_a, obs_b), _, term, trunc, _ = env.step(out[0], out[1])
                if term or trunc:
                    break

    loc = np.abs(np.stack(locs))
    lp = np.abs(np.array(log_probs))
    act = np.abs(np.stack(actions))
    n_joint = loc.size

    print(f"{args.checkpoint}\n{len(log_probs)} policy evaluations over "
          f"{args.episodes} episodes, {n_joint} joint outputs\n")
    print(f"{'|loc|':>26} {'value':>10}")
    for name, value in [("median", np.median(loc)), ("mean", loc.mean()),
                        ("99th pct", np.percentile(loc, 99)), ("max", loc.max())]:
        print(f"{name:>26} {value:10.3f}")
    past = float((loc > SATURATED).mean())
    print(f"\n{100 * past:.2f}% of joint outputs are past |loc| = {SATURATED} "
          f"(tanh already 0.995)")
    print(f"{100 * float((act > 1 - 1e-6).mean()):.2f}% of actions are within "
          f"1e-6 of the bound")

    print(f"\n{'|log_prob|':>26} {'value':>10}")
    for name, value in [("median", np.median(lp)), ("99th pct", np.percentile(lp, 99)),
                        ("max", lp.max())]:
        print(f"{name:>26} {value:10.3f}")

    headroom = OVERFLOW - lp.max()
    print(f"\nfloat32 exp() overflows at a log-ratio of {OVERFLOW:.0f}. The largest "
          f"log-prob here is {lp.max():.1f},")
    if headroom > 0:
        print(f"so a single update would have to move it by {headroom:.1f} to "
              f"overflow. Headroom, not safety.")
    else:
        print(f"which ALONE exceeds it. Any shift in the mean overflows the ratio "
              f"and the loss is infinite.")
    raise SystemExit(0 if lp.max() < OVERFLOW else 1)


if __name__ == "__main__":
    main()
