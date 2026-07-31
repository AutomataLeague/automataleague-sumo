"""How long does the robot last WITHOUT a policy? The bar any standing run must clear.

    python tools/baselines.py

A trained policy's episode length means nothing on its own. The G1 spawns in a
stance it cannot passively hold, so it survives a while and then falls over no
matter what is driving it — and a policy that has learned nothing useful still
produces a curve that rises off its random initialisation and looks like progress.

Measured on sumo-1 against a passive dummy (12 seeds, CPU backend):

    zero action     73.6 steps   the robot simply holds its home pose
    random U(-1,1)  55.5 steps   flailing is worse than doing nothing

So ~76 is the number to beat. The first standing run reached 66 steps with
exploration noise and 50 deterministically after ten million frames, which is to
say it had learned something actively worse than standing still — a fact entirely
invisible in a training curve that was rising the whole time.
"""

from __future__ import annotations

import argparse

import numpy as np

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import RewardConfig, TerminationConfig


def episode_lengths(env, policy, seeds, max_steps):
    lengths, outcomes = [], []
    n = env.action_dim
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        steps, outcome = max_steps, 0
        for t in range(max_steps):
            _, _, term, trunc, info = env.step(policy(rng, n), np.zeros(n))
            if term or trunc:
                steps, outcome = t + 1, info["outcome"]
                break
        lengths.append(steps)
        outcomes.append(outcome)
    return np.array(lengths), outcomes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sumo-1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()

    from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU

    # No pushes: this measures how long the robot lasts under its own dynamics,
    # which is the bar a trained policy has to clear. Shoving it would measure
    # something else and make the bar depend on the perturbation schedule.
    cfg = get_env_spec(args.env).config(
        opponent="zero", push_interval_steps=0, push_speed=0.0)
    tc = TerminationConfig()
    env = SumoEnvCPU(robot=args.robot, cfg=cfg, reward_cfg=RewardConfig(), term_cfg=tc)
    seeds = list(range(args.seeds))

    policies = {
        "zero action": lambda rng, n: np.zeros(n),
        "random U(-1,1)": lambda rng, n: rng.uniform(-1, 1, n),
        "small random U(-0.2,0.2)": lambda rng, n: rng.uniform(-0.2, 0.2, n),
    }
    print(f"{args.env} against a passive dummy, {args.seeds} seeds, "
          f"cap {tc.max_episode_steps} steps\n")
    print(f"{'policy':>26} {'mean':>7} {'std':>6} {'min':>5} {'max':>5}")
    best = 0.0
    for name, fn in policies.items():
        lengths, _ = episode_lengths(env, fn, seeds, tc.max_episode_steps)
        print(f"{name:>26} {lengths.mean():7.1f} {lengths.std():6.1f} "
              f"{lengths.min():5d} {lengths.max():5d}")
        best = max(best, float(lengths.mean()))

    print(f"\nA policy has learned nothing about balance until it beats "
          f"{best:.1f} steps.")
    print(f"That is {100 * best / tc.max_episode_steps:.0f}% of the "
          f"{tc.max_episode_steps}-step cap, so surviving a full episode is a "
          f"{tc.max_episode_steps / best:.1f}x improvement, not a marginal one.")


if __name__ == "__main__":
    main()
