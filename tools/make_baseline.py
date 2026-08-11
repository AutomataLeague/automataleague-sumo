"""Write a scripted baseline artifact a tournament can load like any other entrant.

    python tools/make_baseline.py still -o baselines/still.pt
    python tools/make_baseline.py lean  -o baselines/lean.pt --gain 0.6

A leaderboard needs a fixed floor to be interpretable, and these are it: no
network, no training, a handful of scalars on disk. They also prove the
evaluation contract admits a policy this repo did not train — if a hand-written
baseline cannot enter cleanly, neither will anyone else's SAC.
"""

from __future__ import annotations

import argparse
import os

import torch

from automataleague_sumo.envs.sumo.observation import observation_dim
from automataleague_sumo.policy import check_policy, load_policy
from automataleague_sumo.robots import get_robot
from automataleague_sumo.scripted import KINDS, save_scripted_policy


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=KINDS)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--robot", default="g1")
    p.add_argument("--gain", type=float, default=0.6)
    p.add_argument("--env", default="sumo-1")
    args = p.parse_args()

    out = args.out or f"baselines/{args.kind}.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_scripted_policy(out, args.kind, robot=args.robot, gain=args.gain,
                         env_id=args.env)

    # Load it back through the public path and validate, so a broken artifact is
    # caught here rather than partway through someone's tournament.
    robot = get_robot(args.robot)
    policy = load_policy(out)
    check_policy(policy, obs_dim=observation_dim(robot), act_dim=robot.action_dim)
    print(f"wrote {out}  ({policy.info.algorithm}/{args.kind}, "
          f"{robot.name}, validated)")
    print(f"  torch.load says: {torch.load(out, weights_only=False)}")


if __name__ == "__main__":
    main()
