"""Render two DIFFERENT policies fighting each other.

    python tools/render_versus.py old.pt new.pt -o versus.mp4 --episodes 5

Every other renderer drives both robots with one policy, which shows what a
policy does but never who is better. This puts one checkpoint on each side, so
the video is the head-to-head the round robin scores numerically.

Side A (blue chest) is the first checkpoint, side B (red chest) the second, and
they swap halfway so a result cannot be an artefact of which side of the ring a
policy started on.
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.env import configs_from_cfg
from automataleague_sumo.training.models import build_actor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint_a")
    p.add_argument("checkpoint_b")
    p.add_argument("-o", "--out", default="versus.mp4")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--camera", default="corner")
    p.add_argument("--size", type=int, nargs=2, default=[540, 960])
    return p.parse_args()


def load(path, cfg, robot):
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor = build_actor(cfg, robot, torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()
    frames = int(state.get("collected_frames") or 0)
    return actor, (f"{frames / 1e6:.0f}M" if frames else path.split("/")[-1][:-3])


def main():
    args = parse_args()
    cfg = OmegaConf.create(
        torch.load(args.checkpoint_a, map_location="cpu",
                   weights_only=False)["config"])
    robot = get_robot(cfg.env.robot)
    actor_a, name_a = load(args.checkpoint_a, cfg, robot)
    actor_b, name_b = load(args.checkpoint_b, cfg, robot)

    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc,
                     render_size=tuple(args.size))

    video = []
    wins = {name_a: 0, name_b: 0, "draw": 0}
    print(f"blue = {name_a}   red = {name_b}\n")
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for ep in range(args.episodes):
            # Swap sides halfway: a policy that only wins from one side of the
            # ring has not beaten anything.
            swapped = ep >= args.episodes // 2
            blue, red = (actor_b, actor_a) if swapped else (actor_a, actor_b)
            blue_name = name_b if swapped else name_a
            red_name = name_a if swapped else name_b

            obs_a, obs_b = env.reset(seed=ep)
            steps, outcome = 0, 0
            for _ in range(tc.max_episode_steps):
                act_a = blue(TensorDict(
                    {"observation": torch.as_tensor(obs_a, dtype=torch.float32)[None]},
                    batch_size=[1]))["action"].numpy()[0]
                act_b = red(TensorDict(
                    {"observation": torch.as_tensor(obs_b, dtype=torch.float32)[None]},
                    batch_size=[1]))["action"].numpy()[0]
                (obs_a, obs_b), _, term, trunc, info = env.step(act_a, act_b)
                video.append(env.render(camera=args.camera))
                steps, outcome = steps + 1, info["outcome"]
                if term or trunc:
                    break

            winner = {1: blue_name, 2: red_name}.get(outcome, "draw")
            wins[winner] = wins.get(winner, 0) + 1
            print(f"  ep{ep}: {steps:>3} steps, {winner} wins"
                  f"{'   (sides swapped)' if swapped else ''}")

    imageio.mimsave(args.out, np.stack(video), fps=args.fps)
    print(f"\nwrote {args.out}  ({len(video)} frames)")
    print(f"  {name_a}: {wins.get(name_a, 0)}   {name_b}: {wins.get(name_b, 0)}"
          f"   draws: {wins.get('draw', 0)}")


if __name__ == "__main__":
    main()
