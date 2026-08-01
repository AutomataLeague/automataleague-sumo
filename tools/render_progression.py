"""Render several checkpoints to video, individually and side by side.

    python tools/render_progression.py checkpoints/selfplay_v4/ppo_eval_*.pt \\
        --pick 5 --episodes 3 --out renders/

Writes one mp4 per checkpoint plus a grid mp4 showing all of them at once. The
grid is the one to watch first: a still cannot show balance, recovery or timing,
and a single duel cannot show consistency, so judging a policy from one frame
sequence is judging it on the one thing a frame CAN show, which is pose.

Shorter duels are padded with their final frame so every panel stays in step and
a policy that ends its bouts early is visibly done rather than looping.
"""

from __future__ import annotations

import argparse
import os

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
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--pick", type=int, default=5,
                   help="subsample evenly to this many checkpoints")
    p.add_argument("--episodes", type=int, default=3,
                   help="duels per checkpoint. One duel shows a result, not a policy.")
    p.add_argument("--out", default="renders")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--size", type=int, nargs=2, default=[420, 720])
    p.add_argument("--camera", default="corner")
    return p.parse_args()


def frames_of(path: str) -> int:
    try:
        return int(torch.load(path, map_location="cpu",
                              weights_only=False).get("collected_frames") or 0)
    except Exception:
        return 0


def roll(path: str, episodes: int, size, camera: str):
    """Every frame of `episodes` duels, plus a one-line summary of each."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    actor = build_actor(cfg, get_robot(cfg.env.robot), torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc,
                     render_size=tuple(size))
    names = {1: "blue", 2: "red", 3: "draw"}

    video, summary = [], []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for ep in range(episodes):
            obs_a, obs_b = env.reset(seed=ep)
            steps, outcome = 0, 0
            for _ in range(tc.max_episode_steps):
                td = TensorDict(
                    {"observation": torch.as_tensor(
                        np.stack([obs_a, obs_b]), dtype=torch.float32)}, batch_size=[2])
                act = actor(td)["action"].numpy()
                (obs_a, obs_b), _, term, trunc, info = env.step(act[0], act[1])
                video.append(env.render(camera=camera))
                steps, outcome = steps + 1, info["outcome"]
                if term or trunc:
                    break
            summary.append(f"ep{ep}: {steps} steps, {names.get(outcome, '?')}")
    return np.stack(video), summary


def grid(clips, labels, fps, out_path):
    """One mp4 with every checkpoint playing at once, padded to a common length."""
    longest = max(len(c) for c in clips)
    padded = [np.concatenate([c, np.repeat(c[-1:], longest - len(c), axis=0)])
              if len(c) < longest else c for c in clips]

    cols = min(3, len(padded))
    rows = (len(padded) + cols - 1) // cols
    h, w = padded[0].shape[1:3]
    blank = np.zeros((longest, h, w, 3), dtype=padded[0].dtype)
    cells = padded + [blank] * (rows * cols - len(padded))

    frames = np.concatenate(
        [np.concatenate(cells[r * cols:(r + 1) * cols], axis=2) for r in range(rows)],
        axis=1)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"wrote {out_path}  ({rows}x{cols} grid, {longest} frames)")
    for label in labels:
        print(f"    {label}")


def main():
    args = parse_args()
    paths = sorted(set(args.checkpoints), key=lambda p: (frames_of(p), p))
    if len(paths) > args.pick:
        idx = np.linspace(0, len(paths) - 1, args.pick).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx))]
    os.makedirs(args.out, exist_ok=True)

    clips, labels = [], []
    for path in paths:
        millions = frames_of(path) / 1e6
        tag = f"{millions:.0f}M" if millions else os.path.basename(path)[:-3]
        video, summary = roll(path, args.episodes, args.size, args.camera)
        single = os.path.join(args.out, f"policy_{tag}.mp4")
        imageio.mimsave(single, video, fps=args.fps)
        mean_steps = len(video) / args.episodes
        print(f"wrote {single}  ({len(video)} frames, {mean_steps:.0f} steps/duel)")
        for line in summary:
            print(f"    {line}")
        clips.append(video)
        labels.append(f"{tag}: {mean_steps:.0f} steps/duel  " + "; ".join(summary))

    if len(clips) > 1:
        grid(clips, labels, args.fps, os.path.join(args.out, "progression_grid.mp4"))


if __name__ == "__main__":
    main()
