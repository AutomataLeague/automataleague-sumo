"""Render a trained checkpoint as a video, to see what the policy actually does.

    python tools/render_policy.py checkpoints/sumo1_L0/ppo_best.pt -o duel.mp4

A training curve says a policy survives 66 steps; it does not say whether it is
balancing, toppling slowly, or kneeling. Watch the video before drawing a
conclusion about why a number is where it is.

Runs on the CPU backend, which shares the model and every task-logic module with
the GPU trainer, so what is rendered is the same duel that was trained. Needs
MUJOCO_GL=egl on a headless box.
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import RewardConfig, TerminationConfig
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.models import build_actor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("-o", "--out", default="duel.mp4")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--camera", default="corner")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--both-sides", action="store_true",
                   help="drive BOTH robots with this policy, even at a level whose "
                        "opponent is normally a dummy. The observation carries no "
                        "side identity, so a policy trained on side A should work "
                        "unchanged on side B — this is how to check that it does.")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean. Training used "
                        "sampling, so a deterministic rollout is a different policy "
                        "than the one whose curve you are looking at.")
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    level = int(state.get("level", 0))

    robot = get_robot(cfg.env.robot)
    actor = build_actor(cfg, robot, torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    # Rebuild exactly the configuration this checkpoint trained under, weight
    # overrides included — rendering it under the shipped defaults would show the
    # policy being judged by a reward it never saw.
    sumo_cfg = get_env_spec(cfg.env.name).config(level)
    rc, tc = RewardConfig(), TerminationConfig()
    for group, target in ((getattr(cfg.env, "reward_weights", None), rc),
                          (getattr(cfg.env, "termination", None), tc)):
        if group is None:
            continue
        for key in vars(target):
            value = getattr(group, key, None)
            if value is not None:
                setattr(target, key, type(getattr(target, key))(value))

    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc)
    both_sides = args.both_sides or sumo_cfg.opponent == "self"
    mode = ExplorationType.RANDOM if args.stochastic else ExplorationType.DETERMINISTIC

    frames, summaries = [], []
    for ep in range(args.episodes):
        obs_a, obs_b = env.reset(seed=args.seed + ep)
        steps, outcome = 0, 0
        with set_exploration_type(mode), torch.no_grad():
            for _ in range(tc.max_episode_steps):
                stacked = np.stack([obs_a, obs_b]) if both_sides else obs_a[None]
                from tensordict import TensorDict

                td = TensorDict({"observation": torch.as_tensor(stacked,
                                                                dtype=torch.float32)},
                                batch_size=[stacked.shape[0]])
                act = actor(td)["action"].numpy()
                act_b = act[1] if both_sides else np.zeros_like(act[0])
                (obs_a, obs_b), _, term, trunc, info = env.step(act[0], act_b)
                frames.append(env.render(camera=args.camera))
                steps += 1
                outcome = info["outcome"]
                if term or trunc:
                    break
        summaries.append((ep, steps, outcome))
        print(f"  episode {ep}: {steps} steps, duel outcome code {outcome} "
              f"(0 ongoing, 1 A wins, 2 B wins, 3 draw)")

    imageio.mimsave(args.out, frames, fps=args.fps)
    mean_steps = sum(s for _, s, _ in summaries) / len(summaries)
    print(f"wrote {args.out}  ({len(frames)} frames, "
          f"{mean_steps:.0f} steps/episode mean over {args.episodes} episodes)")


if __name__ == "__main__":
    main()
