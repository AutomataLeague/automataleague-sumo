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
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.envs.sumo.overlay import draw_hud, draw_verdict
from automataleague_sumo.envs.sumo.scene import TEAM_A_RGB, TEAM_B_RGB
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.env import configs_from_cfg
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
                   help="drive BOTH robots with this policy, even when the "
                        "opponent is a dummy. The observation carries no "
                        "side identity, so a policy trained on side A should work "
                        "unchanged on side B — this is how to check that it does.")
    # Clean by DEFAULT. The video is source footage: burnt-in captions cover the
    # robots and cannot be undone downstream, and the media pipeline composes far
    # better overlays from the results JSON this always writes beside the video.
    # --overlay is for eyeballing a policy yourself, not for anything published.
    p.add_argument("--overlay", action="store_true",
                   help="burn a round/score HUD and a winner card into the frames. "
                        "Handy for a quick look; prefer the results JSON and let "
                        "the media pipeline draw its own")
    p.add_argument("--hold", type=float, default=1.4, metavar="SECONDS",
                   help="with --overlay, pause this long on each duel's last frame")
    p.add_argument("--results", default=None, metavar="PATH",
                   help="where to write the per-round facts (default: alongside "
                        "the video, as <out>.json)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the mean. Training used "
                        "sampling, so a deterministic rollout is a different policy "
                        "than the one whose curve you are looking at.")
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])

    robot = get_robot(cfg.env.robot)
    actor = build_actor(cfg, robot, torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    # Rebuild exactly the configuration this checkpoint trained under, arena and
    # weight overrides included. Rendering it under the shipped defaults would show
    # the policy in a duel it never trained in and judged by a reward it never saw.
    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc)
    both_sides = args.both_sides or sumo_cfg.opponent == "self"
    mode = ExplorationType.RANDOM if args.stochastic else ExplorationType.DETERMINISTIC

    frames, summaries, rounds = [], [], []
    tally = {A_WINS: 0, B_WINS: 0, DRAW: 0}
    for ep in range(args.episodes):
        start_frame = len(frames)
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
                frame = env.render(camera=args.camera)
                if args.overlay:
                    frame = draw_hud(
                        frame, f"round {ep + 1} of {args.episodes}",
                        f"blue {tally[A_WINS]}   red {tally[B_WINS]}"
                        + (f"   draw {tally[DRAW]}" if tally[DRAW] else ""))
                frames.append(frame)
                steps += 1
                outcome = info["outcome"]
                if term or trunc:
                    break

        tally[outcome] = tally.get(outcome, 0) + 1
        end_frame = len(frames) - 1
        # Hold the LAST frame, unannotated underneath, so the final pose stays
        # visible while the verdict is on screen.
        held = int(round(args.hold * args.fps))
        if held and args.overlay:
            last = env.render(camera=args.camera)
            card = draw_verdict(last, outcome,
                                f"{steps} steps   {steps / args.fps:.1f} s")
            frames.extend([card] * held)
        summaries.append((ep, steps, outcome))
        rounds.append({
            "round": ep + 1,
            "seed": args.seed + ep,
            # Both the code and the side letter: the code is what the env returns,
            # the letter is what a caption needs, and translating in the consumer
            # is how a "blue won" card ends up on the wrong robot.
            "outcome_code": int(outcome),
            "winner": {A_WINS: "a", B_WINS: "b"}.get(outcome),
            "winner_team": {A_WINS: "blue", B_WINS: "red"}.get(outcome, "draw"),
            "steps": steps,
            "duration_s": round(steps / args.fps, 3),
            # Frame range in the WRITTEN video, so an overlay can be placed
            # without recounting anything.
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
        print(f"  episode {ep}: {steps} steps, duel outcome code {outcome} "
              f"(0 ongoing, 1 A wins, 2 B wins, 3 draw)")

    imageio.mimsave(args.out, frames, fps=args.fps)
    mean_steps = sum(s for _, s, _ in summaries) / len(summaries)
    print(f"wrote {args.out}  ({len(frames)} frames, "
          f"{mean_steps:.0f} steps/episode mean over {args.episodes} episodes)")
    print(f"  blue {tally[A_WINS]}   red {tally[B_WINS]}   draws {tally[DRAW]}")

    results_path = args.results or (os.path.splitext(args.out)[0] + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump({
            "env_id": str(cfg.env.name),
            "robot": str(cfg.env.robot),
            "checkpoint": args.checkpoint,
            "policy": {
                "algorithm": "ppo",
                "frames": int(state.get("collected_frames") or 0) or None,
                "env_version": state.get("env_version"),
            },
            "video": os.path.basename(args.out),
            "fps": args.fps,
            "dt_s": round(1.0 / args.fps, 5),
            "camera": args.camera,
            "overlay_burnt_in": bool(args.overlay),
            # Which chest colour is which side, so a caption cannot be put on the
            # wrong robot. Same source the renderer paints from.
            "teams": {"a": {"name": "blue", "rgb": list(TEAM_A_RGB)},
                      "b": {"name": "red", "rgb": list(TEAM_B_RGB)}},
            "tally": {"a": tally[A_WINS], "b": tally[B_WINS], "draw": tally[DRAW]},
            "episodes": args.episodes,
            "mean_steps": round(mean_steps, 1),
            "rounds": rounds,
        }, fh, indent=2)
    print(f"wrote {results_path}  (per-round facts for overlays)")


if __name__ == "__main__":
    main()
