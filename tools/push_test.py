"""Can a standing policy survive a shove? The prerequisite for any fighting at all.

    python tools/push_test.py checkpoints/sumo1_L0/ppo_best.pt

A policy trained against a collapsed dummy has never been touched. It can hold a
stance for a full episode and still fall over the instant an opponent leans on
it, and no standing metric would show that. This applies a horizontal impulse to
the base partway through an episode and reports whether the robot recovers.

Both robots are driven by the same policy, so this also exercises the side B
transfer the observation's rotation invariance is supposed to give for free.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
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
    p.add_argument("--at-step", type=int, default=200,
                   help="apply the impulse here, late enough that the stance has settled")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--speeds", type=float, nargs="*",
                   default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    actor = build_actor(cfg, get_robot(cfg.env.robot), torch.device("cpu"))
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    # No scripted pushes during the test: this applies ONE controlled impulse and
    # measures recovery, so background shoves would confound the reading.
    sumo_cfg = get_env_spec(cfg.env.name).config(
        opponent="zero", push_interval_steps=0, push_speed=0.0)
    rc, tc = RewardConfig(), TerminationConfig()
    env = SumoEnvCPU(robot=cfg.env.robot, cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc)
    base_dof = env.scene.a.base_dofadr

    print(f"{args.checkpoint}  impulse at step {args.at_step}, {args.seeds} seeds\n")
    print(f"{'shove (m/s)':>12} {'A survived':>12} {'steps after hit':>17}")
    for speed in args.speeds:
        survived, after = 0, []
        for seed in range(args.seeds):
            obs_a, obs_b = env.reset(seed=seed)
            steps = tc.max_episode_steps
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                for t in range(tc.max_episode_steps):
                    if t == args.at_step and speed > 0:
                        # Shove side A straight outward, the direction that loses a
                        # sumo duel. Applied to the base velocity rather than as a
                        # force, so the magnitude is readable in m/s.
                        env.data.qvel[base_dof:base_dof + 2] += np.array([-speed, 0.0])
                    td = TensorDict(
                        {"observation": torch.as_tensor(
                            np.stack([obs_a, obs_b]), dtype=torch.float32)},
                        batch_size=[2])
                    act = actor(td)["action"].numpy()
                    (obs_a, obs_b), _, term, trunc, _ = env.step(act[0], act[1])
                    if term or trunc:
                        steps = t + 1
                        break
            after.append(max(steps - args.at_step, 0))
            survived += int(steps >= tc.max_episode_steps)
        print(f"{speed:12.1f} {survived:>7}/{args.seeds:<4} {np.mean(after):17.0f}")

    print("\nA policy that only survives the 0.0 shove has learned a static pose, not "
          "balance, and will fall the moment an opponent leans on it.")


if __name__ == "__main__":
    main()
