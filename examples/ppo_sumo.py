"""PPO on sumo-1.

    uv run python examples/ppo_sumo.py
    uv run python examples/ppo_sumo.py env.arena.opponent=zero      # bootstrap standing
    uv run python examples/ppo_sumo.py env.num_envs=4096 logger.backend=""

There is no curriculum entry point and no `level` argument. The difficulty of a
competitive game is the opponent, and under self-play it grows with the policy,
so there is nothing to schedule. To start from an existing policy, pass
`init_checkpoint=...`.
"""

import os

import hydra
import numpy as np
import torch

from automataleague_sumo.training import run_ppo


@hydra.main(version_base="1.1", config_path="", config_name="config_ppo")
def main(cfg):
    # Hydra chdirs into its own outputs/ directory; restore the launch directory so
    # checkpoints/ lands where the user ran the command rather than inside a
    # timestamped folder they then have to hunt for.
    os.chdir(hydra.utils.get_original_cwd())
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    best = run_ppo(
        cfg,
        total_frames=cfg.collector.total_frames,
        init_ckpt=cfg.get("init_checkpoint", None),
        run_name=cfg.get("run_name", "sumo1"),
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
