"""PPO on a single sumo-1 curriculum level.

    uv run python examples/ppo_sumo.py level=0
    uv run python examples/ppo_sumo.py level=3 env.num_envs=4096 logger.backend=""
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

    level = int(cfg.get("level", 0))
    best = run_ppo(
        cfg,
        level=level,
        total_frames=cfg.collector.total_frames,
        init_ckpt=cfg.get("init_checkpoint", None),
        run_name=f"sumo1_L{level}",
    )
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
