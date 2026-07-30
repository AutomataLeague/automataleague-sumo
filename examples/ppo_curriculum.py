"""PPO curriculum across sumo-1 levels, warm-starting each stage from the last.

    uv run python examples/ppo_curriculum.py
    uv run python examples/ppo_curriculum.py curriculum.levels=[0,1,3]
"""

import os

import hydra
import numpy as np
import torch

from automataleague_sumo.training import curriculum_from_cfg, run_ppo


@hydra.main(version_base="1.1", config_path="", config_name="config_ppo")
def main(cfg):
    os.chdir(hydra.utils.get_original_cwd())
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    cur = curriculum_from_cfg(cfg)
    prev_best = cfg.get("init_checkpoint", None)
    for i, level in enumerate(cur.levels):
        print(f"=== curriculum level {level} ({cur.frames_per_level[i]:,} frames) ===")
        prev_best = run_ppo(
            cfg,
            level=level,
            total_frames=cur.frames_per_level[i],
            init_ckpt=prev_best if cur.warm_start else None,
            run_name=f"sumo1_curriculum_L{level}",
        )
        print(f"level {level} best -> {prev_best}")
    print(f"curriculum complete; final policy: {prev_best}")


if __name__ == "__main__":
    main()
