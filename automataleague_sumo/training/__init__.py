"""Training stack: PPO over the batched Warp sumo env, plus the curriculum driver.

Importing this package must not import mujoco-warp, so a CPU-only checkout can
still load the config and the model builders. The GPU env is imported inside
``env_maker``/``run_ppo`` at call time.
"""

from automataleague_sumo.training.curriculum import CurriculumConfig, curriculum_from_cfg
from automataleague_sumo.training.models import make_ppo_models
from automataleague_sumo.training.ppo import outcome_rates, run_ppo

__all__ = [
    "CurriculumConfig",
    "curriculum_from_cfg",
    "make_ppo_models",
    "outcome_rates",
    "run_ppo",
]
