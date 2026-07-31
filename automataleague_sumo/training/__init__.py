"""Training stack: PPO over the batched Warp sumo env.

Importing this package must not import mujoco-warp, so a CPU-only checkout can
still load the config and the model builders. The GPU env is imported inside
``env_maker``/``run_ppo`` at call time.

There is no curriculum module. The opponent is the difficulty, and under
self-play it grows with the policy, so there is no authored ladder to walk.
"""

from automataleague_sumo.training.models import build_actor, make_ppo_models
from automataleague_sumo.training.ppo import outcome_rates, run_ppo

__all__ = ["build_actor", "make_ppo_models", "outcome_rates", "run_ppo"]
