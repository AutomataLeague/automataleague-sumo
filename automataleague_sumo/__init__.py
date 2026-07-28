"""Automata League Sumo — two humanoids wrestle inside a circular ring."""

from automataleague_sumo.envs.registry import (
    ENVIRONMENTS,
    EnvSpec,
    get_env_spec,
    list_environments,
    make_env,
)
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW, ONGOING

__version__ = "0.1.0"

__all__ = [
    "A_WINS",
    "B_WINS",
    "DRAW",
    "ENVIRONMENTS",
    "EnvSpec",
    "ONGOING",
    "get_env_spec",
    "list_environments",
    "make_env",
    "__version__",
]
