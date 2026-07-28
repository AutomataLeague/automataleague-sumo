"""Automata League Sumo — two humanoids wrestle inside a circular ring."""

from automataleague_sumo.envs.registry import (
    ENVIRONMENTS,
    EnvSpec,
    get_env_spec,
    list_environments,
    make_env,
)

__version__ = "0.1.0"

__all__ = [
    "ENVIRONMENTS",
    "EnvSpec",
    "get_env_spec",
    "list_environments",
    "make_env",
    "__version__",
]
