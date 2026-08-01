"""Robot registry.

Add a robot by writing a ``make_<robot>() -> RobotSpec`` factory in a module here
and registering it in ``ROBOTS``. Tasks select a robot by name via ``get_robot``.
"""

from __future__ import annotations

from typing import Callable

from automataleague_sumo.robots.base import ExtraCollider, RobotSpec
from automataleague_sumo.robots.g1 import make_g1

ROBOTS: dict[str, Callable[[], RobotSpec]] = {
    "g1": make_g1,
}


def get_robot(name: str) -> RobotSpec:
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot '{name}'. Registered robots: {sorted(ROBOTS)}")
    return ROBOTS[name]()


__all__ = ["ExtraCollider",
    "RobotSpec", "ROBOTS", "get_robot"]
