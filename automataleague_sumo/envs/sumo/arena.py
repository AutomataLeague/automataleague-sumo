"""Dohyo geometry: where the ring is and where the robots start.

Pure geometry, deliberately free of any MuJoCo import, so that the spawn symmetry
that the whole shared-policy self-play argument rests on can be tested without
compiling a model.

Convention: the ring is centred on the world origin. Side A spawns on the -x
side facing +x; side B spawns on the +x side facing -x. Rotating the world by pi
about z maps side A exactly onto side B, which is what makes one policy valid for
both sides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.config import SumoConfig


@dataclass(frozen=True)
class SpawnPose:
    x: float
    y: float
    yaw: float


def spawn_poses(cfg: SumoConfig) -> tuple[SpawnPose, SpawnPose]:
    """Nominal (noise-free) start poses, side A first."""
    r = cfg.spawn_radius
    return (
        SpawnPose(x=-r, y=0.0, yaw=0.0),
        SpawnPose(x=+r, y=0.0, yaw=math.pi),
    )


def yaw_quat(yaw: float) -> list[float]:
    """Quaternion for a rotation of ``yaw`` about world z, MuJoCo (w, x, y, z)."""
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def radial_distance(xy: Tensor) -> Tensor:
    """Distance from the ring centre for a batch of xy positions ``[N, 2]``."""
    return torch.linalg.norm(xy, dim=-1)
