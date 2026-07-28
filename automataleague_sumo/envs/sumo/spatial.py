"""Quaternion and base-frame math, robot agnostic, batched over N.

MuJoCo quaternion convention: (w, x, y, z).
"""

from __future__ import annotations

import torch
from torch import Tensor

_WORLD_DOWN = torch.tensor([0.0, 0.0, -1.0])


def quat_rotate_inverse(q: Tensor, v: Tensor) -> Tensor:
    """Rotate world-frame vectors ``v`` [N,3] into the body frame of ``q`` [N,4]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Inverse rotation = rotation by the conjugate quaternion (negated vector part).
    qv = torch.stack([-x, -y, -z], dim=-1)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)


def projected_gravity(q: Tensor) -> Tensor:
    """Gravity direction [0,0,-1] expressed in the body frame of ``q`` [N,4]."""
    down = _WORLD_DOWN.to(q.device, q.dtype).expand(q.shape[0], 3)
    return quat_rotate_inverse(q, down)


def tilt_angle(q: Tensor) -> Tensor:
    """Angle in radians between body-up and world-up. 0 = perfectly upright."""
    gz = projected_gravity(q)[:, 2].clamp(-1.0, 1.0)
    return torch.arccos(-gz)


def yaw_from_quat(q: Tensor) -> Tensor:
    """Yaw about world z in radians, from quaternion ``q`` [N,4]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
