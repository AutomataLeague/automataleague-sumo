"""Assemble the policy observation for one side of a duel.

Layout (width = 3 * n_joints + 23, robot-derived):

    base_linvel_base(3) | base_angvel(3) | proj_gravity(3)
    | joint_pos - home(n) | joint_vel(n) | prev_action(n)
    | r/R(1) | to_centre_base_xy(2) | (R - r)/R(1)
    | rel_pos_base(3) | rel_vel_base(3) | opp_heading cos,sin(2)
    | (R - r_opp)/R(1) | contact(1)

Every quantity is expressed in the robot's own base frame or is a scalar, so a
rigid rotation of the whole arena leaves the observation unchanged. That is what
makes one shared policy valid for both sides in self-play, and
``tests/sumo/test_observation.py`` asserts it directly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.spatial import (
    projected_gravity,
    quat_rotate_inverse,
    yaw_from_quat,
)
from automataleague_sumo.envs.sumo.state import RobotState
from automataleague_sumo.robots import RobotSpec

# Ring block (4) + opponent block (10). Independent of the robot.
TASK_DIM = 14

# Every observation component is clipped to this magnitude. Contact between two
# humanoids drives joint velocities to spikes far outside their working range —
# measured at 49.5 rad/s against a mean of about 2.5 — and those go straight into
# an unnormalised network. The bound is set above everything the task actually
# uses (the largest non-spike component measured was 21.7) so it truncates the
# collision tail and nothing else.
OBS_CLIP = 25.0


def observation_dim(robot: RobotSpec) -> int:
    """Total observation width for ``robot``, derived from its joint count."""
    return robot.proprio_dim + TASK_DIM


def _planar_to_base(quat: Tensor, vec_xy: Tensor) -> Tensor:
    """Express a world-frame xy vector in the base frame, returning its xy part."""
    zeros = torch.zeros_like(vec_xy[:, :1])
    vec3 = torch.cat([vec_xy, zeros], dim=-1)
    return quat_rotate_inverse(quat, vec3)[:, :2]


def build_observation(
    own: RobotState,
    opp: RobotState,
    prev_action: Tensor,
    home_joint_qpos: Tensor,
    ring_radius: float,
    contact: Tensor,
) -> Tensor:
    home = home_joint_qpos.to(own.joint_pos.device, own.joint_pos.dtype)
    r_own = torch.linalg.norm(own.base_pos[:, :2], dim=-1)
    r_opp = torch.linalg.norm(opp.base_pos[:, :2], dim=-1)

    # Unit vector from the robot toward the ring centre, in its own base frame.
    to_centre = _planar_to_base(own.base_quat, -own.base_pos[:, :2])
    to_centre = to_centre / to_centre.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rel_pos = quat_rotate_inverse(own.base_quat, opp.base_pos - own.base_pos)
    rel_vel = quat_rotate_inverse(
        own.base_quat, opp.base_linvel_world - own.base_linvel_world)
    d_yaw = yaw_from_quat(opp.base_quat) - yaw_from_quat(own.base_quat)

    parts = [
        quat_rotate_inverse(own.base_quat, own.base_linvel_world),   # 3
        own.base_angvel_local,                                       # 3
        projected_gravity(own.base_quat),                            # 3
        own.joint_pos - home,                                        # n
        own.joint_vel,                                               # n
        prev_action,                                                 # n
        (r_own / ring_radius).unsqueeze(-1),                         # 1
        to_centre,                                                   # 2
        ((ring_radius - r_own) / ring_radius).unsqueeze(-1),         # 1
        rel_pos,                                                     # 3
        rel_vel,                                                     # 3
        torch.cos(d_yaw).unsqueeze(-1),                              # 1
        torch.sin(d_yaw).unsqueeze(-1),                              # 1
        ((ring_radius - r_opp) / ring_radius).unsqueeze(-1),         # 1
        contact.reshape(-1, 1).to(own.joint_pos.dtype),              # 1
    ]
    return torch.cat(parts, dim=-1).clamp(-OBS_CLIP, OBS_CLIP)
