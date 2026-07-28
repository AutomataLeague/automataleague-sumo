"""Structured physics state shared by both backends.

Each backend reads raw qpos/qvel, then calls ``extract_duel_state`` to get a pair
of backend-agnostic ``RobotState`` objects that the task logic consumes. Nothing
downstream of here knows whether it is running on CPU MuJoCo or MuJoCo-Warp.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.scene import SideInfo, SumoSceneInfo


@dataclass
class RobotState:
    base_pos: Tensor            # [N,3] world position of the base
    base_quat: Tensor           # [N,4] (w,x,y,z)
    base_linvel_world: Tensor   # [N,3] linear velocity, world frame
    base_angvel_local: Tensor   # [N,3] angular velocity, body frame
    joint_pos: Tensor           # [N, n_joints]
    joint_vel: Tensor           # [N, n_joints]


def extract_state(qpos: Tensor, qvel: Tensor, side: SideInfo) -> RobotState:
    """Slice one side out of raw qpos/qvel into a structured state.

    MuJoCo free-joint convention: qvel[0:3] is linear velocity in the world frame,
    qvel[3:6] is angular velocity in the body frame.
    """
    qa, da = side.base_qposadr, side.base_dofadr
    jq = torch.as_tensor(side.joint_qposadr, device=qpos.device, dtype=torch.long)
    jd = torch.as_tensor(side.joint_dofadr, device=qvel.device, dtype=torch.long)
    return RobotState(
        base_pos=qpos[:, qa:qa + 3],
        base_quat=qpos[:, qa + 3:qa + 7],
        base_linvel_world=qvel[:, da:da + 3],
        base_angvel_local=qvel[:, da + 3:da + 6],
        joint_pos=qpos[:, jq],
        joint_vel=qvel[:, jd],
    )


def extract_duel_state(
    qpos: Tensor, qvel: Tensor, scene: SumoSceneInfo,
) -> tuple[RobotState, RobotState]:
    """Both sides, side A first."""
    return extract_state(qpos, qvel, scene.a), extract_state(qpos, qvel, scene.b)


def contact_flag_cpu(model, data, scene: SumoSceneInfo) -> Tensor:
    """1.0 when any geom of side A touches any geom of side B, else 0.0.

    CPU-only: it walks ``data.contact``. The Warp backend computes the same signal
    from batched contact arrays in Phase C, which is why every consumer takes the
    flag as an argument rather than computing it.
    """
    a_geoms = set(scene.a.geom_ids.tolist())
    b_geoms = set(scene.b.geom_ids.tolist())
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in a_geoms and g2 in b_geoms) or (g1 in b_geoms and g2 in a_geoms):
            return torch.ones(1, dtype=torch.float32)
    return torch.zeros(1, dtype=torch.float32)
