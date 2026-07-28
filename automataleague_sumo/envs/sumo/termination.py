"""Batched loss detection and duel outcome.

A side loses by leaving the ring, by dropping off the platform, or by going down.
A duel ends on the first loss; a timeout is a draw; a simultaneous loss is a draw.

The "down" test is a proxy: base height above the platform plus torso tilt. The
sumo-accurate rule is that touching the ground with anything other than the soles
loses, which needs contact inspection against ``SideInfo.foot_geom_ids``. That is
a deliberate follow-up, not an oversight — the proxy has no false positives while
the robot is upright, which is all the early curriculum needs.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.config import SumoConfig, TerminationConfig
from automataleague_sumo.envs.sumo.spatial import tilt_angle
from automataleague_sumo.envs.sumo.state import RobotState
from automataleague_sumo.robots import RobotSpec

LOSS_NONE = 0
LOSS_RING_OUT = 1
LOSS_FELL_OFF = 2
LOSS_DOWN = 3

ONGOING = 0
A_WINS = 1
B_WINS = 2
DRAW = 3


def side_lost(
    state: RobotState, robot: RobotSpec, cfg: SumoConfig, tc: TerminationConfig,
) -> tuple[Tensor, Tensor]:
    """``(lost [N] bool, code [N] int32)`` for one side."""
    device = state.base_pos.device
    radius = torch.linalg.norm(state.base_pos[:, :2], dim=-1)
    height = state.base_pos[:, 2] - cfg.platform_height

    ring_out = radius > cfg.ring_radius
    fell_off = height < 0.0
    down = (
        (height < tc.fall_height_frac * robot.nominal_height)
        | (tilt_angle(state.base_quat) > math.radians(tc.max_tilt_deg))
    )

    # Priority matters: once the base is outside the rim, `height` is measured
    # against a platform the robot is no longer above, so ring_out must win.
    code = torch.full_like(radius, LOSS_NONE, dtype=torch.int32)
    code = torch.where(down, torch.tensor(LOSS_DOWN, device=device, dtype=torch.int32), code)
    code = torch.where(
        fell_off, torch.tensor(LOSS_FELL_OFF, device=device, dtype=torch.int32), code)
    code = torch.where(
        ring_out, torch.tensor(LOSS_RING_OUT, device=device, dtype=torch.int32), code)
    return ring_out | fell_off | down, code


def compute_termination(
    state_a: RobotState,
    state_b: RobotState,
    robot_a: RobotSpec,
    robot_b: RobotSpec,
    step_count: Tensor,
    cfg: SumoConfig,
    tc: TerminationConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(terminated, truncated, lost_a, lost_b, outcome)``, each ``[N]``."""
    device = state_a.base_pos.device
    lost_a, _ = side_lost(state_a, robot_a, cfg, tc)
    lost_b, _ = side_lost(state_b, robot_b, cfg, tc)

    def const(v):
        return torch.tensor(v, device=device, dtype=torch.int32)

    outcome = torch.full_like(lost_a, ONGOING, dtype=torch.int32)
    outcome = torch.where(lost_b & ~lost_a, const(A_WINS), outcome)
    outcome = torch.where(lost_a & ~lost_b, const(B_WINS), outcome)
    outcome = torch.where(lost_a & lost_b, const(DRAW), outcome)

    terminated = lost_a | lost_b
    truncated = (step_count >= tc.max_episode_steps) & ~terminated
    # A timeout ends the duel with nobody put out, which is a draw. Without this
    # the outcome code would stay ONGOING on the final step and could not
    # describe how the duel ended on its own; every consumer would have to
    # inspect `truncated` separately to tell a draw from a duel still running.
    outcome = torch.where(truncated, const(DRAW), outcome)
    return terminated, truncated, lost_a, lost_b, outcome
