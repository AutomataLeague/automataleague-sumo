"""Batched loss detection and duel outcome.

A side loses by leaving the ring, by dropping off the platform, or by going down.
A duel ends on the first loss; a timeout is a draw; a simultaneous loss is a draw.

The "down" test is a proxy: base height above the platform plus torso tilt. The
sumo-accurate rule is that touching the ground with anything other than the soles
loses, which needs contact inspection against ``SideInfo.foot_geom_ids``. That is
a deliberate follow-up, not an oversight — the proxy has no false positives while
the robot is upright.
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
LOSS_RING_OUT = 1     # the base left the ring
LOSS_FELL_OFF = 2     # the base dropped below the platform
LOSS_DOWN = 3         # too low or too tilted
LOSS_STEP_OUT = 4     # a foot came down off the edge

ONGOING = 0
A_WINS = 1
B_WINS = 2
DRAW = 3

# Row-relative outcome codes: the same duel seen from one contestant's own side.
# A duel-level A_WINS reads as R_WIN on side A's row and R_LOSS on side B's. This
# is what lets a win rate aggregated over a self-play batch mean something, since
# every row is then reporting on itself rather than on a fixed side of the ring.
R_ONGOING = 0
R_WIN = 1
R_LOSS = 2
R_DRAW = 3


def row_outcome(outcome: Tensor, as_side_a: bool) -> Tensor:
    """Recode a duel outcome ``[N]`` from side A's or side B's point of view."""
    win_code, loss_code = (A_WINS, B_WINS) if as_side_a else (B_WINS, A_WINS)
    out = torch.full_like(outcome, R_ONGOING)
    out = torch.where(outcome == win_code, torch.full_like(outcome, R_WIN), out)
    out = torch.where(outcome == loss_code, torch.full_like(outcome, R_LOSS), out)
    return torch.where(outcome == DRAW, torch.full_like(outcome, R_DRAW), out)


def side_lost(
    state: RobotState,
    foot_pos: Tensor,
    robot: RobotSpec,
    cfg: SumoConfig,
    tc: TerminationConfig,
) -> tuple[Tensor, Tensor]:
    """``(lost [N] bool, code [N] int32)`` for one side.

    ``foot_pos`` is ``[N, n_feet, 3]``, the world positions of this side's foot
    geoms. The base tests alone are not enough: the base can sit well inside the
    rim while a foot is planted on the floor beyond it. Measured on a trained
    self-play policy, a foot reached 1.856 m against a 1.5 m ring and sank the
    full 0.30 m to the floor, and a foot was outside the ring on 7.1% of all
    steps with the duel still running.
    """
    device = state.base_pos.device
    radius = torch.linalg.norm(state.base_pos[:, :2], dim=-1)
    height = state.base_pos[:, 2] - cfg.platform_height

    ring_out = radius > cfg.ring_radius
    fell_off = height < 0.0
    down = (
        (height < tc.fall_height_frac * robot.nominal_height)
        | (tilt_angle(state.base_quat) > math.radians(tc.max_tilt_deg))
    )

    # Stepping out: a foot both beyond the rim AND below the ring surface, which
    # is a foot that has come down off the edge. Requiring both is what makes this
    # the sumo rule (touching down outside) rather than a stricter one that would
    # also punish a recovery step swung over the line and brought back in the air.
    foot_r = torch.linalg.norm(foot_pos[..., :2], dim=-1)            # [N, n_feet]
    foot_down = foot_pos[..., 2] < cfg.platform_height
    step_out = ((foot_r > cfg.ring_radius) & foot_down).any(dim=-1)

    # Priority matters: once the base is outside the rim, `height` is measured
    # against a platform the robot is no longer above, so ring_out must win.
    def const(v):
        return torch.tensor(v, device=device, dtype=torch.int32)

    code = torch.full_like(radius, LOSS_NONE, dtype=torch.int32)
    code = torch.where(down, const(LOSS_DOWN), code)
    code = torch.where(step_out, const(LOSS_STEP_OUT), code)
    code = torch.where(fell_off, const(LOSS_FELL_OFF), code)
    code = torch.where(ring_out, const(LOSS_RING_OUT), code)
    return ring_out | fell_off | down | step_out, code


def compute_termination(
    state_a: RobotState,
    state_b: RobotState,
    foot_pos_a: Tensor,
    foot_pos_b: Tensor,
    robot_a: RobotSpec,
    robot_b: RobotSpec,
    step_count: Tensor,
    cfg: SumoConfig,
    tc: TerminationConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(terminated, truncated, lost_a, lost_b, outcome)``, each ``[N]``.

    ``foot_pos_*`` are ``[N, n_feet, 3]`` world positions of each side's foot
    geoms. Required rather than optional: a backend that forgot to supply them
    would silently run without the stepping-out rule, which is the failure this
    signature exists to prevent.
    """
    device = state_a.base_pos.device
    lost_a, _ = side_lost(state_a, foot_pos_a, robot_a, cfg, tc)
    lost_b, _ = side_lost(state_b, foot_pos_b, robot_b, cfg, tc)
    if cfg.dummy_opponent:
        # A zero-action humanoid collapses on its own in about 1.2 s. Letting that
        # count as a loss would hand the learner a free +win roughly 60 steps into
        # every bootstrap episode, and it would learn to wait.
        lost_b = torch.zeros_like(lost_b)

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
