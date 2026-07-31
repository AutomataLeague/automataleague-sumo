"""Reward for one side of a sumo duel.

Only ``win`` is zero sum: side A's win term is exactly the negation of side B's,
in every outcome, which ``tests/sumo/test_rewards.py`` asserts. The shaping terms
are computed per side and are not required to cancel. They exist to give each
side an informative gradient before either has learned to win at all, and they
are multiplied by ``shaping_scale``, which the curriculum decays toward zero so
that the final policy optimizes the actual win condition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.spatial import yaw_from_quat
from automataleague_sumo.envs.sumo.state import RobotState


def survival_margin(
    rc: RewardConfig, ring_radius: float, radius: float, shaping_scale: float = 1.0,
) -> float:
    """Net shaping reward per step for simply staying alive at ``radius``.

    This is the GUARANTEED floor, counting only the two terms paid every step no
    matter what the robot or its opponent does: the alive bonus and the centre
    penalty. ``push`` and ``engage`` depend on the opponent's motion and on
    facing, and ``action``/``joint_vel`` are penalties a still robot drives toward
    zero, so none of them belong in a floor.

    Read the sign accordingly. Positive is a proof that one more step of survival
    pays. Negative is a warning, not a proof of the opposite: ``engage`` can add
    up to ``rc.engage`` per step on top, so the realised margin can still be
    positive. On the first standing run the floor at the 0.9 m spawn radius was
    -0.130 and the realised margin including a favourable ``engage`` was about
    -0.080 — still negative, and the policy responded exactly as asked, creeping
    inward from 0.98 m to 0.40 m while its episode length stayed pinned near 60
    steps for four million frames.
    """
    centre = -rc.center * (radius / ring_radius) ** 2
    return shaping_scale * (rc.alive + centre)


def engage_ceiling(rc: RewardConfig, separation: float) -> float:
    """Largest ``engage`` reward obtainable at a given separation, per step.

    ``engage`` is ``rc.engage * alignment * exp(-separation / engage_range)`` and
    alignment is at most 1, so this is that expression at perfect facing. Bounding
    it by ``rc.engage`` alone ignores the decay and is far too generous to be
    useful: at the 1.8 m spawn separation the decay costs a factor of six.
    """
    return rc.engage * math.exp(-separation / rc.engage_range)


def break_even_radius(rc: RewardConfig, ring_radius: float) -> float:
    """Radius where the centre penalty exactly cancels the alive bonus.

    Inside it, surviving pays; outside it, dying sooner scores better. Compare
    this against ``SumoConfig.spawn_radius``: if the spawn is outside, the reward
    is asking the policy to move before it asks it to survive.
    """
    if rc.center <= 0:
        return float("inf")
    return ring_radius * math.sqrt(rc.alive / rc.center)


def compute_reward(
    own: RobotState,
    opp: RobotState,
    prev_opp_radius: Tensor,
    own_lost: Tensor,
    opp_lost: Tensor,
    action: Tensor,
    ring_radius: float,
    rc: RewardConfig,
    shaping_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return ``(total [N], components)``. The components sum to the total."""
    dtype = own.base_pos.dtype
    r_own = torch.linalg.norm(own.base_pos[:, :2], dim=-1)
    r_opp = torch.linalg.norm(opp.base_pos[:, :2], dim=-1)

    # Facing alignment, decaying with separation.
    to_opp = opp.base_pos[:, :2] - own.base_pos[:, :2]
    dist = torch.linalg.norm(to_opp, dim=-1)
    to_opp_unit = to_opp / dist.clamp_min(1e-6).unsqueeze(-1)
    yaw = yaw_from_quat(own.base_quat)
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    alignment = (heading * to_opp_unit).sum(-1)

    win = rc.win * (opp_lost.to(dtype) - own_lost.to(dtype))

    shaping = {
        # Hold the middle: the further out you drift, the worse.
        "centre": -rc.center * (r_own / ring_radius) ** 2,
        # Drive the opponent outward. Sign follows the change in their radius, so
        # letting them recover the centre is penalized symmetrically.
        "push": rc.push * (r_opp - prev_opp_radius) / ring_radius,
        "engage": rc.engage * alignment * torch.exp(-dist / rc.engage_range),
        "alive": torch.full_like(r_own, rc.alive),
        "action": -rc.action * action.pow(2).mean(-1),
        "joint_vel": -rc.joint_vel * own.joint_vel.pow(2).mean(-1),
    }
    components = {"win": win}
    components.update({k: shaping_scale * v for k, v in shaping.items()})
    total = torch.stack(list(components.values()), dim=0).sum(0)
    return total, components
