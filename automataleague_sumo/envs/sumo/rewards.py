"""Reward for one side of a sumo duel.

Three things are being asked for, in the order they matter:

1. put the opponent out or down            -> ``win``, terminal and zero sum
2. drive them toward the edge              -> ``push``, dense proxy for the above
3. hold the middle, and stay standing      -> ``centre`` and ``alive``

plus two regularizers. Only ``win`` is zero sum: side A's win term is exactly the
negation of side B's in every outcome, which ``tests/sumo/test_rewards.py``
asserts. The shaping terms are per side and are not required to cancel.

Every weight in ``RewardConfig`` is a whole-episode value, so the numbers are
directly comparable to each other and to ``win``. See that class for why.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.state import RobotState


def survival_margin(rc: RewardConfig, ring_radius: float, radius: float) -> float:
    """Whole-episode value of standing at ``radius`` for the entire episode.

    Positive means surviving is worth having. Negative means the per-step
    penalties outweigh the alive bonus, so the highest-scoring behaviour is to end
    the episode sooner — not a hypothetical: the first configuration of this
    reward scored -0.130 per step at the spawn radius, and the policy responded
    exactly as asked, creeping inward from 0.98 m to 0.40 m while its episode
    length stayed pinned near 60 steps for four million frames.

    Exact, not a bound: ``alive`` and ``centre`` are the only terms paid every
    step regardless of what either robot does.
    """
    return rc.alive - rc.centre * (radius / ring_radius) ** 2


def break_even_radius(rc: RewardConfig, ring_radius: float) -> float:
    """Radius where the centre penalty exactly cancels the alive bonus.

    Inside it surviving pays, outside it dying sooner scores better. Compare
    against ``SumoConfig.spawn_radius``: if the spawn is outside, the reward asks
    the policy to move before it asks it to survive.
    """
    if rc.centre <= 0:
        return float("inf")
    return ring_radius * math.sqrt(rc.alive / rc.centre)


def compute_reward(
    own: RobotState,
    opp: RobotState,
    prev_opp_radius: Tensor,
    own_lost: Tensor,
    opp_lost: Tensor,
    action: Tensor,
    ring_radius: float,
    rc: RewardConfig,
    episode_steps: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return ``(total [N], components)``. The components sum to the total."""
    dtype = own.base_pos.dtype
    r_own = torch.linalg.norm(own.base_pos[:, :2], dim=-1)
    r_opp = torch.linalg.norm(opp.base_pos[:, :2], dim=-1)

    # Rate terms are weighted per whole episode, so spread them over the horizon.
    # `push` is excluded: it is a delta that already telescopes to an
    # episode-scale total, so dividing it would shrink it by 750x.
    rate = 1.0 / float(episode_steps)

    components = {
        "win": rc.win * (opp_lost.to(dtype) - own_lost.to(dtype)),
        # Sign follows the change in the opponent's radius, so letting them
        # recover the centre is penalized exactly as much as pushing them out paid.
        "push": rc.push * (r_opp - prev_opp_radius) / ring_radius,
        "centre": -rc.centre * (r_own / ring_radius) ** 2 * rate,
        "alive": torch.full_like(r_own, rc.alive * rate),
        "action": -rc.action * action.pow(2).mean(-1) * rate,
        "joint_vel": -rc.joint_vel * own.joint_vel.pow(2).mean(-1) * rate,
    }
    total = torch.stack(list(components.values()), dim=0).sum(0)
    return total, components
