"""The bootstrap dummy, and the row-relative outcome codes.

A zero-action humanoid collapses under its own weight in about 1.2 s (measured:
~0.45 m of sag against a 0.431 m fall threshold). If that counted as a loss, the
learner would be handed a free +win roughly 60 steps into every bootstrap episode
and would learn to wait rather than to stand.

Every assertion here was run against the mutation it guards.
"""

from __future__ import annotations

import math

import pytest
import torch

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import SumoConfig, TerminationConfig
from automataleague_sumo.envs.sumo.state import RobotState
from automataleague_sumo.envs.sumo.termination import (
    A_WINS,
    B_WINS,
    DRAW,
    ONGOING,
    R_DRAW,
    R_LOSS,
    R_ONGOING,
    R_WIN,
    compute_termination,
    row_outcome,
)
from automataleague_sumo.robots import get_robot

ROBOT = get_robot("g1")


def _state(x=0.0, y=0.0, z=None, tilt_rad=0.0):
    """One robot at a chosen place and lean. z defaults to a healthy stance."""
    cfg = SumoConfig()
    if z is None:
        z = cfg.platform_height + ROBOT.nominal_height
    half = tilt_rad / 2.0
    quat = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    return RobotState(
        base_pos=torch.tensor([[x, y, z]]),
        base_quat=quat,
        base_linvel_world=torch.zeros(1, 3),
        base_angvel_local=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, ROBOT.n_joints),
        joint_vel=torch.zeros(1, ROBOT.n_joints),
    )


UPRIGHT = dict(x=0.0)
# Sagging in place at the centre: base 0.25 m above the platform, so below the
# 0.431 m fall threshold but still ON it. The distinction matters — dropping the
# base under platform_height instead is a different loss condition entirely.
COLLAPSED = dict(x=0.0, z=SumoConfig().platform_height + 0.25)
PUSHED_OUT = dict(x=2.0)                 # beyond ring_radius 1.5


def _feet():
    """Planted on the ring surface: the only foot configuration that is not a
    loss, so these tests keep measuring the dummy rule and nothing else."""
    return torch.tensor([[[0.0, 0.0, SumoConfig().platform_height + 0.02]]])


def _terminate(a_kw, b_kw, opponent, step=1):
    cfg = SumoConfig(opponent=opponent)
    return compute_termination(
        _state(**a_kw), _state(**b_kw), _feet(), _feet(), ROBOT, ROBOT,
        torch.tensor([step]), cfg, TerminationConfig())


# ------------------------------------------------------------ the dummy

def test_a_collapsing_dummy_does_not_end_the_episode():
    """The whole premise of the bootstrap: side B folding up is not a win."""
    terminated, truncated, _, lost_b, outcome = _terminate(
        UPRIGHT, COLLAPSED, "zero")
    assert not bool(lost_b), "a collapsed dummy must not count as lost"
    assert not bool(terminated)
    assert not bool(truncated)
    assert int(outcome) == ONGOING


def test_a_dummy_pushed_clean_out_does_not_end_the_episode_either():
    """The dummy is scenery, not a contestant. Anything else is a second rule set
    to keep straight, and a handicap that can be left on by accident."""
    _, _, _, lost_b, outcome = _terminate(UPRIGHT, PUSHED_OUT, "zero")
    assert not bool(lost_b)
    assert int(outcome) == ONGOING


def test_the_learner_can_still_lose_against_a_dummy():
    """The handicap is one-sided. If it disabled both, bootstrapping would have no
    signal at all and every episode would run to the step cap."""
    terminated, _, lost_a, _, outcome = _terminate(COLLAPSED, UPRIGHT, "zero")
    assert bool(lost_a)
    assert bool(terminated)
    assert int(outcome) == B_WINS


def test_a_bootstrap_episode_still_ends_at_the_step_cap():
    """With side B unable to lose, the only conclusions left are the learner's own
    loss and the timeout."""
    tc = TerminationConfig()
    _, truncated, _, lost_b, outcome = _terminate(
        UPRIGHT, COLLAPSED, "zero", step=tc.max_episode_steps)
    assert bool(truncated)
    assert not bool(lost_b)
    assert int(outcome) == DRAW


# ------------------------------------------------------- a real opponent

def test_a_real_opponent_loses_by_the_ordinary_rules():
    terminated, _, _, lost_b, outcome = _terminate(UPRIGHT, COLLAPSED, "self")
    assert bool(lost_b)
    assert bool(terminated)
    assert int(outcome) == A_WINS


def test_the_handicap_is_derived_from_the_opponent_not_configured():
    """There is no separate field to set, so it cannot be left on by accident
    against a real opponent — which is the failure a standalone knob invites."""
    assert SumoConfig(opponent="zero").dummy_opponent is True
    assert SumoConfig(opponent="self").dummy_opponent is False
    assert not hasattr(SumoConfig(), "opponent_loses_by")


def test_the_registry_default_is_a_real_opponent():
    assert get_env_spec("sumo-1").config().dummy_opponent is False


# ------------------------------------------------- row-relative outcome

@pytest.mark.parametrize(
    "duel,expect_a,expect_b",
    [
        (A_WINS, R_WIN, R_LOSS),
        (B_WINS, R_LOSS, R_WIN),
        (DRAW, R_DRAW, R_DRAW),
        (ONGOING, R_ONGOING, R_ONGOING),
    ],
)
def test_row_outcome_mirrors_between_the_two_sides(duel, expect_a, expect_b):
    """Each row reports on itself. Without this, a self-play win rate would be a
    statement about which half of the batch was labelled A, not about the policy."""
    codes = torch.tensor([duel], dtype=torch.int32)
    assert int(row_outcome(codes, as_side_a=True)) == expect_a
    assert int(row_outcome(codes, as_side_a=False)) == expect_b


def test_row_outcome_win_and_loss_are_not_the_same_code():
    """Guards the degenerate implementation where both sides map to one value."""
    codes = torch.tensor([A_WINS], dtype=torch.int32)
    assert int(row_outcome(codes, as_side_a=True)) != int(
        row_outcome(codes, as_side_a=False))
