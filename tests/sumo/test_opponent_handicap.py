"""The per-level handicap that stops a collapsing dummy from handing out free wins.

Every assertion here was checked against the bug it is meant to catch: each test
was run with the corresponding line of `_filter_opponent_loss` removed or with the
mode wired to the wrong branch, and confirmed to fail before being kept.
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


def _state(x=0.0, y=0.0, z=None, tilt_rad=0.0, n_joints=None):
    """One robot at a chosen place and lean. z defaults to a healthy stance."""
    n = ROBOT.n_joints if n_joints is None else n_joints
    cfg = SumoConfig()
    if z is None:
        z = cfg.platform_height + ROBOT.nominal_height
    half = tilt_rad / 2.0
    # Tilt about x, so the quaternion is (cos, sin, 0, 0).
    quat = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    return RobotState(
        base_pos=torch.tensor([[x, y, z]]),
        base_quat=quat,
        base_linvel_world=torch.zeros(1, 3),
        base_angvel_local=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, n),
        joint_vel=torch.zeros(1, n),
    )


UPRIGHT = dict(x=0.0)
# Sagging in place at the centre of the dohyo: base 0.25 m above the platform, so
# below the 0.431 m fall threshold but still ON the platform. The distinction
# matters — dropping the base under platform_height instead is LOSS_FELL_OFF,
# which "ring_out" mode is supposed to count.
COLLAPSED = dict(x=0.0, z=SumoConfig().platform_height + 0.25)
PUSHED_OUT = dict(x=2.0)                 # beyond ring_radius 1.5


def _terminate(a_kw, b_kw, mode, step=1):
    cfg = SumoConfig(opponent="zero", opponent_loses_by=mode)
    return compute_termination(
        _state(**a_kw), _state(**b_kw), ROBOT, ROBOT,
        torch.tensor([step]), cfg, TerminationConfig())


# --------------------------------------------------------------------- "none"

def test_none_mode_ignores_a_collapsed_dummy():
    """L0's whole premise: the dummy folding up must not end the episode.

    Measured in phase B, a zero-action G1 sags ~0.45 m against a 0.431 m fall
    threshold in about 1.2 s. Under the ordinary rules that is a +10 win handed to
    a policy that has done nothing, ~60 steps into every level 0 episode.
    """
    terminated, truncated, lost_a, lost_b, outcome = _terminate(
        UPRIGHT, COLLAPSED, "none")
    assert not bool(lost_b), "a collapsed dummy must not count as lost at level 0"
    assert not bool(terminated)
    assert not bool(truncated)
    assert int(outcome) == ONGOING


def test_none_mode_ignores_a_dummy_pushed_clean_out():
    """Even putting it out does not score at L0 — the level is not about pushing."""
    _, _, _, lost_b, outcome = _terminate(UPRIGHT, PUSHED_OUT, "none")
    assert not bool(lost_b)
    assert int(outcome) == ONGOING


def test_none_mode_still_lets_the_learner_lose():
    """The handicap is one-sided. If it disabled both, L0 would have no signal."""
    terminated, _, lost_a, _, outcome = _terminate(COLLAPSED, UPRIGHT, "none")
    assert bool(lost_a)
    assert bool(terminated)
    assert int(outcome) == B_WINS


# ----------------------------------------------------------------- "ring_out"

def test_ring_out_mode_ignores_a_dummy_that_merely_falls_over():
    """L1 is 'push it out', so the dummy toppling on its own must not count."""
    _, _, _, lost_b, outcome = _terminate(UPRIGHT, COLLAPSED, "ring_out")
    assert not bool(lost_b)
    assert int(outcome) == ONGOING


def test_ring_out_mode_scores_a_dummy_put_out_of_the_ring():
    """...but actually putting it out is exactly what L1 rewards."""
    terminated, _, _, lost_b, outcome = _terminate(UPRIGHT, PUSHED_OUT, "ring_out")
    assert bool(lost_b)
    assert bool(terminated)
    assert int(outcome) == A_WINS


def test_ring_out_mode_scores_a_dummy_pushed_off_the_platform():
    """Off the raised dohyo counts too — it is the same defeat by another route."""
    below = dict(x=0.5, z=SumoConfig().platform_height - 0.05)
    _, _, _, lost_b, outcome = _terminate(UPRIGHT, below, "ring_out")
    assert bool(lost_b)
    assert int(outcome) == A_WINS


# ---------------------------------------------------------------------- "any"

def test_any_mode_restores_the_ordinary_rules():
    """The default must be untouched, or every existing duel changes meaning."""
    terminated, _, _, lost_b, outcome = _terminate(UPRIGHT, COLLAPSED, "any")
    assert bool(lost_b)
    assert bool(terminated)
    assert int(outcome) == A_WINS


def test_handicap_does_not_suppress_a_timeout_draw():
    """A level 0 episode has to end somehow. With side B unable to lose, the only
    conclusions left are the learner's own loss and the step cap."""
    tc = TerminationConfig()
    _, truncated, _, lost_b, outcome = _terminate(
        UPRIGHT, COLLAPSED, "none", step=tc.max_episode_steps)
    assert bool(truncated)
    assert not bool(lost_b)
    assert int(outcome) == DRAW


# ------------------------------------------------------------------- config

def test_a_real_opponent_cannot_be_handicapped():
    """A duel where one side plays by different rules is not the game we evaluate."""
    with pytest.raises(ValueError, match="real policy-driven contestant"):
        SumoConfig(opponent="self", opponent_loses_by="ring_out")


def test_unknown_handicap_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown opponent_loses_by"):
        SumoConfig(opponent="zero", opponent_loses_by="sometimes")


def test_registry_handicaps_exactly_the_dummy_levels():
    """The schedule must line up with the opponent schedule, entry by entry: a
    handicap on a policy-driven level would not even construct."""
    spec = get_env_spec("sumo-1")
    assert len(spec.opponent_loses_by_level) == spec.n_levels
    for level in range(spec.n_levels):
        cfg = spec.config(level)
        if cfg.opponent == "zero":
            assert cfg.opponent_loses_by in ("none", "ring_out")
        else:
            assert cfg.opponent_loses_by == "any"


# ------------------------------------------------------- row-relative outcome

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
