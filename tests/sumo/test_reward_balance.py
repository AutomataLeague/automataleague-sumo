"""Whether surviving one more step outscores the penalties charged for doing so.

This exists because the first standing run answered "no" and nobody had checked.
The policy spent four million frames creeping from 0.98 m to 0.40 m while its
episode length never left 60 steps, because at the 0.9 m spawn radius the weights
of the day made each extra step of survival worth -0.080.
"""

from __future__ import annotations

import pytest

from automataleague_sumo.envs.sumo.config import RewardConfig, SumoConfig
from automataleague_sumo.envs.sumo.rewards import break_even_radius, survival_margin


def test_margin_is_the_alive_bonus_at_the_exact_centre():
    """At r=0 the centre penalty vanishes, so the margin is the bare alive bonus.
    Anchors the function to a value derived independently of its own formula."""
    rc = RewardConfig()
    assert survival_margin(rc, 1.5, 0.0) == pytest.approx(rc.alive)


def test_margin_falls_as_the_robot_drifts_outward():
    rc = RewardConfig()
    margins = [survival_margin(rc, 1.5, r) for r in (0.0, 0.5, 1.0, 1.5)]
    assert margins == sorted(margins, reverse=True)
    # Strictly decreasing, not merely non-increasing: a `center` weight dropped to
    # zero would keep the ordering above while destroying the term's whole purpose.
    assert all(a > b for a, b in zip(margins, margins[1:]))


def test_break_even_radius_is_where_the_margin_crosses_zero():
    """Derived from the closed form, checked against the numerical definition."""
    rc = RewardConfig(alive=0.2, centre=0.1)
    r = break_even_radius(rc, 1.5)
    assert survival_margin(rc, 1.5, r) == pytest.approx(0.0, abs=1e-9)
    assert survival_margin(rc, 1.5, r - 0.05) > 0
    assert survival_margin(rc, 1.5, r + 0.05) < 0


def test_break_even_is_unbounded_without_a_centre_penalty():
    """With center=0 there is no radius at which dying sooner pays, so the honest
    answer is infinity rather than a divide-by-zero or a bogus 0.0."""
    assert break_even_radius(RewardConfig(centre=0.0), 1.5) == float("inf")


def test_the_shipped_defaults_pay_for_survival_everywhere_in_the_ring():
    """The defect this module exists for, now fixed and pinned so it stays fixed.

    Asserted at the spawn radius AND at the rim: a configuration that only pays
    where the robot starts still collapses the moment it gets pushed outward, and
    being pushed outward is the entire game.
    """
    rc, cfg = RewardConfig(), SumoConfig()
    assert survival_margin(rc, cfg.ring_radius, cfg.spawn_radius) > 0
    assert survival_margin(rc, cfg.ring_radius, cfg.ring_radius) > 0
    assert break_even_radius(rc, cfg.ring_radius) > cfg.ring_radius


def test_the_shaping_budget_cannot_outscore_a_win():
    """Enforced by RewardConfig, checked here on the shipped numbers. A whole
    episode of perfect shaping worth more than one win means the highest-scoring
    policy never tries to win, which is what happened the first time."""
    rc = RewardConfig()
    assert rc.push + rc.alive + rc.centre < rc.win
    with pytest.raises(ValueError, match="shaping budget"):
        RewardConfig(win=1.0, push=3.0, alive=2.0, centre=1.0)
