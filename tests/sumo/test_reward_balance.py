"""Whether surviving one more step outscores the penalties charged for doing so.

This exists because the first level 0 training run answered "no" and nobody had
checked. The policy spent four million frames creeping from 0.98 m to 0.40 m
while its episode length never left 60 steps, because at the 0.9 m spawn radius
the default weights made each extra step of survival worth -0.080.
"""

from __future__ import annotations

import math

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


def test_margin_scales_with_shaping_scale():
    """The curriculum anneals shaping away, so the margin must anneal with it or
    later levels would silently keep a level 0 survival incentive at full strength."""
    rc = RewardConfig()
    full = survival_margin(rc, 1.5, 0.6, shaping_scale=1.0)
    half = survival_margin(rc, 1.5, 0.6, shaping_scale=0.5)
    assert half == pytest.approx(full * 0.5)


def test_break_even_radius_is_where_the_margin_crosses_zero():
    """Derived from the closed form, checked against the numerical definition."""
    rc = RewardConfig(alive=0.2, center=0.1)
    r = break_even_radius(rc, 1.5)
    assert survival_margin(rc, 1.5, r) == pytest.approx(0.0, abs=1e-9)
    assert survival_margin(rc, 1.5, r - 0.05) > 0
    assert survival_margin(rc, 1.5, r + 0.05) < 0


def test_break_even_is_unbounded_without_a_centre_penalty():
    """With center=0 there is no radius at which dying sooner pays, so the honest
    answer is infinity rather than a divide-by-zero or a bogus 0.0."""
    assert break_even_radius(RewardConfig(center=0.0), 1.5) == float("inf")


def test_the_shipped_defaults_pay_the_policy_to_die_at_the_spawn_radius():
    """The measured defect, pinned as a fact rather than a memory.

    This is deliberately an assertion about the CURRENT defaults being wrong for a
    survival level. If someone later fixes RewardConfig, this test fails and tells
    them to update the level 0 recipe in the README rather than silently leaving
    two contradicting stories in the repository.
    """
    rc, cfg = RewardConfig(), SumoConfig()
    margin = survival_margin(rc, cfg.ring_radius, cfg.spawn_radius)
    assert margin < 0, (
        "RewardConfig defaults now pay for survival at the spawn radius. That is an "
        "improvement — update the level 0 override recipe in the README and this test.")
    assert margin == pytest.approx(-0.1300, abs=1e-4)
    # Even the most generous reading stays negative, which is what makes this a
    # defect rather than merely a pessimistic bound. The expected value is worked
    # out here from the weights rather than by calling engage_ceiling, so a bug in
    # that function cannot make this test agree with itself.
    best = margin + rc.engage * math.exp(-2 * cfg.spawn_radius / rc.engage_range)
    assert best == pytest.approx(-0.0804, abs=1e-4)
    assert best < 0
    assert break_even_radius(rc, cfg.ring_radius) < cfg.spawn_radius


def test_the_level_zero_override_recipe_fixes_it():
    """The weights the README recommends for a survival level must actually work.

    Asserted at the spawn radius AND at the rim, because a recipe that only works
    where the robot starts still collapses the moment it gets pushed outward.
    """
    rc = RewardConfig(alive=0.3, center=0.1, push=0.0, engage=0.0)
    cfg = SumoConfig()
    assert survival_margin(rc, cfg.ring_radius, cfg.spawn_radius) > 0
    assert survival_margin(rc, cfg.ring_radius, cfg.ring_radius) > 0
    assert break_even_radius(rc, cfg.ring_radius) > cfg.ring_radius
