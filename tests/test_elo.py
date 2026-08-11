"""Ratings must mean something, survive a permuted field, and stay finite."""

from __future__ import annotations

import numpy as np
import pytest

from automataleague_sumo.elo import (
    DEFAULT_RATING,
    SCALE,
    expected_score,
    fit_ratings,
)


def _matrix(rows):
    return np.array(rows, dtype=float)


def test_the_rating_gap_reproduces_the_observed_win_rate():
    """The defining property. If A beats B 75% of 400 duels, the fitted gap must
    predict 75%, or the rating is just a ranking with decorative numbers."""
    wins = _matrix([[0, 300], [100, 0]])
    r = fit_ratings(wins, prior=1e-6)
    assert expected_score(r[0], r[1]) == pytest.approx(0.75, abs=0.01)


def test_a_400_point_gap_is_ten_to_one():
    """The Elo scale convention, pinned so a change of SCALE cannot pass silently."""
    assert expected_score(1400, 1000) == pytest.approx(10 / 11, abs=1e-9)
    assert SCALE == 400.0


def test_ratings_do_not_depend_on_the_order_of_the_field():
    """The reason this is fitted rather than accumulated.

    Sequential Elo walks match by match and depends on the order they occurred,
    so the same round robin would rate differently depending on how the pairings
    were listed. Permuting the field here must permute the ratings and change
    nothing else.
    """
    wins = _matrix([[0, 60, 90], [40, 0, 70], [10, 30, 0]])
    order = [2, 0, 1]
    direct = fit_ratings(wins)
    permuted = fit_ratings(wins[np.ix_(order, order)])
    assert np.allclose(permuted, direct[order], atol=1e-6)


def test_an_undefeated_competitor_still_gets_a_finite_rating():
    """Its maximum-likelihood strength is unbounded. A rating of +inf is useless
    on a leaderboard and poisons every comparison drawn from it."""
    wins = _matrix([[0, 200, 200], [0, 0, 100], [0, 100, 0]])
    r = fit_ratings(wins)
    assert np.isfinite(r).all()
    assert r[0] == max(r)


def test_a_winless_competitor_still_gets_a_finite_rating():
    wins = _matrix([[0, 0, 0], [200, 0, 100], [200, 100, 0]])
    r = fit_ratings(wins)
    assert np.isfinite(r).all()
    assert r[0] == min(r)


def test_the_prior_shrinks_a_thin_record_more_than_a_thick_one():
    """Two competitors with the same 100% record but 4 duels versus 4000 should
    not be rated equally confident."""
    thin = fit_ratings(_matrix([[0, 4], [0, 0]]))
    thick = fit_ratings(_matrix([[0, 4000], [0, 0]]))
    assert thick[0] - thick[1] > thin[0] - thin[1]


def test_a_prior_of_zero_is_refused():
    """Without it an undefeated entrant is +inf, which is the failure this
    parameter exists to prevent, so it must not be switch-off-able by accident."""
    with pytest.raises(ValueError, match="unbounded"):
        fit_ratings(_matrix([[0, 1], [1, 0]]), prior=0.0)


def test_draws_count_half_to_each_side():
    """Asymmetric on purpose. A symmetric all-draw matrix rates flat under ANY
    draw convention, so it cannot tell half-credit from full credit — the first
    version of this test was exactly that and mutation testing caught it.

    Here A wins 60 of 100 and draws 40, so A's score rate is (60 + 20)/100 = 0.8
    and the fitted gap must predict 0.8. Counting a draw as a full win instead
    gives 100/140 = 0.71 and fails.
    """
    wins = _matrix([[0, 60], [0, 0]])
    draws = _matrix([[0, 40], [40, 0]])
    r = fit_ratings(wins, draws, prior=1e-6)
    assert expected_score(r[0], r[1]) == pytest.approx(0.8, abs=0.01)


def test_a_wholly_drawn_field_rates_flat():
    """The weaker companion to the test above, kept because it pins a different
    thing: no ordering may be invented out of a field that never beat anyone."""
    r = fit_ratings(_matrix([[0, 0], [0, 0]]), _matrix([[0, 100], [100, 0]]))
    assert r[0] == pytest.approx(r[1], abs=1e-6)


def test_asymmetric_draws_are_refused():
    """A draw belongs to both sides; an asymmetric matrix means the caller has
    counted something else and the ratings would be quietly wrong."""
    with pytest.raises(ValueError, match="symmetric"):
        fit_ratings(_matrix([[0, 1], [1, 0]]), _matrix([[0, 5], [3, 0]]))


def test_anchoring_pins_one_competitor_and_shifts_the_rest():
    """Ratings are only determined up to a constant. Pinning the fixed baseline
    is what makes numbers comparable across tournaments."""
    wins = _matrix([[0, 60, 90], [40, 0, 70], [10, 30, 0]])
    r = fit_ratings(wins, anchor=2, anchor_rating=1000.0)
    assert r[2] == pytest.approx(1000.0)
    gaps = r - r[2]
    unanchored = fit_ratings(wins)
    assert np.allclose(gaps, unanchored - unanchored[2], atol=1e-6)


def test_the_default_anchor_rating_is_what_the_tournament_reports():
    """The floor's rating is the reference every other number is read against, so
    a silent change to it would reprice a whole leaderboard."""
    r = fit_ratings(_matrix([[0, 60, 90], [40, 0, 70], [10, 30, 0]]), anchor=2)
    assert r[2] == pytest.approx(DEFAULT_RATING)
    assert DEFAULT_RATING == 1000.0


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        fit_ratings(np.zeros((2, 3)))
