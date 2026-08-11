"""Ratings from a pairwise result matrix.

A round-robin win rate is only meaningful against the field it was measured in.
The same checkpoint scored 76.5% against one field and 67.4% against another with
identical weights, because the second field was stronger. A leaderboard that
grows cannot use that number for anything.

A rating fixes it by modelling each competitor's strength directly, so the
prediction "A beats B 64% of the time" is recoverable from two scalars regardless
of who else was in the tournament.

**Fitted, not accumulated.** The familiar Elo update walks match by match and
nudges two ratings each time, which makes the result depend on the ORDER the
matches happened in. A round robin has no natural order, and replaying the same
tournament with the pairings shuffled would produce different numbers, so a
sequential update is not reproducible here. This fits the Bradley-Terry model to
the whole matrix at once by maximum likelihood. Same input, same ratings, always,
and ``test_elo.py`` asserts exactly that against a permuted field.

The two are the same model: Bradley-Terry strengths are Elo ratings on a log
scale, and the fitted ratings satisfy the usual expectation

    P(i beats j) = 1 / (1 + 10 ** ((R_j - R_i) / 400))

which is the property worth testing, and is tested.
"""

from __future__ import annotations

import numpy as np

SCALE = 400.0          # Elo convention: 400 points is a 10:1 odds ratio
DEFAULT_RATING = 1000.0


def fit_ratings(
    wins: np.ndarray,
    draws: np.ndarray | None = None,
    *,
    prior: float = 2.0,
    anchor: int | None = None,
    anchor_rating: float = DEFAULT_RATING,
    iterations: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Bradley-Terry ratings on the Elo scale, from a wins (and draws) matrix.

    ``wins[i][j]`` is how many duels i won against j; ``draws`` is symmetric and
    each draw counts as half a win to both, the standard convention.

    ``prior`` is a regulariser, and it is not optional in practice. An entrant
    that wins every duel has unbounded maximum-likelihood strength, and a rating
    of +inf is useless on a leaderboard and poisons every comparison. Each
    competitor is given ``prior`` virtual drawn games against an average
    opponent, which keeps ratings finite and shrinks the ones with little
    evidence behind them toward the middle. It costs almost nothing once a
    competitor has played a few hundred real duels.

    ``anchor`` pins one competitor's rating, since the model only determines
    ratings up to a shared constant. Pin the fixed baseline and numbers become
    comparable across tournaments; leave it None to centre the field instead.
    """
    wins = np.asarray(wins, dtype=float)
    n = wins.shape[0]
    if wins.shape != (n, n):
        raise ValueError(f"wins must be square, got {wins.shape}")
    if prior <= 0:
        raise ValueError(
            f"prior must be > 0, got {prior}: an undefeated competitor has "
            f"unbounded strength without it")

    if draws is not None:
        draws = np.asarray(draws, dtype=float)
        if not np.allclose(draws, draws.T):
            raise ValueError("draws must be symmetric: a draw is shared")
        wins = wins + 0.5 * draws

    played = wins + wins.T
    np.fill_diagonal(played, 0.0)
    scored = wins.sum(axis=1)

    # Zermelo/MM iteration. Each step is a closed-form improvement of the
    # likelihood, so it converges monotonically from any positive start.
    strength = np.ones(n, dtype=float)
    for _ in range(iterations):
        previous = strength.copy()
        for i in range(n):
            pair = strength[i] + strength
            pair[i] = 1.0                       # self term excluded below
            expected = np.sum(played[i] / pair)
            # The virtual games, against a fixed average opponent of strength 1.
            expected += prior / (strength[i] + 1.0)
            strength[i] = (scored[i] + prior / 2.0) / max(expected, 1e-12)
        # Only ratios matter; renormalise so the numbers cannot drift off scale.
        strength /= np.exp(np.mean(np.log(strength)))
        if np.max(np.abs(strength - previous)) < tol:
            break

    ratings = SCALE * np.log10(strength)
    if anchor is None:
        return ratings - ratings.mean() + anchor_rating
    return ratings - ratings[anchor] + anchor_rating


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B, the relation the ratings are fitted to."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))
