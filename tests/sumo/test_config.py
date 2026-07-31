import pytest

from automataleague_sumo.envs.sumo.config import (
    OPPONENT_MODES,
    RewardConfig,
    SumoConfig,
    TerminationConfig,
)


def test_defaults():
    cfg = SumoConfig()
    assert cfg.ring_radius == 1.5
    assert cfg.platform_height == 0.3
    # A tachiai start: 0.75 m apart, inside each other's 0.42 m arm reach. It is
    # also the only approach gradient the reward has, since `push` needs contact
    # before it pays anything.
    assert cfg.spawn_frac == 0.25
    # The default opponent is the real game. A default of "zero" would mean every
    # unqualified run silently trains against a corpse.
    assert cfg.opponent == "self"
    assert cfg.frame_skip == 5


def test_there_is_no_difficulty_level_field():
    """The opponent is the difficulty and it grows on its own under self-play. A
    `level` field would be a second difficulty knob fighting the first."""
    assert not hasattr(SumoConfig(), "level")


def test_spawn_radius_is_inside_the_ring():
    cfg = SumoConfig()
    assert cfg.spawn_radius == pytest.approx(0.375)
    assert cfg.spawn_radius < cfg.ring_radius


def test_the_two_robots_spawn_within_reach_of_each_other():
    """Two standing policies that cannot touch have nothing to learn from: `push`
    pays only for a change in the opponent's radius, `win` needs a ring-out, and
    `alive`/`centre` are both maximised by standing still. The measured arm reach
    of the G1 is 0.42 m (tools/measure_reach.py), so a separation past ~0.84 m
    means neither robot can act on the other at the start of an episode."""
    cfg = SumoConfig()
    assert 2 * cfg.spawn_radius < 0.84, (
        f"the robots spawn {2 * cfg.spawn_radius:.2f} m apart, out of reach of "
        f"each other, so the duel has no approach gradient")


def test_reset_noise_cannot_push_a_spawn_outside_the_ring():
    cfg = SumoConfig()
    assert cfg.spawn_radius + cfg.pos_noise < cfg.ring_radius


def test_reset_noise_that_reaches_the_rim_is_rejected():
    """The invariant this module exists to protect: a robot must never be able to
    spawn already outside the ring. Assert the guard actually raises, not merely
    that the defaults happen to satisfy it."""
    with pytest.raises(ValueError, match="reaches the rim"):
        SumoConfig(pos_noise=1.0)          # 0.9 + 1.0 > 1.5


def test_the_rim_guard_rejects_the_exact_boundary():
    """spawn_frac is pinned here rather than taken from the default, so this stays
    a test of the guard rather than of whatever the default spawn happens to be.

    At spawn_frac 0.6 the spawn radius is 0.9 m, and pos_noise 0.6 gives a
    one-dimensional sum of exactly 1.5 — the ring radius. The guard must reject a
    spawn sitting exactly on the rim, not merely one past it.
    """
    with pytest.raises(ValueError, match="reaches the rim"):
        SumoConfig(spawn_frac=0.6, pos_noise=0.6)


@pytest.mark.parametrize("field,value", [
    ("ring_radius", 0.0),
    ("platform_height", -0.1),
    ("spawn_frac", 1.0),
    ("spawn_frac", 0.0),
    ("frame_skip", 0),
])
def test_invalid_values_raise(field, value):
    with pytest.raises(ValueError):
        SumoConfig(**{field: value})


def test_rim_guard_uses_the_true_two_dimensional_worst_case():
    """pos_noise is applied independently on x and y, so the true worst-case spawn
    radius is hypot(spawn_radius + pos_noise, pos_noise), not the one-dimensional
    sum. At spawn_frac 0.6 and pos_noise 0.59 the 1-D sum (0.9 + 0.59 = 1.49) is
    inside a 1.5 m ring, but the real worst case (hypot(1.49, 0.59) ~= 1.602) is
    not. A guard that checks only the sum passes this configuration.

    spawn_frac is pinned so the test keeps exercising the guard regardless of what
    the default spawn becomes.
    """
    with pytest.raises(ValueError, match="reaches the rim"):
        SumoConfig(spawn_frac=0.6, pos_noise=0.59)

    # ...and the same pos_noise IS legal from a closer spawn, which is what makes
    # the assertion above about the guard's geometry rather than about pos_noise.
    SumoConfig(spawn_frac=0.25, pos_noise=0.59)


@pytest.mark.parametrize("field", ["pos_noise", "yaw_noise", "joint_noise"])
def test_negative_reset_noise_raises(field):
    """sumo_cpu.py guards each noise field with `if cfg.<field> > 0`, so a negative
    value would silently disable reset noise instead of erroring — exactly the
    out-of-distribution evaluation failure the reset-noise comment warns about."""
    with pytest.raises(ValueError, match=field):
        SumoConfig(**{field: -0.01})


def test_unknown_opponent_mode_raises():
    with pytest.raises(ValueError, match="opponent"):
        SumoConfig(opponent="telepathy")


def test_all_opponent_modes_are_accepted():
    for mode in OPPONENT_MODES:
        assert SumoConfig(opponent=mode).opponent == mode


def test_the_opponent_modes_describe_who_plays_not_how_hard_it_is():
    """Two modes, and one of them is only a bootstrap. "frozen" and "pool" were
    curriculum machinery; a physical whole-body task has no discrete strategy
    space to cycle in, so naive self-play needs no opponent history to stabilise
    it. If it turns out to need one, win rate against held-out old checkpoints
    will oscillate and say so."""
    assert set(OPPONENT_MODES) == {"zero", "self"}


def test_reward_and_termination_defaults():
    rc = RewardConfig()
    assert rc.win == 10.0
    assert rc.push > 0
    tc = TerminationConfig()
    assert 0.0 < tc.fall_height_frac < 1.0
    assert tc.max_episode_steps == 750
