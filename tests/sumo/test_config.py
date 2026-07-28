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
    assert cfg.spawn_frac == 0.6
    assert cfg.opponent == "zero"
    assert cfg.frame_skip == 5


def test_spawn_radius_is_inside_the_ring():
    cfg = SumoConfig()
    assert cfg.spawn_radius == pytest.approx(0.9)
    assert cfg.spawn_radius < cfg.ring_radius


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
    """The guard uses the true per-axis worst case, hypot(spawn_radius + pos_noise,
    pos_noise), not the one-dimensional sum. At pos_noise=0.6 that worst case is
    hypot(1.5, 0.6) ~= 1.615, which already clears ring_radius 1.5, so this value
    is safely past the boundary rather than sitting exactly on it — but it must
    still raise, and this guards against a regression back to the flatter (and
    wrong) one-dimensional check that this value used to sit exactly on."""
    with pytest.raises(ValueError, match="reaches the rim"):
        SumoConfig(pos_noise=0.6)


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
    """pos_noise is applied independently on x and y in sumo_cpu.py, so the true
    worst-case spawn radius is hypot(spawn_radius + pos_noise, pos_noise), not the
    one-dimensional spawn_radius + pos_noise. At pos_noise=0.59 the 1-D sum
    (0.9 + 0.59 = 1.49) is inside the ring, but the real worst case
    (hypot(1.49, 0.59) ~= 1.602) is not — this must raise."""
    with pytest.raises(ValueError, match="reaches the rim"):
        SumoConfig(pos_noise=0.59)


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


def test_reward_and_termination_defaults():
    rc = RewardConfig()
    assert rc.win == 10.0
    assert rc.push > 0
    tc = TerminationConfig()
    assert 0.0 < tc.fall_height_frac < 1.0
    assert tc.max_episode_steps == 750
