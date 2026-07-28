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
    """spawn_radius 0.9 + pos_noise 0.6 == ring_radius 1.5 exactly. The guard is
    `>=`, so landing precisely on the rim is already out."""
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
