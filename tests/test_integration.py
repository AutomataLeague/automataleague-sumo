"""End-to-end: a full duel driven through the public API only."""

import numpy as np
import pytest

from automataleague_sumo import A_WINS, B_WINS, DRAW, ONGOING, make_env


def test_a_random_duel_runs_to_a_conclusion():
    env = make_env("sumo-1", backend="cpu")
    rng = np.random.default_rng(0)
    env.reset(seed=0)

    outcome = ONGOING
    for _ in range(env.term_cfg.max_episode_steps):
        act_a = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
        act_b = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
        (obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(
            act_a, act_b)
        assert np.isfinite(obs_a).all() and np.isfinite(obs_b).all()
        assert np.isfinite(rew_a) and np.isfinite(rew_b)
        outcome = info["outcome"]
        if terminated or truncated:
            break

    # `terminated`, not merely `terminated or truncated`. The loop runs exactly
    # max_episode_steps iterations and `truncated` is guaranteed true on the last
    # one, so the weaker assertion would pass even if loss detection were
    # completely broken. Flailing humanoids go down within a couple of seconds,
    # far inside the 750-step budget, so a duel that runs the clock out means
    # `side_lost` is not working.
    assert terminated, "the duel ran the clock out; loss detection is not firing"
    assert not truncated
    assert outcome in (A_WINS, B_WINS, DRAW)


def test_outcome_codes_are_part_of_the_public_api():
    """A consumer reading `info["outcome"]` must be able to interpret it without
    importing from an internal module."""
    import automataleague_sumo as als

    assert {"ONGOING", "A_WINS", "B_WINS", "DRAW"} <= set(als.__all__)
    assert len({als.ONGOING, als.A_WINS, als.B_WINS, als.DRAW}) == 4


def test_the_two_sides_see_the_same_world_from_opposite_perspectives():
    env = make_env("sumo-1", backend="cpu")
    obs_a, obs_b = env.reset(seed=7)
    assert obs_a.shape == obs_b.shape
    # Distinct spawn noise means the two views must differ in detail...
    assert not np.allclose(obs_a, obs_b)
    # ...but the opponent-distance column is a shared, symmetric quantity.
    n = env.action_dim
    rel_pos_a = obs_a[9 + 3 * n + 4: 9 + 3 * n + 7]
    rel_pos_b = obs_b[9 + 3 * n + 4: 9 + 3 * n + 7]
    assert np.linalg.norm(rel_pos_a) == pytest.approx(np.linalg.norm(rel_pos_b), abs=1e-4)


def test_a_duel_can_be_replayed_exactly_from_a_seed():
    def run():
        env = make_env("sumo-1", backend="cpu")
        rng = np.random.default_rng(3)
        env.reset(seed=3)
        trace = []
        for _ in range(25):
            a = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
            b = rng.uniform(-1, 1, env.action_dim).astype(np.float32)
            _, rewards, _, _, _ = env.step(a, b)
            trace.append(rewards)
        return np.array(trace)

    assert np.allclose(run(), run())


def test_every_opponent_mode_instantiates():
    """Both opponents a run can actually face. "pool" is registered but its
    machinery is not built yet, so it is expected to refuse loudly rather than
    silently behave like one of the others."""
    for opponent in ("self", "zero"):
        extra = {"opponent_loses_by": "none"} if opponent == "zero" else {}
        env = make_env("sumo-1", backend="cpu", opponent=opponent, **extra)
        obs_a, _ = env.reset(seed=0)
        assert obs_a.shape == (110,)
        assert env.cfg.opponent == opponent


def test_shaping_scale_actually_reaches_the_reward():
    """A smaller shaping_scale must produce strictly smaller shaping components on
    an identical transition. Asserted on two scales rather than on one value, so a
    reward that ignores shaping_scale entirely cannot pass."""
    comps = {}
    for scale in (1.0, 0.2):
        env = make_env("sumo-1", backend="cpu", shaping_scale=scale,
                       pos_noise=0.0, yaw_noise=0.0, joint_noise=0.0)
        env.reset(seed=0)
        zero = np.zeros(env.action_dim, dtype=np.float32)
        _, _, _, _, info = env.step(zero, zero)
        comps[scale] = info["reward_components_a"]
    assert abs(comps[0.2]["centre"]) < abs(comps[1.0]["centre"])
    assert abs(comps[0.2]["alive"]) < abs(comps[1.0]["alive"])
