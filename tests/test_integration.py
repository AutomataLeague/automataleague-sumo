"""End-to-end: a full duel driven through the public API only."""

import numpy as np
import pytest

from automataleague_sumo import make_env
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW, ONGOING


def test_a_random_duel_runs_to_a_conclusion():
    env = make_env("sumo-1", level=0, backend="cpu")
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

    # Flailing humanoids fall over; the duel must resolve, not run forever.
    assert terminated or truncated
    assert outcome in (ONGOING, A_WINS, B_WINS, DRAW)


def test_the_two_sides_see_the_same_world_from_opposite_perspectives():
    env = make_env("sumo-1", level=0, backend="cpu")
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
        env = make_env("sumo-1", level=0, backend="cpu")
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


def test_every_curriculum_level_instantiates():
    for level in range(5):
        env = make_env("sumo-1", level=level, backend="cpu")
        obs_a, _ = env.reset(seed=0)
        assert obs_a.shape == (110,)
        assert env.cfg.level == level


def test_shaping_scale_actually_reaches_the_reward():
    """Level 4 has the smallest shaping weight, so its shaping components must be
    strictly smaller in magnitude than level 0's on an identical transition."""
    comps = {}
    for level in (0, 4):
        env = make_env("sumo-1", level=level, backend="cpu",
                       pos_noise=0.0, yaw_noise=0.0, joint_noise=0.0)
        env.reset(seed=0)
        zero = np.zeros(env.action_dim, dtype=np.float32)
        _, _, _, _, info = env.step(zero, zero)
        comps[level] = info["reward_components_a"]
    assert abs(comps[4]["centre"]) < abs(comps[0]["centre"])
    assert abs(comps[4]["alive"]) < abs(comps[0]["alive"])
