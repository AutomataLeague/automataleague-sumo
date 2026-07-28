import math

import numpy as np
import pytest

from automataleague_sumo.envs.sumo.config import SumoConfig, TerminationConfig
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.envs.sumo.termination import A_WINS, DRAW, ONGOING


@pytest.fixture(scope="module")
def env():
    return SumoEnvCPU("g1")


def _zero(env):
    return np.zeros(env.action_dim, dtype=np.float32)


def test_reset_returns_two_observations_of_the_derived_width(env):
    obs_a, obs_b = env.reset(seed=0)
    assert obs_a.shape == (env.observation_dim,) == (110,)
    assert obs_b.shape == (env.observation_dim,)
    assert np.isfinite(obs_a).all() and np.isfinite(obs_b).all()


def test_both_robots_start_inside_the_ring(env):
    env.reset(seed=0)
    for side in env.scene.sides:
        xy = env.data.qpos[side.base_qposadr:side.base_qposadr + 2]
        assert math.hypot(*xy) < env.cfg.ring_radius


def test_reset_is_noisy(env):
    """Both backends reset with noise. A zero-noise reset would put evaluation out
    of distribution relative to training."""
    a0, _ = env.reset(seed=0)
    a1, _ = env.reset(seed=1)
    assert not np.allclose(a0, a1)


def test_reset_is_reproducible_for_a_given_seed(env):
    a0, b0 = env.reset(seed=42)
    a1, b1 = env.reset(seed=42)
    assert np.allclose(a0, a1) and np.allclose(b0, b1)


def test_noise_free_config_resets_to_the_exact_home_pose():
    quiet = SumoEnvCPU("g1", cfg=SumoConfig(pos_noise=0.0, yaw_noise=0.0, joint_noise=0.0))
    quiet.reset(seed=0)
    assert np.allclose(quiet.data.qpos, quiet.scene.home_qpos, atol=1e-6)


def test_step_returns_a_well_formed_transition(env):
    env.reset(seed=0)
    (obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(
        _zero(env), _zero(env))
    assert obs_a.shape == (env.observation_dim,)
    assert isinstance(rew_a, float) and isinstance(rew_b, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["outcome"] == ONGOING
    assert set(info["reward_components_a"]) == set(info["reward_components_b"])


def test_a_passive_duel_does_not_collapse(env):
    """Zero action means hold the home stance. Both robots must still be standing
    after several seconds — everything downstream assumes a G1 that can stand."""
    env.reset(seed=3)
    for _ in range(200):
        _, _, terminated, _, _ = env.step(_zero(env), _zero(env))
        if terminated:
            break
    assert not terminated, "a passive duel ended early; the stance is unstable"


def test_pushing_a_robot_out_of_the_ring_ends_the_duel_with_a_winner(env):
    env.reset(seed=0)
    # Teleport side B beyond the rim, then take one step.
    env.data.qpos[env.scene.b.base_qposadr] = env.cfg.ring_radius + 0.5
    _, _, terminated, _, info = env.step(_zero(env), _zero(env))
    assert terminated
    assert info["outcome"] == A_WINS
    assert info["rewards"]["a"] > 0 > info["rewards"]["b"]


def test_both_robots_out_is_a_draw(env):
    env.reset(seed=0)
    env.data.qpos[env.scene.a.base_qposadr] = -(env.cfg.ring_radius + 0.5)
    env.data.qpos[env.scene.b.base_qposadr] = env.cfg.ring_radius + 0.5
    _, _, terminated, _, info = env.step(_zero(env), _zero(env))
    assert terminated
    assert info["outcome"] == DRAW


def test_timeout_truncates():
    short = SumoEnvCPU("g1", term_cfg=TerminationConfig(max_episode_steps=3))
    short.reset(seed=0)
    flags = [short.step(_zero(short), _zero(short))[3] for _ in range(3)]
    assert flags[-1] is True
    assert flags[0] is False


def test_actions_are_clipped_to_the_unit_range(env):
    """An out-of-range action must not be able to command a larger joint offset
    than action_scale allows."""
    env.reset(seed=0)
    env.step(np.full(env.action_dim, 10.0, dtype=np.float32), _zero(env))
    side = env.scene.a
    home = side.robot.home_joint_qpos
    commanded = env.data.ctrl[side.actuator_ids]
    assert np.allclose(commanded, home + env.action_scale, atol=1e-6)


def test_reward_components_sum_to_the_reported_reward(env):
    env.reset(seed=0)
    _, (rew_a, _), _, _, info = env.step(_zero(env), _zero(env))
    assert sum(info["reward_components_a"].values()) == pytest.approx(rew_a, abs=1e-5)


def test_render_produces_an_image(env):
    env.reset(seed=0)
    frame = env.render(camera="corner")
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_render_reuses_its_renderer_across_calls(env):
    """render_frame() must not close a caller-supplied renderer: the env owns one
    ``mujoco.Renderer`` and hands it in on every call. If ``render()`` allocated a
    fresh renderer each time, or ``render_frame`` closed the one it was given, a
    second call would either leak renderers or crash on a closed renderer."""
    env.reset(seed=0)
    frame1 = env.render(camera="corner")
    renderer = env._renderer
    assert renderer is not None
    frame2 = env.render(camera="corner")
    assert env._renderer is renderer, "render() allocated a new renderer on reuse"
    assert frame2.ndim == 3 and frame2.shape[2] == 3
    assert frame2.dtype == np.uint8
    assert frame1.shape == frame2.shape
