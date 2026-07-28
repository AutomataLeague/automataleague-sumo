import dataclasses
import math

import numpy as np
import pytest

from automataleague_sumo.envs.sumo.config import SumoConfig, TerminationConfig
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW, ONGOING
from automataleague_sumo.robots import get_robot


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


def test_a_passive_duel_runs_to_a_clean_conclusion(env):
    """Zero action holds the home stance, which a passive humanoid cannot sustain:
    it pitches forward and goes down within a couple of seconds.

    That is expected physics, not a wiring bug. Learning to stand is precisely
    what curriculum level 0 is for, and the same behaviour was measured and
    accepted in the robot and scene tasks. What this test guards is that the
    ENVIRONMENT behaves well while it happens: finite observations and rewards on
    every step, and a clean terminal outcome rather than a hang, a NaN, or an
    episode that never ends.
    """
    env.reset(seed=3)
    terminated = truncated = False
    info = {}
    for _ in range(env.term_cfg.max_episode_steps):
        (obs_a, obs_b), (rew_a, rew_b), terminated, truncated, info = env.step(
            _zero(env), _zero(env))
        assert np.isfinite(obs_a).all() and np.isfinite(obs_b).all()
        assert np.isfinite(rew_a) and np.isfinite(rew_b)
        if terminated or truncated:
            break

    assert terminated or truncated, "the duel never ended"
    assert info["outcome"] in (A_WINS, B_WINS, DRAW)


def test_previous_action_reaches_the_right_side_of_the_next_observation(env):
    """`prev_action` must be the action THAT side took.

    Nothing else in this suite would catch the two halves being swapped, because
    both sides are the same robot and the block is all zeros on the first step.
    Distinct nonzero values per side make the swap visible.
    """
    env.reset(seed=0)
    act_a = np.full(env.action_dim, 0.5, dtype=np.float32)
    act_b = np.full(env.action_dim, -0.5, dtype=np.float32)
    (obs_a, obs_b), *_ = env.step(act_a, act_b)

    n = env.action_dim
    lo, hi = 9 + 2 * n, 9 + 3 * n          # the prev_action block
    assert np.allclose(obs_a[lo:hi], 0.5, atol=1e-5)
    assert np.allclose(obs_b[lo:hi], -0.5, atol=1e-5)


def test_the_push_term_credits_the_side_whose_opponent_moved_out(env):
    """`prev_opp_radius` must be the OPPONENT's previous radius, per side.

    Drive B outward while A stays put: A should be rewarded for pushing, B should
    not. Feeding the wrong side's previous radius into `compute_reward` swaps
    which side gets the credit.
    """
    env.reset(seed=0)
    env.step(_zero(env), _zero(env))                 # establish previous radii
    env.data.qpos[env.scene.b.base_qposadr] += 0.4   # teleport B toward the rim
    _, _, _, _, info = env.step(_zero(env), _zero(env))

    push_a = info["reward_components_a"]["push"]
    push_b = info["reward_components_b"]["push"]
    assert push_a > 0, "A drove its opponent outward and was not credited"
    assert push_a > push_b
    # `push_a > push_b` alone doesn't have teeth here: with seed=0 the two sides'
    # previous radii differ by only ~0.05 m, dwarfed by B's 0.4 m teleport, so
    # swapping which side's previous radius feeds which `compute_reward` call
    # still leaves push_a > push_b (verified: swapping gives push_a=1.17,
    # push_b=0.16, ordering intact). What the swap actually breaks is push_b's
    # absolute value: A never moved, so its opponent-facing push term should be
    # ~0 (measured 0.003), not the ~0.16 the swapped wiring produces.
    assert push_b == pytest.approx(0.0, abs=0.05), \
        "B's opponent (A) didn't move; B should get ~no push credit"


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


def test_cross_robot_matchup_fails_loudly_until_phase_c():
    """action_scale, observation_dim and action_dim are all derived from side A's
    robot alone. A second robot with a different action scale or joint count would
    silently produce a wrong-scale duel rather than an error, so a mismatched pair
    must be rejected up front instead of allowed to run quietly wrong."""
    g1 = get_robot("g1")
    other = dataclasses.replace(g1, name="g1-clone")
    with pytest.raises(NotImplementedError, match="Phase C"):
        SumoEnvCPU("g1", opponent_robot=other)


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
