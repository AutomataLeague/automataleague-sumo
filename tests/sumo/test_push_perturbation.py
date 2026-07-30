"""Push perturbations: the disturbance that turns a held pose into balance.

Every assertion here was run against the bug it guards. In particular the
"schedule fires on the right steps" test was checked with the modulo removed
(pushes every step) and with the ``step_count > 0`` guard removed (a push on the
first step of every episode), and fails on both.
"""

from __future__ import annotations

import numpy as np
import pytest

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import SumoConfig
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU


def _env(**kwargs):
    cfg = SumoConfig(**kwargs)
    return SumoEnvCPU(robot="g1", cfg=cfg)


def _base_speed(env, side):
    adr = side.base_dofadr
    return float(np.linalg.norm(env.data.qvel[adr:adr + 2]))


# ------------------------------------------------------------------- config

def test_a_schedule_without_a_magnitude_is_rejected():
    """Both no-ops look exactly like push training that is switched on, which is
    the worst possible failure: a run reports it trained with pushes and did not."""
    with pytest.raises(ValueError, match="half-configured"):
        SumoConfig(push_interval_steps=100, push_speed=0.0)


def test_a_magnitude_without_a_schedule_is_rejected():
    with pytest.raises(ValueError, match="half-configured"):
        SumoConfig(push_interval_steps=0, push_speed=1.0)


def test_pushes_are_off_by_default():
    """Adding an unrequested disturbance to every existing level would silently
    change what every previous result meant."""
    cfg = SumoConfig()
    assert cfg.push_interval_steps == 0
    assert cfg.push_speed == 0.0


def test_negative_push_speed_is_rejected():
    with pytest.raises(ValueError, match="push_speed must be >= 0"):
        SumoConfig(push_interval_steps=10, push_speed=-1.0)


# ----------------------------------------------------------------- schedule

def test_no_push_arrives_when_the_feature_is_off():
    """The control. Without it, every assertion below could be measuring ordinary
    contact and gravity rather than the perturbation."""
    env = _env()
    env.reset(seed=0)
    zero = np.zeros(env.action_dim)
    jumps = []
    for _ in range(40):
        before = _base_speed(env, env.scene.a)
        env.step(zero, zero)
        jumps.append(_base_speed(env, env.scene.a) - before)
    assert max(jumps) < 0.4, f"unexplained velocity jump with pushes off: {max(jumps)}"


def test_the_push_lands_on_scheduled_steps_and_no_others():
    """Pins WHICH steps get shoved, not merely that some do.

    Magnitude is drawn from U(0, push_speed), so an individual scheduled shove can
    legitimately be too small to detect. The assertion is therefore one-sided in
    the direction that matters: no jump may occur on an UNSCHEDULED step, which is
    what catches a missing modulo or an off-by-one, plus a floor on how many of
    the scheduled steps fire, which catches pushes being absent entirely.
    """
    interval = 10
    env = _env(push_interval_steps=interval, push_speed=6.0)
    env.reset(seed=0)
    zero = np.zeros(env.action_dim)
    jumped = []
    for step in range(41):
        before = _base_speed(env, env.scene.a)
        env.step(zero, zero)
        if _base_speed(env, env.scene.a) - before > 1.0:
            jumped.append(step)
    # step_count is incremented after the step, so the push applied while
    # step_count == 10 happens during the 11th call, i.e. index 10.
    scheduled = {10, 20, 30, 40}
    assert set(jumped) <= scheduled, f"pushed on unscheduled steps: {jumped}"
    assert len(jumped) >= 3, f"barely any scheduled push landed: {jumped}"


def test_the_first_step_of_an_episode_is_never_pushed():
    """step_count is 0 there. Shoving a robot before it has acted once makes the
    spawn distribution a lie and would show up as unexplained early falls.

    Asserted on qvel around a direct _maybe_push call rather than on the speed
    after a full env.step. Magnitude is drawn from U(0, push_speed), so a
    threshold on the observed speed passes whenever the draw happens to be small —
    verified: that version of this test survived deleting the guard it exists to
    protect. Comparing the buffer catches any impulse at all.
    """
    env = _env(push_interval_steps=1, push_speed=6.0)
    env.reset(seed=0)

    before = env.data.qvel.copy()
    env._maybe_push()
    assert np.array_equal(env.data.qvel, before), "pushed before the robot ever acted"

    # And the control: one step later the very same call must do something, or the
    # assertion above would hold just as well with pushes disabled entirely.
    env.step(np.zeros(env.action_dim), np.zeros(env.action_dim))
    before = env.data.qvel.copy()
    env._maybe_push()
    assert not np.array_equal(env.data.qvel, before), "pushes never fire at all"


def test_both_robots_are_pushed_independently():
    """A shared impulse moves the whole scene, which is a change of reference frame
    rather than a disturbance to either robot.

    Compares the DELTA each base receives, not the resulting velocities. The two
    robots' velocities differ anyway from their own dynamics, so comparing those
    passes even when a single impulse is broadcast to both — verified: that
    version of this test survived hoisting the random draw out of the loop.
    """
    env = _env(push_interval_steps=1, push_speed=6.0)
    env.reset(seed=0)
    zero = np.zeros(env.action_dim)
    da, db = env.scene.a.base_dofadr, env.scene.b.base_dofadr

    identical = 0
    for _ in range(12):
        env.step(zero, zero)
        before = env.data.qvel.copy()
        env._maybe_push()
        delta_a = env.data.qvel[da:da + 2] - before[da:da + 2]
        delta_b = env.data.qvel[db:db + 2] - before[db:db + 2]
        assert np.linalg.norm(delta_a) + np.linalg.norm(delta_b) > 0, "no push landed"
        identical += int(np.allclose(delta_a, delta_b, atol=1e-9))
    assert identical == 0, "the two robots received identical impulses"


def test_push_magnitude_scales_with_push_speed():
    """Guards a hard-wired magnitude that ignores the config entirely."""
    def peak(speed):
        env = _env(push_interval_steps=1, push_speed=speed)
        env.reset(seed=3)
        zero = np.zeros(env.action_dim)
        best = 0.0
        for _ in range(15):
            before = _base_speed(env, env.scene.a)
            env.step(zero, zero)
            best = max(best, _base_speed(env, env.scene.a) - before)
        return best

    small, large = peak(1.0), peak(8.0)
    assert large > 3 * small, f"push_speed barely changed the impulse: {small} vs {large}"


# ----------------------------------------------------------------- registry

def test_the_balance_level_actually_enables_pushes():
    """Level 0 exists to teach balance, and balance without a disturbance is a
    held pose. Measured: the first level 0 policy survived a full 750-step episode
    and still fell to a 0.5 m/s shove in 6 of 6 seeds."""
    cfg = get_env_spec("sumo-1").config(0)
    assert cfg.push_interval_steps > 0
    assert cfg.push_speed > 0
    # At least a few shoves per episode, or most episodes never see one.
    from automataleague_sumo.envs.sumo.config import TerminationConfig

    assert TerminationConfig().max_episode_steps // cfg.push_interval_steps >= 3
