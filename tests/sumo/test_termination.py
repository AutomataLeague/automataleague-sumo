import dataclasses
import math

import torch

from automataleague_sumo.envs.sumo.config import SumoConfig, TerminationConfig
from automataleague_sumo.envs.sumo.state import RobotState
from automataleague_sumo.envs.sumo.termination import (
    A_WINS,
    B_WINS,
    DRAW,
    LOSS_DOWN,
    LOSS_FELL_OFF,
    LOSS_NONE,
    LOSS_RING_OUT,
    ONGOING,
    compute_termination,
    side_lost,
)
from automataleague_sumo.robots import get_robot

CFG = SumoConfig()
TC = TerminationConfig()
G1 = get_robot("g1")
N = 29
STANDING_Z = CFG.platform_height + G1.nominal_height   # 0.3 + 0.79


def _state(x=0.0, y=0.0, z=STANDING_Z, tilt=0.0):
    """`tilt` is a rotation about the world x axis, in radians."""
    half = tilt / 2
    return RobotState(
        base_pos=torch.tensor([[x, y, z]]),
        base_quat=torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]]),
        base_linvel_world=torch.zeros(1, 3),
        base_angvel_local=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, N),
        joint_vel=torch.zeros(1, N),
    )


def test_a_standing_robot_in_the_middle_has_not_lost():
    lost, code = side_lost(_state(), G1, CFG, TC)
    assert not lost.item()
    assert code.item() == LOSS_NONE


def test_crossing_the_ring_radius_is_a_ring_out():
    lost, code = side_lost(_state(x=CFG.ring_radius + 0.01), G1, CFG, TC)
    assert lost.item()
    assert code.item() == LOSS_RING_OUT


def test_standing_just_inside_the_rim_is_still_alive():
    lost, _ = side_lost(_state(x=CFG.ring_radius - 0.01), G1, CFG, TC)
    assert not lost.item()


def test_dropping_below_the_platform_surface_is_falling_off():
    lost, code = side_lost(_state(z=CFG.platform_height - 0.05), G1, CFG, TC)
    assert lost.item()
    assert code.item() == LOSS_FELL_OFF


def test_sinking_below_the_height_threshold_is_down():
    z = CFG.platform_height + TC.fall_height_frac * G1.nominal_height - 0.01
    lost, code = side_lost(_state(z=z), G1, CFG, TC)
    assert lost.item()
    assert code.item() == LOSS_DOWN


def test_tipping_past_the_tilt_threshold_is_down():
    lost, code = side_lost(_state(tilt=math.radians(TC.max_tilt_deg + 5)), G1, CFG, TC)
    assert lost.item()
    assert code.item() == LOSS_DOWN


def test_leaning_within_the_tilt_threshold_is_not_down():
    lost, _ = side_lost(_state(tilt=math.radians(TC.max_tilt_deg - 5)), G1, CFG, TC)
    assert not lost.item()


def test_ring_out_takes_priority_over_down():
    """Once the base is outside the rim, the height test is meaningless because it
    is measured against a platform the robot is no longer standing on."""
    _, code = side_lost(
        _state(x=CFG.ring_radius + 0.5, z=CFG.platform_height - 0.4), G1, CFG, TC)
    assert code.item() == LOSS_RING_OUT


def test_b_leaving_the_ring_means_a_wins():
    terminated, truncated, lost_a, lost_b, outcome = compute_termination(
        _state(), _state(x=CFG.ring_radius + 0.1), G1, G1,
        torch.zeros(1, dtype=torch.long), CFG, TC)
    assert terminated.item() and not truncated.item()
    assert not lost_a.item() and lost_b.item()
    assert outcome.item() == A_WINS


def test_a_leaving_the_ring_means_b_wins():
    *_, outcome = compute_termination(
        _state(x=-CFG.ring_radius - 0.1), _state(), G1, G1,
        torch.zeros(1, dtype=torch.long), CFG, TC)
    assert outcome.item() == B_WINS


def test_simultaneous_loss_is_a_draw():
    *_, outcome = compute_termination(
        _state(x=-CFG.ring_radius - 0.1), _state(x=CFG.ring_radius + 0.1), G1, G1,
        torch.zeros(1, dtype=torch.long), CFG, TC)
    assert outcome.item() == DRAW


def test_an_untouched_duel_is_ongoing():
    terminated, truncated, _, _, outcome = compute_termination(
        _state(x=-0.5), _state(x=0.5), G1, G1,
        torch.zeros(1, dtype=torch.long), CFG, TC)
    assert not terminated.item() and not truncated.item()
    assert outcome.item() == ONGOING


def test_timeout_truncates_and_is_recorded_as_a_draw():
    """A timeout ends the duel with nobody put out. `outcome` must say DRAW, so
    that it describes the ending on its own rather than requiring every consumer
    to cross-check `truncated`."""
    terminated, truncated, _, _, outcome = compute_termination(
        _state(x=-0.5), _state(x=0.5), G1, G1,
        torch.tensor([TC.max_episode_steps]), CFG, TC)
    assert not terminated.item()
    assert truncated.item()
    assert outcome.item() == DRAW


def test_a_loss_on_the_final_step_terminates_rather_than_truncates():
    """Pins the `& ~terminated` guard. Without it a duel decided on the very last
    step would report BOTH terminated and truncated, and would be recorded as a
    draw instead of a win."""
    terminated, truncated, _, _, outcome = compute_termination(
        _state(), _state(x=CFG.ring_radius + 0.1), G1, G1,
        torch.tensor([TC.max_episode_steps]), CFG, TC)
    assert terminated.item()
    assert not truncated.item(), "a decided duel must not also report truncation"
    assert outcome.item() == A_WINS


def test_the_rim_belongs_to_the_ring():
    """Pins the ring-out comparison as `>` and not `>=`. Standing exactly on the
    radius is still inside; only beyond it is out."""
    on_rim, _ = side_lost(_state(x=CFG.ring_radius), G1, CFG, TC)
    assert not on_rim.item(), "standing exactly on the rim should still be in"
    beyond, code = side_lost(_state(x=CFG.ring_radius + 1e-4), G1, CFG, TC)
    assert beyond.item()
    assert code.item() == LOSS_RING_OUT


def test_the_down_threshold_scales_with_each_robots_own_height():
    """A taller robot is down at a height a shorter one is fine at. Without this,
    nothing would catch `side_lost` ignoring its `robot` argument."""
    tall = dataclasses.replace(G1, name="tall", nominal_height=1.2)
    st = _state(z=CFG.platform_height + 0.5)          # 0.5 m above the platform
    lost_g1, _ = side_lost(st, G1, CFG, TC)           # 0.5 > 0.55*0.784 = 0.431
    lost_tall, code = side_lost(st, tall, CFG, TC)    # 0.5 < 0.55*1.2  = 0.66
    assert not lost_g1.item()
    assert lost_tall.item()
    assert code.item() == LOSS_DOWN


def test_compute_termination_judges_each_side_by_its_own_robot():
    """Both sides at the same height, but only the taller robot is down. If the
    implementation used `robot_a` for both, this would come out a draw."""
    tall = dataclasses.replace(G1, name="tall", nominal_height=1.2)
    z = CFG.platform_height + 0.5
    _, _, lost_a, lost_b, outcome = compute_termination(
        _state(z=z), _state(z=z), tall, G1,
        torch.zeros(1, dtype=torch.long), CFG, TC)
    assert lost_a.item()
    assert not lost_b.item()
    assert outcome.item() == B_WINS


def test_termination_is_batched():
    def batch(st, n):
        return RobotState(
            base_pos=st.base_pos.repeat(n, 1), base_quat=st.base_quat.repeat(n, 1),
            base_linvel_world=st.base_linvel_world.repeat(n, 1),
            base_angvel_local=st.base_angvel_local.repeat(n, 1),
            joint_pos=st.joint_pos.repeat(n, 1), joint_vel=st.joint_vel.repeat(n, 1))

    terminated, truncated, lost_a, lost_b, outcome = compute_termination(
        batch(_state(x=-0.5), 5), batch(_state(x=0.5), 5), G1, G1,
        torch.zeros(5, dtype=torch.long), CFG, TC)
    for t in (terminated, truncated, lost_a, lost_b, outcome):
        assert t.shape == (5,)
