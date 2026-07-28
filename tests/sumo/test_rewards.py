import math

import pytest
import torch

from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.rewards import compute_reward
from automataleague_sumo.envs.sumo.state import RobotState

RING = 1.5
N = 29


def _state(x, y, yaw=0.0, joint_vel=0.0):
    half = yaw / 2
    return RobotState(
        base_pos=torch.tensor([[x, y, 1.05]]),
        base_quat=torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]]),
        base_linvel_world=torch.zeros(1, 3),
        base_angvel_local=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, N),
        joint_vel=torch.full((1, N), joint_vel),
    )


def _reward(own, opp, prev_opp_r=None, own_lost=False, opp_lost=False,
            action=None, rc=None, shaping_scale=1.0):
    if prev_opp_r is None:
        prev_opp_r = torch.linalg.norm(opp.base_pos[:, :2], dim=-1)
    return compute_reward(
        own, opp, prev_opp_r,
        torch.tensor([own_lost]), torch.tensor([opp_lost]),
        action if action is not None else torch.zeros(1, N),
        RING, rc or RewardConfig(), shaping_scale,
    )


def test_components_sum_to_the_total():
    """Structural check only. `compute_reward` builds its total by summing the
    very dict it returns, so this cannot fail for any implementation following
    that pattern. It guards against a future refactor that computes the total
    separately and lets the two drift. The real value pinning is in
    `test_every_shaping_term_matches_its_configured_weight` below."""
    total, comps = _reward(_state(-0.5, 0.0), _state(0.5, 0.0, math.pi))
    assert torch.allclose(sum(comps.values()), total, atol=1e-6)


def test_every_shaping_term_matches_its_configured_weight():
    """Pin each term to a value derived independently from its config field.

    Sign and ordering tests cannot catch a coefficient SWAP between two terms
    that share a sign: `rc.action` (0.01) and `rc.joint_vel` (0.001) are both
    positive penalties, as are `rc.center` (0.5) and `rc.engage` (0.3). Swap
    either pair and every other test in this file still passes. Nor is `alive`
    pinned anywhere else: the `shaping_scale=0` test forces it to zero, which is
    trivially true of any scaled term regardless of its magnitude.
    """
    rc = RewardConfig()
    own = _state(-0.75, 0.0, yaw=0.0, joint_vel=0.5)   # r_own = 0.75 = R/2
    opp = _state(0.9, 0.0, math.pi)                    # r_opp = 0.9
    action = torch.ones(1, N)

    total, c = _reward(own, opp, prev_opp_r=torch.tensor([0.6]), action=action, rc=rc)

    dist = 1.65                                        # 0.9 - (-0.75)
    assert c["win"].item() == pytest.approx(0.0)
    assert c["centre"].item() == pytest.approx(-rc.center * 0.5 ** 2, abs=1e-6)
    assert c["push"].item() == pytest.approx(rc.push * (0.9 - 0.6) / RING, abs=1e-6)
    # own faces +x and the opponent is straight ahead, so alignment is exactly 1.
    assert c["engage"].item() == pytest.approx(
        rc.engage * math.exp(-dist / rc.engage_range), abs=1e-6)
    assert c["alive"].item() == pytest.approx(rc.alive, abs=1e-9)
    assert c["action"].item() == pytest.approx(-rc.action * 1.0, abs=1e-9)
    assert c["joint_vel"].item() == pytest.approx(-rc.joint_vel * 0.25, abs=1e-9)
    assert total.item() == pytest.approx(sum(v.item() for v in c.values()), abs=1e-6)


def test_win_term_is_strictly_zero_sum():
    """A's win term and B's win term must cancel exactly, in every outcome."""
    a, b = _state(-0.5, 0.0), _state(0.5, 0.0, math.pi)
    for a_lost, b_lost in ((True, False), (False, True), (True, True), (False, False)):
        _, ca = _reward(a, b, own_lost=a_lost, opp_lost=b_lost)
        _, cb = _reward(b, a, own_lost=b_lost, opp_lost=a_lost)
        assert (ca["win"] + cb["win"]).abs().item() < 1e-6


def test_winning_pays_and_losing_costs():
    a, b = _state(-0.5, 0.0), _state(0.5, 0.0, math.pi)
    rc = RewardConfig()
    _, won = _reward(a, b, opp_lost=True)
    _, lost = _reward(a, b, own_lost=True)
    assert won["win"].item() == pytest.approx(rc.win)
    assert lost["win"].item() == pytest.approx(-rc.win)


def test_push_term_rewards_driving_the_opponent_outward():
    a, b = _state(-0.5, 0.0), _state(0.9, 0.0, math.pi)
    _, out = _reward(a, b, prev_opp_r=torch.tensor([0.6]))   # opponent moved out
    _, back = _reward(a, b, prev_opp_r=torch.tensor([1.2]))  # opponent came back in
    assert out["push"].item() > 0
    assert back["push"].item() < 0


def test_push_term_is_zero_when_the_opponent_does_not_move():
    a, b = _state(-0.5, 0.0), _state(0.9, 0.0, math.pi)
    _, comps = _reward(a, b, prev_opp_r=torch.tensor([0.9]))
    assert comps["push"].item() == pytest.approx(0.0, abs=1e-6)


def test_centre_term_penalizes_drifting_toward_the_rim():
    b = _state(0.5, 0.0, math.pi)
    _, near = _reward(_state(-0.1, 0.0), b)
    _, far = _reward(_state(-1.4, 0.0), b)
    assert near["centre"].item() > far["centre"].item()
    assert far["centre"].item() < 0


def test_engage_term_rewards_facing_the_opponent():
    b = _state(0.5, 0.0, math.pi)
    _, facing = _reward(_state(-0.5, 0.0, yaw=0.0), b)          # looking at B
    _, away = _reward(_state(-0.5, 0.0, yaw=math.pi), b)        # looking away
    assert facing["engage"].item() > 0 > away["engage"].item()


def test_engage_term_decays_with_distance():
    _, close = _reward(_state(-0.2, 0.0), _state(0.2, 0.0, math.pi))
    _, distant = _reward(_state(-1.4, 0.0), _state(1.4, 0.0, math.pi))
    assert close["engage"].item() > distant["engage"].item()


def test_action_and_joint_velocity_are_penalized():
    a, b = _state(-0.5, 0.0), _state(0.5, 0.0, math.pi)
    _, quiet = _reward(a, b, action=torch.zeros(1, N))
    _, loud = _reward(a, b, action=torch.ones(1, N))
    assert loud["action"].item() < quiet["action"].item() <= 0
    _, spinning = _reward(_state(-0.5, 0.0, joint_vel=5.0), b)
    assert spinning["joint_vel"].item() < 0


def test_shaping_scale_zero_leaves_only_the_sparse_win_term():
    a, b = _state(-1.2, 0.0), _state(0.5, 0.0, math.pi)
    total, comps = _reward(a, b, opp_lost=True, shaping_scale=0.0)
    assert total.item() == pytest.approx(RewardConfig().win)
    for name, value in comps.items():
        if name != "win":
            assert value.item() == pytest.approx(0.0, abs=1e-6)


def test_shaping_scale_scales_every_shaping_term_but_not_the_win_term():
    a, b = _state(-1.2, 0.0), _state(0.5, 0.0, math.pi)
    _, full = _reward(a, b, opp_lost=True, shaping_scale=1.0)
    _, half = _reward(a, b, opp_lost=True, shaping_scale=0.5)
    assert half["win"].item() == pytest.approx(full["win"].item())
    assert half["centre"].item() == pytest.approx(0.5 * full["centre"].item())


def test_reward_is_batched():
    own = RobotState(
        base_pos=torch.rand(6, 3), base_quat=torch.tensor([[1.0, 0, 0, 0]]).repeat(6, 1),
        base_linvel_world=torch.zeros(6, 3), base_angvel_local=torch.zeros(6, 3),
        joint_pos=torch.zeros(6, N), joint_vel=torch.zeros(6, N))
    opp = RobotState(
        base_pos=torch.rand(6, 3), base_quat=torch.tensor([[1.0, 0, 0, 0]]).repeat(6, 1),
        base_linvel_world=torch.zeros(6, 3), base_angvel_local=torch.zeros(6, 3),
        joint_pos=torch.zeros(6, N), joint_vel=torch.zeros(6, N))
    total, comps = compute_reward(
        own, opp, torch.zeros(6), torch.zeros(6, dtype=torch.bool),
        torch.zeros(6, dtype=torch.bool), torch.zeros(6, N), RING, RewardConfig())
    assert total.shape == (6,)
    assert all(v.shape == (6,) for v in comps.values())
