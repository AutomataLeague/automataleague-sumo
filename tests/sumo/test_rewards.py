import math

import pytest
import torch

from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.rewards import compute_reward
from automataleague_sumo.envs.sumo.state import RobotState

RING = 1.5
N = 29
HORIZON = 750          # matches TerminationConfig.max_episode_steps
RATE = 1.0 / HORIZON   # rate terms are whole-episode weights spread over it


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
            action=None, rc=None, horizon=HORIZON):
    if prev_opp_r is None:
        prev_opp_r = torch.linalg.norm(opp.base_pos[:, :2], dim=-1)
    return compute_reward(
        own, opp, prev_opp_r,
        torch.tensor([own_lost]), torch.tensor([opp_lost]),
        action if action is not None else torch.zeros(1, N),
        RING, rc or RewardConfig(), horizon,
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
    that share a sign: `rc.action` and `rc.joint_vel` are both positive penalties,
    as are `rc.centre` and `rc.alive`. Swap either pair and every other test in
    this file still passes.

    It also pins WHICH terms are spread over the episode horizon and which are
    not. `push` telescopes, so it is already an episode-scale quantity; dividing
    it too would shrink it 750-fold and make the one dense proxy for winning
    invisible.
    """
    rc = RewardConfig()
    own = _state(-0.75, 0.0, yaw=0.0, joint_vel=0.5)   # r_own = 0.75 = R/2
    opp = _state(0.9, 0.0, math.pi)                    # r_opp = 0.9
    action = torch.ones(1, N)

    total, c = _reward(own, opp, prev_opp_r=torch.tensor([0.6]), action=action, rc=rc)

    assert c["win"].item() == pytest.approx(0.0)
    # rate terms: whole-episode weight, spread over the horizon
    assert c["centre"].item() == pytest.approx(-rc.centre * 0.5 ** 2 * RATE, rel=1e-6)
    assert c["alive"].item() == pytest.approx(rc.alive * RATE, rel=1e-6)
    assert c["action"].item() == pytest.approx(-rc.action * 1.0 * RATE, rel=1e-6)
    assert c["joint_vel"].item() == pytest.approx(-rc.joint_vel * 0.25 * RATE, rel=1e-6)
    # delta term: NOT spread, because it telescopes over the episode by itself
    assert c["push"].item() == pytest.approx(rc.push * (0.9 - 0.6) / RING, abs=1e-6)
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


def test_there_is_no_facing_reward():
    """`engage` paid for pointing at the opponent and decayed with distance. Over
    a 750-step episode it integrated to 136 against a win of 10, so the
    highest-scoring behaviour was to stand close and pose rather than to fight.
    Reintroducing a dense positional term is how that comes back."""
    _, comps = _reward(_state(-0.5, 0.0), _state(0.5, 0.0, math.pi))
    assert "engage" not in comps
    assert set(comps) == {"win", "push", "centre", "alive", "action", "joint_vel"}


def test_action_and_joint_velocity_are_penalized():
    a, b = _state(-0.5, 0.0), _state(0.5, 0.0, math.pi)
    _, quiet = _reward(a, b, action=torch.zeros(1, N))
    _, loud = _reward(a, b, action=torch.ones(1, N))
    assert loud["action"].item() < quiet["action"].item() <= 0
    _, spinning = _reward(_state(-0.5, 0.0, joint_vel=5.0), b)
    assert spinning["joint_vel"].item() < 0


def test_winning_outweighs_a_whole_episode_of_shaping():
    """The property the weights exist to guarantee, checked on real numbers
    rather than trusted to the config validator alone: a full episode of the best
    possible shaping must still be worth less than one win."""
    rc = RewardConfig()
    a, b = _state(-0.1, 0.0), _state(0.5, 0.0, math.pi)
    _, comps = _reward(a, b)
    per_step_best = comps["alive"].item() + comps["centre"].item()
    best_case = per_step_best * HORIZON + rc.push
    assert best_case < rc.win, (
        f"a whole episode of shaping is worth {best_case:.2f} against a win of "
        f"{rc.win} — farming beats fighting")


def test_the_rate_terms_scale_with_the_horizon_and_push_does_not():
    """A longer episode must not make the per-step terms worth more in total, and
    must not shrink the delta term. This is what keeps every weight comparable to
    `win` no matter what max_episode_steps is set to."""
    a, b = _state(-0.75, 0.0), _state(0.9, 0.0, math.pi)
    _, short = _reward(a, b, prev_opp_r=torch.tensor([0.6]), horizon=100)
    _, long = _reward(a, b, prev_opp_r=torch.tensor([0.6]), horizon=1000)
    assert short["alive"].item() == pytest.approx(10 * long["alive"].item())
    assert short["centre"].item() == pytest.approx(10 * long["centre"].item())
    assert short["push"].item() == pytest.approx(long["push"].item())


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
        torch.zeros(6, dtype=torch.bool), torch.zeros(6, N), RING, RewardConfig(),
        HORIZON)
    assert total.shape == (6,)
    assert all(v.shape == (6,) for v in comps.values())
