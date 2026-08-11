"""Scripted baselines, and the proof that the contract admits a non-PPO policy.

These are as far from the trained actor as a competitor gets: no weights, no
torchrl, no hydra config, an artifact of five scalars. If they cannot enter a
tournament cleanly, neither will anyone's SAC.
"""

from __future__ import annotations

import pytest
import torch

from automataleague_sumo.envs.sumo.observation import observation_dim
from automataleague_sumo.policy import check_policy, load_policy
from automataleague_sumo.robots import get_robot
from automataleague_sumo.scripted import KINDS, ScriptedPolicy, save_scripted_policy

ROBOT = get_robot("g1")
OBS, ACT = observation_dim(ROBOT), ROBOT.action_dim


def _obs(batch=4, opponent_ahead=True):
    """A batch with the opponent placed in front of, or behind, the robot."""
    obs = torch.zeros(batch, OBS)
    rel = ROBOT.proprio_dim + 4          # rel_pos_base, in the robot's own frame
    obs[:, rel] = 1.0 if opponent_ahead else -1.0
    return obs


@pytest.mark.parametrize("kind", KINDS)
def test_every_scripted_kind_satisfies_the_contract(kind):
    """The whole point of these existing before the leaderboard does."""
    check_policy(ScriptedPolicy(kind, ROBOT), obs_dim=OBS, act_dim=ACT)


def test_still_does_exactly_nothing():
    """It is the floor a rating is read against, so it must be the null action
    rather than merely a small one."""
    action = ScriptedPolicy("still", ROBOT).act(_obs())
    assert torch.equal(action, torch.zeros_like(action))


def test_lean_drives_the_ankles_toward_the_opponent():
    """Direction is measured, not assumed: -0.8 on both ankle pitches moves the
    base +0.215 m along its own forward axis, +0.8 moves it -0.213 m. So facing
    an opponent in front, the ankle action must be NEGATIVE."""
    policy = ScriptedPolicy("lean", ROBOT)
    ahead = policy.act(_obs(opponent_ahead=True))
    behind = policy.act(_obs(opponent_ahead=False))
    assert (ahead[:, policy._ankle] < 0).all(), "should lean into a front opponent"
    assert (behind[:, policy._ankle] > 0).all(), "should lean back at one behind"


def test_lean_ignores_how_far_away_the_opponent_is():
    """It uses the unit direction, so a distant opponent gets the same lean as a
    close one. Scaling by raw distance would make the action depend on the ring
    size, which is a config knob."""
    policy = ScriptedPolicy("lean", ROBOT)
    near, far = _obs(), _obs()
    far[:, ROBOT.proprio_dim + 4] = 50.0
    assert torch.allclose(policy.act(near), policy.act(far))


def test_a_wrong_width_observation_says_what_it_expected():
    """A silent shape mismatch would drive the wrong joints. The message has to
    name both numbers or the next person has to go reading source."""
    policy = ScriptedPolicy("still", ROBOT)
    with pytest.raises(ValueError, match=f"expects {OBS}"):
        policy.act(torch.zeros(2, OBS - 1))


def test_an_unknown_kind_is_refused_with_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown scripted kind"):
        ScriptedPolicy("teleport", ROBOT)


def test_an_artifact_round_trips_through_the_public_loader(tmp_path):
    """Saved by one tool, loaded by the tournament, with nothing imported in
    between: exactly the path a third-party submission takes."""
    path = tmp_path / "lean.pt"
    save_scripted_policy(str(path), "lean", robot="g1", gain=0.4)
    policy = load_policy(str(path))
    check_policy(policy, obs_dim=OBS, act_dim=ACT)
    assert policy.info.algorithm == "scripted"
    assert policy.info.robot == "g1"
    assert policy.info.extra["gain"] == 0.4


def test_saving_an_invalid_baseline_fails_before_writing(tmp_path):
    """Otherwise a broken artifact sits on disk until a tournament trips over it."""
    path = tmp_path / "nope.pt"
    with pytest.raises(ValueError):
        save_scripted_policy(str(path), "not-a-kind")
    assert not path.exists()
