import math

import pytest
import torch

from automataleague_sumo.envs.sumo.observation import build_observation, observation_dim
from automataleague_sumo.envs.sumo.state import RobotState
from automataleague_sumo.robots import get_robot

RING = 1.5
N_JOINTS = 29


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw / 2
    z = torch.zeros_like(half)
    return torch.stack([torch.cos(half), z, z, torch.sin(half)], dim=-1)


def _make_state(x, y, yaw, seed):
    """A state with arbitrary but reproducible joint and velocity values."""
    g = torch.Generator().manual_seed(seed)
    n = 1
    return RobotState(
        base_pos=torch.tensor([[x, y, 1.05]]),
        base_quat=_quat_from_yaw(torch.tensor([yaw])),
        base_linvel_world=torch.rand(n, 3, generator=g) - 0.5,
        base_angvel_local=torch.rand(n, 3, generator=g) - 0.5,
        joint_pos=torch.rand(n, N_JOINTS, generator=g) - 0.5,
        joint_vel=torch.rand(n, N_JOINTS, generator=g) - 0.5,
    )


def _rotate_state(st: RobotState, theta: float) -> RobotState:
    """Rotate a state rigidly about the world z axis through the ring centre."""
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    yaw = torch.atan2(
        2 * (st.base_quat[:, 0] * st.base_quat[:, 3]),
        1 - 2 * st.base_quat[:, 3] ** 2,
    )
    return RobotState(
        base_pos=st.base_pos @ rot.T,
        base_quat=_quat_from_yaw(yaw + theta),
        base_linvel_world=st.base_linvel_world @ rot.T,
        base_angvel_local=st.base_angvel_local,   # already body frame
        joint_pos=st.joint_pos,
        joint_vel=st.joint_vel,
    )


@pytest.fixture
def pair():
    own = _make_state(-0.9, 0.0, 0.0, seed=1)
    opp = _make_state(0.6, 0.3, math.pi, seed=2)
    prev_action = torch.zeros(1, N_JOINTS)
    home = torch.tensor(get_robot("g1").home_joint_qpos)
    contact = torch.zeros(1)
    return own, opp, prev_action, home, contact


def _obs(own, opp, prev_action, home, contact):
    return build_observation(own, opp, prev_action, home, RING, contact)


def test_observation_dim_is_derived_from_joint_count():
    g1 = get_robot("g1")
    assert observation_dim(g1) == 3 * g1.n_joints + 23
    assert observation_dim(g1) == 110


def test_observation_width_matches_observation_dim(pair):
    obs = _obs(*pair)
    assert obs.shape == (1, observation_dim(get_robot("g1")))


@pytest.mark.parametrize("theta", [0.4, 1.7, -2.2, math.pi])
def test_observation_is_invariant_to_rotating_the_whole_world(pair, theta):
    """The arena is rotationally symmetric, so a rigid rotation of both robots
    about the ring centre must not change either observation. This is what proves
    the observation carries no absolute world direction."""
    own, opp, prev_action, home, contact = pair
    before = _obs(own, opp, prev_action, home, contact)
    after = _obs(_rotate_state(own, theta), _rotate_state(opp, theta),
                 prev_action, home, contact)
    assert torch.allclose(before, after, atol=1e-5)


def test_mirrored_configurations_produce_identical_observations(pair):
    """If B is exactly A rotated by pi about the ring centre, both sides must see
    the same thing. This is the invariant that makes one shared self-play policy
    valid for both sides."""
    _, _, prev_action, home, contact = pair
    a = _make_state(-0.9, 0.2, 0.3, seed=7)
    b = _rotate_state(a, math.pi)
    obs_a = _obs(a, b, prev_action, home, contact)
    obs_b = _obs(b, a, prev_action, home, contact)
    assert torch.allclose(obs_a, obs_b, atol=1e-5)


def test_swapping_the_arguments_swaps_the_perspective(pair):
    own, opp, prev_action, home, contact = pair
    obs_a = _obs(own, opp, prev_action, home, contact)
    obs_b = _obs(opp, own, prev_action, home, contact)
    assert not torch.allclose(obs_a, obs_b), "asymmetric states must differ"


def test_relative_position_block_points_at_the_opponent(pair):
    """Own robot at -x facing +x, opponent ahead: relative position is +x in the
    own base frame."""
    _, _, prev_action, home, contact = pair
    own = _make_state(-0.9, 0.0, 0.0, seed=3)
    opp = _make_state(0.9, 0.0, math.pi, seed=4)
    obs = _obs(own, opp, prev_action, home, contact)
    start = 9 + 3 * N_JOINTS + 4          # after proprio and the ring block
    rel_pos = obs[0, start:start + 3]
    assert rel_pos[0].item() == pytest.approx(1.8, abs=1e-4)
    assert rel_pos[1].item() == pytest.approx(0.0, abs=1e-4)


def test_ring_block_reports_normalized_radius_and_edge_distance(pair):
    _, _, prev_action, home, contact = pair
    own = _make_state(-0.75, 0.0, 0.0, seed=5)     # exactly half the ring radius
    opp = _make_state(0.9, 0.0, math.pi, seed=6)
    obs = _obs(own, opp, prev_action, home, contact)
    ring = obs[0, 9 + 3 * N_JOINTS: 9 + 3 * N_JOINTS + 4]
    assert ring[0].item() == pytest.approx(0.5, abs=1e-5)       # r / R
    assert ring[3].item() == pytest.approx(0.5, abs=1e-5)       # (R - r) / R
    # Unit vector toward the centre, in the own base frame: own faces +x at -x,
    # so the centre is straight ahead.
    assert ring[1].item() == pytest.approx(1.0, abs=1e-5)
    assert ring[2].item() == pytest.approx(0.0, abs=1e-5)


def test_contact_flag_is_the_last_column(pair):
    own, opp, prev_action, home, _ = pair
    off = _obs(own, opp, prev_action, home, torch.zeros(1))
    on = _obs(own, opp, prev_action, home, torch.ones(1))
    assert off[0, -1].item() == 0.0
    assert on[0, -1].item() == 1.0
    assert torch.allclose(off[0, :-1], on[0, :-1])


def test_centre_direction_is_stable_at_the_exact_centre(pair):
    """No NaN when the robot stands precisely on the ring centre."""
    _, _, prev_action, home, contact = pair
    own = _make_state(0.0, 0.0, 0.0, seed=8)
    opp = _make_state(0.9, 0.0, math.pi, seed=9)
    obs = _obs(own, opp, prev_action, home, contact)
    assert torch.isfinite(obs).all()


def test_observation_is_batched(pair):
    own, opp, prev_action, home, contact = pair

    def rep(st):
        return RobotState(
            *[getattr(st, f).repeat(8, *([1] * (getattr(st, f).dim() - 1)))
              for f in ("base_pos", "base_quat", "base_linvel_world",
                        "base_angvel_local", "joint_pos", "joint_vel")])

    obs = build_observation(rep(own), rep(opp), prev_action.repeat(8, 1), home,
                            RING, contact.repeat(8))
    assert obs.shape == (8, 110)
