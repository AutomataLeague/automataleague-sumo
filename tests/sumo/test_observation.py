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
    the same thing.

    Note for the reader: this is a COROLLARY of the rotation-invariance test at
    theta = pi, not an independent check. Since R_pi is an involution,
    obs_b = f(R_pi a, R_pi(R_pi a)) collapses to f(a, R_pi a) = obs_a for any
    rotation-invariant f. It is kept because it states the shared-policy claim in
    the form a reader cares about, but the claim is actually carried by the
    rotation test plus the concrete-value tests below."""
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


def test_relative_position_uses_the_full_3d_base_frame_not_just_yaw(pair):
    """With a pitched base the opponent's relative position must be expressed in
    the true 3D body frame.

    A yaw-only approximation agrees with the correct transform whenever the robot
    is upright, and diverges the moment it leans — which in sumo is most of the
    time. Rotation invariance can never catch this: a yaw-only implementation is
    still invariant under a world z rotation, because the yaw and the world
    vector rotate together and cancel. Only a value test with a tilted base
    distinguishes them.
    """
    _, _, prev_action, home, contact = pair
    pitch = math.radians(30.0)
    own = _make_state(0.0, 0.0, 0.0, seed=11)
    # Pitch the base 30 degrees about its own y axis.
    own.base_pos = torch.tensor([[0.0, 0.0, 1.0]])
    own.base_quat = torch.tensor(
        [[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    opp = _make_state(1.0, 0.0, math.pi, seed=12)
    opp.base_pos = torch.tensor([[1.0, 0.0, 1.0]])   # one metre straight ahead

    obs = _obs(own, opp, prev_action, home, contact)
    start = 9 + 3 * N_JOINTS + 4
    rel_pos = obs[0, start:start + 3]
    # R(q)^T applied to the world offset (1,0,0) with q = pitch about y gives
    # (cos p, 0, sin p). A yaw-only transform would return (1, 0, 0).
    assert rel_pos[0].item() == pytest.approx(math.cos(pitch), abs=1e-5)
    assert rel_pos[1].item() == pytest.approx(0.0, abs=1e-5)
    assert rel_pos[2].item() == pytest.approx(math.sin(pitch), abs=1e-5)


def test_proprioception_block_carries_its_values(pair):
    """Pin the proprioceptive slots by value.

    Rotation invariance can never guard `projected_gravity`: gravity lies on the
    rotation axis, so that block is invariant for free and a rotation test would
    pass even if it were zeroed. The joint_vel and prev_action slots are likewise
    unguarded unless they hold DISTINCT nonzero values, otherwise a swap between
    them is invisible.
    """
    _, opp, _, home, contact = pair
    own = _make_state(-0.5, 0.0, 0.0, seed=13)
    own.base_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])          # upright
    own.base_angvel_local = torch.tensor([[0.1, 0.2, 0.3]])
    own.joint_pos = home.clone().unsqueeze(0) + 0.25
    own.joint_vel = torch.full((1, N_JOINTS), 0.5)
    prev_action = torch.full((1, N_JOINTS), -0.75)                # distinct value

    obs = _obs(own, opp, prev_action, home, contact)
    n = N_JOINTS
    assert torch.allclose(obs[0, 3:6], torch.tensor([0.1, 0.2, 0.3]), atol=1e-5)
    # Upright base: projected gravity points straight down in the body frame.
    assert torch.allclose(obs[0, 6:9], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5)
    assert torch.allclose(obs[0, 9:9 + n], torch.full((n,), 0.25), atol=1e-5)
    assert torch.allclose(obs[0, 9 + n:9 + 2 * n], torch.full((n,), 0.5), atol=1e-5)
    assert torch.allclose(obs[0, 9 + 2 * n:9 + 3 * n], torch.full((n,), -0.75), atol=1e-5)


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


# ------------------------------------------------------------------- clipping

def test_the_observation_is_bounded():
    """Contact between two humanoids drives joint velocities far outside their
    working range — measured at 49.5 rad/s against a mean of about 2.5 — and those
    feed an unnormalised network. A 1B-frame run died at 500M when the losses went
    non-finite and clip_grad_norm_ multiplied every weight by the NaN.
    """
    from automataleague_sumo.envs.sumo.observation import OBS_CLIP

    robot = get_robot("g1")
    n = robot.n_joints
    wild = RobotState(
        base_pos=torch.tensor([[0.3, 0.0, 1.0]]),
        base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        base_linvel_world=torch.full((1, 3), 500.0),
        base_angvel_local=torch.full((1, 3), -900.0),
        joint_pos=torch.full((1, n), 40.0),
        joint_vel=torch.full((1, n), -3000.0),
    )
    obs = build_observation(wild, wild, torch.zeros(1, n),
                            torch.zeros(n), 1.5, torch.zeros(1))
    assert torch.isfinite(obs).all()
    assert float(obs.abs().max()) <= OBS_CLIP + 1e-6


def test_the_clip_does_not_touch_ordinary_play():
    """A bound that clipped normal signal would be silently destroying the
    observation rather than protecting it. The largest non-spike component
    measured in real duels was 21.7, so the bound has to sit above that."""
    from automataleague_sumo.envs.sumo.observation import OBS_CLIP

    assert OBS_CLIP > 21.7, "the clip would truncate signal the task actually uses"

    robot = get_robot("g1")
    n = robot.n_joints
    normal = RobotState(
        base_pos=torch.tensor([[0.4, 0.1, 1.0]]),
        base_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        base_linvel_world=torch.full((1, 3), 1.5),
        base_angvel_local=torch.full((1, 3), 2.0),
        joint_pos=torch.full((1, n), 0.3),
        joint_vel=torch.full((1, n), 3.0),
    )
    obs = build_observation(normal, normal, torch.zeros(1, n),
                            torch.zeros(n), 1.5, torch.zeros(1))
    assert float(obs.abs().max()) < OBS_CLIP, "ordinary play is being clipped"
