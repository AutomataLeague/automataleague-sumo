import mujoco
import pytest
import torch

from automataleague_sumo.envs.sumo.scene import build_sumo_model
from automataleague_sumo.envs.sumo.state import (
    contact_flag_cpu,
    extract_duel_state,
    extract_state,
)


@pytest.fixture(scope="module")
def built():
    model, info = build_sumo_model("g1")
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)
    return model, data, info


def _tensors(data):
    return (torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0),
            torch.tensor(data.qvel, dtype=torch.float32).unsqueeze(0))


def test_extract_state_shapes(built):
    _, data, info = built
    qpos, qvel = _tensors(data)
    st = extract_state(qpos, qvel, info.a)
    n = info.a.robot.n_joints
    assert st.base_pos.shape == (1, 3)
    assert st.base_quat.shape == (1, 4)
    assert st.base_linvel_world.shape == (1, 3)
    assert st.base_angvel_local.shape == (1, 3)
    assert st.joint_pos.shape == (1, n)
    assert st.joint_vel.shape == (1, n)


def test_duel_state_reads_the_two_sides_apart(built):
    _, data, info = built
    qpos, qvel = _tensors(data)
    sa, sb = extract_duel_state(qpos, qvel, info)
    # Spawns are diametrically opposite, so the x coordinates have opposite signs.
    assert sa.base_pos[0, 0].item() < 0 < sb.base_pos[0, 0].item()
    assert sa.base_pos[0, 2].item() == pytest.approx(sb.base_pos[0, 2].item())


def test_state_matches_the_home_pose(built):
    _, data, info = built
    qpos, qvel = _tensors(data)
    sa, _ = extract_duel_state(qpos, qvel, info)
    assert torch.allclose(
        sa.joint_pos[0], torch.tensor(info.a.robot.home_joint_qpos), atol=1e-5)
    assert torch.allclose(sa.joint_vel, torch.zeros_like(sa.joint_vel))


def test_base_linear_and_angular_velocity_land_in_the_right_slots(built):
    """At rest every velocity field reads zero, so swapping the linear and
    angular slices would be completely invisible. Write distinct nonzero values
    and check each lands where its NAME says it does.

    MuJoCo free-joint convention: qvel[0:3] is linear velocity in the WORLD
    frame, qvel[3:6] is angular velocity in the BODY frame.
    """
    model, data, info = built
    qpos = torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0)
    qvel = torch.zeros(1, model.nv)
    da = info.a.base_dofadr
    qvel[0, da:da + 3] = torch.tensor([1.0, 2.0, 3.0])        # linear, world
    qvel[0, da + 3:da + 6] = torch.tensor([4.0, 5.0, 6.0])    # angular, body

    st = extract_state(qpos, qvel, info.a)
    assert torch.allclose(st.base_linvel_world[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(st.base_angvel_local[0], torch.tensor([4.0, 5.0, 6.0]))


def test_joint_velocity_is_read_from_the_joint_dofs_not_the_base(built):
    """A wrong dof offset would pull base velocity into the joint block."""
    model, data, info = built
    qpos = torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0)
    qvel = torch.zeros(1, model.nv)
    jd = torch.as_tensor(info.a.joint_dofadr, dtype=torch.long)
    expected = torch.arange(1, len(jd) + 1, dtype=torch.float32)
    qvel[0, jd] = expected

    st = extract_state(qpos, qvel, info.a)
    assert torch.allclose(st.joint_vel[0], expected)
    assert torch.allclose(st.base_linvel_world, torch.zeros(1, 3))
    assert torch.allclose(st.base_angvel_local, torch.zeros(1, 3))


def test_each_side_reads_its_own_joints(built):
    """Both robots share a home pose, so comparing against home cannot catch a
    side-swap in the joint indices. Perturb only side A and check B is untouched."""
    model, data, info = built
    qpos = torch.tensor(data.qpos, dtype=torch.float32).unsqueeze(0)
    qvel = torch.zeros(1, model.nv)
    qpos[0, torch.as_tensor(info.a.joint_qposadr, dtype=torch.long)] += 0.5

    sa, sb = extract_duel_state(qpos, qvel, info)
    assert not torch.allclose(sa.joint_pos, sb.joint_pos)
    assert torch.allclose(
        sb.joint_pos[0], torch.tensor(info.b.robot.home_joint_qpos), atol=1e-5)


def test_extract_state_is_batched(built):
    _, data, info = built
    qpos, qvel = _tensors(data)
    st = extract_state(qpos.repeat(4, 1), qvel.repeat(4, 1), info.a)
    assert st.base_pos.shape == (4, 3)
    assert st.joint_pos.shape == (4, info.a.robot.n_joints)


def test_no_contact_between_robots_at_spawn(built):
    model, data, info = built
    assert contact_flag_cpu(model, data, info).item() == 0.0


def test_contact_is_detected_when_the_robots_are_shoved_together(built):
    model, _, info = built
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    # Slide both robots to the ring centre so they interpenetrate.
    data.qpos[info.a.base_qposadr] = -0.1
    data.qpos[info.b.base_qposadr] = 0.1
    mujoco.mj_forward(model, data)
    assert contact_flag_cpu(model, data, info).item() == 1.0
