import math

import mujoco
import pytest

from automataleague_sumo.envs.sumo.config import SumoConfig
from automataleague_sumo.envs.sumo.scene import build_sumo_model


@pytest.fixture(scope="module")
def built():
    cfg = SumoConfig()
    model, info = build_sumo_model("g1", cfg=cfg)
    return model, info, cfg


def test_model_holds_exactly_two_robots(built):
    model, info, _ = built
    n = info.a.robot.n_joints
    assert model.nu == 2 * n                      # 29 actuators per side
    assert model.nq == 2 * (7 + n)                # free joint + hinges, per side


def test_both_prefixes_resolve(built):
    model, info, _ = built
    for side, prefix in ((info.a, "a/"), (info.b, "b/")):
        assert side.prefix == prefix
        assert side.base_body_id >= 0
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + side.robot.base_body) == side.base_body_id
        assert len(side.actuator_ids) == side.robot.n_joints
        assert len(side.joint_qposadr) == side.robot.n_joints
        assert len(side.joint_dofadr) == side.robot.n_joints


def test_side_address_blocks_do_not_overlap(built):
    _, info, _ = built
    assert set(info.a.joint_qposadr).isdisjoint(set(info.b.joint_qposadr))
    assert set(info.a.actuator_ids).isdisjoint(set(info.b.actuator_ids))
    assert info.a.base_qposadr != info.b.base_qposadr


def test_geom_ownership_partitions_the_two_robots(built):
    _, info, _ = built
    assert len(info.a.geom_ids) > 0
    assert set(info.a.geom_ids).isdisjoint(set(info.b.geom_ids))
    assert set(info.a.foot_geom_ids).issubset(set(info.a.geom_ids))
    assert len(info.a.foot_geom_ids) == len(info.a.robot.foot_geoms)


def test_home_qpos_places_both_robots_on_the_platform(built):
    _, info, cfg = built
    for side in info.sides:
        base = info.home_qpos[side.base_qposadr:side.base_qposadr + 3]
        assert math.hypot(base[0], base[1]) == pytest.approx(cfg.spawn_radius, abs=1e-5)
        assert base[2] == pytest.approx(cfg.platform_height + side.robot.nominal_height)


def test_home_qpos_faces_the_robots_at_each_other(built):
    _, info, _ = built
    yaws = []
    for side in info.sides:
        w, x, y, z = info.home_qpos[side.base_qposadr + 3:side.base_qposadr + 7]
        yaws.append(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    assert abs(abs(yaws[0] - yaws[1]) - math.pi) < 1e-5


def test_robots_do_not_interpenetrate_at_spawn(built):
    model, info, _ = built
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)
    a_geoms, b_geoms = set(info.a.geom_ids), set(info.b.geom_ids)
    crossing = [
        c for c in data.contact[:data.ncon]
        if (c.geom1 in a_geoms and c.geom2 in b_geoms)
        or (c.geom1 in b_geoms and c.geom2 in a_geoms)
    ]
    assert not crossing, f"{len(crossing)} robot-robot contacts at spawn"


def test_both_robots_stand_on_the_platform_without_falling(built):
    model, info, cfg = built
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    for side in info.sides:
        data.ctrl[side.actuator_ids] = side.robot.home_joint_qpos
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    for side in info.sides:
        z = data.qpos[side.base_qposadr + 2] - cfg.platform_height
        assert z > 0.6, f"{side.prefix} collapsed to {z:.3f} above the platform"


def test_platform_is_a_cylinder_of_the_configured_radius(built):
    model, info, cfg = built
    assert model.geom_type[info.platform_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom_size[info.platform_geom_id][0] == pytest.approx(cfg.ring_radius)
    assert model.geom_size[info.platform_geom_id][1] == pytest.approx(cfg.platform_height / 2)


def test_explicit_opponent_robot_builds():
    """Naming the second robot is the cross-robot evaluation path. Passing the same
    robot twice must be equivalent to leaving it None."""
    model, info = build_sumo_model("g1", "g1", cfg=SumoConfig())
    assert info.a.robot.name == "g1"
    assert info.b.robot.name == "g1"
    assert model.nu == 58


def test_ring_radius_is_honoured():
    _, info = build_sumo_model("g1", cfg=SumoConfig(ring_radius=2.0))
    for side in info.sides:
        base = info.home_qpos[side.base_qposadr:side.base_qposadr + 3]
        assert math.hypot(base[0], base[1]) == pytest.approx(1.2, abs=1e-5)


def test_compiled_model_keeps_the_vendored_solver_tuning(built):
    """MjSpec.attach's default conflict policy keeps the *parent's* option
    fields on any mismatch with the child, so a freshly created MjSpec() would
    silently revert the compiled model to timestep=0.002/iterations=100 instead
    of the vendored g1_mjx.xml's timestep=0.004/iterations=5 the robot was
    tuned with, and would leave eulerdamp enabled instead of disabled. That
    changes control rate (SumoConfig.frame_skip=5 assumes a 0.004s step, giving
    50 Hz) and joint damping behaviour, so the scene builder must set these
    explicitly rather than let them fall back to MuJoCo defaults.
    """
    model, _, _ = built
    assert model.opt.timestep == pytest.approx(0.004)
    assert model.opt.iterations == 5
    assert model.opt.ls_iterations == 8
    assert bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
