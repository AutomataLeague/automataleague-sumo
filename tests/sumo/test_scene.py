import math

import mujoco
import numpy as np
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


def test_the_assembled_scene_is_numerically_stable(built):
    """The scene must not blow up. This is about the solver, not about balance.

    A passive humanoid sinking into a crouch is expected physics; learning to
    stand is what the standing bootstrap is for. A humanoid going AIRBORNE, or its
    pelvis reaching an absurd coordinate, means contact forces are diverging.
    An earlier revision of this scene inherited `cone=ELLIPTIC, impratio=100`
    from the quadruped parkour repo; against this model's `iterations=5` that
    launched the pelvis to z = -4839 m. Hence the explicit checks below.
    """
    model, info, cfg = built
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    for side in info.sides:
        data.ctrl[side.actuator_ids] = side.robot.home_joint_qpos
    mujoco.mj_forward(model, data)

    airborne = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qvel).all(), "solver diverged (non-finite qvel)"
        if data.ncon == 0:
            airborne += 1
        for side in info.sides:
            z = data.qpos[side.base_qposadr + 2] - cfg.platform_height
            assert -1.0 < z < 2.0, f"{side.prefix} left the world at z={z:.1f}"

    assert airborne == 0, f"robots went airborne on {airborne}/500 steps"


def test_both_robots_settle_upright_on_the_platform(built):
    """After settling, both robots are still on top of the platform and have not
    sunk through it. Not a balance test: the threshold is low on purpose."""
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
        assert z > 0.1, f"{side.prefix} sank into the platform: z={z:.3f}"
        xy = data.qpos[side.base_qposadr:side.base_qposadr + 2]
        assert math.hypot(*xy) < cfg.ring_radius, f"{side.prefix} drifted out of the ring"


def test_the_two_sides_behave_symmetrically(built):
    """The two robots start in mirror-image poses under identical control, so
    they should settle to near-identical heights. A large divergence means the
    simulation is chaotic, which is the signature of an unstable solver."""
    model, info, cfg = built
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    for side in info.sides:
        data.ctrl[side.actuator_ids] = side.robot.home_joint_qpos
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    z_a = data.qpos[info.a.base_qposadr + 2]
    z_b = data.qpos[info.b.base_qposadr + 2]
    assert abs(z_a - z_b) < 0.15, f"sides diverged: {z_a:.3f} vs {z_b:.3f}"


def test_the_vendored_solver_settings_survive_attach(built):
    """`MjSpec.attach` does not carry the child's <option> block up to the parent.
    Without mirroring it, the compiled model silently uses MuJoCo's default
    timestep of 0.002, which breaks SumoConfig's frame_skip=5 => 50 Hz control
    rate. Pin the values that matter."""
    model, _, _ = built
    assert model.opt.timestep == pytest.approx(0.004)
    assert model.opt.iterations == 5
    assert model.opt.ls_iterations == 8


def test_the_scene_does_not_inherit_quadruped_contact_settings(built):
    """Regression guard. `cone=ELLIPTIC` with `impratio=100` is correct for the
    parkour quadruped and catastrophic here: against this model's iterations=5 it
    launched the pelvis to z = -4839 m. The vendored defaults must win."""
    model, _, _ = built
    assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert model.opt.impratio == pytest.approx(1.0)


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
