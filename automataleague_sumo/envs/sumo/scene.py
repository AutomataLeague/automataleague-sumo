"""Programmatic sumo scene: a raised circular dohyo with two robots attached.

Built as an ``MjSpec`` rather than a static XML so that the ring radius used by
the scene and the ring radius used by the out-of-bounds test cannot drift apart —
they are the same ``SumoConfig`` field.

Both robots are grafted on with ``MjSpec.attach`` under the prefixes ``a/`` and
``b/``. ``SumoSceneInfo`` resolves every index a task or renderer needs, so no
downstream code ever parses a prefixed name.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from automataleague_sumo.envs.sumo.arena import SpawnPose, spawn_poses, yaw_quat
from automataleague_sumo.envs.sumo.config import SumoConfig
from automataleague_sumo.robots import RobotSpec, get_robot

# Restrained palette: clay ring, single bright rim band, cool dark surround.
_CLAY = [0.72, 0.55, 0.36, 1.0]
_BAND = [0.96, 0.95, 0.92, 1.0]
_PAINT_HALF_Z = 0.001
_BAND_WIDTH = 0.08      # painted rim band, purely visual


@dataclass
class SideInfo:
    """Resolved handles for one side of the duel."""

    robot: RobotSpec
    prefix: str
    spawn: SpawnPose
    base_body_id: int
    actuator_ids: np.ndarray     # (n_joints,) in robot.joint_names order
    joint_qposadr: np.ndarray    # (n_joints,) qpos address of each actuated joint
    joint_dofadr: np.ndarray     # (n_joints,) dof address of each actuated joint
    base_qposadr: int            # qpos address of the free joint (base pose)
    base_dofadr: int             # dof address of the free joint (base velocity)
    geom_ids: np.ndarray         # every geom belonging to this robot
    foot_geom_ids: np.ndarray    # the subset allowed to touch the ground


@dataclass
class SumoSceneInfo:
    a: SideInfo
    b: SideInfo
    cfg: SumoConfig
    home_qpos: np.ndarray        # combined reset pose for both robots
    platform_geom_id: int

    @property
    def sides(self) -> tuple[SideInfo, SideInfo]:
        return (self.a, self.b)


def _add_floor(spec: mujoco.MjSpec, cfg: SumoConfig) -> None:
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.width = 300
    tex.height = 300
    tex.nchannel = 3
    tex.rgb1 = [0.18, 0.18, 0.20]
    tex.rgb2 = [0.24, 0.24, 0.26]

    mat = spec.add_material()
    mat.name = "grid"
    mat.texrepeat = [4, 4]
    mat.reflectance = 0.05
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"

    extent = cfg.ring_radius * 4.0
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[extent, extent, 0.1], pos=[0.0, 0.0, 0.0], material="grid",
    )


def _add_dohyo(spec: mujoco.MjSpec, cfg: SumoConfig) -> None:
    """The raised platform plus a painted rim band.

    The band is two stacked flat decal cylinders: a full-radius bright disc with a
    slightly smaller clay disc on top of it, leaving a bright annulus at the rim.
    Decals are contact free (``contype=0``), so they are purely visual.

    All three geoms share an explicit low-specular, low-shininess material. The
    clay dohyo is a matte surface; it has nothing to gain from a specular
    highlight, and MuJoCo's material defaults (specular=0.5, shininess=0.5) turn
    the overhead key light into a glossy hotspot that reads as a blown-out disc
    from the near-vertical top camera. Geom ``rgba`` still wins for color; the
    material only supplies the low-gloss shading parameters.
    """
    h = cfg.platform_height
    mat = spec.add_material()
    mat.name = "clay"
    mat.specular = 0.02
    mat.shininess = 0.02
    mat.reflectance = 0.0
    spec.worldbody.add_geom(
        name="dohyo", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.ring_radius, h / 2.0], pos=[0.0, 0.0, h / 2.0],
        rgba=_CLAY, material="clay",
    )
    spec.worldbody.add_geom(
        name="ring_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.ring_radius, _PAINT_HALF_Z], pos=[0.0, 0.0, h + _PAINT_HALF_Z],
        rgba=_BAND, material="clay", contype=0, conaffinity=0,
    )
    spec.worldbody.add_geom(
        name="ring_inner", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.ring_radius - _BAND_WIDTH, _PAINT_HALF_Z],
        pos=[0.0, 0.0, h + 3 * _PAINT_HALF_Z], rgba=_CLAY, material="clay",
        contype=0, conaffinity=0,
    )


def _add_lights(spec: mujoco.MjSpec, cfg: SumoConfig) -> None:
    """Key + fill lights, both off the vertical axis and both non-specular.

    The key light used to sit directly overhead (``pos=[0,0,4], dir=[0,0,-1]``),
    which put its specular lobe squarely on the top camera's line of sight
    (``elevation=-89``, i.e. looking almost straight up the light's boresight)
    and blew the platform out to near-white in that view. Moving it off-axis,
    and zeroing specular on both lights (the clay/floor materials are matte and
    don't need a highlight), fixes that without dimming the corner/side views,
    which are lit primarily by diffuse, not specular.
    """
    spec.worldbody.add_light(
        pos=[1.4 * cfg.ring_radius, -1.8 * cfg.ring_radius, 3.5],
        dir=[-0.55, 0.7, -1.35],
        diffuse=[0.8, 0.8, 0.8], specular=[0.0, 0.0, 0.0], castshadow=True,
    )
    spec.worldbody.add_light(
        pos=[0.0, 2.5 * cfg.ring_radius, 3.0], dir=[0, -0.6, -1],
        diffuse=[0.35, 0.35, 0.35], specular=[0.0, 0.0, 0.0], castshadow=False,
    )


def build_sumo_model(
    robot_a: str | RobotSpec = "g1",
    robot_b: str | RobotSpec | None = None,
    cfg: SumoConfig | None = None,
) -> tuple[mujoco.MjModel, SumoSceneInfo]:
    """Compose arena + two robots into a compiled ``MjModel`` and a ``SumoSceneInfo``.

    ``robot_b`` defaults to ``robot_a``, which is the symmetric self-play case.
    Passing two different robots builds a cross-robot matchup for evaluation.
    """
    cfg = cfg or SumoConfig()
    spec_a = robot_a if isinstance(robot_a, RobotSpec) else get_robot(robot_a)
    if robot_b is None:
        spec_b = spec_a
    else:
        spec_b = robot_b if isinstance(robot_b, RobotSpec) else get_robot(robot_b)

    spec = mujoco.MjSpec()
    spec.modelname = f"sumo_{spec_a.name}_vs_{spec_b.name}"
    # Mirror the vendored g1_mjx.xml <option> block. MjSpec.attach does not carry
    # the child's solver settings up to the parent spec, so without this the
    # compiled model silently falls back to MuJoCo's defaults (timestep 0.002),
    # which breaks SumoConfig's frame_skip=5 => 50 Hz assumption.
    #
    # Do NOT set `cone = ELLIPTIC` or `impratio = 100` here. Those are the right
    # choices for a quadruped with small round feet (they are what the parkour
    # repo uses) and they are catastrophic for this model: combined with the MJX
    # model's `iterations=5`, contact forces fail to converge and launch the
    # robot. Measured over 1250 steps of a passive stance, elliptic+impratio=100
    # sent the pelvis to z = -4839 m with 312 airborne steps, while the vendored
    # settings below stay airborne for 0 steps and settle smoothly. The vendored
    # <option> block is authoritative; leave it alone.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.004
    spec.option.iterations = 5
    spec.option.ls_iterations = 8
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

    _add_floor(spec, cfg)
    _add_dohyo(spec, cfg)
    _add_lights(spec, cfg)

    pose_a, pose_b = spawn_poses(cfg)
    placed = []
    for robot, prefix, pose in ((spec_a, "a/", pose_a), (spec_b, "b/", pose_b)):
        frame = spec.worldbody.add_frame(
            pos=[pose.x, pose.y, cfg.platform_height])
        spec.attach(robot.load_spec(), prefix=prefix, frame=frame)
        placed.append((robot, prefix, pose))

    model, sides, home = _compile_and_place(spec, placed, cfg)

    # MuJoCo's automatic contact exclusion only covers directly-connected
    # parent/child body pairs. Non-adjacent links that pass close to each
    # other at the home stance (e.g. pelvis vs. a hip-roll link two joints
    # down, or an elbow vs. a wrist link two joints down) are not covered by
    # that rule and can show up as spurious self-contact. Detect any such
    # same-robot body pairs in contact at the assembled home pose and exclude
    # them explicitly, then recompile once so the excludes take effect.
    self_pairs = _find_self_collisions(model, sides, home)
    if self_pairs:
        for name1, name2 in self_pairs:
            exclude = spec.add_exclude()
            exclude.bodyname1 = name1
            exclude.bodyname2 = name2
        model, sides, home = _compile_and_place(spec, placed, cfg)

    return model, SumoSceneInfo(
        a=sides[0], b=sides[1], cfg=cfg, home_qpos=home,
        platform_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dohyo"),
    )


def _compile_and_place(
    spec: mujoco.MjSpec,
    placed: list[tuple[RobotSpec, str, SpawnPose]],
    cfg: SumoConfig,
) -> tuple[mujoco.MjModel, list[SideInfo], np.ndarray]:
    """Compile ``spec`` and build the home pose for the already-attached robots."""
    model = spec.compile()
    home = mujoco.MjData(model).qpos.copy().astype(np.float32)
    sides = []
    for robot, prefix, pose in placed:
        side = _resolve_side(model, robot, prefix, pose)
        home[side.base_qposadr:side.base_qposadr + 3] = [
            pose.x, pose.y, cfg.platform_height + robot.nominal_height]
        home[side.base_qposadr + 3:side.base_qposadr + 7] = yaw_quat(pose.yaw)
        home[side.joint_qposadr] = robot.home_joint_qpos
        sides.append(side)
    return model, sides, home


def _find_self_collisions(
    model: mujoco.MjModel, sides: list[SideInfo], home_qpos: np.ndarray,
) -> set[tuple[str, str]]:
    """Body name pairs, within a single side, that touch at the home pose."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)

    pairs: set[tuple[str, str]] = set()
    for side in sides:
        own_geoms = set(side.geom_ids.tolist())
        for c in data.contact[:data.ncon]:
            if c.geom1 not in own_geoms or c.geom2 not in own_geoms:
                continue
            body1, body2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if body1 == body2:
                continue
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body1))
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body2))
            pairs.add(tuple(sorted((name1, name2))))
    return pairs


def _resolve_side(
    model: mujoco.MjModel, robot: RobotSpec, prefix: str, pose: SpawnPose,
) -> SideInfo:
    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)

    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, prefix + robot.base_body)
    base_jnt = int(model.body_jntadr[base_body_id])

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
         for n in robot.joint_names], dtype=np.int64)
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + a)
         for a in robot.actuator_names], dtype=np.int64)

    # Every geom whose kinematic-tree root is this robot's base body belongs to it.
    # That is how a contact is attributed to a side without parsing geom names.
    geom_ids = np.flatnonzero(
        model.body_rootid[model.geom_bodyid] == base_body_id).astype(np.int64)

    return SideInfo(
        robot=robot,
        prefix=prefix,
        spawn=pose,
        base_body_id=base_body_id,
        actuator_ids=actuator_ids,
        joint_qposadr=model.jnt_qposadr[joint_ids].astype(np.int64),
        joint_dofadr=model.jnt_dofadr[joint_ids].astype(np.int64),
        base_qposadr=int(model.jnt_qposadr[base_jnt]),
        base_dofadr=int(model.jnt_dofadr[base_jnt]),
        geom_ids=geom_ids,
        foot_geom_ids=np.array([gid(n) for n in robot.foot_geoms], dtype=np.int64),
    )


if __name__ == "__main__":
    m, i = build_sumo_model("g1")
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom}")
    print(f"side a: base_qposadr={i.a.base_qposadr} geoms={len(i.a.geom_ids)}")
    print(f"side b: base_qposadr={i.b.base_qposadr} geoms={len(i.b.geom_ids)}")
