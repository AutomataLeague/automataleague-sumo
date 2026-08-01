"""Robot abstraction for the Automata League.

A ``RobotSpec`` is everything a task needs to know about a robot without knowing
its internals: where its model lives, how it stands, the order of its actuated
joints, and how big an action step is. Tasks are written against this contract,
so a new robot is a new ``RobotSpec`` and nothing else.

Observation and action dimensionality are *derived* from the joint count, never
hardcoded, so different robots naturally produce differently sized policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

# MuJoCo geom groups used for collision proxies in Menagerie models: 3 is the
# convention for collision capsules, 4 for the box feet in the MJX variants.
_COLLISION_GROUPS = (3, 4)


@dataclass
class ExtraCollider:
    """A collision primitive to graft onto a body that upstream left bare.

    Menagerie's MJX variants strip the collision model down to what a locomotion
    task needs, which is essentially the feet. For two humanoids wrestling the
    upper body IS the contact surface, so the missing primitives are exactly where
    the sport happens: replaying a trained policy through both collision models
    showed 56% of all robot-to-robot contact was simply not happening.

    Positions are in the target body's frame, measured from its visual mesh. A
    capsule is given by ``fromto``, a sphere by ``pos``; ``size`` is the radius in
    both cases.
    """

    body: str
    size: float
    pos: tuple[float, float, float] | None = None
    fromto: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self):
        if (self.pos is None) == (self.fromto is None):
            raise ValueError(
                f"{self.body}: give exactly one of pos (sphere) or fromto (capsule)")
        if self.size <= 0:
            raise ValueError(f"{self.body}: size must be > 0, got {self.size}")


@dataclass
class RobotSpec:
    """Static description of a robot for use by any task.

    Attributes:
        name: Registry key, e.g. ``"g1"``.
        mjcf_path: Absolute path to the robot's MJCF file (with its mesh assets).
        base_body: Name of the floating-base body, used for base-frame transforms
            and fall detection. Unprefixed; the scene composer adds a prefix.
        nominal_height: Standing height of the base in metres, measured above
            whatever surface the robot stands on.
        joint_names: Actuated joints in canonical order. Defines the ordering of
            joint-position and joint-velocity observations and of the action vector.
        actuator_names: Actuators in the same order as ``joint_names``.
        home_joint_qpos: Default-stance angle for each actuated joint in radians,
            same order as ``joint_names``. The action offsets from this pose.
        action_scale: Magnitude in radians of the position offset a unit action
            applies around ``home_joint_qpos``.
        joint_scale: Per-joint multipliers on ``action_scale``, keyed by a
            substring of the joint name. Different limbs want different amounts
            of range: legs need a small window for fine balance control, arms need
            a large one to reach at all. A single number cannot serve both, and
            the G1 is the case in point — its home pose has the elbows bent 73
            degrees, so at a uniform 0.5 rad the arm can never get within 45
            degrees of straight and simply hangs.
        foot_geoms: Collision geoms that are legitimately allowed to touch the
            ground. Everything else touching the ground means the robot is down.
        team_colour_meshes: Visual meshes that carry the team colour, so the two
            sides can be told apart on sight. Unprefixed names; the scene composer
            adds the prefix. Meshes rather than bodies, because a body can carry
            several: the G1 hangs its head and its chest logo off the same
            ``torso_link`` body, so body-level selection cannot paint the chest
            without also painting the head. Declared here because mesh names are a
            property of the robot, not of the task. Empty means the whole robot is
            tinted instead.
    """

    name: str
    mjcf_path: str
    base_body: str
    nominal_height: float
    joint_names: list[str]
    actuator_names: list[str]
    home_joint_qpos: np.ndarray
    action_scale: float = 0.5
    joint_scale: dict[str, float] = field(default_factory=dict)
    foot_geoms: list[str] = field(default_factory=list)
    team_colour_meshes: list[str] = field(default_factory=list)
    extra_colliders: list[ExtraCollider] = field(default_factory=list)

    def __post_init__(self):
        self.home_joint_qpos = np.asarray(self.home_joint_qpos, dtype=np.float32)
        if len(self.home_joint_qpos) != len(self.joint_names):
            raise ValueError(
                f"{self.name}: home_joint_qpos has {len(self.home_joint_qpos)} entries "
                f"but there are {len(self.joint_names)} joints"
            )
        if len(self.actuator_names) != len(self.joint_names):
            raise ValueError(
                f"{self.name}: {len(self.actuator_names)} actuators vs "
                f"{len(self.joint_names)} joints — must match 1:1"
            )
        for key, multiplier in self.joint_scale.items():
            if multiplier <= 0:
                raise ValueError(
                    f"{self.name}: joint_scale[{key!r}] = {multiplier}, must be > 0")
            if not any(key in name for name in self.joint_names):
                raise ValueError(
                    f"{self.name}: joint_scale key {key!r} matches no joint, so it "
                    f"would silently do nothing. Joints: {self.joint_names}")

    def scale_vector(self, action_scale: float | None = None) -> np.ndarray:
        """Per-joint action scale in radians, in ``joint_names`` order.

        The offset a unit action applies to joint i is ``scale_vector()[i]``, so
        this is the whole action-to-pose mapping in one array. ``action_scale``
        overrides the robot's own base value (that is how ``SumoConfig`` retunes a
        run) while keeping the per-joint multipliers, since those describe the
        robot's proportions rather than a training choice.
        """
        base = self.action_scale if action_scale is None else float(action_scale)
        out = np.full(self.n_joints, base, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            for key, multiplier in self.joint_scale.items():
                if key in name:
                    out[i] = base * multiplier
                    break
        return out

    # --- derived dimensions -------------------------------------------------
    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def action_dim(self) -> int:
        return self.n_joints

    @property
    def proprio_dim(self) -> int:
        """Robot-only observation width.

        base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
        + joint_pos(n) + joint_vel(n) + prev_action(n).

        Task-specific blocks are added by the task, not here — see
        ``envs/sumo/observation.py::observation_dim``.
        """
        return 9 + 3 * self.n_joints

    # --- model loading ------------------------------------------------------
    def load_spec(self) -> mujoco.MjSpec:
        """Load the robot as an editable ``MjSpec``, with collision re-enabled.

        Menagerie's MJX humanoid models disable contacts globally
        (``contype=0 conaffinity=0`` on the robot's default class) and re-enable
        exactly the foot-to-floor pairs in their own ``scene_mjx.xml``. We do not
        vendor that scene, because it brings its own floor and its pair list
        names a ``floor`` geom that our arena does not have — and an explicit
        pair list cannot express robot-against-robot contact at all, which is the
        entire point of this task.

        So we re-enable ordinary automatic collision on the collision geoms here,
        programmatically, and leave the vendored XML byte-identical to upstream
        so it stays diffable against future Menagerie releases. Visual geoms
        (group 2) keep ``contype=0`` and stay non-colliding.

        Self-collision between adjacent links is handled by MuJoCo's automatic
        parent-child contact exclusion; the scene builder adds explicit
        ``exclude`` pairs if any spurious self-contact shows up.
        """
        spec = mujoco.MjSpec.from_file(self.mjcf_path)
        for geom in spec.geoms:
            if geom.group in _COLLISION_GROUPS:
                geom.contype = 1
                geom.conaffinity = 1
                geom.condim = 3
        self._add_extra_colliders(spec)
        return spec

    def _add_extra_colliders(self, spec: mujoco.MjSpec) -> None:
        """Graft on the collision primitives upstream left out.

        ``mass=0`` is belt and braces, not the mechanism. Every body in the G1
        carries an explicit ``<inertial>``, and MuJoCo ignores geom mass entirely
        when one is present — setting it to 5 kg changes the model's total mass by
        nothing, which was verified rather than assumed. It is set anyway so a
        future robot WITHOUT explicit inertials does not silently gain weight from
        a collider that exists only to make contact happen.
        """
        if not self.extra_colliders:
            return
        bodies = {body.name: body for body in spec.bodies}
        for extra in self.extra_colliders:
            body = bodies.get(extra.body)
            if body is None:
                raise ValueError(
                    f"{self.name}: extra collider names body {extra.body!r}, which "
                    f"does not exist, so it would silently add nothing. "
                    f"Bodies: {sorted(bodies)}")
            geom = body.add_geom()
            geom.group = _COLLISION_GROUPS[0]
            geom.contype, geom.conaffinity, geom.condim = 1, 1, 3
            geom.mass = 0.0
            if extra.fromto is not None:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                geom.fromto = list(extra.fromto)
                geom.size = [extra.size, 0.0, 0.0]
            else:
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                geom.pos = list(extra.pos)
                geom.size = [extra.size, 0.0, 0.0]
