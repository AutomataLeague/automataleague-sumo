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
        foot_geoms: Collision geoms that are legitimately allowed to touch the
            ground. Everything else touching the ground means the robot is down.
    """

    name: str
    mjcf_path: str
    base_body: str
    nominal_height: float
    joint_names: list[str]
    actuator_names: list[str]
    home_joint_qpos: np.ndarray
    action_scale: float = 0.5
    foot_geoms: list[str] = field(default_factory=list)

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
        return spec
