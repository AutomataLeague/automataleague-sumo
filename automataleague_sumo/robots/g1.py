"""Unitree G1 — first humanoid in the sumo league.

Model vendored from MuJoCo Menagerie (``unitree_g1``). We load ``g1_mjx.xml``,
not ``g1.xml``: the MJX variant replaces mesh colliders with capsules, spheres
and box feet, which is what makes two humanoids in sustained contact tractable
under MuJoCo-Warp. Both backends load this same file so that CPU evaluation and
GPU training cannot disagree about the physics.

29 position-controlled joints: 12 leg, 3 waist, 14 arm. The arms are actuated
and matter here — shoving is done with them.

See ``assets/unitree_g1/LICENSE`` for attribution.
"""

from __future__ import annotations

import os

import numpy as np

from automataleague_sumo.robots.base import RobotSpec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_G1_XML = os.path.join(_ROOT, "assets", "unitree_g1", "g1_mjx.xml")

# Canonical joint order == actuator order in g1_mjx.xml.
_LEG = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw"]
_JOINTS = (
    [f"left_{j}_joint" for j in _LEG]
    + [f"right_{j}_joint" for j in _LEG]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"left_{j}_joint" for j in _ARM]
    + [f"right_{j}_joint" for j in _ARM]
)

# From the "home" keyframe of upstream's own scene_mjx.xml, which is the stance
# this MJX model is built around. Do NOT use g1.xml's "stand" keyframe: that pose
# has straight legs (all leg joints zero), which is a locked-knee inverted
# pendulum with no ankle leverage. It topples in about 1.5 s under a passive
# position hold, and no gain tuning fixes that honestly.
#
# Slightly bent: hip_pitch -0.1, knee 0.3, ankle_pitch -0.2 per leg.
_LEG_HOME = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0)
_HOME_QPOS = np.array(
    list(_LEG_HOME) * 2                             # left leg, right leg
    + [0.0] * 3                                     # waist
    + [0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0]          # left arm
    + [0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0],        # right arm (roll mirrored)
    dtype=np.float32,
)

# Geoms allowed to touch the ground. Everything else touching means "down".
_FEET = [f"{side}_foot{tag}_collision"
         for side in ("left", "right") for tag in ("_box", "1", "2", "3")]


def make_g1() -> RobotSpec:
    return RobotSpec(
        name="g1",
        mjcf_path=_G1_XML,
        base_body="pelvis",
        nominal_height=0.784,    # pelvis height in the MJX "home" keyframe
        joint_names=list(_JOINTS),
        actuator_names=list(_JOINTS),   # actuators share joint names in g1_mjx.xml
        home_joint_qpos=_HOME_QPOS,
        action_scale=0.5,        # provisional; MEASURE it with tools/measure_reach.py
        foot_geoms=list(_FEET),
        # The chest only. It is the largest continuous surface and stays visible
        # from every camera angle. Deliberately NOT `head_link` or `logo_link`,
        # which hang off the same torso_link body — hence selecting by mesh.
        team_colour_meshes=["torso_link"],
    )
