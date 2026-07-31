"""What does `action_scale` actually buy the robot? Measure it, do not guess it.

    python tools/measure_reach.py
    python tools/measure_reach.py --scales 0.2 0.4 0.6 0.8 --samples 8000

`action_scale` is the maximum radians any joint can be commanded away from the
home stance: ``q_target = home + action_scale * action`` with action in [-1, 1].
It is not a gain and not a speed limit. It is a hard geometric cap on the poses
the policy is able to ask for, and therefore on the stance width, stride and
crouch depth available to it.

Reported as pure kinematics, on purpose. The commanded pose is exactly
``clip(home + scale * action, joint_range)``, so the envelope below is a
statement about the ACTION SPACE rather than about any particular controller or
any particular amount of training. Whether the robot can hold a pose under load
is a separate question about actuator strength; whether it can ever request the
pose at all is this one.

Read the table against two things:

* the down-rule. A base that drops past 45% of nominal height loses the duel, so
  past some scale the action space CONTAINS poses that lose immediately. That is
  not automatically wrong, but it is worth knowing before wondering why a policy
  keeps ending episodes.
* where the curve flattens. If the leg joints hit their mechanical limits, more
  scale buys nothing and is pure downside: the same reachable poses, reached by
  more violent commands. On the G1 over 0.2 to 1.0 it never flattens, so the
  choice is a trade against controllability rather than against geometry.

stance and stride are the WIDEST reachable foot spreads, which are full-splits
poses. A usable walking step is a fraction of the stride column; treat it as an
upper bound on gait, not as a step length.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from automataleague_sumo.envs.sumo.scene import build_sumo_model
from automataleague_sumo.robots import get_robot

# Joint-name fragments that matter for each measurement. Restricting the search
# to the relevant joints is what makes a random search over a 29-dimensional
# action space find the real extreme rather than a mediocre interior point.
_LEG = ("hip", "knee", "ankle")
_UPPER = ("waist", "shoulder", "elbow", "wrist")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="g1")
    p.add_argument("--scales", type=float, nargs="*",
                   default=[0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    p.add_argument("--samples", type=int, default=6000,
                   help="random action samples per measurement per scale")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class Envelope:
    """Kinematic reach of one robot's action space, with its base held fixed.

    The base is pinned rather than stood on the floor so the measurement is of
    the limbs alone. A robot that cannot lift a foot because it is standing on it
    tells you about gravity, not about the action space.
    """

    def __init__(self, robot_name: str):
        self.robot = get_robot(robot_name)
        model, scene = build_sumo_model(robot_name)
        self.model, self.side = model, scene.a
        self.data = mujoco.MjData(model)

        self.n = self.robot.n_joints
        self.home = np.asarray(self.robot.home_joint_qpos, dtype=np.float64)
        self.qadr = self.side.joint_qposadr
        jnt = model.jnt_qposadr.tolist()
        self.jrange = np.array([
            model.jnt_range[jnt.index(a)] for a in self.qadr], dtype=np.float64)
        self.limited = np.array([
            bool(model.jnt_limited[jnt.index(a)]) for a in self.qadr])

        gid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i
               for i in range(model.ngeom)}
        self.left_foot = gid[f"a/{self.robot.foot_geoms[0]}"]
        self.right_foot = gid[f"a/{self.robot.foot_geoms[4]}"]
        self.pelvis = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"a/{self.robot.base_body}")
        self.hands = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"a/{side}_wrist_yaw_link")
            for side in ("left", "right")]

        self.mask = {
            "leg": np.array([any(k in n for k in _LEG) for n in self.robot.joint_names]),
            "upper": np.array([any(k in n for k in _UPPER)
                               for n in self.robot.joint_names]),
        }
        self.home_geometry = self._geometry(np.zeros(self.n), scale=0.0)

    def _geometry(self, action: np.ndarray, scale: float) -> dict:
        """Pose the robot at the commanded joint targets and read its geometry.

        Uses the robot's per-joint scale vector, not the bare scalar, so this
        measures the mapping the env actually applies. A scalar here reported an
        arm reach that did not move when the arm multiplier was introduced, which
        is a tool quietly measuring a different robot from the one being trained.
        """
        target = self.home + self.robot.scale_vector(scale) * action
        target = np.where(self.limited,
                          np.clip(target, self.jrange[:, 0], self.jrange[:, 1]), target)
        self.data.qpos[:] = 0.0
        self.data.qpos[self.side.base_qposadr + 3] = 1.0        # identity quaternion
        self.data.qpos[self.side.base_qposadr + 2] = 1.0        # hold the base at 1 m
        self.data.qpos[self.qadr] = target
        mujoco.mj_kinematics(self.model, self.data)

        lf = self.data.geom_xpos[self.left_foot]
        rf = self.data.geom_xpos[self.right_foot]
        base = self.data.xpos[self.pelvis]
        return {
            "stance": abs(lf[1] - rf[1]),                       # lateral foot spread
            "stride": abs(lf[0] - rf[0]),                       # fore/aft foot spread
            # How much shorter the leg gets. With the feet planted instead, this
            # is exactly how far the base can drop.
            "crouch": base[2] - max(lf[2], rf[2]),
            "reach": max(float(np.linalg.norm(self.data.xpos[h][:2] - base[:2]))
                         for h in self.hands),
            "saturated": float(np.mean(
                np.isclose(target, self.home + scale * action, atol=1e-9) == 0)),
        }

    def measure(self, scale: float, samples: int, rng) -> dict:
        """Best value of each metric reachable at this scale."""
        out = {}
        for metric, group in (("stance", "leg"), ("stride", "leg"),
                              ("crouch", "leg"), ("reach", "upper")):
            mask = self.mask[group]
            acts = rng.uniform(-1.0, 1.0, size=(samples, self.n)) * mask
            # The corners of the involved subspace, where an envelope extreme
            # almost always sits, plus the random interior samples.
            acts = np.vstack([acts, np.eye(self.n)[mask], -np.eye(self.n)[mask]])
            values = [self._geometry(a, scale)[metric] for a in acts]
            out[metric] = (min(values) if metric == "crouch" else max(values))
        # Fraction of joint commands the mechanical limits clipped away.
        probe = rng.uniform(-1.0, 1.0, size=(samples, self.n))
        target = self.home + self.robot.scale_vector(scale) * probe
        clipped = np.where(self.limited,
                           np.clip(target, self.jrange[:, 0], self.jrange[:, 1]), target)
        out["clipped"] = float(np.mean(~np.isclose(target, clipped, atol=1e-9)))
        return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = Envelope(args.robot)
    home = env.home_geometry

    print(f"{args.robot}: kinematic envelope of the action space "
          f"({args.samples} samples per metric)\n")
    print(f"  at the home stance:  stance {home['stance']:.3f} m   "
          f"stride {home['stride']:.3f} m   leg length {home['crouch']:.3f} m\n")
    print(f"{'scale':>6} {'rad':>6} | {'stance':>8} {'stride':>8} {'crouch':>8} "
          f"{'reach':>8} | {'clipped':>8}")
    print(f"{'':>6} {'(deg)':>6} | {'width m':>8} {'m':>8} {'drop m':>8} "
          f"{'m':>8} | {'by limits':>9}")
    print("  " + "-" * 68)

    rows = []
    for scale in args.scales:
        m = env.measure(scale, args.samples, rng)
        crouch_drop = home["crouch"] - m["crouch"]
        rows.append((scale, m, crouch_drop))
        print(f"{scale:6.2f} {np.degrees(scale):6.1f} | {m['stance']:8.3f} "
              f"{m['stride']:8.3f} {crouch_drop:8.3f} {m['reach']:8.3f} | "
              f"{100 * m['clipped']:7.0f}%")

    print("\nreading the table")
    print("  stance and stride are the WIDEST reachable foot spreads, which are")
    print("  full-splits poses. A usable walking step is a fraction of the stride")
    print("  column, so treat it as an upper bound on gait, not as a step length.")

    from automataleague_sumo.envs.sumo.config import TerminationConfig

    tc = TerminationConfig()
    nominal = env.robot.nominal_height
    legal_drop = nominal - tc.fall_height_frac * nominal
    print("\nthe crouch column against the down-rule")
    print(f"  the base starts at {nominal:.3f} m and loses below "
          f"{tc.fall_height_frac * nominal:.3f} m, so a drop past "
          f"{legal_drop:.3f} m loses the duel outright.")
    for scale, _, drop in rows:
        verdict = ("safe: cannot crouch into a loss" if drop < legal_drop else
                   "the action space CONTAINS poses that lose immediately")
        print(f"    scale {scale:.2f}: max drop {drop:.3f} m   {verdict}")

    print("\ndoes more scale keep buying anything?")
    flat = False
    for (s0, m0, c0), (s1, m1, c1) in zip(rows, rows[1:]):
        gains = {k: m1[k] - m0[k] for k in ("stance", "stride", "reach")}
        gains["crouch"] = c1 - c0
        best = max(abs(v) for v in gains.values())
        note = "  <- flat, joint limits bind before the action scale does" \
            if best < 0.01 else ""
        flat = flat or best < 0.01
        print(f"  {s0:.2f} -> {s1:.2f}: largest gain {best:+.3f} m"
              f"   ({100 * m1['clipped']:.0f}% of commands clipped){note}")
    if not flat:
        print("  nothing flattened over this range: the mechanical joint limits are")
        print("  NOT the binding constraint here, action_scale is. Every increase")
        print("  buys real reach, so the choice is a trade against control, not")
        print("  against the robot's geometry.")


if __name__ == "__main__":
    main()
