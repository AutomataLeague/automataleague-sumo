"""Render preview stills of the sumo arena. Phase A design-review tool.

    MUJOCO_GL=egl uv run python tools/render_scene.py

``--settle`` defaults to 0: the render shows exactly ``home_qpos``, the pose an
episode actually resets to. A nonzero ``--settle`` instead runs that many
passive physics steps first, which is useful for checking solver stability but
is not a design preview — the G1's PD-held stance pitches forward well past a
"slight crouch" over a few hundred steps of unlearned balance, since standing
is what the standing bootstrap is for, not something the home pose guarantees on
its own.
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import mujoco

from automataleague_sumo.envs.sumo.config import SumoConfig
from automataleague_sumo.envs.sumo.render import CAMERAS, render_frame
from automataleague_sumo.envs.sumo.scene import build_sumo_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="g1")
    p.add_argument("--opponent", default=None, help="defaults to --robot")
    p.add_argument("--ring-radius", type=float, default=SumoConfig().ring_radius)
    p.add_argument(
        "--settle", type=int, default=0,
        help="physics steps to run before rendering. 0 (default) renders the "
             "exact spawn pose (home_qpos) the env resets to; a nonzero value "
             "shows passive settling instead, for checking solver stability",
    )
    p.add_argument("--out", default="renders/arena")
    args = p.parse_args()

    cfg = SumoConfig(ring_radius=args.ring_radius)
    model, info = build_sumo_model(args.robot, args.opponent, cfg=cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    for side in info.sides:
        data.ctrl[side.actuator_ids] = side.robot.home_joint_qpos
    mujoco.mj_forward(model, data)
    for _ in range(args.settle):
        mujoco.mj_step(model, data)

    os.makedirs(args.out, exist_ok=True)
    for name in CAMERAS:
        path = os.path.join(args.out, f"{name}.png")
        imageio.imwrite(path, render_frame(model, data, info, camera=name))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
