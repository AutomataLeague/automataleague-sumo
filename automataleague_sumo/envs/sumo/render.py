"""Camera presets for rendering a duel.

Distances are expressed as a multiple of the ring radius, so a preset frames the
arena correctly whatever ``ring_radius`` is configured.
"""

from __future__ import annotations

import mujoco
import numpy as np

from automataleague_sumo.envs.sumo.scene import SumoSceneInfo

CAMERAS: dict[str, dict] = {
    # Broadside, level with the action: the clearest read on who is pushing whom.
    "side": dict(azimuth=90.0, elevation=-12.0, distance_frac=3.2, lookat_z=0.7),
    # Three-quarter view: shows both the ring edge and the robots' facing.
    "corner": dict(azimuth=135.0, elevation=-22.0, distance_frac=3.0, lookat_z=0.6),
    # Straight down: the honest view of the out-of-ring condition.
    "top": dict(azimuth=90.0, elevation=-89.0, distance_frac=2.6, lookat_z=0.3),
}


def camera_for(name: str, info: SumoSceneInfo) -> mujoco.MjvCamera:
    if name not in CAMERAS:
        raise ValueError(f"Unknown camera '{name}'. Available: {sorted(CAMERAS)}")
    preset = CAMERAS[name]
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.0, 0.0, preset["lookat_z"]])
    cam.azimuth = preset["azimuth"]
    cam.elevation = preset["elevation"]
    cam.distance = preset["distance_frac"] * info.cfg.ring_radius
    return cam


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: SumoSceneInfo,
    camera: str = "corner",
    size: tuple[int, int] = (720, 1280),
    renderer: mujoco.Renderer | None = None,
) -> np.ndarray:
    """Render one RGB frame of shape ``(height, width, 3)``.

    Pass ``renderer`` to reuse an allocated one across frames; creating a
    ``mujoco.Renderer`` per frame is slow enough to dominate video export.
    """
    own = renderer is None
    if own:
        h, w = size
        renderer = mujoco.Renderer(model, height=h, width=w)
    try:
        renderer.update_scene(data, camera=camera_for(camera, info))
        return renderer.render()
    finally:
        if own:
            renderer.close()
