import mujoco
import numpy as np
import pytest

from automataleague_sumo.envs.sumo.render import CAMERAS, camera_for, render_frame
from automataleague_sumo.envs.sumo.scene import build_sumo_model


@pytest.fixture(scope="module")
def built():
    model, info = build_sumo_model("g1")
    data = mujoco.MjData(model)
    data.qpos[:] = info.home_qpos
    mujoco.mj_forward(model, data)
    return model, data, info


def test_camera_presets_exist(built):
    _, _, info = built
    assert {"side", "corner", "top"} <= set(CAMERAS)
    for name in CAMERAS:
        cam = camera_for(name, info)
        assert cam.distance > 0


def test_unknown_camera_raises(built):
    _, _, info = built
    with pytest.raises(ValueError, match="Unknown camera"):
        camera_for("nope", info)


def test_camera_distance_scales_with_the_ring(built):
    from automataleague_sumo.envs.sumo.config import SumoConfig

    _, small = build_sumo_model("g1", cfg=SumoConfig(ring_radius=1.5))
    _, big = build_sumo_model("g1", cfg=SumoConfig(ring_radius=3.0, pos_noise=0.05))
    assert camera_for("corner", big).distance > camera_for("corner", small).distance


@pytest.mark.parametrize("camera", ["side", "corner", "top"])
def test_render_returns_an_image_of_the_requested_size(built, camera):
    model, data, info = built
    frame = render_frame(model, data, info, camera=camera, size=(240, 320))
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_render_is_not_a_blank_frame(built):
    model, data, info = built
    frame = render_frame(model, data, info, camera="corner", size=(240, 320))
    assert frame.std() > 5.0, "frame looks blank — lighting or camera is wrong"
