"""The two robots must be told apart on sight.

They are the same model attached twice, so without this a video shows two
identical silver humanoids and there is no way to say which one was pushed out.

Only the meshes a robot declares in ``RobotSpec.team_colour_meshes`` are painted.
Colouring the whole robot turns it into a solid block and loses the light and
dark shading that shows which way a limb is pointing.

Selection is by mesh, not by body: the G1 hangs its head and its chest logo off
the same ``torso_link`` body, so body-level selection cannot paint the chest
without also painting the head.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import automataleague_sumo.envs.sumo.scene as scene
from automataleague_sumo.robots import get_robot


def _model(meshes=None, tint=None):
    """Build a model, optionally overriding the robot's team meshes and strength."""
    robot = get_robot("g1")
    if meshes is not None:
        robot.team_colour_meshes = list(meshes)
    original = scene._TINT
    try:
        if tint is not None:
            scene._TINT = tint
        return scene.build_sumo_model(robot)
    finally:
        scene._TINT = original


def _materials(model):
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i): model.mat_rgba[i][:3]
        for i in range(model.nmat)
    }


def _painted_meshes(model, scene_info, side):
    """Mesh names whose visual geoms wear that side's team material."""
    team = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"{side}/team")
    if team < 0:
        return set()
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g]))
        for g in scene_info.a.geom_ids if model.geom_matid[g] == team
    }


@pytest.fixture(scope="module")
def built():
    return _model()


def test_the_two_teams_are_different_colours(built):
    mats = _materials(built[0])
    assert not np.allclose(mats["a/team"], mats["b/team"], atol=1e-3)


def test_side_a_is_blue_and_side_b_is_red(built):
    """Pins WHICH team gets which colour, not merely that they differ. Without
    this the two could swap between builds and every video would be mislabelled."""
    mats = _materials(built[0])
    assert mats["a/team"][2] > mats["a/team"][0], f"a is not blue: {mats['a/team']}"
    assert mats["b/team"][0] > mats["b/team"][2], f"b is not red: {mats['b/team']}"


def test_only_the_declared_meshes_are_painted():
    """The point of painting parts rather than the whole robot. Checked against
    the exact declared list, so painting one mesh too many or too few fails."""
    wanted = ["torso_link", "pelvis"]
    model, info = _model(meshes=wanted)
    assert _painted_meshes(model, info, "a") == {f"a/{n}" for n in wanted}


def test_the_head_is_not_painted():
    """What mesh-level selection is FOR. The G1's head and chest logo hang off the
    same torso_link body, so a body-level implementation paints all three and this
    is the only assertion that notices."""
    model, info = _model()
    painted = _painted_meshes(model, info, "a")
    assert "a/torso_link" in painted, "the chest is not painted at all"
    assert "a/head_link" not in painted, "the head got painted with the chest"
    assert "a/logo_link" not in painted, "the chest logo got painted over"


def test_a_mesh_name_that_matches_nothing_is_an_error():
    """A typo would otherwise leave the robot in its default colour, which is
    exactly the indistinguishable-robots failure this module exists to prevent —
    and it would be silent."""
    with pytest.raises(ValueError, match="matched a visual mesh"):
        _model(meshes=["chest_plate"])


def test_the_limbs_keep_their_original_colour():
    """The limbs carry no team colour, so their materials must be untouched and
    identical between the two sides. If they differed, the paint would be leaking."""
    model, _ = _model()
    mats = _materials(model)
    for suffix in ("silver", "black"):
        assert np.allclose(mats[f"a/{suffix}"], mats[f"b/{suffix}"], atol=1e-6), (
            f"{suffix} differs between the sides — the paint leaked past the "
            f"declared bodies")


def test_the_paint_shades_like_the_rest_of_the_robot(built):
    """The painted chest inherits the original material's specular and shininess,
    so it catches the light like the surrounding parts instead of reading as a
    flat sticker. At full strength the colour is pure, so this is the only thing
    keeping the panel looking like part of the machine."""
    model = built[0]
    team = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "a/team")
    silver = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "a/silver")
    assert model.mat_specular[team] == pytest.approx(model.mat_specular[silver])
    assert model.mat_shininess[team] == pytest.approx(model.mat_shininess[silver])


def test_the_tint_strength_actually_changes_the_colour():
    """Guards a `strength` argument that is accepted and ignored."""
    weak = _materials(_model(tint=0.3)[0])["a/team"]
    strong = _materials(_model(tint=1.0)[0])["a/team"]
    assert not np.allclose(weak, strong, atol=1e-3)
    # At full strength the paint is the pure team colour, which is the shipped look.
    assert np.allclose(strong, scene._TEAM_A, atol=1e-6)


def test_the_arena_keeps_its_own_colours(built):
    """Painting is matched by body name, so a sloppy match would recolour the
    dohyo and the floor. Compared against an untinted build, because asserting
    the materials merely still EXIST does not catch a repaint."""
    tinted = _materials(built[0])
    plain = _materials(_model(tint=0.0)[0])
    for name in ("clay", "grid"):
        assert np.allclose(tinted[name], plain[name], atol=1e-6), (
            f"arena material {name} was repainted")


def test_a_zero_tint_paints_nothing():
    """The control. Without it every assertion above could be describing the
    upstream model's own colours rather than anything this code did.

    Also why `_tint_teams` reads the module constant at call time instead of
    taking it as a default argument: a default binds at definition, so setting it
    from here would silently do nothing and this test would pass against a fully
    painted model.
    """
    model, info = _model(tint=0.0)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "a/team") < 0
    assert _painted_meshes(model, info, "a") == set()


def test_a_robot_naming_no_parts_still_gets_a_colour():
    """A newly added robot must be distinguishable before anyone has picked out
    its meshes, so an empty list falls back to tinting the whole robot."""
    mats = _materials(_model(meshes=[])[0])
    assert not np.allclose(mats["a/silver"], mats["b/silver"], atol=1e-3)
