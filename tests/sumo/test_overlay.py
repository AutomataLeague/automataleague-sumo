"""Captions must name the side that actually won.

Getting this backwards is the worst kind of bug this repo produces: the video
looks completely normal and asserts the opposite of the truth, and anyone
watching believes it over the numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from automataleague_sumo.envs.sumo.overlay import draw_hud, draw_verdict, verdict
from automataleague_sumo.envs.sumo.scene import TEAM_A_RGB, TEAM_B_RGB
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW


def _rgb255(c):
    return tuple(int(round(255 * v)) for v in c)


def test_side_a_is_blue_and_side_b_is_red():
    """The mapping itself, stated once so a swap fails loudly here rather than
    silently in a published video."""
    assert verdict(A_WINS) == ("BLUE WINS", _rgb255(TEAM_A_RGB))
    assert verdict(B_WINS) == ("RED WINS", _rgb255(TEAM_B_RGB))


def test_the_caption_colour_is_the_colour_on_that_robots_chest():
    """Read from `scene`, not restated. A retuned team tint must move the caption
    with it, or the word and the robot drift apart."""
    _, blue = verdict(A_WINS)
    _, red = verdict(B_WINS)
    assert blue == _rgb255(TEAM_A_RGB) and red == _rgb255(TEAM_B_RGB)
    assert blue != red, "the two teams must not caption the same colour"


def test_a_draw_is_neither_team_colour():
    text, colour = verdict(DRAW)
    assert text == "DRAW"
    assert colour not in (_rgb255(TEAM_A_RGB), _rgb255(TEAM_B_RGB))


def test_an_unfinished_duel_does_not_claim_a_winner():
    """Outcome 0 means ongoing. Captioning that as a win would invent a result."""
    text, _ = verdict(0)
    assert "WINS" not in text


@pytest.mark.parametrize("fn,args", [(draw_hud, ("round 1 of 10", "blue 0  red 0")),
                                     (draw_verdict, (A_WINS, "71 steps"))])
def test_drawing_preserves_the_frame_shape_and_dtype(fn, args):
    """The result is fed straight to the encoder; a changed shape or dtype turns
    into a corrupt video rather than an exception."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = fn(frame, *args)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_the_verdict_actually_marks_the_frame():
    """Without this the whole module could be a no-op and every test above would
    still pass, since they only check the text/colour lookup."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert draw_verdict(frame, A_WINS).any(), "verdict drew nothing at all"
