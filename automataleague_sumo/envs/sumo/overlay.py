"""Captions burned into rendered frames, so a duel reads without a commentary track.

Two robots in a ring look symmetrical. Watching ten duels back to back, it is
genuinely hard to tell who just lost, because the losing robot leaves the frame
in the same instant the next round starts. The `outcome` code is printed to the
terminal, but by then the video has moved on.

The team colours come from ``scene`` rather than being restated here, so the word
"BLUE" is drawn in exactly the colour on that robot's chest.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from automataleague_sumo.envs.sumo.scene import TEAM_A_RGB, TEAM_B_RGB
from automataleague_sumo.envs.sumo.termination import A_WINS, B_WINS, DRAW

_INK = (245, 243, 238)
_SHADE = (12, 12, 14)
_NEUTRAL = (170, 165, 155)

# Tried in order; any missing font just falls through to Pillow's bitmap default,
# which is ugly but never raises. A renderer that crashes on a headless box for
# want of a typeface would be a poor trade.
_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
)


def _rgb255(colour: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(255 * c)) for c in colour)


def _font(size: int):
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def verdict(outcome: int) -> tuple[str, tuple[int, int, int]]:
    """Winner text and colour for an outcome code."""
    if outcome == A_WINS:
        return "BLUE WINS", _rgb255(TEAM_A_RGB)
    if outcome == B_WINS:
        return "RED WINS", _rgb255(TEAM_B_RGB)
    if outcome == DRAW:
        return "DRAW", _NEUTRAL
    return "UNDECIDED", _NEUTRAL


def draw_hud(frame: np.ndarray, left: str, right: str = "") -> np.ndarray:
    """A thin status line across the top: round on the left, score on the right."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    size = max(14, im.height // 34)
    font = _font(size)
    pad = size // 2
    d.rectangle([0, 0, im.width, size + 2 * pad], fill=(*_SHADE, 190))
    d.text((pad + 2, pad), left, font=font, fill=_INK)
    if right:
        w = d.textlength(right, font=font)
        d.text((im.width - w - pad - 2, pad), right, font=font, fill=_NEUTRAL)
    return np.asarray(im)


def draw_verdict(frame: np.ndarray, outcome: int, subtitle: str = "") -> np.ndarray:
    """A centred card naming the winner, for the held frames after a duel ends."""
    text, colour = verdict(outcome)
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")

    # Dim the whole frame so the card reads over a bright dohyo as well as over
    # the dark floor; the pose underneath stays visible, which is the point.
    d.rectangle([0, 0, im.width, im.height], fill=(*_SHADE, 110))

    big = _font(max(26, im.height // 11))
    small = _font(max(14, im.height // 30))
    tw = d.textlength(text, font=big)
    # Low in the frame, over the empty half of the dohyo. Centred vertically it
    # lands squarely on the two robots and hides the finishing pose, which is the
    # one thing a viewer is trying to see.
    x, y = (im.width - tw) / 2, im.height * 0.60

    bar_h = max(4, im.height // 150)
    d.rectangle([x, y - bar_h * 3, x + tw, y - bar_h * 2], fill=colour)
    d.text((x, y), text, font=big, fill=colour)
    if subtitle:
        sw = d.textlength(subtitle, font=small)
        d.text(((im.width - sw) / 2, y + big.size * 1.35), subtitle,
               font=small, fill=_INK)
    return np.asarray(im)
