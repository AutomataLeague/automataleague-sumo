"""Figures explaining the collision-model fix, for external write-ups.

    python tools/make_contact_visuals.py --out renders/visuals

Three charts, each answering one question:

  1. what the head's collider actually covered
  2. how much contact was missing
  3. what that cost, as measured progress

All numbers are measured, and each is sourced in the caption drawn on the figure
so a reader can tell a measurement from an illustration.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

# Categorical slots 1 and 2 of the validated palette: clears the colour-vision
# and normal-vision separation floors on a light surface with room to spare.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# --- measured on the vendored g1_mjx.xml, at the home pose (world z, metres) ---
HEAD_LO, HEAD_HI = 1.409, 1.615          # visual mesh extent
HEAD_HALF_W = 0.078                      # visual half-width, side to side
SPHERE_Z, SPHERE_R = 1.558, 0.060        # upstream's collision sphere

# --- champion policy replayed through both collision models, 761 steps ---
CONTACTS_OLD, CONTACTS_NEW = 1877, 4225

# --- round robin: win rate against the whole field of checkpoints ---
PHANTOM = [(20, 19.0), (140, 34.2), (260, 35.7), (400, 58.4), (500, 55.5),
           (620, 66.5), (760, 59.5), (880, 53.1), (1000, 58.1)]
CORRECTED = [(20, 11.6), (160, 23.4), (300, 44.5), (440, 50.5), (580, 54.3),
             (720, 61.2), (860, 72.0), (1000, 76.5)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def caption(fig, text):
    fig.text(0.012, 0.015, text, fontsize=9, color=MUTED, ha="left")


def head_coverage(path):
    """The head, and the part of it that could be touched."""
    fig, ax = plt.subplots(figsize=(6.4, 7.2), facecolor=SURFACE)
    style(ax)
    ax.add_patch(Rectangle((-HEAD_HALF_W, HEAD_LO), 2 * HEAD_HALF_W,
                           HEAD_HI - HEAD_LO, facecolor="#d8d7d2",
                           edgecolor=MUTED, linewidth=1.2, zorder=1))
    ax.add_patch(Circle((0.0, SPHERE_Z), SPHERE_R, facecolor=BLUE, alpha=0.55,
                        edgecolor=BLUE, linewidth=2, zorder=2))

    gap_top = SPHERE_Z - SPHERE_R
    ax.add_patch(Rectangle((-HEAD_HALF_W, HEAD_LO), 2 * HEAD_HALF_W,
                           gap_top - HEAD_LO, facecolor=ORANGE, alpha=0.30,
                           edgecolor=ORANGE, linewidth=1.6, hatch="///", zorder=3))

    pct = 100 * (gap_top - HEAD_LO) / (HEAD_HI - HEAD_LO)
    ax.annotate(f"nothing here\n{gap_top - HEAD_LO:.3f} m, {pct:.0f}% of the head",
                xy=(0.0, (HEAD_LO + gap_top) / 2), xytext=(0.16, 1.44),
                fontsize=11, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    ax.annotate("upstream's collision sphere\nr = 0.060 m, sits 45 mm too high",
                xy=(0.0, SPHERE_Z), xytext=(0.16, 1.65),
                fontsize=11, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))

    ax.set_xlim(-0.25, 0.52)
    ax.set_ylim(HEAD_LO - 0.05, HEAD_HI + 0.09)
    ax.set_xlabel("metres, side to side", fontsize=10, color=MUTED)
    ax.set_ylabel("height above the floor (m)", fontsize=10, color=MUTED)
    ax.set_title("The head could only be hit on top",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    caption(fig, "Grey: the head you can see. Blue: the shape that could be "
                 "touched.\nMeasured from the vendored g1_mjx.xml at the home pose.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


def contact_bars(path):
    """How much contact the old model was not registering."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6), facecolor=SURFACE)
    style(ax)
    bars = ax.barh(["stripped model\n(what we trained on)", "corrected model"],
                   [CONTACTS_OLD, CONTACTS_NEW],
                   color=[ORANGE, BLUE], height=0.5)
    for bar, value in zip(bars, (CONTACTS_OLD, CONTACTS_NEW)):
        ax.text(value + 70, bar.get_y() + bar.get_height() / 2, f"{value:,}",
                va="center", fontsize=13, fontweight="bold", color=INK)
    missing = CONTACTS_NEW - CONTACTS_OLD
    ax.text(CONTACTS_OLD + missing / 2, 0.62,
            f"{missing:,} contacts missing\n{100 * missing / CONTACTS_NEW:.0f}% of the real total",
            ha="center", fontsize=12, fontweight="bold", color=ORANGE)
    ax.set_xlim(0, CONTACTS_NEW * 1.18)
    ax.set_xlabel("robot-to-robot contacts", fontsize=10, color=MUTED)
    ax.set_title("Over half the contact was not happening",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    caption(fig, "The same trained policy's exact motion replayed through both "
                 "collision models, 761 steps.\nSo the difference is contact that "
                 "was missing, not contact caused by the change.")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


def progress_curves(path):
    """What the missing contact cost, measured as progress."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE)
    style(ax)
    for data, colour, label in ((PHANTOM, ORANGE, "stripped collision model"),
                                (CORRECTED, BLUE, "corrected collision model")):
        xs, ys = zip(*data)
        ax.plot(xs, ys, color=colour, linewidth=2.6, marker="o", markersize=6,
                label=label)
        ax.annotate(f"{ys[-1]:.1f}%", xy=(xs[-1], ys[-1]), xytext=(12, -4),
                    textcoords="offset points", fontsize=12, fontweight="bold",
                    color=colour)
    ax.axhline(50, color=GRID, linewidth=1.2, zorder=0)
    ax.text(30, 51, "even against the field", fontsize=9, color=MUTED)
    # The stripped model does not fall off a cliff at 400M, it stops TRENDING:
    # it wanders between 53% and 66% for the remaining 600M and ends below its own
    # peak. An arrow at one point would imply a collapse that did not happen.
    lo = min(y for x, y in PHANTOM if x >= 400)
    hi = max(y for x, y in PHANTOM if x >= 400)
    ax.axhspan(lo, hi, xmin=(400 - 0) / 1050, xmax=1.0,
               color=ORANGE, alpha=0.12, zorder=0)
    ax.text(700, lo - 7.5,
            f"no trend after 400M:\nwanders {lo:.0f}-{hi:.0f}%, ends below its own peak",
            fontsize=10.5, color=ORANGE, fontweight="bold", ha="center")
    ax.text(700, 79, "improves at every step, still climbing at 1B",
            fontsize=10.5, color=BLUE, fontweight="bold", ha="center")
    ax.set_xlabel("training frames (millions)", fontsize=10, color=MUTED)
    ax.set_ylabel("win rate against every other checkpoint", fontsize=10, color=MUTED)
    ax.set_ylim(0, 85)
    ax.set_title("Fixing the contact model removed the ceiling",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=11, labelcolor=MUTED, loc="upper left")
    caption(fig, "Every saved policy played against every other one, 256 duels per "
                 "pairing, both sides.\nSelf-play win rate is pinned at 50% by "
                 "construction, so this is the only absolute measure of progress.")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="renders/visuals")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    head_coverage(os.path.join(args.out, "1_head_coverage.png"))
    contact_bars(os.path.join(args.out, "2_missing_contact.png"))
    progress_curves(os.path.join(args.out, "3_progress_curves.png"))


if __name__ == "__main__":
    main()
