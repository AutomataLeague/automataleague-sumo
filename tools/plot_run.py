"""Plot a training run from its metrics.jsonl.

    python tools/plot_run.py checkpoints/sumo1_L0/metrics.jsonl -o run.png

Small multiples rather than one crowded axis: the metrics live on wildly
different scales (steps, metres, rates, frames per second), and putting two of
them on one pair of axes would mean a second y scale, which makes the crossing
point of the two lines an artefact of the scaling rather than a fact about the
run.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1 and 2 of the validated default palette. Two series is the
# most any panel here carries, and this pair clears the colour-vision-deficiency
# and normal-vision separation floors on a light surface with room to spare.
PRIMARY = "#2a78d6"
SECONDARY = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# (panel title, y label, [(metric key, series label, colour)])
PANELS = [
    ("Episode length", "control steps", [
        ("train/episode_length", "train", PRIMARY),
        ("eval/episode_length", "eval", SECONDARY)]),
    ("Episode reward", "return", [
        ("train/reward", "train", PRIMARY),
        ("eval/reward", "eval", SECONDARY)]),
    ("How duels ended", "fraction of episodes", [
        ("train/loss_rate", "learner lost", PRIMARY),
        ("train/draw_rate", "survived to timeout", SECONDARY)]),
    ("Final distance from centre", "metres", [
        ("train/final_radius", "learner", PRIMARY)]),
    ("Value loss", "critic loss", [("train/loss_critic", "train", PRIMARY)]),
    ("Throughput", "world steps / s", [
        ("train/sim_steps_per_sec", "physics", PRIMARY)]),
]


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} is empty — has the run produced a batch yet?")
    return rows


def series(rows, key):
    xs, ys = [], []
    for row in rows:
        if key in row:
            xs.append(row["collected_frames"] / 1e6)
            ys.append(row[key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="one or more metrics.jsonl files")
    ap.add_argument("-o", "--out", default="run.png")
    ap.add_argument("--title", default="sumo-1")
    args = ap.parse_args()

    runs = [(p.split("/")[-2], load(p)) for p in args.metrics]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor=SURFACE)
    fig.suptitle(args.title, fontsize=17, fontweight="bold", color=INK,
                 x=0.012, ha="left", y=0.975)
    fig.text(0.012, 0.928,
             "  ".join(f"{name}: {len(rows)} batches, "
                       f"{rows[-1]['collected_frames'] / 1e6:.1f}M frames"
                       for name, rows in runs),
             fontsize=10, color=MUTED, ha="left")

    for ax, (title, ylabel, specs) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

        drawn = 0
        for key, label, colour in specs:
            for run_i, (_, rows) in enumerate(runs):
                xs, ys = series(rows, key)
                if not xs:
                    continue
                ax.plot(xs, ys, color=colour, linewidth=2.0,
                        linestyle=("-", "--", ":")[run_i % 3],
                        marker="o" if len(xs) < 25 else None, markersize=4,
                        label=label if len(runs) == 1 else f"{label} ({runs[run_i][0]})")
                drawn += 1

        # Return is measured in whatever units that run's weights define, so two
        # runs with different reward_weights cannot be compared on this panel. Say
        # so on the chart rather than trusting the reader to remember.
        if title == "Episode reward" and len(runs) > 1:
            title += "  (not comparable across runs)"
        ax.set_title(title, fontsize=12, fontweight="bold", color=INK,
                     loc="left", pad=8)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        ax.set_xlabel("million frames", fontsize=9, color=MUTED)
        if drawn == 0:
            ax.text(0.5, 0.5, "not recorded yet", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=10)
        # A legend only earns its space with more than one series; with one, the
        # panel title already names it.
        if drawn > 1:
            ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="best")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
