"""Is surviving one more step actually worth more than the penalties for doing so?

    python tools/reward_balance.py
    python tools/reward_balance.py --alive 0.3 --center 0.1

Run this before a level, not after. The first sumo-1 level 0 run spent four
million frames learning to creep toward the middle while its episode length sat
at 60 steps, because the default weights made survival at the spawn radius worth
-0.080 per step. The training curves looked like slow progress; the policy was in
fact optimising exactly what it had been asked to.
"""

from __future__ import annotations

import argparse

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.rewards import (
    break_even_radius,
    engage_ceiling,
    survival_margin,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sumo-1")
    ap.add_argument("--levels", type=int, nargs="*", default=None)
    for field, default in vars(RewardConfig()).items():
        ap.add_argument(f"--{field.replace('_', '-')}", type=float, default=None,
                        help=f"override (default {default})")
    args = ap.parse_args()

    rc = RewardConfig()
    for field in vars(rc):
        value = getattr(args, field, None)
        if value is not None:
            setattr(rc, field, value)

    spec = get_env_spec(args.env)
    levels = args.levels if args.levels else list(range(spec.n_levels))
    print(f"{args.env}  win={rc.win}  alive={rc.alive}  center={rc.center}\n")

    bad = []
    for level in levels:
        cfg = spec.config(level)
        R, s = cfg.ring_radius, cfg.shaping_scale
        spawn = cfg.spawn_radius
        floor = survival_margin(rc, R, spawn, s)
        # `engage` is the only per-step term that can push the floor back up. Its
        # ceiling depends on how far apart the two robots are, and at spawn they
        # are a full diameter of the spawn circle apart.
        best = floor + s * engage_ceiling(rc, 2.0 * spawn)
        break_even = break_even_radius(rc, R)
        flag = "ok" if floor > 0 else ("borderline" if best > 0 else "NEGATIVE")
        print(f"level {level}  opponent={cfg.opponent:<7} shaping_scale={s:<4}")
        print(f"  survival margin at the {spawn:.2f} m spawn : {floor:+.4f} / step "
              f"guaranteed, {best:+.4f} best case  [{flag}]")
        print(f"  break-even radius                        : {break_even:.3f} m "
              f"({'inside' if break_even < spawn else 'outside'} the spawn)")
        # A whole episode of survival against the one-off cost of losing, which is
        # the trade the policy is actually being offered.
        horizon = 750
        print(f"  {horizon} steps of survival               : {floor * horizon:+.1f} "
              f"vs {-rc.win:+.1f} for losing\n")
        if best <= 0:
            bad.append(level)

    if bad:
        raise SystemExit(
            f"levels {bad} pay the policy to end the episode sooner at the spawn "
            f"radius, even in the best case. Raise `alive`, lower `center`, or move "
            f"the spawn inward before training them.")
    print("every level pays for survival at its spawn radius")


if __name__ == "__main__":
    main()
