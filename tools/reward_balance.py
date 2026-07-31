"""Does the reward actually pay for the behaviour it is asking for?

    python tools/reward_balance.py
    python tools/reward_balance.py --alive 4 --centre 0.5

Run this before a run, not after. The first standing run spent four million
frames learning to creep toward the middle while its episode length sat at 60
steps, because the weights of the day made survival at the spawn radius worth
-0.130 per step. The training curves looked like slow progress; the policy was
optimising exactly what it had been asked to.

Every weight is a whole-episode value (see RewardConfig), so everything printed
here is directly comparable to `win`.
"""

from __future__ import annotations

import argparse

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import RewardConfig
from automataleague_sumo.envs.sumo.rewards import break_even_radius, survival_margin


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sumo-1")
    for field, default in vars(RewardConfig()).items():
        ap.add_argument(f"--{field.replace('_', '-')}", type=float, default=None,
                        help=f"override (default {default})")
    args = ap.parse_args()

    rc = RewardConfig()
    for field in vars(rc):
        value = getattr(args, field, None)
        if value is not None:
            setattr(rc, field, value)
    rc.__post_init__()          # re-validate the overridden combination

    cfg = get_env_spec(args.env).config()
    R, spawn = cfg.ring_radius, cfg.spawn_radius

    print(f"{args.env}   every number below is a whole-episode value\n")
    print(f"  win (terminal, zero sum)        {rc.win:+7.2f}")
    print(f"  push them centre -> rim         {rc.push:+7.2f}")
    print(f"  a whole episode alive           {rc.alive:+7.2f}")
    print(f"  a whole episode pinned at rim   {-rc.centre:+7.2f}")
    print(f"  shaping budget vs win           "
          f"{rc.push + rc.alive + rc.centre:7.2f} / {rc.win:.2f}\n")

    print(f"{'radius (m)':>12} {'survive a whole episode':>25}")
    for radius in (0.0, spawn, R):
        margin = survival_margin(rc, R, radius)
        flag = "ok" if margin > 0 else "NEGATIVE"
        label = " (spawn)" if radius == spawn else (" (rim)" if radius == R else "")
        print(f"{radius:12.2f} {margin:+25.3f}  [{flag}]{label}")

    break_even = break_even_radius(rc, R)
    where = "outside the ring" if break_even > R else (
        "inside the spawn" if break_even < spawn else "between spawn and rim")
    print(f"\n  break-even radius {break_even:.3f} m ({where}); "
          f"spawn is {spawn:.2f} m, rim is {R:.2f} m")

    if survival_margin(rc, R, R) <= 0:
        raise SystemExit(
            "surviving at the rim scores worse than ending the episode, so a "
            "policy that is being pushed out is paid to give up. Raise `alive` "
            "or lower `centre`.")
    print("  survival pays everywhere inside the ring")


if __name__ == "__main__":
    main()
