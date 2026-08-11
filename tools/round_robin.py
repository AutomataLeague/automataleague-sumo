"""Play every checkpoint against every other one. The only absolute progress curve.

    python tools/round_robin.py checkpoints/selfplay_v3/ppo_eval_*.pt
    python tools/round_robin.py checkpoints/*/ppo_final.pt --duels 512

Self-play metrics are self-referential: a policy is only ever measured against a
copy of itself, so its win rate is pinned at exactly 0.5 by construction and
carries no information. Scoring each checkpoint against a FIXED field of other
checkpoints is what turns "it changed" into "it improved".

It also answers whether naive self-play is cycling. If later checkpoints beat
earlier ones consistently, improvement is transitive and a single opponent is
enough. If instead A beats B, B beats C and C beats A, the policy is going in
circles by trading one exploitable weakness for another, and training against a
POOL of past snapshots is the fix. That question is why this tool exists.

Runs on the batched Warp backend. Its ``[2N]`` self-play layout already puts side
A in the first half of the batch and side B in the second, so pitting two
different policies against each other is just applying one actor to each half:
every pairing runs as N simultaneous duels.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.envs.sumo.termination import R_LOSS, R_WIN
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.models import build_actor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+",
                   help="paths, or LABEL=path to name and order them by hand "
                        "(needed across warm starts, whose frame counters reset)")
    p.add_argument("--duels", type=int, default=256,
                   help="simultaneous duels per ordered pairing")
    p.add_argument("--max-checkpoints", type=int, default=10,
                   help="subsample evenly to this many, since the cost is quadratic")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def frames_of(path: str) -> int:
    """Frames the checkpoint records, or 0 if it does not say."""
    try:
        return int(torch.load(path, map_location="cpu",
                              weights_only=False).get("collected_frames") or 0)
    except Exception:
        return 0


def label(path: str, frames: int) -> str:
    if frames:
        return f"{frames / 1e6:.0f}M"
    return path.split("/")[-1].removesuffix(".pt")


def duel(env, actor_a, actor_b, max_steps: int) -> tuple[int, int, int]:
    """Run one batch of duels to conclusion. Returns (a_wins, b_wins, draws).

    Only each world's FIRST conclusion counts. Worlds auto-reset and would
    otherwise contribute an unequal number of duels depending on how fast they
    finish, which would silently weight the fast pairings more heavily.
    """
    n = env.num_worlds
    td = env.reset()
    settled = torch.zeros(n, dtype=torch.bool, device=env.device)
    result = torch.zeros(n, dtype=torch.int32, device=env.device)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(max_steps):
            obs = td["observation"]
            act = torch.cat([
                actor_a(TensorDict({"observation": obs[:n]}, batch_size=[n]))["action"],
                actor_b(TensorDict({"observation": obs[n:]}, batch_size=[n]))["action"],
            ], dim=0)
            td["action"] = act
            transition, td = env.step_and_maybe_reset(td)
            done = transition["next", "done"].squeeze(-1)[:n]
            fresh = done & ~settled
            if fresh.any():
                result[fresh] = transition["next", "outcome"].squeeze(-1)[:n][fresh]
                settled |= fresh
            if bool(settled.all()):
                break

    a = int((result == R_WIN).sum())
    b = int((result == R_LOSS).sum())
    return a, b, int(settled.sum()) - a - b


def main():
    args = parse_args()
    # Sort by frame count, NOT by filename. Lexicographic order puts
    # ppo_eval_100007936 (100M) before ppo_eval_10027008 (10M), which makes an
    # "evenly spaced" subsample a lopsided one and quietly ruins the progress
    # curve this tool exists to draw.
    # `LABEL=path` overrides both the name and the ordering. A warm-started run
    # restarts its own frame counter, so a lineage spanning three runs sorts and
    # labels wrongly by collected_frames alone: v6's 1000M would sort AFTER v8's
    # 300M even though v8 continues from it. Explicit labels are the only honest
    # way to draw a progress curve across a chain of warm starts.
    entries, explicit = [], False
    for arg in args.checkpoints:
        name, sep, path = arg.partition("=")
        if sep and os.path.exists(path):
            entries.append((name, path))
            explicit = True
        else:
            entries.append((None, arg))
    if explicit and any(name is None for name, _ in entries):
        raise SystemExit(
            "label every checkpoint as LABEL=path or none of them; a mix would "
            "order the labelled ones by hand and the rest by frame count")

    if explicit:
        seen, paths, given = set(), [], {}
        for name, path in entries:
            if path not in seen:
                seen.add(path)
                paths.append(path)
                given[path] = name
        counts = {path: frames_of(path) for path in paths}
    else:
        unique = sorted({path for _, path in entries})
        counts = {path: frames_of(path) for path in unique}
        paths = sorted(unique, key=lambda p: (counts[p], p))
        given = {}
    if len(paths) > args.max_checkpoints:
        idx = np.linspace(0, len(paths) - 1, args.max_checkpoints).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx))]
    if len(paths) < 2:
        raise SystemExit("need at least two checkpoints to hold a round robin")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(first["config"])

    from automataleague_sumo.envs.sumo.sumo_warp import SumoEnvWarp
    from automataleague_sumo.training.env import configs_from_cfg

    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    env = SumoEnvWarp(robot=cfg.env.robot, num_envs=args.duels, device=args.device,
                      cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc,
                      nconmax=int(cfg.env.nconmax), njmax=int(cfg.env.njmax))

    robot = get_robot(cfg.env.robot)
    actors, names = [], []
    for path in paths:
        state = torch.load(path, map_location=device, weights_only=False)
        # Each actor is rebuilt from ITS OWN stored config, not the first
        # checkpoint's. They differ in ways that change behaviour: a policy
        # trained with network.max_loc was shaped by that bound, and replaying it
        # without one produces means it was never trained to emit. Sharing one
        # config would silently misrepresent every checkpoint but the first.
        actor = build_actor(OmegaConf.create(state["config"]), robot, device)
        actor.load_state_dict(state["actor_state_dict"])
        actor.eval()
        actors.append(actor)
        names.append(given.get(path) or label(path, counts[path]))

    k = len(actors)
    wins = np.zeros((k, k))          # wins[i][j] = duels i won against j
    played = np.zeros((k, k))
    print(f"{k} checkpoints, {args.duels} duels per ordering, "
          f"both orderings per pair ({k * (k - 1)} runs)\n")

    for i, j in itertools.permutations(range(k), 2):
        # Both orderings are run, so any residual advantage to being side A
        # cancels instead of being read as one checkpoint beating another.
        a, b, d = duel(env, actors[i], actors[j], tc.max_episode_steps)
        wins[i][j] += a
        wins[j][i] += b
        played[i][j] += a + b + d
        played[j][i] += a + b + d
        print(f"  {names[i]:>6} vs {names[j]:<6}  {a:>4} - {b:<4}  ({d} draws)")

    rate = np.divide(wins, played, out=np.full_like(wins, np.nan), where=played > 0)
    overall = np.nansum(wins, axis=1) / np.maximum(np.nansum(played, axis=1), 1)

    print("\nwin rate against the whole field")
    print(f"{'checkpoint':>10} {'win rate':>10}   {'':<24}")
    for i in np.argsort(-overall):
        bar = "#" * int(round(40 * overall[i]))
        print(f"{names[i]:>10} {overall[i]:>9.1%}   {bar}")

    # Transitivity. A cycling policy is one where beating the current opponent
    # means losing to an older one, which is exactly what an opponent pool fixes
    # and what its absence risks.
    beats = rate > 0.5
    cycles = total = 0
    for a, b, c in itertools.permutations(range(k), 3):
        if played[a][b] and played[b][c] and played[a][c]:
            total += 1
            if beats[a][b] and beats[b][c] and beats[c][a]:
                cycles += 1
    print(f"\ntransitivity: {cycles} of {total} ordered triples cycle "
          f"({100 * cycles / max(total, 1):.1f}%)")
    if cycles == 0:
        print("  no cycles at all: improvement is a strict ordering, so training")
        print("  against the current policy alone is enough and an opponent pool")
        print("  would buy nothing.")
    else:
        print("  cycles present: later policies lose to older ones they had already")
        print("  beaten, which is the case for sampling opponents from a pool of")
        print("  past snapshots rather than only from the present.")

    # Does later actually mean better? `paths` is already in frame order.
    ranked = overall
    ascending = sum(1 for x, y in zip(ranked, ranked[1:]) if y >= x)
    print(f"\nmonotonicity: {ascending} of {len(ranked) - 1} consecutive steps "
          f"improve or hold")


if __name__ == "__main__":
    main()
