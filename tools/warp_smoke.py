"""Validate and benchmark the batched Warp backend before spending GPU hours.

    uv run python tools/warp_smoke.py --num-envs 2048
    uv run python tools/warp_smoke.py --num-envs 4096 --opponent zero --steps 300

Checks, in order of how expensive it is to discover them late:

1. The physics does not diverge. The parkour repo's solver settings, transplanted
   into this model, sent a pelvis to z = -4839 m; that is silent until a training
   run produces nonsense, so it is checked explicitly here.
2. Contact buffers are large enough. MuJoCo-Warp DROPS contacts past ``nconmax``
   rather than raising, so an undersized buffer looks like two robots that pass
   through each other. This reports the observed peak against the cap.
3. The two sides are actually symmetric. Both robots are the same model mirrored
   through the origin, so under mirrored actions their trajectories must match to
   solver tolerance. Asymmetry here means an index is crossed somewhere between
   the scene, the state slicing and the ctrl write — a bug that self-play would
   otherwise convert into a policy that quietly prefers one side of the ring.
4. Throughput, reported as physics steps/s and as policy rows/s.
"""

from __future__ import annotations

import argparse
import time

import torch

from automataleague_sumo.envs.registry import get_env_spec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="sumo-1")
    p.add_argument("--robot", default="g1")
    p.add_argument("--opponent", default="self",
                   help="who drives side B: self (the real game) or zero (a dummy)")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--nconmax", type=int, default=160)
    p.add_argument("--njmax", type=int, default=600)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-symmetry", action="store_true")
    p.add_argument(
        "--inject-crossed-index", action="store_true",
        help="Deliberately roll side B's actuator mapping by one joint. This is the "
             "exact bug the symmetry check exists to catch; run it to see what "
             "magnitude a real crossing produces before trusting the threshold.")
    return p.parse_args()


def build(args, num_envs=None, seed=0):
    from automataleague_sumo.envs.sumo.sumo_warp import SumoEnvWarp

    torch.manual_seed(seed)
    overrides = {"opponent": args.opponent}
    if args.opponent == "zero":
        overrides["opponent_loses_by"] = "none"
    cfg = get_env_spec(args.env).config(**overrides)
    return SumoEnvWarp(
        robot=args.robot, num_envs=num_envs or args.num_envs, device=args.device,
        cfg=cfg, nconmax=args.nconmax, njmax=args.njmax,
    )


def check_symmetry(args) -> None:
    """Mirrored inputs must give mirrored outputs.

    Side A spawns at (+x, yaw 0) and side B at (-x, yaw pi), so the whole duel is
    symmetric under a 180 degree rotation about z. Driving both sides with the SAME
    joint-space action is that mirror in joint space, so their base heights, tilts
    and radii must track each other. Comparing radius rather than raw position is
    what makes this a statement about the mirror instead of about the coordinates.
    """
    env = build(args, num_envs=8, seed=0)
    if not env.two_sided:
        # A dummy opponent drives side B at zero, so force the symmetric case.
        print("  (dummy opponent; symmetry is checked with both sides driven "
              "identically, which is the same physics question)")

    # Deterministic, noise-free start: this check is about index wiring, and spawn
    # noise would mask a real crossing behind a plausible-looking difference.
    env.cfg.pos_noise = env.cfg.yaw_noise = env.cfg.joint_noise = 0.0
    env._reset_worlds(torch.ones(env.num_worlds, dtype=torch.bool, device=env.device))

    if args.inject_crossed_index:
        env._act_cols["b"] = env._act_cols["b"].roll(1)
        print("    [injected] side B's actuator mapping rolled by one joint")

    import warp as wp

    from automataleague_sumo.envs.sumo.state import extract_duel_state

    act = torch.zeros(env.num_worlds, env.action_spec.shape[-1], device=env.device)
    history = []
    for step in range(120):
        act = 0.3 * torch.sin(torch.full_like(act, 0.05 * step))
        env._write_ctrl(act, act)
        wp.capture_launch(env._graph)
        wp.synchronize()

        qpos, qvel = env._state_tensors()
        sa, sb = extract_duel_state(qpos, qvel, env.scene)
        r_a = torch.linalg.norm(sa.base_pos[:, :2], dim=-1)
        r_b = torch.linalg.norm(sb.base_pos[:, :2], dim=-1)
        history.append(max((sa.base_pos[:, 2] - sb.base_pos[:, 2]).abs().max().item(),
                           (r_a - r_b).abs().max().item()))

    probes = [1, 5, 10, 20, 40, 80, 120]
    print("    step:  " + "  ".join(f"{s:>9d}" for s in probes))
    print("    |A-B|: " + "  ".join(f"{history[s - 1]:9.2e}" for s in probes))

    # Only the FIRST step is diagnostic. A crossed index shows up immediately at
    # full magnitude; later differences are dominated by chaos, since two collapsing
    # humanoids amplify any perturbation exponentially (measured: 4e-6 grows to 4e-3
    # over 120 steps all on its own).
    #
    # The threshold is measured, not guessed. Run --inject-crossed-index to
    # reproduce: rolling side B's actuator map by one joint gives 4.4e-3 on step 1,
    # while correct wiring gives 4.2e-6 — three orders of magnitude apart, so 1e-4
    # sits comfortably between them.
    #
    # The residual 4e-6 is not rounding noise; it is reproducible run to run. It
    # comes from the solver being capped at iterations=5 and therefore stopping
    # before convergence, so the two robots' different constraint orderings break
    # ties differently. Both sides are solved to the same accuracy, just not to the
    # same tie-break, which is harmless for training.
    early, late = history[0], max(history)
    verdict = "OK" if early < 1e-4 else "MISMATCH"
    print(f"    first step {early:.2e} [{verdict}], worst over 120 steps {late:.2e}")
    if early >= 1e-4:
        raise SystemExit(
            "side A and side B differ on the FIRST step under mirrored actions — an "
            "index is crossed between scene.py, state.py and the ctrl write")


def main():
    args = parse_args()
    print(f"building {args.env} (opponent={args.opponent!r}) with {args.num_envs} "
          f"duels on {args.device} ...")
    t0 = time.time()
    env = build(args)
    print(f"  built in {time.time() - t0:.1f}s "
          f"(includes MuJoCo-Warp kernel JIT on a cold cache)")
    print(f"  worlds={env.num_worlds}  policy rows={env.batch_size[0]}  "
          f"two_sided={env.two_sided}")
    print(f"  obs={env.observation_spec['observation'].shape[-1]}  "
          f"act={env.action_spec.shape[-1]}  "
          f"opponent={env.cfg.opponent!r}  loses_by={env.cfg.opponent_loses_by!r}")

    td = env.reset()
    rows = env.batch_size[0]
    act_dim = env.action_spec.shape[-1]

    peak_contacts = 0
    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        td["action"] = torch.zeros(rows, act_dim, device=env.device).uniform_(-0.3, 0.3)
        _, td = env.step_and_maybe_reset(td)
        peak_contacts = max(peak_contacts, env.contact_headroom()["active_contacts"])
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    obs = td["observation"]
    z = None
    from automataleague_sumo.envs.sumo.state import extract_duel_state

    qpos, qvel = env._state_tensors()
    sa, sb = extract_duel_state(qpos, qvel, env.scene)
    z = torch.cat([sa.base_pos[:, 2], sb.base_pos[:, 2]])

    print("\nhealth")
    print(f"  obs finite: {bool(torch.isfinite(obs).all())}  "
          f"|obs|max = {obs.abs().max().item():.2f}")
    print(f"  base z: min {z.min().item():.3f}  max {z.max().item():.3f}  "
          f"(a diverging solver shows up here as a huge negative)")
    if not torch.isfinite(obs).all() or z.min().item() < -5.0:
        raise SystemExit("physics diverged — do not train on this configuration")

    cap = env.contact_headroom()["capacity"]
    print("\ncontacts")
    print(f"  peak active {peak_contacts} / capacity {cap} "
          f"({100 * peak_contacts / cap:.0f}% of nconmax={args.nconmax} per world)")
    if peak_contacts > 0.9 * cap:
        print("  WARNING: within 10% of the cap. MuJoCo-Warp DROPS contacts past it "
              "rather than raising, so raise --nconmax before training.")

    print("\nthroughput")
    sim_sps = args.steps * env.num_worlds / elapsed
    print(f"  {args.steps} control steps in {elapsed:.2f}s")
    print(f"  {sim_sps:,.0f} world-steps/s   ({sim_sps * env.cfg.frame_skip:,.0f} "
          f"physics substeps/s)")
    print(f"  {args.steps * rows / elapsed:,.0f} policy rows/s")

    if not args.skip_symmetry:
        print("\nsymmetry")
        check_symmetry(args)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
