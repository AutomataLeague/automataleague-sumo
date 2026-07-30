"""Environment registry — named, versioned task environments.

An ENV is a registry id (``"sumo-1"``) mapped to an ``EnvSpec``: a default
``SumoConfig`` factory over the shared ``envs/sumo`` engine, plus season metadata
and the per-level curriculum schedule. Users import an env by id:

    from automataleague_sumo import make_env, list_environments
    env = make_env("sumo-1", robot="g1", level=0, backend="cpu")

Add a future season by adding one ``EnvSpec`` entry; the engine is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from automataleague_sumo.envs.sumo.config import (
    RewardConfig,
    SumoConfig,
    TerminationConfig,
)


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    season: int
    description: str
    ring_radius: float
    n_levels: int                                # difficulty levels 0..n_levels-1
    action_scale_by_level: tuple[float, ...]     # q-target scale (radians)
    shaping_scale_by_level: tuple[float, ...]    # anneals the shaping terms away
    opponent_by_level: tuple[str, ...]           # see config.OPPONENT_MODES
    opponent_loses_by_level: tuple[str, ...]     # see config.OPPONENT_LOSS_MODES
    push_speed_by_level: tuple[float, ...]       # m/s, unobserved shove magnitude
    push_interval_by_level: tuple[int, ...]      # control steps between shoves

    def config(self, level: int, **overrides) -> SumoConfig:
        """Default ``SumoConfig`` for this env at ``level``, before hydra overrides."""
        if not 0 <= level < self.n_levels:
            raise ValueError(
                f"{self.env_id}: level {level} out of range 0..{self.n_levels - 1}")
        known = {f.name for f in fields(SumoConfig)}
        for k in overrides:
            if k not in known:
                raise ValueError(f"Unknown SumoConfig field '{k}'")
        defaults = dict(
            ring_radius=self.ring_radius,
            level=level,
            opponent=self.opponent_by_level[level],
            opponent_loses_by=self.opponent_loses_by_level[level],
            push_speed=self.push_speed_by_level[level],
            push_interval_steps=self.push_interval_by_level[level],
            shaping_scale=self.shaping_scale_by_level[level],
            action_scale=self.action_scale_by_level[level],
        )
        # Construct once with the overrides merged in, so SumoConfig.__post_init__
        # validates the final combination rather than the pre-override one.
        return SumoConfig(**{**defaults, **overrides})


ENVIRONMENTS: dict[str, EnvSpec] = {
    "sumo-1": EnvSpec(
        env_id="sumo-1",
        season=0,
        description=(
            "Season 0 — raised circular dohyo, two humanoids, 5 curriculum levels "
            "from balance to league self-play."),
        ring_radius=1.5,
        n_levels=5,
        # Provisional and uniform. tools/measure_reach.py sets the real schedule in
        # Phase C by measuring step length and push impulse against action_scale.
        # The parkour lesson is that this number must be measured with margin over
        # what the task demands, never guessed.
        action_scale_by_level=(0.5, 0.5, 0.5, 0.5, 0.5),
        # L0 balance, L1 push a dummy, L2 beat a frozen L1 snapshot, L3 naive
        # self-play, L4 league play against a checkpoint pool.
        shaping_scale_by_level=(1.0, 1.0, 0.7, 0.4, 0.2),
        opponent_by_level=("zero", "zero", "frozen", "self", "pool"),
        # A zero-action dummy collapses on its own in ~1.2 s, so at L0 it cannot
        # lose at all (the task is purely balance) and at L1 it loses only by
        # being put out (the task is purely pushing). From L2 the opponent is a
        # real policy and plays by the ordinary rules.
        opponent_loses_by_level=("none", "ring_out", "any", "any", "any"),
        # Balance is only balance if something disturbs it. Level 0 with no
        # perturbation taught a held pose: the resulting policy stood for a full
        # 750-step episode and fell to a 0.5 m/s shove in 6 of 6 seeds. The
        # fighting levels supply their own disturbance — an opponent — so the
        # scripted shoves taper off rather than compounding with real contact.
        push_speed_by_level=(1.0, 1.0, 0.5, 0.5, 0.5),
        push_interval_by_level=(75, 75, 150, 150, 150),
    ),
}


def get_env_spec(env_id: str) -> EnvSpec:
    if env_id not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{env_id}'. Registered: {sorted(ENVIRONMENTS)}")
    return ENVIRONMENTS[env_id]


def list_environments(season: int | None = None) -> list[EnvSpec]:
    specs = list(ENVIRONMENTS.values())
    if season is not None:
        specs = [s for s in specs if s.season == season]
    return sorted(specs, key=lambda s: (s.season, s.env_id))


def make_env(
    env_id: str,
    robot: str = "g1",
    opponent_robot: str | None = None,
    level: int | None = None,
    backend: str = "cpu",
    num_envs: int | None = None,
    reward_cfg: RewardConfig | None = None,
    term_cfg: TerminationConfig | None = None,
    backend_kwargs: dict | None = None,
    **cfg_overrides,
):
    """Instantiate a registered env.

    backend: ``"cpu"`` (one duel, renderable) or ``"warp"`` (GPU, batched).
    ``opponent_robot`` defaults to ``robot``, the symmetric self-play case.
    ``num_envs`` only has meaning for ``backend="warp"``; the CPU backend is a
    single renderable duel and rejects any explicit ``num_envs`` rather than
    silently ignoring it.

    ``cfg_overrides`` are ``SumoConfig`` fields and apply to every backend.
    ``backend_kwargs`` are the chosen backend's own constructor arguments
    (``device``, ``nconmax``, ``njmax`` for Warp); keeping the two separate is
    what makes an unknown ``SumoConfig`` field an error instead of silently
    becoming a backend argument, or the reverse.
    """
    backend_kwargs = dict(backend_kwargs or {})
    spec = get_env_spec(env_id)
    lvl = spec.n_levels - 1 if level is None else int(level)
    cfg = spec.config(lvl, **cfg_overrides)

    if backend == "cpu":
        if backend_kwargs:
            raise ValueError(
                f"backend='cpu' got unexpected backend_kwargs "
                f"{sorted(backend_kwargs)} — those are Warp-only arguments.")
        if num_envs is not None:
            raise ValueError(
                f"backend='cpu' is a single renderable duel and does not support "
                f"num_envs (got {num_envs}). Use backend='warp' for batched duels."
            )
        from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU

        return SumoEnvCPU(robot=robot, opponent_robot=opponent_robot, cfg=cfg,
                          reward_cfg=reward_cfg, term_cfg=term_cfg)
    if backend == "warp":
        # Imported lazily: mujoco-warp is a GPU-only dependency, and importing the
        # registry must stay possible on a CPU-only install.
        from automataleague_sumo.envs.sumo.sumo_warp import SumoEnvWarp

        return SumoEnvWarp(robot=robot, opponent_robot=opponent_robot, cfg=cfg,
                           reward_cfg=reward_cfg, term_cfg=term_cfg,
                           **({} if num_envs is None else {"num_envs": num_envs}),
                           **backend_kwargs)
    raise ValueError(f"Unknown backend '{backend}' (use 'cpu' or 'warp')")
