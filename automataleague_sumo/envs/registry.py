"""Environment registry — named, versioned task environments.

An ENV is a registry id (``"sumo-1"``) mapped to an ``EnvSpec``: the default
``SumoConfig`` for that season's arena, over the shared ``envs/sumo`` engine.
Users import an env by id:

    from automataleague_sumo import make_env, list_environments
    env = make_env("sumo-1", robot="g1", backend="cpu")

Add a future season by adding one ``EnvSpec`` entry; the engine is shared.

There is deliberately no difficulty ladder. In a competitive game the difficulty
IS the opponent, and under self-play the opponent improves exactly as fast as the
policy does, so a hand-authored schedule of environment difficulty is a second
difficulty knob fighting the first. That structure came from the parkour sibling,
where difficulty genuinely is a property of the world (a taller obstacle) rather
than of who you are fighting.
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
    action_scale: float          # q-target scale in radians
    push_speed: float            # m/s, unobserved shove magnitude
    push_interval_steps: int     # control steps between shoves

    def config(self, **overrides) -> SumoConfig:
        """Default ``SumoConfig`` for this env, before any hydra overrides."""
        known = {f.name for f in fields(SumoConfig)}
        for key in overrides:
            if key not in known:
                raise ValueError(f"Unknown SumoConfig field '{key}'")
        defaults = dict(
            ring_radius=self.ring_radius,
            action_scale=self.action_scale,
            push_speed=self.push_speed,
            push_interval_steps=self.push_interval_steps,
        )
        # Construct once with the overrides merged in, so SumoConfig.__post_init__
        # validates the final combination rather than the pre-override one.
        return SumoConfig(**{**defaults, **overrides})


ENVIRONMENTS: dict[str, EnvSpec] = {
    "sumo-1": EnvSpec(
        env_id="sumo-1",
        season=0,
        description=(
            "Season 0 — raised circular dohyo, two humanoids, self-play against a "
            "growing pool of past policies."),
        ring_radius=1.5,
        # Provisional. tools/measure_reach.py must MEASURE this: the parkour lesson
        # is that a robot's capability needs margin over what the task demands, and
        # an action scale that is quietly too small reads as an exploration plateau.
        action_scale=0.5,
        # Unobserved shoves. Balance without a disturbance is a held pose: the first
        # policy trained without these survived a full 750-step episode and still
        # fell to a 0.5 m/s shove in 6 of 6 seeds. Retraining with them gave 6/6 at
        # 1.0 m/s. Two policies wrestling supply their own disturbance, but these
        # cost nothing and keep the floor honest.
        push_speed=1.0,
        push_interval_steps=75,
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

    ``cfg_overrides`` are ``SumoConfig`` fields and apply to every backend, which
    is how a run selects an opponent (``opponent="zero"`` to bootstrap standing
    against a dummy, ``"self"`` for the real game).
    ``backend_kwargs`` are the chosen backend's own constructor arguments
    (``device``, ``nconmax``, ``njmax`` for Warp); keeping the two separate is
    what makes an unknown ``SumoConfig`` field an error instead of silently
    becoming a backend argument, or the reverse.
    """
    backend_kwargs = dict(backend_kwargs or {})
    cfg = get_env_spec(env_id).config(**cfg_overrides)

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
