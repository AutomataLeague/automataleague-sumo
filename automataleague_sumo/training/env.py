"""Env construction for training and evaluation, resolved through the registry."""

from __future__ import annotations

from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter

from automataleague_sumo.envs.registry import get_env_spec
from automataleague_sumo.envs.sumo.config import RewardConfig, TerminationConfig

# SumoConfig fields a run is allowed to override from its yaml. Anything outside
# this list is a typo rather than an intention, and SumoConfig rejects it.
_ARENA_KEYS = (
    "ring_radius", "platform_height", "spawn_frac", "opponent", "action_scale",
    "frame_skip", "push_interval_steps", "push_speed",
    "pos_noise", "yaw_noise", "joint_noise",
)


def log_metrics(logger, metrics, step):
    """Log a whole batch of metrics at ``step``.

    For wandb this writes the entire dict in ONE call with an explicit step,
    rather than one ``log_scalar`` per metric. TorchRL's WandbLogger performs its
    own per-group step injection and does not forward our step to wandb's global
    counter, so every call landed on ``_step: 0`` and overwrote the last: a
    1B-frame run with 15,259 batches arrived as a single history row holding only
    the final values, while the local metrics.jsonl had all of them.

    One call per batch instead of twenty is also the shape wandb expects, since a
    dict logged together becomes one row.
    """
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log"):
        experiment.log(dict(metrics), step=int(step))
        return
    for name, value in metrics.items():
        logger.log_scalar(name, value, step)


def configs_from_cfg(cfg):
    """``(SumoConfig, RewardConfig, TerminationConfig)`` for one run.

    The registry supplies the season's defaults; the yaml may override any field
    on top. A yaml value of ``null`` means "keep whatever the registry chose", so
    a config file does not have to restate everything to change one knob.
    """
    overrides = {}
    if hasattr(cfg.env, "arena"):
        for key in _ARENA_KEYS:
            if hasattr(cfg.env.arena, key):
                value = getattr(cfg.env.arena, key)
                if value is not None:
                    overrides[key] = value

    sumo_cfg = get_env_spec(cfg.env.name).config(**overrides)

    rc = RewardConfig()
    if hasattr(cfg.env, "reward_weights"):
        for key in vars(rc):
            if hasattr(cfg.env.reward_weights, key):
                value = getattr(cfg.env.reward_weights, key)
                if value is not None:
                    setattr(rc, key, float(value))

    tc = TerminationConfig()
    if hasattr(cfg.env, "termination"):
        for key in vars(tc):
            if hasattr(cfg.env.termination, key):
                value = getattr(cfg.env.termination, key)
                if value is not None:
                    setattr(tc, key, type(getattr(tc, key))(value))
    return sumo_cfg, rc, tc


def env_maker(cfg, num_envs=None):
    from automataleague_sumo.envs.sumo.sumo_warp import SumoEnvWarp

    sumo_cfg, rc, tc = configs_from_cfg(cfg)
    return SumoEnvWarp(
        robot=cfg.env.robot,
        num_envs=int(num_envs if num_envs is not None else cfg.env.num_envs),
        device=cfg.network.device or "cuda",
        cfg=sumo_cfg, reward_cfg=rc, term_cfg=tc,
        nconmax=int(cfg.env.nconmax), njmax=int(cfg.env.njmax),
    )


def apply_env_transforms(env, max_episode_steps):
    return TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            RewardSum(),
        ),
    )


def make_environment(cfg):
    """Train and eval envs.

    Evaluation runs the identical configuration, including the spawn noise. A
    noise-free eval would be measuring a starting state the policy never trained
    on, which is exactly how the parkour project ended up with evaluations that
    disagreed with real performance for days.
    """
    _, _, tc = configs_from_cfg(cfg)
    train_env = apply_env_transforms(env_maker(cfg), tc.max_episode_steps)
    eval_env = apply_env_transforms(
        env_maker(cfg, num_envs=int(cfg.logger.eval_envs)), tc.max_episode_steps)
    return train_env, eval_env


