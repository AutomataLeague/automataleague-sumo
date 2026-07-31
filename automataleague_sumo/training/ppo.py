"""Reusable PPO training loop over the GPU-parallel Warp sumo env."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss, group_optimizers
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger

from automataleague_sumo.envs.sumo.termination import R_DRAW, R_LOSS, R_WIN
from automataleague_sumo.training.env import log_metrics, make_environment
from automataleague_sumo.training.models import make_ppo_models

_OUTCOME_NAMES = {R_WIN: "win", R_LOSS: "loss", R_DRAW: "draw"}


def outcome_rates(codes: torch.Tensor) -> dict[str, float]:
    """Fractions of concluded episodes that were a win, a loss and a draw.

    Rates are over CONCLUDED episodes only, so they sum to 1 and each is directly
    readable. Dividing by all transitions instead would make every rate a function
    of episode length, and a policy that merely survived longer would look like it
    was winning less.
    """
    total = int(codes.numel())
    if total == 0:
        return {}
    return {name: float((codes == code).sum().item()) / total
            for code, name in _OUTCOME_NAMES.items()}


def _save(path, actor, critic, collected_frames, cfg):
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "collected_frames": collected_frames,
        "config": dict(cfg),
    }, path)


def run_ppo(cfg, *, total_frames, init_ckpt=None, run_name="ppo",
            checkpoints_root="checkpoints") -> str:
    """Train one PPO run for ``total_frames``; return the best checkpoint path."""
    device = torch.device(
        cfg.network.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = os.path.join(checkpoints_root, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(checkpoint_dir, "metrics.jsonl")

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="ppo_logging",
            experiment_name=generate_exp_name("PPO", f"{cfg.logger.exp_name}_{run_name}"),
            wandb_kwargs={
                "mode": cfg.logger.mode, "config": dict(cfg),
                "project": cfg.logger.project_name, "group": cfg.logger.group_name,
            },
        )

    train_env, eval_env = make_environment(cfg)
    actor, critic = make_ppo_models(cfg, train_env, device)

    if init_ckpt:
        state = torch.load(init_ckpt, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor_state_dict"])
        critic.load_state_dict(state["critic_state_dict"])
        torchrl_logger.info(f"Warm-started from {init_ckpt}")

    adv_module = GAE(gamma=cfg.loss.gamma, lmbda=cfg.loss.gae_lambda,
                     value_network=critic, average_gae=False, device=device)
    loss_module = ClipPPOLoss(
        actor_network=actor, critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon, loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coeff=cfg.loss.entropy_coeff, critic_coeff=cfg.loss.critic_coeff,
        normalize_advantage=True,
    )
    optim = group_optimizers(
        torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5),
        torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5),
    )

    # frames_per_batch counts POLICY rows, not worlds. Under self-play each world
    # contributes two rows per step, so a given frames_per_batch is half as many
    # simulated steps as it would be against a dummy — the collector derives the
    # step count from the env's real batch size rather than assuming.
    rows = train_env.batch_size[0]
    num_worlds = train_env.base_env.num_worlds
    frames_per_batch = int(cfg.collector.frames_per_batch)
    steps_per_batch = max(frames_per_batch // rows, 1)
    frames_per_batch = steps_per_batch * rows

    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.loss.mini_batch_size,
    )
    num_mini_batches = max(frames_per_batch // cfg.loss.mini_batch_size, 1)
    total_network_updates = max(
        (int(total_frames) // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches, 1)

    ppo_epochs = int(cfg.loss.ppo_epochs)
    losses = TensorDict(batch_size=[ppo_epochs, num_mini_batches])

    start_time = time.time()
    collected_frames = 0
    num_network_updates = 0
    best_score = float("-inf")
    pbar = tqdm.tqdm(total=int(total_frames))
    td = train_env.reset()

    while collected_frames < int(total_frames):
        collect_start = time.time()
        rollout = []
        for _ in range(steps_per_batch):
            with torch.no_grad():
                td = actor(td)
            transition, td = train_env.step_and_maybe_reset(td)
            rollout.append(transition.clone())
        data = torch.stack(rollout, dim=1)
        collect_time = time.time() - collect_start

        collected_frames += data.numel()
        pbar.update(data.numel())

        metrics = {
            "train/collect_time": collect_time,
            "train/fps": data.numel() / collect_time,
            # Physics steps per second: what the GPU actually simulated, which is
            # the number comparable against other MuJoCo-Warp workloads. Each world
            # advances once per collected step regardless of how many of its robots
            # the policy controls, so this is NOT train/fps under self-play.
            "train/sim_steps_per_sec": steps_per_batch * num_worlds / collect_time,
        }

        done = data["next", "done"].squeeze(-1)
        if done.any():
            metrics["train/reward"] = data["next", "episode_reward"][done].mean().item()
            metrics["train/episode_length"] = (
                data["next", "step_count"][done].float().mean().item())
            # Honest task progress, independent of shaping: how far out each side
            # ended up. A policy that farms the shaping terms without ever putting
            # anyone out shows up here as a flat opp_radius.
            metrics["train/final_radius"] = data["next", "radius"][done].mean().item()
            metrics["train/final_opp_radius"] = (
                data["next", "opp_radius"][done].mean().item())
            for name, rate in outcome_rates(data["next", "outcome"][done]).items():
                metrics[f"train/{name}_rate"] = rate

        train_start = time.time()
        for j in range(ppo_epochs):
            with torch.no_grad():
                data = adv_module(data)
            data_buffer.extend(data.reshape(-1))
            for k, batch in enumerate(data_buffer):
                alpha = 1.0
                if cfg.loss.anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg.optim.lr * alpha
                if cfg.loss.anneal_clip_epsilon:
                    loss_module.clip_epsilon.copy_(cfg.loss.clip_epsilon * alpha)
                num_network_updates += 1

                optim.zero_grad(set_to_none=True)
                loss = loss_module(batch)
                (loss["loss_objective"] + loss["loss_entropy"]
                 + loss["loss_critic"]).backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.loss.max_grad_norm)
                optim.step()
                losses[j, k] = loss.detach().select(
                    "loss_critic", "loss_entropy", "loss_objective")
        metrics["train/training_time"] = time.time() - train_start
        for key, value in losses.apply(
                lambda x: x.float().mean(), batch_size=[]).items():
            metrics[f"train/{key}"] = value.item()

        if abs(collected_frames % cfg.logger.eval_iter) < frames_per_batch:
            metrics.update(_evaluate(actor, eval_env, cfg))
            score = metrics["eval/score"]
            if score > best_score:
                best_score = score
                _save(os.path.join(checkpoint_dir, "ppo_best.pt"),
                      actor, critic, collected_frames, cfg)
                torchrl_logger.info(f"New best policy (score={score:.3f}) -> ppo_best.pt")
            metrics["eval/best_score"] = best_score
            _save(os.path.join(checkpoint_dir, f"ppo_eval_{collected_frames}.pt"),
                  actor, critic, collected_frames, cfg)

        if logger is not None:
            log_metrics(logger, metrics, collected_frames)
        # Always keep a local copy. The dashboard is the nice view, but a run whose
        # metrics only exist in a remote service cannot be replotted or diffed
        # against another run from the box that produced it.
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({"collected_frames": collected_frames, **metrics}) + "\n")

    _save(os.path.join(checkpoint_dir, "ppo_final.pt"),
          actor, critic, collected_frames, cfg)
    torchrl_logger.info(f"Training took {time.time() - start_time:.1f}s")
    if logger is not None and cfg.logger.backend == "wandb":
        import wandb

        wandb.finish()
    pbar.close()
    return os.path.join(checkpoint_dir, "ppo_best.pt")


def _evaluate(actor, eval_env, cfg) -> dict[str, float]:
    """Deterministic evaluation over a batch of duels.

    The score that selects the best checkpoint is *survival first, then pushing*:
    a policy that stays in is worth more than one that lunges and falls out. Under
    self-play the win and loss rates are mirror images by construction (every duel
    produces one of each), so ranking on win rate alone would be ranking on noise —
    which is why the score is built from episode length and the opponent's final
    radius instead.
    """
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_start = time.time()
        actor.eval()
        rollout = eval_env.rollout(int(cfg.logger.eval_steps), actor,
                                   auto_cast_to_device=True, break_when_any_done=False)
        actor.train()

    done = rollout["next", "done"].squeeze(-1)
    metrics = {
        "eval/reward": rollout["next", "reward"].sum(-2).mean().item(),
        "eval/time": time.time() - eval_start,
    }
    if done.any():
        metrics["eval/episode_length"] = (
            rollout["next", "step_count"][done].float().mean().item())
        metrics["eval/final_opp_radius"] = (
            rollout["next", "opp_radius"][done].mean().item())
        metrics.update({f"eval/{k}_rate": v
                        for k, v in outcome_rates(rollout["next", "outcome"][done]).items()})
    else:
        # No duel concluded inside the eval window: everybody survived the whole
        # rollout. Report that as the full length rather than silently omitting the
        # metric, or the checkpoint score would read as 0 for the best policy so far.
        metrics["eval/episode_length"] = float(rollout.batch_size[-1])
        metrics["eval/final_opp_radius"] = rollout["next", "opp_radius"].mean().item()

    # Credit pushing the opponent out only where the opponent can actually be put
    # out. Against a dummy that cannot lose, its final radius is decided by how it
    # happens to topple, so including it would rank checkpoints partly on a
    # quantity the learner has no influence over — noise worth 9 to 20 points
    # against an episode length of 50 to 100.
    scores_pushing = eval_env.base_env.cfg.opponent_loses_by != "none"
    metrics["eval/score"] = (
        metrics["eval/episode_length"]
        + 100.0 * metrics.get("eval/win_rate", 0.0)
        + (10.0 * metrics["eval/final_opp_radius"] if scores_pushing else 0.0)
    )
    return metrics
