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

from automataleague_sumo import __version__
from automataleague_sumo.envs.sumo.termination import R_DRAW, R_LOSS, R_WIN
from automataleague_sumo.training import policy_ppo
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


_MAX_CONSECUTIVE_BAD_UPDATES = 25

# A batch where most importance ratios are absurd is a diverged policy, not a
# rough patch. Two consecutive such batches, rather than one, so a single violent
# update that the next batch recovers from does not end a run.
_SATURATION_ABORT_FRACTION = 0.25
_MAX_CONSECUTIVE_SATURATED_BATCHES = 2

# exp() of a float32 log-ratio overflows to inf above ~88, and TorchRL bounds the
# ratio on only ONE of PPO's two branches. In `ClipPPOLoss`:
#
#     gain1 = log_weight.exp() * advantage           # unbounded
#     gain2 = log_weight.clamp(*clip_bounds).exp() * advantage
#     gain  = min(gain1, gain2)
#
# The pessimistic `min` selects the CLIPPED branch when the advantage is positive
# and the UNCLIPPED one when it is negative. So a single sample whose ratio
# overflows lands on -inf and the whole objective becomes +inf, no matter how
# healthy the network is. Measured exactly that at 12.8M frames: entropy 7.98,
# explained_variance 0.85, critic loss 1.59, all finite, with max_ratio inf and
# ESS 1/8192 — one sample in the minibatch, everything else fine.
#
# The ratios get that large because the policy is a TanhNormal and an action at
# 0.9999990 sits deep in the saturation, where log_prob is dominated by the
# -log(1 - a^2) jacobian and is violently sensitive to a small shift in the mean.
#
# DO NOT CLAMP log_weight TO FIX THIS. That was tried, and it destroyed a run
# far more thoroughly than the overflow it prevented.
#
# `torch.clamp` has ZERO gradient outside its range, so clamping removes the
# corrective force on exactly the samples whose ratios have run away — the ones
# PPO most needs to pull back. It is self-reinforcing: a few clamped samples let
# the policy step somewhere it should not, which saturates more samples, which
# removes more of the gradient. Measured, against an unclamped run warm-started
# from the same checkpoint with the same seed:
#
#     frames   unclamped            clamped
#      6.7M    healthy              9 saturated samples, healthy
#     13.2M    ep_len 113, healthy  ep_len 26, ALL 253952 samples saturated
#     40.8M    (not reached)        ep_len 15, entropy loss -2.1e17, sigma 0.0027
#
# With every sample clamped the policy gradient is identically zero, leaving the
# entropy bonus and critic unopposed; the actor's scale collapsed monotonically
# (0.36 -> 0.0027) while its mean head grew without bound. sigma held at 0.36-0.39
# across a full 1B run and across 290M of its parent, so this was new.
#
# The overflow is instead handled where it belongs, in the training loop: a
# non-finite loss skips that minibatch. Every other sample keeps its gradient and
# nothing is silently zeroed.
#
# 20 remains the threshold at which a ratio is *reported* as absurd: it is e^20,
# eight orders of magnitude outside the 1 +- clip_epsilon trust region.
_LOG_WEIGHT_ALARM = 20.0


class SaturationCountingPPOLoss(ClipPPOLoss):
    """``ClipPPOLoss`` that reports how many importance ratios ran away.

    Observation only. It must never alter the loss, because the one time this
    class changed a number it cost 40M frames — see the note above and
    ``test_counting_never_changes_the_loss``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saturated = None

    def _log_weight(self, *args, **kwargs):
        out = super()._log_weight(*args, **kwargs)
        # Counted on device and summed across minibatches; `pop_saturated` does
        # the single host sync per collector batch. Reading it here would sync
        # once per minibatch, which is 32 stalls a batch to log one integer.
        with torch.no_grad():
            n = (out[0].abs() > _LOG_WEIGHT_ALARM).sum()
            self._saturated = n if self._saturated is None else self._saturated + n
        return out

    def pop_saturated(self) -> int:
        """Runaway ratios seen since the last call, and reset."""
        count = 0 if self._saturated is None else int(self._saturated)
        self._saturated = None
        return count


def _abort_on_nonfinite(loss, batch, collected_frames, checkpoint_dir,
                        note: str = "", log_prob_key: str = "sample_log_prob") -> None:
    """Raise with enough detail to identify the cause without a rerun."""
    parts = {k: float(v) for k, v in loss.items() if v.numel() == 1}
    log_prob = batch.get(log_prob_key, None)
    diag = {
        "collected_frames": collected_frames,
        "note": note,
        "losses": parts,
        "obs_absmax": float(batch["observation"].abs().max()),
        "obs_finite": bool(torch.isfinite(batch["observation"]).all()),
        "action_absmax": float(batch["action"].abs().max()),
        "actions_at_bound": int((batch["action"].abs() >= 1.0 - 1e-7).sum()),
        "advantage_absmax": float(batch["advantage"].abs().max())
        if "advantage" in batch.keys() else None,
        "scale_min": float(batch["scale"].min()) if "scale" in batch.keys() else None,
        "scale_max": float(batch["scale"].max()) if "scale" in batch.keys() else None,
        # Looked up through the loss module's own key rather than a hardcoded
        # name: TorchRL renamed this to "action_log_prob", so the literal
        # "sample_log_prob" silently reported None on the one diagnostic dump
        # that most needed it.
        "log_prob_absmax": (None if log_prob is None
                            else float(log_prob.abs().max())),
        "log_prob_key": str(log_prob_key),
    }
    path = os.path.join(checkpoint_dir, "nonfinite_loss.json")
    with open(path, "w") as fh:
        json.dump(diag, fh, indent=2)
    raise FloatingPointError(
        f"non-finite loss at {collected_frames:,} frames; diagnostics written to "
        f"{path}\n{json.dumps(diag, indent=2)}")


def _save(path, actor, critic, collected_frames, cfg):
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "collected_frames": collected_frames,
        "config": dict(cfg),
        # Which contract this artifact satisfies, so an evaluator can load it
        # without knowing it came from here. See automataleague_sumo/policy.py.
        "format": policy_ppo.FORMAT,
        # The task itself has changed under us before: correcting the G1's
        # collision model added 56% more contact and made policies from either
        # side of that change incomparable, with nothing in any metric to say so.
        # A rating that mixes versions is meaningless, so the version travels
        # with the weights.
        "env_version": __version__,
    }, path)


def run_ppo(cfg, *, total_frames, init_ckpt=None, init_critic=True, run_name="ppo",
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
        if init_critic:
            critic.load_state_dict(state["critic_state_dict"])
        else:
            # A critic is only meaningful against the reward it was fitted to. Carrying
            # one across a change in reward SCALE is worse than starting fresh: GAE
            # would subtract a baseline from a distribution that no longer exists, so
            # the advantages are wrong in sign structure and not merely in magnitude.
            # The critic refits in a few hundred thousand frames; a poisoned one can
            # wreck the policy in the first few updates.
            torchrl_logger.info("Warm start: actor only, critic left fresh")
        torchrl_logger.info(f"Warm-started from {init_ckpt}")

    adv_module = GAE(gamma=cfg.loss.gamma, lmbda=cfg.loss.gae_lambda,
                     value_network=critic, average_gae=False, device=device)
    loss_module = SaturationCountingPPOLoss(
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
    # Every sample is re-evaluated once per epoch, so this is the denominator
    # the saturation count has to be read against.
    evaluations_per_batch = frames_per_batch * ppo_epochs
    losses = TensorDict(batch_size=[ppo_epochs, num_mini_batches])

    start_time = time.time()
    collected_frames = 0
    num_network_updates = 0
    bad_updates = consecutive_bad = consecutive_saturated = 0
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
                total = (loss["loss_objective"] + loss["loss_entropy"]
                         + loss["loss_critic"])
                # A non-finite loss produces non-finite gradients, and
                # clip_grad_norm_ then multiplies EVERY parameter by that NaN, so
                # one bad batch destroys the whole network in a single step. This
                # happened once at 500M frames and the run went on producing
                # garbage for another 500M because nothing checked.
                #
                # Skip rather than abort, for the same reason as the gradient
                # guard below: aborting killed a 300M-frame run 6 minutes in over
                # a single outlier sample out of 8192, with every other quantity
                # in the dump healthy. The weights are untouched at this point.
                if not torch.isfinite(total):
                    optim.zero_grad(set_to_none=True)
                    bad_updates += 1
                    consecutive_bad += 1
                    if consecutive_bad >= _MAX_CONSECUTIVE_BAD_UPDATES:
                        _abort_on_nonfinite(
                            loss, batch, collected_frames, checkpoint_dir,
                            note=f"{consecutive_bad} consecutive non-finite "
                                 f"losses; the policy is diverging, not hitting "
                                 f"one bad sample",
                            log_prob_key=loss_module.tensor_keys.sample_log_prob)
                    continue
                total.backward()
                # clip_grad_norm_ returns the norm BEFORE clipping, and a
                # non-finite one is fatal: clipping scales every parameter by
                # norm/max_grad_norm, so an inf or NaN norm turns the whole
                # network to NaN in a single step. A finite LOSS does not imply a
                # finite gradient — checking only the loss is what let a run reach
                # 292M frames and then destroy itself anyway, with clip_fraction
                # at 1.0 (every importance ratio outside the clip range) because
                # PPO clips the objective, not the gradient magnitude.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.loss.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    # Skip rather than abort: one violent batch should not end a
                    # multi-hour run, and the weights are still intact at this
                    # point precisely because the step is being skipped.
                    optim.zero_grad(set_to_none=True)
                    bad_updates += 1
                    consecutive_bad += 1
                    if consecutive_bad >= _MAX_CONSECUTIVE_BAD_UPDATES:
                        _abort_on_nonfinite(
                            loss, batch, collected_frames, checkpoint_dir,
                            note=f"{consecutive_bad} consecutive non-finite "
                                 f"gradients; the policy is diverging, not "
                                 f"hitting one bad batch",
                            log_prob_key=loss_module.tensor_keys.sample_log_prob)
                    continue
                consecutive_bad = 0
                optim.step()
                losses[j, k] = loss.detach().select(
                    "loss_critic", "loss_entropy", "loss_objective")
        metrics["train/training_time"] = time.time() - train_start
        # Surfaced as a metric, not just a log line: a run quietly skipping a
        # rising fraction of its updates is diverging, and that is visible here
        # long before it becomes fatal.
        metrics["train/skipped_updates"] = bad_updates
        # Ratios so far outside the trust region that the policy cannot justify
        # the step it took. A handful is noise; a large FRACTION of the batch
        # means the current policy disowns the data it just collected, which is
        # divergence however finite every individual number still looks.
        saturated = loss_module.pop_saturated()
        metrics["train/saturated_ratios"] = saturated
        metrics["train/saturated_fraction"] = saturated / evaluations_per_batch
        # This is the guard that was missing. The run this was written for spent
        # 30M frames with EVERY ratio saturated, episode length collapsing 96 to
        # 15 and the entropy loss growing tenfold every 3M frames, while
        # train/reward ROSE from -1.46 to -0.54 (shorter episodes accrue less of
        # the per-episode shaping cost) and skipped_updates sat at 0. Nothing
        # already logged said anything was wrong.
        if saturated > _SATURATION_ABORT_FRACTION * evaluations_per_batch:
            consecutive_saturated += 1
            if consecutive_saturated >= _MAX_CONSECUTIVE_SATURATED_BATCHES:
                raise FloatingPointError(
                    f"{consecutive_saturated} consecutive batches with over "
                    f"{100 * _SATURATION_ABORT_FRACTION:.0f}% of importance "
                    f"ratios beyond e^{_LOG_WEIGHT_ALARM:.0f} at "
                    f"{collected_frames:,} frames. The policy has diverged from "
                    f"the data it collected; continuing only produces garbage "
                    f"checkpoints. Last metrics: {json.dumps(metrics)}")
        else:
            consecutive_saturated = 0
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

    # What "better" means depends on who is at the other end of the duel, and
    # getting this backwards silently selects the wrong checkpoint.
    #
    # Against a dummy the task is survival, so episode length IS the score.
    #
    # Against a real opponent it is the opposite. `win_rate` is pinned at exactly
    # 0.5 under self-play by construction — every duel produces one winner and one
    # loser and both rows are in the same batch — so it carries no information at
    # all. And a long episode is a STALEMATE, not a success: measured on a 450M
    # frame run, the old episode-length score picked a 140M checkpoint that drew
    # 65% of its duels over the 450M one that drew 0% and drove its opponent from
    # 0.375 m out to 1.21 m against a 1.5 m rim.
    #
    # So a real opponent is scored on how far the loser gets driven and on how
    # decisive the duels are. Neither is an absolute measure of skill — nothing
    # computed from self-play alone can be, since the opponent improves in step
    # with the policy — but both point the same way as the task instead of
    # against it.
    if eval_env.base_env.cfg.dummy_opponent:
        metrics["eval/score"] = metrics["eval/episode_length"]
    else:
        metrics["eval/score"] = (
            10.0 * metrics["eval/final_opp_radius"]
            + 10.0 * (1.0 - metrics.get("eval/draw_rate", 0.0))
        )
    return metrics
