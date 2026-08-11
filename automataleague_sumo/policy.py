"""What it means to compete in a sumo env, independent of how you trained.

A tournament needs to drive two policies against each other. Until now the only
thing it could drive was *this* repo's PPO actor: ``tools/round_robin.py``
rebuilt an MLP from the checkpoint's own hydra config and loaded a state dict
into it. Anything trained with a different algorithm, or a different network, or
outside this repo entirely, could not be evaluated at all.

This module is the contract that removes that coupling. A competitor is anything
that maps a batch of observations to a batch of actions:

    actions = policy.act(observations)      # [B, obs_dim] -> [B, act_dim]

plus a ``PolicyInfo`` describing what it is, so a leaderboard can say what it is
ranking. Nothing here imports torchrl, hydra or mujoco: a submission may use none
of them, and the contract has to be loadable to check that.

**Deliberately not a training interface.** There is no update, no loss, no
optimiser. Training is the submitter's business and stays in their repo; this is
only the surface an evaluator needs.

``check_policy`` is the important part. Each of its assertions corresponds to a
way a policy can be quietly wrong rather than loudly broken, which is the failure
mode this project keeps meeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor

# The action space every sumo env exposes. `q_target = home + action_scale * a`,
# so an action outside this range is not "a bit strong", it is a different action
# scale, and a duel between a compliant policy and a non-compliant one measures
# nothing. Checked rather than clamped: clamping would hide the bug, and this
# project has already lost a run to a clamp that silently changed behaviour.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


@dataclass(frozen=True)
class PolicyInfo:
    """What a leaderboard needs to know about a submission.

    ``env_version`` is not bookkeeping. The task itself has changed under us
    before: correcting the G1's collision model added 56% more contact, which
    made policies from before and after the change incomparable in a way no
    metric revealed. A rating that mixes them is meaningless, so the version a
    policy was trained against travels with it.
    """

    env_id: str                     # e.g. "sumo-1"
    robot: str                      # e.g. "g1". Duels are same-robot only.
    algorithm: str                  # free text: "ppo", "sac", "td3", "scripted"
    label: str                      # short display name, unique on a leaderboard
    env_version: str | None = None  # automataleague_sumo version it trained against
    frames: int | None = None       # environment frames trained, if meaningful
    extra: dict = field(default_factory=dict)   # anything the submitter wants kept


@runtime_checkable
class Policy(Protocol):
    """A competitor. Batched, deterministic, and stateless across calls.

    All three properties are load-bearing for evaluation rather than stylistic:

    * **Batched**, because the GPU backend evaluates ``2N`` rows at once, both
      contestants of every duel in one call.
    * **Deterministic**, because a tournament result has to be reproducible from
      the artifact. A stochastic policy should expose its mean here and keep its
      sampling for training.
    * **Stateless across calls**, because rows in one batch belong to different,
      unrelated duels. Anything that carries state between calls, or between rows,
      makes a duel depend on who else happened to be in the batch.
    """

    info: PolicyInfo

    def act(self, observation: Tensor) -> Tensor:
        """``[B, obs_dim]`` in, ``[B, act_dim]`` in ``[-1, 1]`` out, same device."""
        ...


LoaderFn = Callable[[str, torch.device], Policy]
_LOADERS: dict[str, LoaderFn] = {}


def register_loader(fmt: str, fn: LoaderFn) -> None:
    """Teach ``load_policy`` about a new artifact format."""
    _LOADERS[fmt] = fn


def load_policy(path: str, device: torch.device | str = "cpu") -> Policy:
    """Load a policy artifact, dispatching on the ``format`` it declares.

    Checkpoints written before this contract existed carry no ``format`` key and
    are this repo's PPO actor, so they resolve to that loader. Being able to keep
    ranking the existing checkpoints is the whole reason for the fallback.
    """
    device = torch.device(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    fmt = state.get("format", "ppo-torchrl") if isinstance(state, dict) else "ppo-torchrl"
    if fmt not in _LOADERS:
        # Imported here, not at module scope: the PPO loader needs torchrl, which
        # is an optional extra, and this module must import without it.
        if fmt == "ppo-torchrl":
            from automataleague_sumo.training import policy_ppo  # noqa: F401
        if fmt not in _LOADERS:
            raise ValueError(
                f"No loader registered for policy format {fmt!r}. Known: "
                f"{sorted(_LOADERS)}. Register one with "
                f"automataleague_sumo.policy.register_loader.")
    return _LOADERS[fmt](path, device)


def check_policy(
    policy: Policy,
    *,
    obs_dim: int,
    act_dim: int,
    device: torch.device | str = "cpu",
    batch: int = 8,
    seed: int = 0,
) -> None:
    """Reject a policy that would corrupt a tournament, before it enters one.

    Raises ``ValueError`` with a specific reason. Every check here stands for a
    failure that produces a plausible-looking result rather than an error:

    * **shape / dtype** — a wrong action width silently drives the wrong joints.
    * **finite** — a NaN action propagates into the physics and reads as a robot
      that simply falls over. Three runs in this project died on non-finite
      numbers and none of them announced themselves.
    * **range** — an action outside ``[-1, 1]`` is a different action scale, so
      the duel measures the scale rather than the policy.
    * **determinism** — a tournament that cannot be re-run is not a measurement.
    * **row independence** — both contestants of every duel go through one call.
      A policy with batch-coupled state (BatchNorm left in train mode is the
      classic) makes each robot's action depend on its opponent's observation,
      which is not a duel at all and would never show up as an error.
    """
    device = torch.device(device)
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(batch, obs_dim, generator=gen).to(device)

    with torch.no_grad():
        action = policy.act(obs)

        if not isinstance(action, Tensor):
            raise ValueError(f"act() must return a Tensor, got {type(action).__name__}")
        if action.shape != (batch, act_dim):
            raise ValueError(
                f"act() returned {tuple(action.shape)}, expected {(batch, act_dim)}")
        if action.dtype != torch.float32:
            raise ValueError(f"act() must return float32, got {action.dtype}")
        if not torch.isfinite(action).all():
            raise ValueError("act() returned non-finite values")
        if float(action.min()) < ACTION_LOW or float(action.max()) > ACTION_HIGH:
            raise ValueError(
                f"actions must lie in [{ACTION_LOW}, {ACTION_HIGH}], got "
                f"[{float(action.min()):.4f}, {float(action.max()):.4f}]. They are "
                f"not clipped for you: out-of-range actions mean a different "
                f"action scale, not a stronger policy.")

        again = policy.act(obs)
        if not torch.equal(action, again):
            raise ValueError(
                f"act() is not deterministic: the same observations gave a "
                f"different action (max difference "
                f"{float((action - again).abs().max()):.3g}). Expose the mean for "
                f"evaluation and keep sampling for training.")

        one_at_a_time = torch.cat([policy.act(obs[i:i + 1]) for i in range(batch)])
        if not torch.allclose(action, one_at_a_time, atol=1e-4):
            raise ValueError(
                f"act() couples rows: evaluating the batch differs from "
                f"evaluating each row alone by up to "
                f"{float((action - one_at_a_time).abs().max()):.3g}. Both robots "
                f"in a duel share one call, so a row must not see the others. "
                f"BatchNorm left in train mode is the usual cause.")
