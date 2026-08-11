"""Adapter presenting this repo's PPO checkpoints as `policy.Policy`.

This is the reference implementation of the contract, and it is what keeps every
checkpoint already on disk rankable. It lives under ``training/`` rather than
beside the contract because it needs torchrl, which is an optional extra; the
contract itself must import without it so a submission that uses neither torchrl
nor hydra can still be validated.

Anyone adding SAC or TD3 writes the equivalent of this file, in their own repo,
and registers it. Nothing about their training belongs here.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import Tensor
from torchrl.envs import ExplorationType, set_exploration_type

from automataleague_sumo.policy import Policy, PolicyInfo, register_loader
from automataleague_sumo.robots import get_robot
from automataleague_sumo.training.models import build_actor

FORMAT = "ppo-torchrl"


class TorchRLActorPolicy(Policy):
    """Wraps a torchrl ``ProbabilisticActor`` as a batched deterministic policy."""

    def __init__(self, actor, info: PolicyInfo, device: torch.device):
        self.actor = actor
        self.info = info
        self.device = device

    def act(self, observation: Tensor) -> Tensor:
        # DETERMINISTIC, not the training-time sample: a tournament result has to
        # be reproducible from the artifact, and the contract requires act() to
        # return the same action for the same observation.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td = TensorDict({"observation": observation},
                            batch_size=[observation.shape[0]])
            return self.actor(td)["action"]


def load_ppo_policy(path: str, device: torch.device) -> TorchRLActorPolicy:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    # Each actor is rebuilt from ITS OWN stored config. Two checkpoints in one
    # tournament can differ in ways that change behaviour — network.max_loc is
    # one — and sharing a config would misrepresent every one but the first.
    actor = build_actor(cfg, get_robot(cfg.env.robot), device)
    actor.load_state_dict(state["actor_state_dict"])
    actor.eval()

    frames = int(state.get("collected_frames") or 0) or None
    info = PolicyInfo(
        env_id=str(cfg.env.name),
        robot=str(cfg.env.robot),
        algorithm="ppo",
        label=f"{frames / 1e6:.0f}M" if frames else path.split("/")[-1].removesuffix(".pt"),
        # Absent on checkpoints written before the contract existed. None is
        # honest; inventing the current version would claim a provenance we
        # cannot actually check.
        env_version=state.get("env_version"),
        frames=frames,
    )
    return TorchRLActorPolicy(actor, info, device)


register_loader(FORMAT, load_ppo_policy)
