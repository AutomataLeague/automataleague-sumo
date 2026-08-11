"""Hand-written baselines. No network, no training, no torchrl.

Two jobs.

**A floor for the leaderboard.** A rating is only interpretable against something
fixed. ``still`` is that something: it does nothing at all, and any policy that
cannot beat it has learned nothing. The first "successful" standing policy this
project trained scored *below* the equivalent baseline, and its training curve
rose the whole way.

**A test of the evaluation contract.** These enter a tournament through exactly
the same path a SAC or TD3 submission would, and they are about as far from the
PPO actor as a competitor can get: no weights, no torchrl, no hydra config, and
an artifact that is a handful of scalars. If the contract cannot admit them it
will not admit anyone else's either, which is why they exist before the
leaderboard rather than after it.

The joint directions in ``lean`` are measured, not guessed. Driving both ankle
pitches to -0.8 for 40 control steps moves the base +0.215 m along its own
forward axis; +0.8 moves it -0.213 m. Hip pitch behaves the same way with about
a third of the authority.

``lean`` is nevertheless **worse than doing nothing**: over 192 duels a side it
loses to ``still`` 17-173, and it scored 2.5% against a mixed field where
``still`` scored 22.3%. Leaning into an opponent topples you long before it
pushes them anywhere. That is not a bug to fix, it is the point of measuring
against a do-nothing baseline, and it is the same result this project got the
first time it trained a policy: a plausible idea that is measurably worse than
inaction. Both are kept because a floor made of two different bad strategies is
more informative than one.
"""

from __future__ import annotations

import torch
from torch import Tensor

from automataleague_sumo.envs.sumo.observation import TASK_DIM, observation_dim
from automataleague_sumo.policy import PolicyInfo, register_loader
from automataleague_sumo.robots import RobotSpec, get_robot

FORMAT = "scripted"
KINDS = ("still", "lean")

# Offsets into the task block, which begins at `robot.proprio_dim`. See the
# layout in envs/sumo/observation.py: r/R, to-centre xy, (R-r)/R, then the
# opponent block starting with its position in this robot's own base frame.
_REL_POS = 4


class ScriptedPolicy:
    """A `policy.Policy` with no learned parameters."""

    def __init__(self, kind: str, robot: str | RobotSpec = "g1", *,
                 gain: float = 0.6, env_id: str = "sumo-1", label: str | None = None):
        if kind not in KINDS:
            raise ValueError(f"Unknown scripted kind {kind!r}. Valid: {list(KINDS)}")
        self.robot = robot if isinstance(robot, RobotSpec) else get_robot(robot)
        self.kind = kind
        self.gain = float(gain)
        self._rel_pos = self.robot.proprio_dim + _REL_POS
        self._n = self.robot.n_joints

        joints = {name: i for i, name in enumerate(self.robot.joint_names)}
        self._ankle = [i for name, i in joints.items() if "ankle_pitch" in name]
        self._hip = [i for name, i in joints.items() if "hip_pitch" in name]

        self.info = PolicyInfo(
            env_id=env_id, robot=self.robot.name, algorithm="scripted",
            label=label or kind, frames=0,
            extra={"kind": kind, "gain": self.gain},
        )

    def act(self, observation: Tensor) -> Tensor:
        expected = observation_dim(self.robot)
        if observation.shape[-1] != expected:
            raise ValueError(
                f"{self.kind!r} expects {expected} observations for "
                f"{self.robot.name} ({self.robot.proprio_dim} proprioceptive + "
                f"{TASK_DIM} task), got {observation.shape[-1]}")

        action = torch.zeros(observation.shape[0], self._n,
                             dtype=torch.float32, device=observation.device)
        if self.kind == "still":
            return action

        # Where the opponent is, in THIS robot's base frame, so no absolute
        # heading is needed and the same rule works on either side of the ring.
        rel = observation[:, self._rel_pos:self._rel_pos + 2]
        unit = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        forward = unit[:, :1]

        # Negative pitch drives the base forward (measured, see module docstring),
        # so lean INTO the opponent. Ankles carry it and the hips add a third.
        action[:, self._ankle] = -self.gain * forward
        action[:, self._hip] = -self.gain / 3.0 * forward
        # Safe to clamp: there is no gradient anywhere near this, which is the
        # only reason a clamp is acceptable in this repo at all.
        return action.clamp(-1.0, 1.0)


def load_scripted_policy(path: str, device: torch.device) -> ScriptedPolicy:
    spec = torch.load(path, map_location="cpu", weights_only=False)
    return ScriptedPolicy(
        kind=spec["kind"],
        robot=spec.get("robot", "g1"),
        gain=float(spec.get("gain", 0.6)),
        env_id=spec.get("env_id", "sumo-1"),
        label=spec.get("label"),
    )


def save_scripted_policy(path: str, kind: str, robot: str = "g1",
                         gain: float = 0.6, env_id: str = "sumo-1") -> None:
    """Write an artifact a tournament can load like any other entrant."""
    ScriptedPolicy(kind, robot, gain=gain, env_id=env_id)   # validate before writing
    torch.save({"format": FORMAT, "kind": kind, "robot": robot,
                "gain": gain, "env_id": env_id}, path)


register_loader(FORMAT, load_scripted_policy)
