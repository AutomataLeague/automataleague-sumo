"""GPU-batched sumo duels on MuJoCo Warp.

Steps ``num_envs`` parallel *worlds*, each world holding BOTH robots of one duel,
while observation, reward and termination run as batched PyTorch on the same
tensors the CPU backend uses. Nothing in ``envs/sumo`` below this file knows which
backend it is running under, which is what keeps CPU evaluation and GPU training
from disagreeing.

Batch layout
------------
The physics batch is always ``num_envs`` worlds. What varies is how many robots in
each world the *policy* controls, and that is what sets the TorchRL batch size:

* ``opponent="self"`` — both robots are the learner. Batch is ``[2N]``: rows
  ``0..N-1`` are side A of each world and rows ``N..2N-1`` are side B of the same
  worlds, in the same order. One shared policy acting on all ``2N`` rows *is*
  naive self-play, with no opponent bookkeeping anywhere. This works only because
  the observation is expressed entirely in each robot's own base frame, so side B
  cannot tell it is side B.
* ``opponent="zero"`` — a passive dummy, only for bootstrapping standing on a
  fresh robot. Side B is held at its home pose and the batch is ``[N]``.

Both rows of a world share that world's ``done``, because a duel ends for both
contestants at once.
"""

from __future__ import annotations

import math

import mujoco_warp as mjw
import torch
import warp as wp
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from automataleague_sumo.envs.sumo.config import (
    RewardConfig,
    SumoConfig,
    TerminationConfig,
)
from automataleague_sumo.envs.sumo.observation import build_observation, observation_dim
from automataleague_sumo.envs.sumo.rewards import compute_reward
from automataleague_sumo.envs.sumo.scene import SideInfo, build_sumo_model
from automataleague_sumo.envs.sumo.state import extract_duel_state
from automataleague_sumo.envs.sumo.termination import compute_termination, row_outcome


class SumoEnvWarp(EnvBase):
    """Batched GPU sumo environment. All worlds step simultaneously."""

    def __init__(
        self,
        robot: str = "g1",
        opponent_robot: str | None = None,
        num_envs: int = 1024,
        device: str = "cuda",
        cfg: SumoConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        term_cfg: TerminationConfig | None = None,
        # Per-world contact/constraint buffer sizes. Two humanoids in a clinch are
        # far more contact-rich than one quadruped on terrain; MuJoCo-Warp silently
        # drops constraints past these caps, so they are measured, not guessed —
        # tools/warp_smoke.py reports the observed peak.
        nconmax: int = 160,
        njmax: int = 600,
    ):
        self._num_worlds = int(num_envs)
        self._device = torch.device(device)
        self.cfg = cfg or SumoConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.term_cfg = term_cfg or TerminationConfig()

        self._mjm, self.scene = build_sumo_model(robot, opponent_robot, self.cfg)
        if self.scene.a.robot.name != self.scene.b.robot.name:
            raise NotImplementedError(
                f"SumoEnvWarp assumes both sides share a robot (got "
                f"a={self.scene.a.robot.name!r}, b={self.scene.b.robot.name!r}). "
                f"action_scale and the observation width are derived from side A, so "
                f"a cross-robot matchup would silently run at the wrong action scale."
            )
        self.robot = self.scene.a.robot

        # Both sides learn only under true self-play; otherwise side B is a dummy
        # driven by the env and never appears in the policy's batch.
        self._two_sided = self.cfg.opponent == "self"
        self._num_rows = self._num_worlds * (2 if self._two_sided else 1)

        self._mjw_model = mjw.put_model(self._mjm)
        self._mjw_data = mjw.make_data(
            self._mjm, nworld=self._num_worlds, nconmax=nconmax, njmax=njmax
        )

        self._obs_dim = observation_dim(self.robot)
        self._act_dim = self.robot.action_dim
        # Per joint, not one number: the legs need a small window for balance and
        # the arms a large one to reach at all. See RobotSpec.joint_scale.
        self._action_scale = torch.as_tensor(
            self.robot.scale_vector(self.cfg.action_scale),
            dtype=torch.float32, device=self._device)

        self._setup_device_tensors()
        self._setup_contact_lookup()

        # Per-world task state.
        d, N = self._device, self._num_worlds
        self.step_count = torch.zeros(N, dtype=torch.long, device=d)
        self.prev_action = {
            "a": torch.zeros(N, self._act_dim, device=d),
            "b": torch.zeros(N, self._act_dim, device=d),
        }
        self.prev_radius = {"a": torch.zeros(N, device=d), "b": torch.zeros(N, device=d)}

        self._graph = None
        self._capture_cuda_graph()

        super().__init__(device=self._device, batch_size=torch.Size([self._num_rows]))
        self._make_spec()

    # ------------------------------------------------------------------ setup
    def _setup_device_tensors(self) -> None:
        """Hoist every index and constant the step loop needs onto the device.

        These are rebuilt from numpy on each access otherwise, which is a
        host-to-device copy per step per side — cheap in absolute terms but paid
        50 times a second against a step that is itself only a couple of
        milliseconds at large batch.
        """
        d = self._device
        self._home_qpos = torch.as_tensor(
            self.scene.home_qpos, dtype=torch.float32, device=d)
        self._sides: dict[str, SideInfo] = {"a": self.scene.a, "b": self.scene.b}
        self._act_cols, self._home_joint, self._joint_qadr, self._base_qadr = {}, {}, {}, {}
        for key, side in self._sides.items():
            self._act_cols[key] = torch.as_tensor(
                side.actuator_ids, dtype=torch.long, device=d)
            self._home_joint[key] = torch.as_tensor(
                side.robot.home_joint_qpos, dtype=torch.float32, device=d)
            self._joint_qadr[key] = torch.as_tensor(
                side.joint_qposadr, dtype=torch.long, device=d)
            self._base_qadr[key] = int(side.base_qposadr)
        self._base_dofadr = {k: int(s.base_dofadr) for k, s in self._sides.items()}
        self._foot_ids = {k: torch.as_tensor(s.foot_geom_ids, dtype=torch.long, device=d)
                          for k, s in self._sides.items()}

    def _setup_contact_lookup(self) -> None:
        """Per-geom side tag plus the buffers for the batched A-touches-B test.

        MuJoCo-Warp packs active contacts into the first ``nacon`` rows of a dense
        ``nconmax * nworld`` array and leaves the remaining rows holding stale data
        from previous steps — including a ``worldid`` of 0, which would otherwise
        attribute every dead slot to world 0. The validity mask therefore has to
        come from ``nacon``, and since ``nacon`` already lives on the device the
        whole test runs there with no host synchronization.
        """
        d = self._device
        side_tag = torch.zeros(self._mjm.ngeom, dtype=torch.uint8, device=d)
        side_tag[torch.as_tensor(self.scene.a.geom_ids, device=d)] = 1
        side_tag[torch.as_tensor(self.scene.b.geom_ids, device=d)] = 2
        self._geom_side = side_tag
        n_slots = self._mjw_data.contact.geom.shape[0]
        self._contact_slots = torch.arange(n_slots, device=d)

    def _capture_cuda_graph(self) -> None:
        """Record the whole frame_skip physics burst as one replayable CUDA graph."""
        mjw.step(self._mjw_model, self._mjw_data)
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(self.cfg.frame_skip):
                mjw.step(self._mjw_model, self._mjw_data)
        self._graph = capture.graph

    def _make_spec(self) -> None:
        d, B = self._device, self._num_rows
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(B, self._obs_dim), dtype=torch.float32, device=d),
            shape=(B,),
        )
        self.action_spec = Composite(
            action=Bounded(
                low=-torch.ones(self._act_dim, device=d).expand(B, -1),
                high=torch.ones(self._act_dim, device=d).expand(B, -1),
                dtype=torch.float32, device=d),
            shape=(B,),
        )
        self.reward_spec = Unbounded(shape=(B, 1), dtype=torch.float32, device=d)
        self.done_spec = Composite(
            done=Unbounded(shape=(B, 1), dtype=torch.bool, device=d),
            terminated=Unbounded(shape=(B, 1), dtype=torch.bool, device=d),
            truncated=Unbounded(shape=(B, 1), dtype=torch.bool, device=d),
            shape=(B,),
        )

    @property
    def num_worlds(self) -> int:
        """Parallel duels being simulated. Distinct from ``batch_size``, which
        counts policy-controlled robots and is twice this under self-play."""
        return self._num_worlds

    @property
    def two_sided(self) -> bool:
        return self._two_sided

    # -------------------------------------------------------------- utilities
    def _state_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return wp.to_torch(self._mjw_data.qpos), wp.to_torch(self._mjw_data.qvel)

    def _contact_flag(self) -> torch.Tensor:
        """``[N]`` 1.0 where any geom of side A touches any geom of side B.

        The CPU backend answers the same question by walking ``data.contact``; both
        feed the identical observation slot, so a policy trained here evaluates
        unchanged on CPU.
        """
        c = self._mjw_data.contact
        geom = wp.to_torch(c.geom).long()                    # [slots, 2]
        worldid = wp.to_torch(c.worldid).long()              # [slots]
        nacon = wp.to_torch(self._mjw_data.nacon)            # [1], stays on device
        valid = self._contact_slots < nacon
        # Dead slots can hold out-of-range ids; clamp before the gather so the
        # lookup is always in bounds, and let `valid` do the actual rejecting.
        tag1 = self._geom_side[geom[:, 0].clamp_(0, self._mjm.ngeom - 1)]
        tag2 = self._geom_side[geom[:, 1].clamp_(0, self._mjm.ngeom - 1)]
        cross = valid & (((tag1 == 1) & (tag2 == 2)) | ((tag1 == 2) & (tag2 == 1)))
        flag = torch.zeros(self._num_worlds, device=self._device)
        flag.index_put_((worldid.clamp_(0, self._num_worlds - 1),),
                        cross.to(flag.dtype), accumulate=True)
        return (flag > 0).to(torch.float32)

    def _maybe_push(self) -> None:
        """Shove both bases in a random horizontal direction, on schedule.

        Applied to ``qvel`` before the physics burst rather than as a force, so the
        magnitude reads directly in m/s and is comparable to ``tools/push_test.py``.
        Deliberately absent from the observation: a disturbance the policy can see
        coming is a control input, not a disturbance, and would let it pre-brace
        instead of learning to recover.
        """
        cfg = self.cfg
        if cfg.push_interval_steps <= 0:
            return
        due = (self.step_count > 0) & (self.step_count % cfg.push_interval_steps == 0)
        if not bool(due.any()):
            return
        qvel = wp.to_torch(self._mjw_data.qvel)
        N, d = self._num_worlds, self._device
        for key in ("a", "b"):
            adr = self._base_dofadr[key]
            # Independent heading and magnitude per robot: a shared impulse would
            # push both the same way, which is a moving reference frame rather than
            # a disturbance to either.
            theta = torch.rand(N, device=d) * (2.0 * math.pi)
            speed = torch.rand(N, device=d) * cfg.push_speed
            delta = torch.stack([speed * torch.cos(theta),
                                 speed * torch.sin(theta)], dim=-1)
            qvel[:, adr:adr + 2] += torch.where(
                due.unsqueeze(-1), delta, torch.zeros_like(delta))

    def _write_ctrl(self, act_a: torch.Tensor, act_b: torch.Tensor) -> None:
        ctrl = wp.to_torch(self._mjw_data.ctrl)              # [N, nu]
        for key, act in (("a", act_a), ("b", act_b)):
            ctrl[:, self._act_cols[key]] = (
                self._home_joint[key] + self._action_scale * act)

    def _observations(self, contact: torch.Tensor | None = None):
        qpos, qvel = self._state_tensors()
        sa, sb = extract_duel_state(qpos, qvel, self.scene)
        if contact is None:
            contact = self._contact_flag()
        obs_a = build_observation(sa, sb, self.prev_action["a"], self._home_joint["a"],
                                  self.cfg.ring_radius, contact)
        obs_b = build_observation(sb, sa, self.prev_action["b"], self._home_joint["b"],
                                  self.cfg.ring_radius, contact)
        return obs_a, obs_b, sa, sb

    def _stack_rows(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Side A's rows then side B's, or A alone when B is a dummy."""
        return torch.cat([a, b], dim=0) if self._two_sided else a

    def _world_mask(self, row_mask: torch.Tensor) -> torch.Tensor:
        """Collapse a ``[rows]`` reset mask onto the ``[N]`` worlds it refers to."""
        if not self._two_sided:
            return row_mask
        return row_mask[:self._num_worlds] | row_mask[self._num_worlds:]

    # ------------------------------------------------------------------ reset
    def _reset_worlds(self, mask: torch.Tensor) -> None:
        """Re-home the masked worlds with the same spawn noise the CPU env uses.

        Both backends reset with noise on purpose: a zero-noise reset makes a
        deterministic evaluation out of distribution relative to training, which is
        how the parkour project spent days trusting evaluations that disagreed with
        real performance.
        """
        if not bool(mask.any()):
            return
        qpos, qvel = self._state_tensors()
        d, N, cfg = self._device, self._num_worlds, self.cfg
        new_qpos = self._home_qpos.unsqueeze(0).expand(N, -1).clone()

        for key in ("a", "b"):
            qa = self._base_qadr[key]
            if cfg.pos_noise > 0:
                new_qpos[:, qa:qa + 2] += (
                    torch.rand(N, 2, device=d) * 2.0 - 1.0) * cfg.pos_noise
            if cfg.yaw_noise > 0:
                # The home pose is a pure yaw, so recomposing from the perturbed
                # yaw reproduces it exactly rather than approximating it.
                w, z = new_qpos[:, qa + 3], new_qpos[:, qa + 6]
                yaw = 2.0 * torch.atan2(z, w)
                yaw = yaw + (torch.rand(N, device=d) * 2.0 - 1.0) * cfg.yaw_noise
                new_qpos[:, qa + 3] = torch.cos(yaw / 2)
                new_qpos[:, qa + 4] = 0.0
                new_qpos[:, qa + 5] = 0.0
                new_qpos[:, qa + 6] = torch.sin(yaw / 2)
            if cfg.joint_noise > 0:
                jq = self._joint_qadr[key]
                new_qpos[:, jq] += torch.randn(
                    N, jq.shape[0], device=d) * cfg.joint_noise

        qpos[mask] = new_qpos[mask]
        qvel[mask] = 0.0
        self.step_count[mask] = 0
        for key in ("a", "b"):
            self.prev_action[key][mask] = 0.0

    def _reset(self, td: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        if td is not None and "_reset" in td.keys():
            mask = self._world_mask(td["_reset"].reshape(self._num_rows))
        else:
            mask = torch.ones(self._num_worlds, dtype=torch.bool, device=self._device)
        self._reset_worlds(mask)

        # The contact arrays still describe the pre-reset step; a world that just
        # reset would otherwise start its episode reporting the contact that ended
        # the previous one. Re-running collision detection to fix that would cost a
        # full forward per reset, and it is unnecessary: the two spawns are a ring
        # diameter apart, so a freshly reset world is never in contact.
        contact = self._contact_flag()
        contact[mask] = 0.0
        obs_a, obs_b, sa, sb = self._observations(contact)
        self.prev_radius["a"] = torch.linalg.norm(sa.base_pos[:, :2], dim=-1)
        self.prev_radius["b"] = torch.linalg.norm(sb.base_pos[:, :2], dim=-1)

        B, d = self._num_rows, self._device
        zeros = torch.zeros(B, 1, dtype=torch.bool, device=d)
        return TensorDict(
            {
                "observation": self._stack_rows(obs_a, obs_b),
                "done": zeros.clone(),
                "terminated": zeros.clone(),
                "truncated": zeros.clone(),
            },
            batch_size=self.batch_size,
            device=d,
        )

    # ------------------------------------------------------------------- step
    def _step(self, td: TensorDictBase) -> TensorDictBase:
        actions = td["action"].clamp(-1.0, 1.0)
        if self._two_sided:
            act_a, act_b = actions[:self._num_worlds], actions[self._num_worlds:]
        else:
            # A zero action holds the dummy at its home pose, which is what makes
            # it collapse. It cannot lose by doing so (see SumoConfig.dummy_opponent).
            act_a = actions
            act_b = torch.zeros_like(act_a)
        self._write_ctrl(act_a, act_b)
        self._maybe_push()

        wp.capture_launch(self._graph)
        wp.synchronize()
        self.step_count += 1

        qpos, qvel = self._state_tensors()
        sa, sb = extract_duel_state(qpos, qvel, self.scene)
        geom_xpos = wp.to_torch(self._mjw_data.geom_xpos)          # [N, ngeom, 3]
        terminated, truncated, lost_a, lost_b, outcome = compute_termination(
            sa, sb,
            geom_xpos[:, self._foot_ids["a"]], geom_xpos[:, self._foot_ids["b"]],
            self.scene.a.robot, self.scene.b.robot,
            self.step_count, self.cfg, self.term_cfg)

        horizon = self.term_cfg.max_episode_steps
        rew_a, _ = compute_reward(
            sa, sb, self.prev_radius["b"], lost_a, lost_b, act_a,
            self.cfg.ring_radius, self.reward_cfg, horizon)
        rew_b, _ = compute_reward(
            sb, sa, self.prev_radius["a"], lost_b, lost_a, act_b,
            self.cfg.ring_radius, self.reward_cfg, horizon)

        r_a = torch.linalg.norm(sa.base_pos[:, :2], dim=-1)
        r_b = torch.linalg.norm(sb.base_pos[:, :2], dim=-1)
        self.prev_radius["a"], self.prev_radius["b"] = r_a, r_b
        # Clone, do not alias. act_a/act_b are views into the caller's action
        # tensor, and _reset_worlds zeroes prev_action for finished worlds — which
        # would reach through the view and rewrite the actions already recorded in
        # this batch, silently training on actions that were never taken.
        self.prev_action["a"], self.prev_action["b"] = act_a.clone(), act_b.clone()

        obs_a, obs_b, _, _ = self._observations()
        done = terminated | truncated

        # No auto-reset here. TorchRL's step_and_maybe_reset calls _reset with the
        # done mask, so the observation returned for a finished duel is the real
        # terminal state. Resetting in place instead would hand the value function
        # a post-reset observation as the bootstrap target on every truncation.
        return TensorDict(
            {
                "observation": self._stack_rows(obs_a, obs_b),
                "reward": self._stack_rows(rew_a, rew_b).unsqueeze(-1),
                "done": self._stack_rows(done, done).unsqueeze(-1),
                "terminated": self._stack_rows(terminated, terminated).unsqueeze(-1),
                "truncated": self._stack_rows(truncated, truncated).unsqueeze(-1),
                "outcome": self._row_outcome(outcome).unsqueeze(-1),
                "radius": self._stack_rows(r_a, r_b).unsqueeze(-1),
                "opp_radius": self._stack_rows(r_b, r_a).unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self._device,
        )

    def _row_outcome(self, outcome: torch.Tensor) -> torch.Tensor:
        """Duel outcome recoded from each learner row's own point of view."""
        as_a = row_outcome(outcome, as_side_a=True)
        if not self._two_sided:
            return as_a
        return torch.cat([as_a, row_outcome(outcome, as_side_a=False)], dim=0)

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    # --------------------------------------------------------------- reporting
    def contact_headroom(self) -> dict[str, int]:
        """Observed contact usage against the allocated cap.

        MuJoCo-Warp does not raise when a world needs more contacts than
        ``nconmax``; it drops them, and the duel quietly runs with less collision
        than it should. This is the number to watch when tuning the buffers.
        """
        nacon = int(wp.to_torch(self._mjw_data.nacon)[0].item())
        return {
            "active_contacts": nacon,
            "capacity": int(self._mjw_data.contact.geom.shape[0]),
            "per_world_mean": nacon // max(self._num_worlds, 1),
        }
