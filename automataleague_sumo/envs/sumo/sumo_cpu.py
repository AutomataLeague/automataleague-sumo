"""Single-duel sumo on classic MuJoCo (CPU). Steppable and renderable.

Runs the shared task logic at batch size 1. Used for local validation, for
rendering, and for head-to-head evaluation of two checkpoints. The batched
MuJoCo-Warp backend used for training arrives in Phase C and consumes exactly the
same task logic modules.

The API is duel level: ``step`` takes both actions. Wrapping this as a
single-agent env with a frozen opponent belongs with the self-play machinery.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from automataleague_sumo.envs.sumo.config import (
    RewardConfig,
    SumoConfig,
    TerminationConfig,
)
from automataleague_sumo.envs.sumo.observation import build_observation, observation_dim
from automataleague_sumo.envs.sumo.render import render_frame
from automataleague_sumo.envs.sumo.rewards import compute_reward
from automataleague_sumo.envs.sumo.scene import build_sumo_model
from automataleague_sumo.envs.sumo.state import contact_flag_cpu, extract_duel_state
from automataleague_sumo.envs.sumo.termination import compute_termination


class SumoEnvCPU:
    def __init__(
        self,
        robot: str = "g1",
        opponent_robot: str | None = None,
        cfg: SumoConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        term_cfg: TerminationConfig | None = None,
        render_size: tuple[int, int] = (720, 1280),
    ):
        self.cfg = cfg or SumoConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.term_cfg = term_cfg or TerminationConfig()
        self.model, self.scene = build_sumo_model(robot, opponent_robot, self.cfg)
        self.data = mujoco.MjData(self.model)

        self.action_scale = (
            self.cfg.action_scale if self.cfg.action_scale is not None
            else self.scene.a.robot.action_scale)
        self._home = {
            side.prefix: torch.tensor(side.robot.home_joint_qpos, dtype=torch.float32)
            for side in self.scene.sides
        }
        self._render_size = render_size
        self._renderer = None
        self._rng = np.random.default_rng()

    # --- dimensions ---------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return observation_dim(self.scene.a.robot)

    @property
    def action_dim(self) -> int:
        return self.scene.a.robot.action_dim

    # --- internals ----------------------------------------------------------
    def _tensors(self):
        return (torch.tensor(self.data.qpos, dtype=torch.float32).unsqueeze(0),
                torch.tensor(self.data.qvel, dtype=torch.float32).unsqueeze(0))

    def _observations(self):
        qpos, qvel = self._tensors()
        sa, sb = extract_duel_state(qpos, qvel, self.scene)
        contact = contact_flag_cpu(self.model, self.data, self.scene)
        obs_a = build_observation(sa, sb, self._prev_action["a/"], self._home["a/"],
                                  self.cfg.ring_radius, contact)
        obs_b = build_observation(sb, sa, self._prev_action["b/"], self._home["b/"],
                                  self.cfg.ring_radius, contact)
        return obs_a.squeeze(0).numpy(), obs_b.squeeze(0).numpy(), sa, sb

    def _apply_reset_noise(self) -> None:
        """Perturb both spawns. Training and evaluation must share this noise, or
        a deterministic evaluation is out of distribution and reports the wrong
        thing — the failure mode that made parkour evaluations misleading."""
        cfg = self.cfg
        for side in self.scene.sides:
            qa = side.base_qposadr
            if cfg.pos_noise > 0:
                self.data.qpos[qa:qa + 2] += self._rng.uniform(
                    -cfg.pos_noise, cfg.pos_noise, size=2)
            if cfg.yaw_noise > 0:
                w, x, y, z = self.data.qpos[qa + 3:qa + 7]
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                yaw += self._rng.uniform(-cfg.yaw_noise, cfg.yaw_noise)
                self.data.qpos[qa + 3:qa + 7] = [
                    np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            if cfg.joint_noise > 0:
                self.data.qpos[side.joint_qposadr] += self._rng.normal(
                    0.0, cfg.joint_noise, size=side.robot.n_joints)

    # --- gym-ish API --------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.data.qpos[:] = self.scene.home_qpos
        self.data.qvel[:] = 0.0
        self._apply_reset_noise()
        mujoco.mj_forward(self.model, self.data)

        n = self.action_dim
        self._prev_action = {"a/": torch.zeros(1, n), "b/": torch.zeros(1, n)}
        self.step_count = torch.zeros(1, dtype=torch.long)

        obs_a, obs_b, sa, sb = self._observations()
        self._prev_radius = {
            "a/": torch.linalg.norm(sa.base_pos[:, :2], dim=-1),
            "b/": torch.linalg.norm(sb.base_pos[:, :2], dim=-1),
        }
        return obs_a, obs_b

    def step(self, action_a, action_b):
        actions = {}
        for key, side, raw in (("a/", self.scene.a, action_a),
                               ("b/", self.scene.b, action_b)):
            act = np.clip(np.asarray(raw, dtype=np.float64), -1.0, 1.0)
            self.data.ctrl[side.actuator_ids] = (
                side.robot.home_joint_qpos + self.action_scale * act)
            actions[key] = torch.tensor(act, dtype=torch.float32).unsqueeze(0)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        qpos, qvel = self._tensors()
        sa, sb = extract_duel_state(qpos, qvel, self.scene)
        terminated, truncated, lost_a, lost_b, outcome = compute_termination(
            sa, sb, self.scene.a.robot, self.scene.b.robot,
            self.step_count, self.cfg, self.term_cfg)

        rew_a, comps_a = compute_reward(
            sa, sb, self._prev_radius["b/"], lost_a, lost_b, actions["a/"],
            self.cfg.ring_radius, self.reward_cfg, self.cfg.shaping_scale)
        rew_b, comps_b = compute_reward(
            sb, sa, self._prev_radius["a/"], lost_b, lost_a, actions["b/"],
            self.cfg.ring_radius, self.reward_cfg, self.cfg.shaping_scale)

        self._prev_action = actions
        self._prev_radius = {
            "a/": torch.linalg.norm(sa.base_pos[:, :2], dim=-1),
            "b/": torch.linalg.norm(sb.base_pos[:, :2], dim=-1),
        }
        obs_a, obs_b, _, _ = self._observations()

        info = {
            "outcome": int(outcome.item()),
            "rewards": {"a": float(rew_a.item()), "b": float(rew_b.item())},
            "reward_components_a": {k: float(v.item()) for k, v in comps_a.items()},
            "reward_components_b": {k: float(v.item()) for k, v in comps_b.items()},
        }
        return ((obs_a, obs_b), (float(rew_a.item()), float(rew_b.item())),
                bool(terminated.item()), bool(truncated.item()), info)

    def render(self, camera: str = "corner") -> np.ndarray:
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        return render_frame(self.model, self.data, self.scene, camera=camera,
                            renderer=self._renderer)
