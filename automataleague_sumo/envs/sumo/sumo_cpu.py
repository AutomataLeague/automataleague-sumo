"""Single-duel sumo on classic MuJoCo (CPU). Steppable and renderable.

Runs the shared task logic at batch size 1. Used for local validation, for
rendering, and for head-to-head evaluation of two checkpoints. The batched
MuJoCo-Warp backend consumes exactly the same task logic modules at batch N,
where N is the number of parallel WORLDS, each holding BOTH robots of one duel.
Only the stacked policy view over both sides is 2N; one robot per world would
build 2N single-robot worlds whose robots never collide.

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
        if self.scene.a.robot.name != self.scene.b.robot.name:
            raise NotImplementedError(
                f"A duel is one robot against the SAME robot (got "
                f"a={self.scene.a.robot.name!r}, b={self.scene.b.robot.name!r}). "
                f"That is a rule of the competition, not a missing feature: a "
                f"league only tests the algorithm if both sides run the same "
                f"hardware, and one shared policy can drive both contestants only "
                f"while their widths match. action_scale, observation_dim and "
                f"action_dim all come from side A's robot here, "
                f"so a mixed duel would silently run at the wrong scale. See "
                f"github.com/AutomataLeague/automataleague-sumo/issues/1."
            )
        self.data = mujoco.MjData(self.model)

        # Per joint, not one number: the legs need a small window for balance and
        # the arms a large one to reach at all. See RobotSpec.joint_scale.
        self.action_scale = self.scene.a.robot.scale_vector(self.cfg.action_scale)
        self._home = {
            side.prefix: torch.tensor(side.robot.home_joint_qpos, dtype=torch.float32)
            for side in self.scene.sides
        }
        self._foot_ids = {side.prefix: side.foot_geom_ids for side in self.scene.sides}
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

    def _foot_positions(self, prefix: str) -> torch.Tensor:
        """``[1, n_feet, 3]`` world positions of one side's foot geoms."""
        return torch.tensor(
            self.data.geom_xpos[self._foot_ids[prefix]], dtype=torch.float32).unsqueeze(0)

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

    def _maybe_push(self) -> None:
        """Shove both bases in a random horizontal direction, on schedule.

        Must match the Warp backend exactly, or an evaluation would be measuring a
        policy under disturbances it never trained against — the same class of
        mismatch that a zero-noise reset creates. See SumoConfig.push_speed.
        """
        cfg = self.cfg
        if cfg.push_interval_steps <= 0:
            return
        step = int(self.step_count.item())
        if step == 0 or step % cfg.push_interval_steps != 0:
            return
        for side in self.scene.sides:
            theta = self._rng.uniform(0.0, 2.0 * np.pi)
            speed = self._rng.uniform(0.0, cfg.push_speed)
            da = side.base_dofadr
            self.data.qvel[da:da + 2] += speed * np.array(
                [np.cos(theta), np.sin(theta)])

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

        self._maybe_push()
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        qpos, qvel = self._tensors()
        sa, sb = extract_duel_state(qpos, qvel, self.scene)
        terminated, truncated, lost_a, lost_b, outcome = compute_termination(
            sa, sb, self._foot_positions("a/"), self._foot_positions("b/"),
            self.scene.a.robot, self.scene.b.robot,
            self.step_count, self.cfg, self.term_cfg)

        horizon = self.term_cfg.max_episode_steps
        rew_a, comps_a = compute_reward(
            sa, sb, self._prev_radius["b/"], lost_a, lost_b, actions["a/"],
            self.cfg.ring_radius, self.reward_cfg, horizon)
        rew_b, comps_b = compute_reward(
            sb, sa, self._prev_radius["a/"], lost_b, lost_a, actions["b/"],
            self.cfg.ring_radius, self.reward_cfg, horizon)

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
