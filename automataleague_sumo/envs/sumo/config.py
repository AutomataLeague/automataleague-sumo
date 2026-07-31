"""Configuration for the sumo task.

These values are the single source of truth for both the *scene geometry* (ring
radius, platform height, spawn placement) and the *task logic* (out-of-ring test,
reward normalization). The scene builder and the env read the same config, so
they cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Who drives side B. This is not a difficulty setting: under self-play the
# opponent's strength tracks the policy's own, which is the whole reason this
# task needs no authored difficulty ladder.
#   "zero"  - a passive dummy. Only for bootstrapping standing on a fresh robot.
#   "self"  - the current policy drives both sides. The real game.
#   "pool"  - sampled from a growing set of past snapshots plus the current
#             policy, so improvement is monotone instead of a cycle in which the
#             policy beats its present self by exploiting a hole it then trains
#             away. This is where "stronger and growing" actually comes from.
OPPONENT_MODES: tuple[str, ...] = ("zero", "self", "pool")

# Which loss conditions count against side B. Only relevant to the passive dummy
# used for bootstrapping: it collapses on its own in about 1.2 s (~0.45 m of sag
# against a 0.431 m fall threshold), so under the ordinary rules the learner is
# handed a free win roughly 60 steps in and never has to act.
#   "none"     - side B cannot lose; only the learner's own loss ends the duel,
#                which makes the bootstrap purely "stay upright".
#   "ring_out" - side B loses only by leaving the ring or the platform, so the
#                learner has to actually push it out rather than wait.
#   "any"      - the ordinary rules, and the only legal value against a real
#                opponent.
OPPONENT_LOSS_MODES: tuple[str, ...] = ("none", "ring_out", "any")


@dataclass
class SumoConfig:
    # --- arena geometry (metres) ---
    # The human dohyo is 2.275 m in radius for a ~1.8 m wrestler, a ratio near
    # 1.26. Applied to the G1's 1.32 m standing height that gives 1.66 m; we round
    # down to keep duels short and contact frequent.
    ring_radius: float = 1.5
    platform_height: float = 0.3    # raised dohyo, so falling out is physical
    band_width: float = 0.08        # painted rim band (visual only)

    # --- spawn ---
    # Both robots start diametrically opposite at spawn_frac * ring_radius,
    # facing each other along the x axis.
    spawn_frac: float = 0.6

    # --- who drives side B ---
    # NOT a difficulty level. The difficulty of a competitive game is the
    # opponent, and under self-play it grows with the policy on its own. This
    # selects who is at the other end of the duel, nothing more.
    opponent: str = "self"
    # See OPPONENT_LOSS_MODES. Only meaningful while side B is a dummy; a real
    # opponent must play by the ordinary rules, which __post_init__ enforces.
    opponent_loses_by: str = "any"
    # Multiplies every shaping term, never the sparse win term. Lower means the
    # policy optimizes closer to the actual win condition.
    shaping_scale: float = 1.0

    # --- action range: q_target = home + action_scale * action, action in [-1,1].
    # None => use the robot's default. tools/measure_reach.py must MEASURE this,
    # following the parkour lesson that capability must have margin over what the
    # task demands and must never be guessed. ---
    action_scale: float | None = None

    # --- control rate: model timestep is 0.004 s, so frame_skip 5 => 50 Hz ---
    frame_skip: int = 5

    # --- push perturbations ---
    # Random horizontal impulses on each robot's base, unobserved, applied every
    # `push_interval_steps` control steps with a magnitude drawn from
    # U(0, push_speed) in m/s and a uniformly random heading.
    #
    # Without them, bootstrapping against a passive dummy has no disturbance in it
    # at all: the opponent never makes contact and the only variation inside an
    # episode comes from the reset. The cheapest solution is then one fixed stance,
    # and that is what gets learned. Measured on the first such policy, which held
    # its pose for a full 750-step episode: a 0.5 m/s shove toppled it in 6 of 6
    # seeds within 57 steps. Both default to 0 so a config opts in.
    push_interval_steps: int = 0
    push_speed: float = 0.0

    # --- reset noise. Both backends reset with noise. A zero-noise reset makes
    # evaluation out of distribution relative to training, which in the parkour
    # work produced deterministic evals that disagreed with real performance. ---
    pos_noise: float = 0.10     # metres, uniform on the spawn xy
    yaw_noise: float = 0.25     # radians, uniform on the spawn heading
    joint_noise: float = 0.05   # radians, gaussian on each actuated joint

    def __post_init__(self):
        if self.ring_radius <= 0:
            raise ValueError(f"ring_radius must be > 0, got {self.ring_radius}")
        if self.platform_height < 0:
            raise ValueError(
                f"platform_height must be >= 0, got {self.platform_height}"
            )
        if not 0.0 < self.spawn_frac < 1.0:
            raise ValueError(f"spawn_frac must be in (0, 1), got {self.spawn_frac}")
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")
        if self.opponent not in OPPONENT_MODES:
            raise ValueError(
                f"Unknown opponent mode '{self.opponent}'. Valid: {list(OPPONENT_MODES)}"
            )
        if self.opponent_loses_by not in OPPONENT_LOSS_MODES:
            raise ValueError(
                f"Unknown opponent_loses_by mode '{self.opponent_loses_by}'. "
                f"Valid: {list(OPPONENT_LOSS_MODES)}"
            )
        if self.opponent != "zero" and self.opponent_loses_by != "any":
            raise ValueError(
                f"opponent_loses_by='{self.opponent_loses_by}' handicaps side B, but "
                f"opponent='{self.opponent}' is a real policy-driven contestant, not a "
                f"dummy. A duel where one side plays by different rules is not the game "
                f"being evaluated. Use opponent_loses_by='any' against a real opponent."
            )
        if self.push_interval_steps < 0:
            raise ValueError(
                f"push_interval_steps must be >= 0, got {self.push_interval_steps}")
        # Checked before the half-configured test below, so a negative magnitude
        # reports what is actually wrong with it rather than being described as a
        # missing schedule.
        if self.push_speed < 0:
            raise ValueError(f"push_speed must be >= 0, got {self.push_speed}")
        if (self.push_interval_steps > 0) != (self.push_speed > 0):
            raise ValueError(
                f"push perturbations are half-configured: push_interval_steps="
                f"{self.push_interval_steps}, push_speed={self.push_speed}. Either "
                f"both are positive or both are zero — a schedule with no magnitude "
                f"and a magnitude with no schedule are both silently no-ops, which "
                f"would look exactly like push training that is switched on."
            )
        for name in ("pos_noise", "yaw_noise", "joint_noise", "push_speed"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        # pos_noise is applied independently on x and y (see sumo_cpu.py's
        # _apply_reset_noise), so the true worst-case spawn radius is the corner of
        # a pos_noise x pos_noise square offset outward from spawn_radius along one
        # axis, not the one-dimensional sum of the two budgets.
        worst_case = math.hypot(self.spawn_radius + self.pos_noise, self.pos_noise)
        if worst_case >= self.ring_radius:
            raise ValueError(
                f"spawn_radius {self.spawn_radius:.2f} + pos_noise {self.pos_noise:.2f} "
                f"(applied per-axis) reaches the rim at {self.ring_radius:.2f}: "
                f"worst-case spawn radius is {worst_case:.3f} — a robot could spawn out"
            )

    @property
    def spawn_radius(self) -> float:
        """Distance of each spawn point from the ring centre."""
        return self.spawn_frac * self.ring_radius


@dataclass
class RewardConfig:
    """Weights for the sumo reward (see envs/sumo/rewards.py).

    Only ``win`` is zero sum. The shaping terms are computed per side and are not
    required to cancel; they are what gives each side an informative gradient
    before either has learned to win.
    """

    win: float = 10.0          # terminal, +win on opponent loss, -win on own loss
    center: float = 0.5        # penalty on own squared normalized radius
    push: float = 5.0          # reward for increasing the opponent's radius
    engage: float = 0.3        # reward for facing the opponent, decaying with distance
    engage_range: float = 1.0  # metres; the decay length of the engage term
    alive: float = 0.05        # per-step survival bonus
    action: float = 0.01       # penalty on mean squared action
    joint_vel: float = 0.001   # penalty on mean squared joint velocity


@dataclass
class TerminationConfig:
    # A side is "down" when its base drops below this fraction of nominal height
    # above the platform, or tilts past max_tilt_deg.
    fall_height_frac: float = 0.55
    max_tilt_deg: float = 50.0
    # 750 steps at 50 Hz = 15 s, then the duel is a draw.
    max_episode_steps: int = 750
