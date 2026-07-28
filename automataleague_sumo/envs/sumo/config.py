"""Configuration for the sumo task.

These values are the single source of truth for both the *scene geometry* (ring
radius, platform height, spawn placement) and the *task logic* (out-of-ring test,
reward normalization). The scene builder and the env read the same config, so
they cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# How the opponent side is driven. Only "zero" is implemented in Phase B; the
# rest are wired up with the self-play machinery in Phase C.
OPPONENT_MODES: tuple[str, ...] = ("zero", "frozen", "self", "pool")


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

    # --- curriculum ---
    level: int = 0
    opponent: str = "zero"
    # Multiplies every shaping term (not the sparse win term), so the reward
    # anneals toward the true win condition as the curriculum advances.
    shaping_scale: float = 1.0

    # --- action range: q_target = home + action_scale * action, action in [-1,1].
    # None => use the robot's default. Refined per level by tools/measure_reach.py
    # in Phase C, following the parkour lesson that capability must be measured
    # and must have margin, not be guessed. ---
    action_scale: float | None = None

    # --- control rate: model timestep is 0.004 s, so frame_skip 5 => 50 Hz ---
    frame_skip: int = 5

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
        if self.spawn_radius + self.pos_noise >= self.ring_radius:
            raise ValueError(
                f"spawn_radius {self.spawn_radius:.2f} + pos_noise {self.pos_noise:.2f} "
                f"reaches the rim at {self.ring_radius:.2f} — a robot could spawn out"
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
