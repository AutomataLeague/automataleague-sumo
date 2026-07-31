"""Configuration for the sumo task.

These values are the single source of truth for both the *scene geometry* (ring
radius, platform height, spawn placement) and the *task logic* (out-of-ring test,
reward normalization). The scene builder and the env read the same config, so
they cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Who drives side B. This is not a difficulty setting: the difficulty of a
# competitive game is the opponent, and under self-play it tracks the policy's
# own strength, which is why this task needs no authored difficulty ladder.
#   "self" - the current policy drives both robots. The real game.
#   "zero" - a passive dummy that cannot lose. Only for bootstrapping standing on
#            a fresh robot, since two robots that both collapse in 1.5 s produce
#            nothing but simultaneous losses and no win signal to learn from.
OPPONENT_MODES: tuple[str, ...] = ("self", "zero")


@dataclass
class SumoConfig:
    # --- arena geometry (metres) ---
    # The human dohyo is 2.275 m in radius for a ~1.8 m wrestler, a ratio near
    # 1.26. Applied to the G1's 1.32 m standing height that gives 1.66 m; we round
    # down to keep duels short and contact frequent.
    ring_radius: float = 1.5
    platform_height: float = 0.3    # raised dohyo, so falling out is physical

    # --- spawn ---
    # Both robots start diametrically opposite at spawn_frac * ring_radius,
    # facing each other along the x axis.
    spawn_frac: float = 0.6

    # --- who drives side B (see OPPONENT_MODES) ---
    opponent: str = "self"

    # --- action range: q_target = home + action_scale * action, action in [-1,1].
    # None => use the robot's default. tools/measure_reach.py must MEASURE this,
    # following the parkour lesson that capability needs margin over what the task
    # demands and must never be guessed. ---
    action_scale: float | None = None

    # --- control rate: model timestep is 0.004 s, so frame_skip 5 => 50 Hz ---
    frame_skip: int = 5

    # --- push perturbations ---
    # Random horizontal impulses on each base, unobserved, every
    # `push_interval_steps` control steps, magnitude U(0, push_speed) m/s with an
    # independent random heading per robot.
    #
    # A disturbance the policy can see coming is a control input, not a
    # disturbance, so these are deliberately absent from the observation. Without
    # them, balance is a held pose: the first policy trained without them survived
    # a full 750-step episode and still fell to a 0.5 m/s shove in 6 of 6 seeds.
    # Retraining with them gave 6 of 6 at 1.0 m/s.
    push_interval_steps: int = 75
    push_speed: float = 1.0

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
                f"platform_height must be >= 0, got {self.platform_height}")
        if not 0.0 < self.spawn_frac < 1.0:
            raise ValueError(f"spawn_frac must be in (0, 1), got {self.spawn_frac}")
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {self.frame_skip}")
        if self.opponent not in OPPONENT_MODES:
            raise ValueError(
                f"Unknown opponent mode '{self.opponent}'. Valid: {list(OPPONENT_MODES)}")
        if self.push_interval_steps < 0:
            raise ValueError(
                f"push_interval_steps must be >= 0, got {self.push_interval_steps}")
        for name in ("pos_noise", "yaw_noise", "joint_noise", "push_speed"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if (self.push_interval_steps > 0) != (self.push_speed > 0):
            raise ValueError(
                f"push perturbations are half-configured: push_interval_steps="
                f"{self.push_interval_steps}, push_speed={self.push_speed}. Either "
                f"both are positive or both are zero — a schedule with no magnitude "
                f"and a magnitude with no schedule are both silent no-ops, which "
                f"looks exactly like push training that is switched on.")
        # pos_noise is applied independently on x and y, so the worst-case spawn is
        # the corner of a pos_noise square offset outward along one axis, not the
        # one-dimensional sum of the two budgets.
        worst_case = math.hypot(self.spawn_radius + self.pos_noise, self.pos_noise)
        if worst_case >= self.ring_radius:
            raise ValueError(
                f"spawn_radius {self.spawn_radius:.2f} + pos_noise {self.pos_noise:.2f} "
                f"(applied per-axis) reaches the rim at {self.ring_radius:.2f}: "
                f"worst-case spawn radius is {worst_case:.3f} — a robot could spawn out")

    @property
    def spawn_radius(self) -> float:
        """Distance of each spawn point from the ring centre."""
        return self.spawn_frac * self.ring_radius

    @property
    def dummy_opponent(self) -> bool:
        """True when side B is scenery rather than a contestant.

        A zero-action humanoid collapses under its own weight in about 1.2 s, so a
        dummy that could lose would hand the learner a free win roughly 60 steps
        into every episode. It therefore cannot lose at all. Derived from the
        opponent mode rather than configured separately: a handicap that can be
        set independently of who is playing is a handicap that can be left on by
        accident against a real opponent.
        """
        return self.opponent == "zero"


@dataclass
class RewardConfig:
    """Reward weights, all denominated in the same units as ``win``.

    Every weight is the value of that term **over a whole episode**, so each one
    reads directly against ``win`` and against the others. That is the property
    whose absence caused the one reward bug this project has had: with per-step
    weights, a term worth 0.3 a step quietly outscored a terminal +10 by thirteen
    times over a 750-step episode, and nothing in the numbers said so.

    Two kinds of term, treated differently inside ``compute_reward``:

    * ``push`` is a *delta*. It pays the change in the opponent's radius each
      step, so it telescopes to the total change across the episode and is
      already an episode-scale quantity. It cannot be farmed by shoving them out
      and letting them back in.
    * everything else is a *rate*, paid every step, and is divided by the episode
      horizon so its weight means "value if this held for the whole episode".
    """

    # Terminal and zero sum: +win when the opponent goes out or down, -win when
    # you do. The actual objective, and deliberately the largest number here.
    win: float = 10.0

    # Driving the opponent from the centre all the way over the rim. A dense
    # proxy for `win`, worth about a third as much.
    push: float = 3.0

    # A whole episode spent surviving, and a whole episode spent pinned against
    # the rim. Both small against `win`: they exist to give a gradient before
    # either side can win at all, not to be an alternative way of scoring.
    alive: float = 2.0
    centre: float = 1.0

    # Regularizers, in the same episode units.
    action: float = 0.5
    joint_vel: float = 0.5

    def __post_init__(self):
        if self.win <= 0:
            raise ValueError(f"win must be > 0, got {self.win}")
        for name in ("push", "alive", "centre", "action", "joint_vel"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        # Shaping exists to bootstrap, not to compete with the objective. If the
        # whole shaping budget can outscore a win, the highest-scoring policy is
        # one that never tries to win — which is exactly what happened here once,
        # and took a training run and a video to notice.
        budget = self.push + self.alive + self.centre
        if budget > self.win:
            raise ValueError(
                f"the shaping budget ({budget:.1f} = push {self.push} + alive "
                f"{self.alive} + centre {self.centre}) exceeds win ({self.win}). "
                f"Every weight here is a whole-episode value, so that means farming "
                f"the shaping beats winning the duel.")


@dataclass
class TerminationConfig:
    # A side is "down" when its base drops below this fraction of nominal height
    # above the platform, or tilts past max_tilt_deg.
    fall_height_frac: float = 0.55
    max_tilt_deg: float = 50.0
    # 750 steps at 50 Hz = 15 s, then the duel is a draw.
    max_episode_steps: int = 750
