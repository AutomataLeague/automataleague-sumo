"""The numerical guards in the PPO loop, each reproducing a failure that happened.

Three runs of this project were destroyed by non-finite numbers, and every time
the guard that existed checked a quantity adjacent to the one that did the
damage. These tests construct the failing condition directly rather than waiting
for a 300M-frame run to hit it.
"""

from __future__ import annotations

import pytest
import torch

# torchrl and tensordict ship in the `train` extra, so a CPU-only checkout skips
# these rather than failing collection — the same contract test_sumo_warp.py has
# for mujoco-warp.
pytest.importorskip("torchrl", reason="torchrl ships in the `train` extra")

from tensordict import TensorDict  # noqa: E402
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule  # noqa: E402
from torchrl.data import Bounded, Composite  # noqa: E402
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator  # noqa: E402
from torchrl.objectives import ClipPPOLoss  # noqa: E402

from automataleague_sumo.training.ppo import (  # noqa: E402
    _LOG_WEIGHT_ALARM,
    _SATURATION_ABORT_FRACTION,
    SaturationCountingPPOLoss,
)

OBS_DIM, ACT_DIM, BATCH = 6, 3, 16


def _build_loss(loss_cls):
    """A minimum viable PPO loss over a TanhNormal actor, matching the real one."""
    policy_mlp = torch.nn.Sequential(
        MLP(in_features=OBS_DIM, out_features=ACT_DIM, num_cells=[8]),
        AddStateIndependentNormalScale(ACT_DIM, scale_lb=1e-8),
    )
    actor = ProbabilisticActor(
        TensorDictModule(policy_mlp, in_keys=["observation"],
                         out_keys=["loc", "scale"]),
        spec=Composite(action=Bounded(low=-torch.ones(ACT_DIM),
                                      high=torch.ones(ACT_DIM))),
        in_keys=["loc", "scale"], distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True, default_interaction_type="random",
    )
    critic = ValueOperator(
        MLP(in_features=OBS_DIM, out_features=1, num_cells=[8]),
        in_keys=["observation"])
    return loss_cls(actor_network=actor, critic_network=critic,
                    clip_epsilon=0.2, normalize_advantage=False)


def _batch(loss_module, *, stale_log_prob, advantage):
    """A collected batch whose stored log prob is `stale_log_prob` off the truth.

    The importance ratio is exp(current - stored), so making the stored value
    absurdly small is exactly what a policy that has moved a long way looks like
    to the loss, without needing to actually train one into that state.
    """
    obs = torch.randn(BATCH, OBS_DIM)
    action = torch.tanh(torch.randn(BATCH, ACT_DIM))
    key = loss_module.tensor_keys.sample_log_prob
    return TensorDict({
        "observation": obs,
        "action": action,
        key: torch.full((BATCH,), stale_log_prob),
        "advantage": torch.full((BATCH, 1), advantage),
        "value_target": torch.zeros(BATCH, 1),
        "next": TensorDict({
            "observation": obs,
            "reward": torch.zeros(BATCH, 1),
            "done": torch.zeros(BATCH, 1, dtype=torch.bool),
            "terminated": torch.zeros(BATCH, 1, dtype=torch.bool),
        }, batch_size=[BATCH]),
    }, batch_size=[BATCH])


def test_stock_clip_ppo_loss_overflows_on_a_negative_advantage():
    """The bug, in the upstream class. If this ever passes, the fix is obsolete.

    PPO takes the pessimistic min of the clipped and unclipped branches. With a
    NEGATIVE advantage the unclipped branch is the smaller one, so the clip that
    is supposed to bound the update does not apply and an overflowing ratio goes
    straight into the objective.
    """
    torch.manual_seed(0)
    loss_module = _build_loss(ClipPPOLoss)
    batch = _batch(loss_module, stale_log_prob=-500.0, advantage=-1.0)
    assert not torch.isfinite(loss_module(batch)["loss_objective"])


def test_stock_clip_ppo_loss_survives_the_same_ratio_when_the_advantage_is_positive():
    """Pins WHY it overflows, so the test above cannot pass for an unrelated reason.

    Same ratio, opposite advantage sign: here min() picks the clipped branch and
    the objective is finite. The sign of the advantage is the whole mechanism.
    """
    torch.manual_seed(0)
    loss_module = _build_loss(ClipPPOLoss)
    batch = _batch(loss_module, stale_log_prob=-500.0, advantage=1.0)
    assert torch.isfinite(loss_module(batch)["loss_objective"])


def test_counting_never_changes_the_loss():
    """The regression test for the worst bug this loop has had.

    A previous version of this class clamped log_weight to bound the ratio. It
    bounded it, and because torch.clamp has zero gradient outside its range it
    also deleted the policy gradient on every saturated sample. The entropy bonus
    ran unopposed, the actor's scale collapsed 0.36 -> 0.0027 and 30M frames were
    trained on garbage while train/reward ROSE. The class is now observation only,
    and this asserts it on the exact batch that used to be altered.
    """
    for stale in (-500.0, -3.0, 0.0):
        for advantage in (-1.0, 1.0):
            torch.manual_seed(0)
            stock = _build_loss(ClipPPOLoss)
            torch.manual_seed(0)
            counting = _build_loss(SaturationCountingPPOLoss)
            torch.manual_seed(1)
            batch = _batch(stock, stale_log_prob=stale, advantage=advantage)
            a = stock(batch.clone())["loss_objective"]
            b = counting(batch.clone())["loss_objective"]
            assert torch.equal(a, b), (
                f"loss changed at stale={stale} advantage={advantage}: {a} vs {b}")


def test_the_counter_sees_the_runaway_it_must_not_fix():
    """Counting has to fire on exactly the batch the loss cannot survive.

    A counter that stayed silent here would leave the divergence abort blind, and
    the abort is the only thing standing between a diverged policy and 30M frames
    of garbage checkpoints.
    """
    torch.manual_seed(0)
    loss_module = _build_loss(SaturationCountingPPOLoss)
    batch = _batch(loss_module, stale_log_prob=-500.0, advantage=-1.0)
    # Unchanged from stock: still non-finite. The training loop skips it.
    assert not torch.isfinite(loss_module(batch)["loss_objective"])
    assert loss_module.pop_saturated() == BATCH


def test_the_counter_stays_quiet_on_a_healthy_batch():
    """Otherwise the divergence abort would kill every run it was added to."""
    torch.manual_seed(0)
    loss_module = _build_loss(SaturationCountingPPOLoss)
    loss_module(_batch(loss_module, stale_log_prob=-3.0, advantage=-1.0))
    assert loss_module.pop_saturated() == 0


def test_pop_saturated_resets():
    """A count that never resets turns into a cumulative total in the metrics,
    where a single early spike would read as a permanently unhealthy run."""
    torch.manual_seed(0)
    loss_module = _build_loss(SaturationCountingPPOLoss)
    loss_module(_batch(loss_module, stale_log_prob=-500.0, advantage=-1.0))
    assert loss_module.pop_saturated() == BATCH
    assert loss_module.pop_saturated() == 0


def test_the_alarm_threshold_is_far_outside_any_trust_region():
    """The alarm must never fire on an update PPO would consider reasonable.

    clip_epsilon is at most a few tenths, so a log ratio of 20 is orders of
    magnitude beyond anything a healthy run produces. A threshold near the clip
    range would make the divergence abort fire on ordinary training.
    """
    assert _LOG_WEIGHT_ALARM > 10.0
    assert 0.0 < _SATURATION_ABORT_FRACTION < 1.0
