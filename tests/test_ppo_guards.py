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

from automataleague_sumo.training.ppo import _MAX_LOG_WEIGHT, BoundedRatioPPOLoss  # noqa: E402

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


def test_bounded_ratio_loss_stays_finite_on_the_batch_that_killed_the_run():
    torch.manual_seed(0)
    loss_module = _build_loss(BoundedRatioPPOLoss)
    batch = _batch(loss_module, stale_log_prob=-500.0, advantage=-1.0)
    out = loss_module(batch)
    assert torch.isfinite(out["loss_objective"])
    assert torch.isfinite(out["loss_critic"])
    assert loss_module.pop_saturated() == BATCH


def test_bounded_ratio_loss_is_identical_inside_the_trust_region():
    """The clamp must be inert for any update PPO would actually take.

    A backstop that changed ordinary updates would be a silent second clip, and
    its effect would be indistinguishable from a hyperparameter change.
    """
    torch.manual_seed(0)
    stock = _build_loss(ClipPPOLoss)
    torch.manual_seed(0)
    bounded = _build_loss(BoundedRatioPPOLoss)

    torch.manual_seed(1)
    batch = _batch(stock, stale_log_prob=-3.0, advantage=-1.0)
    a = stock(batch.clone())["loss_objective"]
    b = bounded(batch.clone())["loss_objective"]
    assert torch.equal(a, b), f"clamp changed a healthy update: {a} vs {b}"
    assert bounded.pop_saturated() == 0


def test_pop_saturated_resets():
    """A count that never resets turns into a cumulative total in the metrics,
    where a single early spike would read as a permanently unhealthy run."""
    torch.manual_seed(0)
    loss_module = _build_loss(BoundedRatioPPOLoss)
    loss_module(_batch(loss_module, stale_log_prob=-500.0, advantage=-1.0))
    assert loss_module.pop_saturated() == BATCH
    assert loss_module.pop_saturated() == 0


def test_max_log_weight_cannot_overflow_float32():
    """exp() of the bound must be representable, or the clamp achieves nothing."""
    assert torch.exp(torch.tensor(_MAX_LOG_WEIGHT, dtype=torch.float32)).isfinite()
    # And it must stay far outside any trust region a run could configure, so it
    # never becomes a clip in disguise.
    assert _MAX_LOG_WEIGHT > 10.0
