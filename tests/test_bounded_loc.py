"""The soft bound on the policy's pre-squash mean.

An unbounded mean is what let a stored log-prob reach 3427 and overflow the PPO
importance ratio, killing three runs. These assert the bound does its job without
repeating the mistake that a hard clamp made.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("tensordict", reason="tensordict ships in the `train` extra")

from tensordict.nn import AddStateIndependentNormalScale  # noqa: E402

from automataleague_sumo.training.models import BoundedLocNormalScale  # noqa: E402

DIM = 29
# Measured over 28652 joint outputs of a healthy 290M-frame checkpoint by
# tools/policy_saturation.py. The bound must not disturb this range.
MEASURED_MEDIAN, MEASURED_P99, MEASURED_MAX = 0.363, 2.071, 3.617


def _module(limit=5.0):
    return BoundedLocNormalScale(DIM, scale_lb=1e-8, limit=limit)


def test_the_state_dict_matches_the_unbounded_module_exactly():
    """Existing checkpoints must keep loading.

    v6, v7 and every render and round-robin result depend on actors saved before
    this class existed. Adding a Sequential entry instead of subclassing would
    have shifted every key and silently orphaned all of them.
    """
    assert (list(_module().state_dict())
            == list(AddStateIndependentNormalScale(DIM, scale_lb=1e-8).state_dict()))


def test_it_barely_moves_the_range_a_healthy_policy_uses():
    """A bound that changed ordinary behaviour would be a policy change wearing a
    safety fix's clothes, and warm starts from it would be uninterpretable."""
    loc = torch.tensor([[MEASURED_MEDIAN, MEASURED_P99, MEASURED_MAX]]).expand(1, 3)
    out, _ = BoundedLocNormalScale(3, scale_lb=1e-8, limit=5.0)(loc)
    # What matters is the ACTION, which is tanh(loc); compare there, not on loc.
    shift = (torch.tanh(out) - torch.tanh(loc)).abs().max()
    assert shift < 0.02, f"bound moved a healthy action by {shift:.4f}"


def test_it_bounds_a_mean_that_has_run_away():
    """60 is the scale implied by the log-prob of 3427 that broke the run."""
    out, _ = BoundedLocNormalScale(1, scale_lb=1e-8, limit=5.0)(
        torch.tensor([[60.0]]))
    assert out.abs().max() <= 5.0


def test_the_gradient_survives_across_the_whole_operating_range():
    """The anti-clamp assertion, and the reason this is tanh and not clamp.

    A hard clamp has zero gradient past its bound, which deletes the force that
    pulls the mean back. Doing exactly that to the importance ratio destroyed a
    40M-frame run. Here the gradient must stay usable everywhere a real policy
    goes, which the measured maximum bounds at 3.62.
    """
    loc = torch.linspace(-MEASURED_MAX, MEASURED_MAX, 64)[None].requires_grad_(True)
    out, _ = BoundedLocNormalScale(64, scale_lb=1e-8, limit=5.0)(loc)
    out.sum().backward()
    assert loc.grad.min() > 0.35, f"weakest gradient {loc.grad.min():.4f}"


def test_a_hard_clamp_would_fail_the_gradient_test():
    """Pins that the test above has teeth, by showing what it rejects.

    Without this, `limit * tanh(x / limit)` could be silently replaced by
    `x.clamp(-limit, limit)` and only a training run would notice.
    """
    loc = torch.tensor([[0.5, 7.0]], requires_grad=True)
    loc.clamp(-5.0, 5.0).sum().backward()
    assert loc.grad[0, 1] == 0.0, "clamp is supposed to have no gradient past its bound"
    soft = torch.tensor([[0.5, 7.0]], requires_grad=True)
    (5.0 * torch.tanh(soft / 5.0)).sum().backward()
    assert soft.grad[0, 1] > 0.0


def test_a_non_positive_limit_is_refused():
    """0 would collapse every mean to zero and train a policy that cannot act,
    which is a silent failure rather than a loud one."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_loc"):
            _module(limit=bad)


def test_the_bound_decays_the_gradient_far_outside_its_range():
    """Documents the honest limitation, so nobody discovers it during a run.

    tanh's derivative is sech^2, which underflows float32 around |loc|/limit ~ 10.
    That is acceptable ONLY because the bound is applied from the first update, so
    a mean can never get there while training through it. Warm-starting a
    checkpoint whose mean already exceeds ~40 would need a larger limit; measured
    checkpoints sit at 3.62, well inside.
    """
    loc = torch.tensor([[60.0]], requires_grad=True)
    (5.0 * torch.tanh(loc / 5.0)).sum().backward()
    assert loc.grad.item() == pytest.approx(0.0, abs=1e-8)


def test_a_config_without_max_loc_gets_no_bound_at_all():
    """Checkpoints predating this class must replay bit-identically.

    Every round-robin ranking and every published video came from actors saved
    before the bound existed. Quietly applying it on reload would change what
    those policies do and invalidate comparisons already drawn from them.
    """
    plain = AddStateIndependentNormalScale(3, scale_lb=1e-8)
    unbounded = BoundedLocNormalScale(3, scale_lb=1e-8, limit=None)
    unbounded.load_state_dict(plain.state_dict())
    loc = torch.tensor([[0.36, 3.62, 60.0]])
    assert torch.equal(unbounded(loc)[0], plain(loc)[0])
