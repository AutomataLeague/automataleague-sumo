"""The evaluation contract, and the validator that keeps a tournament honest.

`check_policy` exists to reject submissions that would produce a plausible result
rather than an error. Every test below builds a policy broken in exactly one way
and asserts it is caught, because a validator nobody has watched fail is a
hypothesis.
"""

from __future__ import annotations

import pytest
import torch

from automataleague_sumo.policy import (
    Policy,
    PolicyInfo,
    check_policy,
    load_policy,
    register_loader,
)

OBS, ACT, BATCH = 110, 29, 8


def _info(**kw) -> PolicyInfo:
    base = dict(env_id="sumo-1", robot="g1", algorithm="test", label="t")
    return PolicyInfo(**{**base, **kw})


class Good(Policy):
    """A minimal well-behaved competitor: batched, deterministic, row-independent."""

    def __init__(self):
        self.info = _info()
        torch.manual_seed(0)
        self.w = torch.randn(OBS, ACT) * 0.05

    def act(self, observation):
        return torch.tanh(observation @ self.w)


def _check(policy, **kw):
    check_policy(policy, obs_dim=OBS, act_dim=ACT, batch=BATCH, **kw)


def test_a_well_behaved_policy_passes():
    _check(Good())


def test_the_contract_needs_no_torchrl():
    """A submission may use neither torchrl nor hydra, and the contract has to be
    importable to validate it. Asserted in a subprocess with both blocked."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['torchrl'] = None; sys.modules['hydra'] = None;"
        "import automataleague_sumo.policy as p;"
        "print(p.ACTION_LOW, p.ACTION_HIGH)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "-1.0 1.0" in out.stdout


# --- each of these is a way to be quietly wrong rather than loudly broken ---

def test_wrong_action_width_is_rejected():
    """Silently drives the wrong joints. Nothing downstream would notice."""
    class WrongWidth(Good):
        def act(self, observation):
            return super().act(observation)[:, : ACT - 1]

    with pytest.raises(ValueError, match="expected"):
        _check(WrongWidth())


def test_non_finite_actions_are_rejected():
    """A NaN action reads as a robot that falls over, not as an error."""
    class Nan(Good):
        def act(self, observation):
            out = super().act(observation)
            out[0, 0] = float("nan")
            return out

    with pytest.raises(ValueError, match="non-finite"):
        _check(Nan())


def test_out_of_range_actions_are_rejected_not_clipped():
    """Out of range is a different action scale, so the duel would measure the
    scale rather than the policy. Rejected rather than clipped: clipping hides it,
    and a clamp has already cost this project a 40M-frame run."""
    class TooBig(Good):
        def act(self, observation):
            return super().act(observation) * 1.5

    with pytest.raises(ValueError, match=r"must lie in"):
        _check(TooBig())


def test_a_stochastic_policy_is_rejected():
    """A tournament that cannot be re-run is not a measurement."""
    class Sampling(Good):
        def act(self, observation):
            return torch.tanh(super().act(observation) + 0.1 * torch.randn(
                observation.shape[0], ACT))

    with pytest.raises(ValueError, match="not deterministic"):
        _check(Sampling())


def test_a_policy_that_couples_rows_is_rejected():
    """The subtle one, and the reason this check exists.

    Both robots in a duel are rows of a single call. A policy that normalises
    across the batch makes each robot's action depend on its opponent's
    observation. It is deterministic, correctly shaped, in range, finite, and
    completely meaningless as a duel. BatchNorm in train mode does exactly this.
    """
    class BatchCoupled(Good):
        def act(self, observation):
            centred = observation - observation.mean(dim=0, keepdim=True)
            return torch.tanh(centred @ self.w)

    with pytest.raises(ValueError, match="couples rows"):
        _check(BatchCoupled())


def test_policy_info_records_what_a_leaderboard_must_not_mix():
    """Correcting the collision model made policies from either side of the change
    incomparable, with no metric to say so. The version travels with the weights."""
    info = _info(env_version="0.1.0", frames=1_000_000)
    assert (info.env_id, info.robot, info.env_version) == ("sumo-1", "g1", "0.1.0")
    with pytest.raises(Exception):        # frozen: metadata cannot drift after load
        info.env_version = "9.9.9"


def test_an_unknown_format_names_the_registered_ones(tmp_path):
    path = tmp_path / "weird.pt"
    torch.save({"format": "not-a-real-format"}, path)
    with pytest.raises(ValueError, match="No loader registered"):
        load_policy(str(path))


def test_a_third_party_format_can_register_itself(tmp_path):
    """The point of the contract: a SAC or TD3 submission plugs in from outside
    this repo without any change here."""
    register_loader("test-format", lambda path, device: Good())
    path = tmp_path / "third_party.pt"
    torch.save({"format": "test-format"}, path)
    policy = load_policy(str(path))
    _check(policy)
    assert policy.info.env_id == "sumo-1"
