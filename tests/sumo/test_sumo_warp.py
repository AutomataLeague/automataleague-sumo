"""Contract tests for the batched Warp backend. Require a CUDA device + mujoco-warp.

    ./.venv/bin/python -m pytest tests/sumo/test_sumo_warp.py -q -m gpu

Kept small on purpose: each test builds an env, and building one JIT-compiles
MuJoCo-Warp kernels on a cold cache. The shared fixtures below are what keep this
file from costing minutes.
"""

from __future__ import annotations

import pytest
import torch

from automataleague_sumo.envs.registry import get_env_spec

pytestmark = pytest.mark.gpu

mjw = pytest.importorskip("mujoco_warp", reason="mujoco-warp is a GPU-only dependency")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from automataleague_sumo.envs.sumo.state import extract_duel_state  # noqa: E402
from automataleague_sumo.envs.sumo.sumo_warp import SumoEnvWarp  # noqa: E402

N = 8


def _env(level):
    return SumoEnvWarp(robot="g1", num_envs=N, device="cuda:0",
                       cfg=get_env_spec("sumo-1").config(level))


@pytest.fixture(scope="module")
def dummy_env():
    """Level 0: a zero-action dummy opponent, so the policy batch is [N]."""
    return _env(0)


@pytest.fixture(scope="module")
def selfplay_env():
    """Level 3: naive self-play, so the policy batch is [2N]."""
    return _env(3)


# ------------------------------------------------------------------ batch shape

def test_dummy_level_exposes_one_row_per_world(dummy_env):
    assert dummy_env.num_worlds == N
    assert dummy_env.batch_size[0] == N
    assert not dummy_env.two_sided


def test_selfplay_level_exposes_two_rows_per_world(selfplay_env):
    """The [2N] flattening is the whole self-play mechanism: side B's rows are
    ordinary policy rows, so one shared network is both wrestlers."""
    assert selfplay_env.num_worlds == N
    assert selfplay_env.batch_size[0] == 2 * N
    assert selfplay_env.two_sided


@pytest.mark.parametrize("fixture", ["dummy_env", "selfplay_env"])
def test_every_spec_matches_the_batch(fixture, request):
    env = request.getfixturevalue(fixture)
    rows = env.batch_size[0]
    td = env.reset()
    assert td["observation"].shape == (rows, env.observation_spec["observation"].shape[-1])
    assert td["done"].shape == (rows, 1)
    td["action"] = torch.zeros(rows, env.action_spec.shape[-1], device=env.device)
    out = env.step(td)["next"]
    for key in ("reward", "done", "terminated", "truncated", "outcome"):
        assert out[key].shape == (rows, 1), key


# ----------------------------------------------------------- physics integrity

def test_stepping_does_not_diverge(dummy_env):
    """The parkour solver settings, transplanted here, sent a pelvis to z=-4839 m.
    Divergence is silent, so it gets an explicit assertion rather than a comment."""
    env = dummy_env
    td = env.reset()
    for _ in range(150):
        td["action"] = torch.zeros(
            env.batch_size[0], env.action_spec.shape[-1], device=env.device
        ).uniform_(-0.5, 0.5)
        _, td = env.step_and_maybe_reset(td)
    qpos, _ = env._state_tensors()
    assert torch.isfinite(qpos).all()
    sa, sb = extract_duel_state(*env._state_tensors(), env.scene)
    z = torch.cat([sa.base_pos[:, 2], sb.base_pos[:, 2]])
    assert float(z.min()) > -1.0, f"a base fell to z={float(z.min())}"
    assert torch.isfinite(td["observation"]).all()


def test_contacts_stay_inside_the_allocated_buffer(dummy_env):
    """MuJoCo-Warp drops contacts past nconmax instead of raising, which reads as
    two robots passing through each other rather than as an error."""
    head = dummy_env.contact_headroom()
    assert head["active_contacts"] < head["capacity"], head


# ------------------------------------------------------------------- wiring

def test_side_b_is_held_at_home_under_a_dummy_opponent(dummy_env):
    """A dummy level must ignore whatever the policy would have said for side B.

    Asserted through the ctrl buffer rather than through a trajectory, because a
    trajectory difference could equally come from noise: this pins the exact
    mechanism, that side B's ctrl equals its home pose whatever the action was.
    """
    env = dummy_env
    env.reset()
    rows, act_dim = env.batch_size[0], env.action_spec.shape[-1]
    td = env.reset()
    td["action"] = torch.full((rows, act_dim), 0.9, device=env.device)
    env.step(td)

    import warp as wp

    ctrl = wp.to_torch(env._mjw_data.ctrl)
    home_b = env._home_joint["b"]
    got_b = ctrl[:, env._act_cols["b"]]
    assert torch.allclose(got_b, home_b.expand_as(got_b), atol=1e-6)

    # ...and side A must NOT be at home, or the test above would also pass with
    # the action pipeline disconnected entirely.
    got_a = ctrl[:, env._act_cols["a"]]
    assert not torch.allclose(got_a, env._home_joint["a"].expand_as(got_a), atol=1e-3)


def test_the_two_policy_row_blocks_describe_different_robots(selfplay_env):
    """Rows [0:N] are side A and rows [N:2N] are side B of the SAME worlds. If the
    stacking were wrong (say both halves built from side A) self-play would train
    against a mirror of itself and every duel would be a draw by construction."""
    env = selfplay_env
    td = env.reset()
    obs = td["observation"]
    obs_a, obs_b = obs[:N], obs[N:]
    assert not torch.allclose(obs_a, obs_b), "both halves describe the same robot"

    # The opponent-relative block must be antisymmetric: A sees B where B sees A.
    # rel_pos starts at index proprio_dim + 4 (ring block is 4 wide).
    start = env.robot.proprio_dim + 4
    rel_a, rel_b = obs_a[:, start:start + 3], obs_b[:, start:start + 3]
    assert torch.allclose(rel_a.norm(dim=-1), rel_b.norm(dim=-1), atol=1e-3), (
        "the separation between the two robots must be the same seen from either side")


def test_done_is_shared_by_both_rows_of_a_world(selfplay_env):
    """A duel ends for both contestants at once."""
    env = selfplay_env
    td = env.reset()
    for _ in range(60):
        td["action"] = torch.zeros(
            env.batch_size[0], env.action_spec.shape[-1], device=env.device)
        out = env.step(td)["next"]
        done = out["done"].squeeze(-1)
        assert torch.equal(done[:N], done[N:])
        td = env.reset(out.set("_reset", out["done"])) if bool(done.any()) else out


def test_reset_clears_the_stale_contact_flag(selfplay_env):
    """The contact arrays describe the step before the reset. Carrying that into a
    fresh episode would tell a policy it is already in a clinch at spawn, when the
    two spawns are a ring diameter apart."""
    env = selfplay_env
    td = env.reset()
    contact_col = td["observation"][:, -1]
    assert float(contact_col.abs().max()) == 0.0
