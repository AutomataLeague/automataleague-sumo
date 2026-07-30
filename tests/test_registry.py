import pytest

from automataleague_sumo import list_environments, make_env
from automataleague_sumo.envs.registry import ENVIRONMENTS, get_env_spec
from automataleague_sumo.envs.sumo.sumo_cpu import SumoEnvCPU


def test_sumo_1_is_registered():
    assert "sumo-1" in ENVIRONMENTS
    spec = get_env_spec("sumo-1")
    assert spec.season == 0
    assert spec.n_levels == 5


def test_per_level_tuples_cover_every_level():
    spec = get_env_spec("sumo-1")
    for name in ("action_scale_by_level", "shaping_scale_by_level", "opponent_by_level"):
        assert len(getattr(spec, name)) == spec.n_levels, name


def test_shaping_anneals_downward_across_the_curriculum():
    scales = get_env_spec("sumo-1").shaping_scale_by_level
    assert scales[0] == 1.0
    assert scales[-1] < scales[0]
    assert all(b <= a for a, b in zip(scales, scales[1:])), "shaping must not increase"


def test_opponent_progression_matches_the_curriculum():
    assert get_env_spec("sumo-1").opponent_by_level == (
        "zero", "zero", "frozen", "self", "pool")


def test_config_carries_the_level_settings():
    spec = get_env_spec("sumo-1")
    cfg = spec.config(3)
    assert cfg.level == 3
    assert cfg.opponent == spec.opponent_by_level[3]
    assert cfg.shaping_scale == spec.shaping_scale_by_level[3]
    assert cfg.action_scale == spec.action_scale_by_level[3]
    assert cfg.ring_radius == spec.ring_radius


def test_out_of_range_level_raises():
    with pytest.raises(ValueError, match="out of range"):
        get_env_spec("sumo-1").config(5)
    with pytest.raises(ValueError, match="out of range"):
        get_env_spec("sumo-1").config(-1)


def test_overrides_are_applied_and_validated():
    cfg = get_env_spec("sumo-1").config(0, ring_radius=2.0)
    assert cfg.ring_radius == 2.0
    with pytest.raises(ValueError, match="Unknown SumoConfig field"):
        get_env_spec("sumo-1").config(0, radius=2.0)


def test_unknown_env_id_raises():
    with pytest.raises(ValueError, match="Unknown environment"):
        get_env_spec("sumo-99")


def test_list_environments_filters_by_season():
    assert [s.env_id for s in list_environments()] == ["sumo-1"]
    assert list_environments(season=0)
    assert list_environments(season=99) == []


def test_make_env_builds_a_working_cpu_env():
    env = make_env("sumo-1", level=0, backend="cpu")
    assert isinstance(env, SumoEnvCPU)
    # Pin the level actually reached the config, not just "some env was built" —
    # otherwise a make_env that silently ignores `level` still passes this test.
    assert env.cfg.level == 0
    obs_a, obs_b = env.reset(seed=0)
    assert obs_a.shape == (env.observation_dim,)
    assert obs_b.shape == obs_a.shape


def test_make_env_forwards_robot_and_opponent_robot_without_crossing_them(monkeypatch):
    # Only one robot ("g1") is registered today, so a real two-robot duel can't
    # distinguish "opponent_robot forwarded correctly" from "robot passed to both
    # slots". Stub the CPU env to capture what make_env actually forwards.
    captured = {}

    class FakeEnv:
        def __init__(self, robot, opponent_robot, cfg, reward_cfg, term_cfg):
            captured["robot"] = robot
            captured["opponent_robot"] = opponent_robot

    monkeypatch.setattr(
        "automataleague_sumo.envs.sumo.sumo_cpu.SumoEnvCPU", FakeEnv)
    make_env("sumo-1", robot="g1", opponent_robot="other-bot", level=0, backend="cpu")
    assert captured["robot"] == "g1"
    assert captured["opponent_robot"] == "other-bot"


def test_make_env_defaults_to_the_hardest_level():
    spec = get_env_spec("sumo-1")
    env = make_env("sumo-1", backend="cpu")
    assert env.cfg.level == spec.n_levels - 1


def test_importing_the_registry_does_not_import_mujoco_warp():
    """mujoco-warp is a GPU-only dependency. If the registry pulled it in at import
    time, a CPU-only checkout could not so much as list the available environments.

    Run in a fresh interpreter on purpose. Asserting on this process's sys.modules
    would pass for the wrong reason in both directions: trivially where mujoco-warp
    is not installed at all, and by accident of test ordering where it is.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from automataleague_sumo.envs import registry\n"
         "registry.list_environments()\n"
         "assert 'mujoco_warp' not in sys.modules, sorted(sys.modules)[:0] or 'imported'\n"
         "print('clean')"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_cpu_backend_rejects_warp_only_backend_kwargs():
    """backend_kwargs are the backend's own constructor arguments. Accepting and
    ignoring them on CPU would let a typo'd Warp run silently execute one duel."""
    with pytest.raises(ValueError, match="Warp-only"):
        make_env("sumo-1", backend="cpu", backend_kwargs={"njmax": 600})


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        make_env("sumo-1", backend="quantum")


def test_num_envs_on_cpu_backend_raises_instead_of_being_ignored():
    """The CPU backend is a single renderable duel. Silently ignoring num_envs
    would let `num_envs=2048, backend="cpu"` quietly hand back one duel instead
    of the 2048 the caller asked for, and a benchmark would read 2048x too fast."""
    with pytest.raises(ValueError, match="does not support"):
        make_env("sumo-1", backend="cpu", num_envs=2048)
