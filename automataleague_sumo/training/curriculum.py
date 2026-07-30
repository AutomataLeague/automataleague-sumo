"""Curriculum config: train across levels, warm-starting each stage from the last."""

from __future__ import annotations

from dataclasses import dataclass

from automataleague_sumo.envs.registry import get_env_spec


@dataclass
class CurriculumConfig:
    levels: list[int]
    frames_per_level: list[int]
    warm_start: bool = True


def curriculum_from_cfg(cfg) -> CurriculumConfig:
    c = cfg.curriculum
    cur = CurriculumConfig(
        levels=[int(x) for x in c.levels],
        frames_per_level=[int(x) for x in c.frames_per_level],
        warm_start=bool(getattr(c, "warm_start", True)),
    )
    if len(cur.levels) != len(cur.frames_per_level):
        raise ValueError(
            f"curriculum lists must be equal length: levels has {len(cur.levels)} "
            f"entries, frames_per_level has {len(cur.frames_per_level)}")
    if not cur.levels:
        raise ValueError("curriculum.levels is empty — nothing to train")

    spec = get_env_spec(cfg.env.name)
    for level in cur.levels:
        if not 0 <= level < spec.n_levels:
            raise ValueError(
                f"{spec.env_id}: curriculum level {level} out of range "
                f"0..{spec.n_levels - 1}")
    # Warm-starting only transfers if the observation and action widths match, which
    # they do across levels of one env by construction. Going BACKWARDS through the
    # levels is not an error, but it is almost always a typo in the yaml, so say so.
    if cur.warm_start and cur.levels != sorted(cur.levels):
        raise ValueError(
            f"curriculum.levels {cur.levels} is not ascending while warm_start is on. "
            f"Each stage initialises from the previous stage's best policy, so a "
            f"descending schedule would hand a harder-trained policy to an easier "
            f"level. Set warm_start=false if that is really the intent.")
    return cur
