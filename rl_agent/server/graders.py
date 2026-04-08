"""Explicit task grader entry points for RxGym."""

from __future__ import annotations

from .rl_agent_environment import RlAgentEnvironment


def grade_phase_i_dosing(env: RlAgentEnvironment) -> float:
    return float(env.grade_episode("phase_i_dosing"))


def grade_allometric_scaling(env: RlAgentEnvironment) -> float:
    return float(env.grade_episode("allometric_scaling"))


def grade_combo_ddi(env: RlAgentEnvironment) -> float:
    return float(env.grade_episode("combo_ddi"))


TASK_GRADERS = {
    "phase_i_dosing": grade_phase_i_dosing,
    "allometric_scaling": grade_allometric_scaling,
    "combo_ddi": grade_combo_ddi,
}
