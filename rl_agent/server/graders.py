"""Explicit task grader entry points for RxGym."""

from __future__ import annotations

from typing import Optional

from .rl_agent_environment import RlAgentEnvironment


def _fresh_env_with_episode(steps: int = 3) -> RlAgentEnvironment:
    """Create a default environment and run a short episode so graders have data."""
    try:
        from ..models import RlAgentAction
    except ImportError:
        from models import RlAgentAction

    # use_llm=False prevents DoctorAgent (OpenAI client) from being created,
    # avoiding slow SSL/DNS calls during grader smoke tests.
    env = RlAgentEnvironment(use_llm=False)
    env.reset()
    for _ in range(steps):
        action = RlAgentAction(next_dose=2.0, cohort_size=3, escalate=True)
        result = env.step(action)
        if result.done:
            break
    return env


def grade_phase_i_dosing(env: Optional[RlAgentEnvironment] = None) -> float:
    if env is None:
        env = _fresh_env_with_episode()
    return float(env.grade_episode("phase_i_dosing"))


def grade_allometric_scaling(env: Optional[RlAgentEnvironment] = None) -> float:
    if env is None:
        env = _fresh_env_with_episode()
    return float(env.grade_episode("allometric_scaling"))


def grade_combo_ddi(env: Optional[RlAgentEnvironment] = None) -> float:
    if env is None:
        env = _fresh_env_with_episode()
    return float(env.grade_episode("combo_ddi"))


TASK_GRADERS = {
    "phase_i_dosing": grade_phase_i_dosing,
    "allometric_scaling": grade_allometric_scaling,
    "combo_ddi": grade_combo_ddi,
}
