"""Repository-root grader wrappers for validator discovery."""

# Import graders - try absolute import first (Docker/installed mode),
# then relative/direct import (development/validation mode)
try:
    from rl_agent.server.graders import (
        TASK_GRADERS,
        grade_allometric_scaling,
        grade_combo_ddi,
        grade_phase_i_dosing,
    )
except (ImportError, ModuleNotFoundError):
    # Fallback: import directly from rl_agent submodules
    import sys
    from pathlib import Path
    
    # Add rl_agent to path if not already there
    rl_agent_path = Path(__file__).parent / "rl_agent"
    if str(rl_agent_path) not in sys.path:
        sys.path.insert(0, str(rl_agent_path))
    
    from server.graders import (
        TASK_GRADERS,
        grade_allometric_scaling,
        grade_combo_ddi,
        grade_phase_i_dosing,
    )

__all__ = [
    "TASK_GRADERS",
    "grade_phase_i_dosing",
    "grade_allometric_scaling",
    "grade_combo_ddi",
]
