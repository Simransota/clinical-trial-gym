from rl_agent.server.graders import (
    grade_phase_i_dosing,
    grade_allometric_scaling,
    grade_combo_ddi,
)

TASK_GRADERS = {
    "phase_i_dosing": grade_phase_i_dosing,
    "allometric_scaling": grade_allometric_scaling,
    "combo_ddi": grade_combo_ddi,
}