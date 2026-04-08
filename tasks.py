"""Repository-root task registry for RxGym hackathon validation."""

from server.graders import (
    grade_allometric_scaling,
    grade_combo_ddi,
    grade_phase_i_dosing,
)


TASKS = [
    {
        "id": "phase_i_dosing",
        "difficulty": "easy",
        "description": "Single-agent Phase I dose escalation to identify an RP2D from molecule-derived PK/PD and toxicity signals.",
        "grader": grade_phase_i_dosing,
    },
    {
        "id": "allometric_scaling",
        "difficulty": "medium",
        "description": "Translate a preclinical rat dose into a human-equivalent dose and refine it from observed human PK.",
        "grader": grade_allometric_scaling,
    },
    {
        "id": "combo_ddi",
        "difficulty": "hard",
        "description": "Schedule a two-drug regimen while controlling mechanistic CYP-mediated drug-drug interaction risk.",
        "grader": grade_combo_ddi,
    },
]


TASK_IDS = [task["id"] for task in TASKS]
