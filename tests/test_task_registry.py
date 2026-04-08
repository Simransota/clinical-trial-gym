"""Validator-facing tests for task and grader discovery."""

from tasks import TASKS
from server.graders import TASK_GRADERS


def test_three_tasks_are_declared():
    assert len(TASKS) >= 3
    assert {task["id"] for task in TASKS} >= {
        "phase_i_dosing",
        "allometric_scaling",
        "combo_ddi",
    }


def test_each_task_has_a_grader():
    for task in TASKS:
        grader = TASK_GRADERS.get(task["id"])
        assert grader is not None
        assert callable(grader)
