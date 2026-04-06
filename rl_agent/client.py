"""
client.py — Python client for the rxgym Clinical Trial environment.
"""
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import RlAgentAction, RlAgentObservation
except ImportError:
    from models import RlAgentAction, RlAgentObservation


class RlAgentEnv(EnvClient[RlAgentAction, RlAgentObservation, State]):
    """
    Client for the rxgym Clinical Trial Gym environment.

    Example:
        with RlAgentEnv(base_url="http://localhost:8000") as client:
            result = client.reset()
            print(result.observation.dose_level)
            print(result.observation.dlt_count)

            result = client.step(RlAgentAction(
                next_dose=5.0,
                cohort_size=3,
                escalate=True
            ))
            print(result.observation.plasma_conc)
            print(result.observation.doctor_recommendation)
    """

    def _step_payload(self, action: RlAgentAction) -> Dict:
        return {
            "next_dose":   action.next_dose,
            "cohort_size": action.cohort_size,
            "escalate":    action.escalate,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RlAgentObservation]:
        obs_data = payload.get("observation", payload)
        observation = RlAgentObservation(
            phase=obs_data.get("phase", "phase_i"),
            cohort_size=obs_data.get("cohort_size", 3),
            dose_level=obs_data.get("dose_level", 0.0),
            plasma_conc=obs_data.get("plasma_conc", 0.0),
            dlt_count=obs_data.get("dlt_count", 0),
            dlt_grade=obs_data.get("dlt_grade", []),
            hepatocyte_signal=obs_data.get("hepatocyte_signal", 0.0),
            immune_signal=obs_data.get("immune_signal", 0.0),
            renal_signal=obs_data.get("renal_signal", 1.0),
            doctor_recommendation=obs_data.get("doctor_recommendation", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
