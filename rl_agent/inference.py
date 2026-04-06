"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# from my_env_v4 import MyEnvV4Action, MyEnvV4Env  # Not needed, using HTTP
import requests

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Set OPENAI_API_KEY for the OpenAI client
os.environ["OPENAI_API_KEY"] = API_KEY

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "phase_i_dosing")   # easy | medium | hard
BENCHMARK = os.getenv("ENV_NAME", "gym_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = 0.2
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: based on trial success
MAX_TOTAL_REWARD = 1.0  # Assuming score is normalized

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI clinical pharmacologist running a Phase I drug trial.
    You see the current trial state and must decide the next dose.

    Rules you MUST follow:
    - Start low, escalate slowly (max 50% increase per step)
    - If DLT count >= 2 in a cohort of 3, DE-ESCALATE immediately
    - If DLT count >= 2 in a cohort of 6, STOP escalating
    - Never exceed 50 mg/kg
    - Target the dose where 0-1 patients have DLTs (this is the RP2D)

    You MUST respond with ONLY valid JSON in this exact format:
    {"next_dose": 5.0, "cohort_size": 3, "escalate": true}

    No explanation. No markdown. Just the JSON object.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs: dict, step: int) -> str:
    return textwrap.dedent(f"""
        Step {step} — Current trial status:
        - Current dose: {obs.get('dose_level', 0):.1f} mg/kg
        - Patients with serious side effects: {obs.get('dlt_count', 0)}/{obs.get('cohort_size', 3)}
        - Peak blood concentration: {obs.get('plasma_conc', 0):.2f} mg/L
        - Liver stress: {obs.get('hepatocyte_signal', 0):.0%}
        - Kidney function: {obs.get('renal_signal', 1):.0%}
        - Immune reaction: {obs.get('immune_signal', 0):.0%}
        - Doctor says: {obs.get('doctor_recommendation', '')}

        What is your next action?
        Respond with ONLY JSON: {{"next_dose": float, "cohort_size": int, "escalate": bool}}
    """).strip()


def get_model_message(client: OpenAI, obs: dict, step: int) -> str:
    user_prompt = build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"next_dose": 2.0, "cohort_size": 3, "escalate": true}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"next_dose": 2.0, "cohort_size": 3, "escalate": true}'


def env_reset() -> dict:
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
    resp = requests.post(f"{ENV_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
    resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action(response_text: str, current_dose: float) -> dict:
    """
    Parse LLM response into action dict.
    Falls back to safe default if LLM response is malformed.
    """
    try:
        # Strip markdown code fences if present
        text = response_text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        action = json.loads(text)
        # Validate fields exist and are reasonable
        next_dose = float(action.get("next_dose", current_dose * 1.3))
        next_dose = max(0.1, min(50.0, next_dose))
        return {
            "next_dose":    next_dose,
            "cohort_size":  int(action.get("cohort_size", 3)),
            "escalate":     bool(action.get("escalate", True)),
        }
    except Exception:
        # Safe fallback: small escalation
        return {
            "next_dose":   min(current_dose * 1.3, 50.0),
            "cohort_size": 3,
            "escalate":    True,
        }


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env_reset()
        obs = result
        current_dose = obs.get("dose_level", 1.0)

        for step in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            message = get_model_message(client, obs, step)
            action = parse_action(message, current_dose)
            result = env_step(action)
            obs = result
            current_dose = obs.get("dose_level", current_dose)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=json.dumps(action),
                     reward=reward, done=done, error=None)
            if done:
                break

        score = min(max(sum(rewards) / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


if __name__ == "__main__":
    main()
