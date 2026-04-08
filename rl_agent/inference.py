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
import sys
import math
import time
from typing import List, Optional
from openai import OpenAI

# from my_env_v4 import MyEnvV4Action, MyEnvV4Env  # Not needed, using HTTP
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Set OPENAI_API_KEY for the OpenAI client
if API_KEY:
    os.environ["OPENAI_API_KEY"] = API_KEY

# Create robust requests session with retry strategy
def _create_session_with_retries():
    """Create a requests session with exponential backoff retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Exponential backoff: 1s, 2s, 4s
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Global session for connection pooling
_SESSION = _create_session_with_retries()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "phase_i_dosing")   # easy | medium | hard
BENCHMARK = os.getenv("ENV_NAME", "gym_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = 0.2
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]
MEDICINE_SMILES = os.getenv("MEDICINE_SMILES")
MEDICINE_NAME = os.getenv("MEDICINE_NAME", "user_compound")
SOURCE_SPECIES = os.getenv("SOURCE_SPECIES", "rat")
ANIMAL_DOSE_MGKG = float(os.getenv("ANIMAL_DOSE_MGKG", "8.0"))

# Max possible reward: based on trial success
MAX_TOTAL_REWARD = 1.0  # Assuming score is normalized
USE_LLM_POLICY = os.getenv("USE_LLM_POLICY", "true").lower() in ("1", "true", "yes")
DEBUG_DRUG = os.getenv("DEBUG_DRUG", "0").lower() in ("1", "true", "yes")
DEFAULT_ENV_URL = os.getenv("DEFAULT_ENV_URL", "https://simransota-clinical-trial-env.hf.space")

DEFAULT_DRUGS = {
    "phase_i_dosing": {
        "name": "Acetaminophen",
        "smiles": "CC(=O)NC1=CC=C(O)C=C1",
        "source_species": "rat",
        "animal_dose_mgkg": 10.0,
    },
    "allometric_scaling": {
        "name": "Naproxen",
        "smiles": "CC(C1=CC=CC=C1)C(O)=O",
        "source_species": "rat",
        "animal_dose_mgkg": 8.0,
    },
    "combo_ddi": {
        "name": "Diazepam",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "source_species": "rat",
        "animal_dose_mgkg": 2.0,
    },
}

# Centralized policy configuration (task-aware escalation/hold logic).
POLICY_CONFIG = {
    "phase_i_dosing": {
        "far_mult": (1.6, 2.2),
        "mid_mult": (1.3, 1.5),
        "near_mult": (1.1, 1.25),
        "max_safe_risk": 0.40,
        "refine_band": (0.88, 1.15),  # ratio to target where refinement/stop is preferred
        "late_catchup_ratio": 0.55,
    },
    "allometric_scaling": {
        "far_mult": (1.8, 2.5),
        "mid_mult": (1.3, 1.6),
        "near_mult": (1.05, 1.2),
        "max_safe_risk": 0.45,
        "stop_after_first_aligned": True,
        "align_tolerance": 0.12,
    },
    "combo_ddi": {
        "far_mult": (1.45, 1.8),      # ← bump this
        "mid_mult": (1.18, 1.35),     # ← slight bump
        "near_mult": (1.02, 1.12),
        "max_safe_risk": 0.35,
        "late_catchup_ratio": 0.60,
        "late_catchup_mult": 1.35,    # ← MOST IMPORTANT
    },  
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI clinical pharmacologist optimizing benchmark score across three tasks:
    phase_i_dosing, allometric_scaling, combo_ddi.

    Critical objective:
    - Optimize final benchmark score, not just immediate per-step safety.
    - Avoid low-dose stagnation when clearly under target and safety is clean.
    - Use task-aware logic. Do not reuse one generic small-escalation policy.

    Core trial constraints:
    - Never exceed 50 mg/kg.
    - If DLT count >= 2 or DLT rate > 0.33, de-escalate or stop.
    - If organ risk trends worsen, reduce step size / hold / de-escalate.

    Task intent:
    - phase_i_dosing: move efficiently toward RP2D (~12 mg/kg), then refine/stop in plausible RP2D zone.
    - allometric_scaling: first meaningful dose should be close to HED anchor, then stop.
    - combo_ddi: advance with CYP/DDI caution, but avoid chronic underdosing below effective range.

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
    # Always emit at least one reward value so the parser regex matches even
    # when an episode aborts before any successful env.step() call.
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def debug_log(message: str) -> None:
    """Debug log routed to stderr so benchmark stdout format remains unchanged."""
    if DEBUG_DRUG:
        print(f"[DEBUG] {message}", file=sys.stderr, flush=True)


def build_user_prompt(
    obs: dict,
    step: int,
    task_name: str,
    hed: float,
    dose_history: List[float],
    reward_history: List[float],
    cumulative_dlt_rate: float,
    organ_risk_trend: float,
) -> str:
    steps_remaining = max(0, MAX_STEPS - step + 1)
    return textwrap.dedent(f"""
        Step {step} — Task: {task_name}
        Steps remaining: {steps_remaining}
        HED anchor: {hed:.4f} mg/kg

        Current trial status:
        - Current dose: {obs.get('dose_level', 0):.1f} mg/kg
        - Patients with serious side effects: {obs.get('dlt_count', 0)}/{obs.get('cohort_size', 3)}
        - Peak blood concentration: {obs.get('plasma_conc', 0):.2f} mg/L
        - Liver stress: {obs.get('hepatocyte_signal', 0):.0%}
        - Kidney function: {obs.get('renal_signal', 1):.0%}
        - Immune reaction: {obs.get('immune_signal', 0):.0%}
        - Organ risk trend (positive=worse): {organ_risk_trend:.4f}
        - Doctor says: {obs.get('doctor_recommendation', '')}
        - Previous doses: {dose_history[-6:]}
        - Previous rewards: {[round(r, 3) for r in reward_history[-6:]]}

        Decision guidance:
        - If below 40-50% of likely target and recent cohorts are safe, prefer a substantial catch-up increase.
        - Do not stay in a flat low-dose safe regime.
        - Different tasks require different logic and target anchors.
        - phase_i_dosing: once in plausible RP2D zone, use tiny refinement or stop (do not overshoot).
        - allometric_scaling: after first HED-aligned proposal, stop escalation behaviorally.
        - combo_ddi: preserve DDI caution but still reach effective range when safety is clean.

        What is your next action?
        Respond with ONLY JSON: {{"next_dose": float, "cohort_size": int, "escalate": bool}}
    """).strip()


def get_model_message(
    client: Optional[OpenAI],
    obs: dict,
    step: int,
    task_name: str,
    hed: float,
    dose_history: List[float],
    reward_history: List[float],
    cumulative_dlt_rate: float,
    organ_risk_trend: float,
) -> str:
    if client is None:
        raise RuntimeError("OpenAI client is unavailable because no API key was provided.")
    user_prompt = build_user_prompt(
        obs=obs,
        step=step,
        task_name=task_name,
        hed=hed,
        dose_history=dose_history,
        reward_history=reward_history,
        cumulative_dlt_rate=cumulative_dlt_rate,
        organ_risk_trend=organ_risk_trend,
    )
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
    """Reset the environment with retry logic and increased timeout."""
    ENV_URL = os.getenv("ENV_URL", DEFAULT_ENV_URL)
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _SESSION.post(f"{ENV_URL}/reset", timeout=60, allow_redirects=True)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
            if attempt < max_attempts:
                wait_time = min(30, 2 ** (attempt - 1))  # 1s, 2s, 4s, 8s, 16s, 30s
                debug_log(f"env_reset attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"env_reset failed after {max_attempts} attempts: {e}") from e

def env_configure_drug(smiles: str, name: str, source_species: str, animal_dose_mgkg: float) -> dict:
    """Configure drug with retry logic."""
    ENV_URL = os.getenv("ENV_URL", DEFAULT_ENV_URL)
    payload = {
        "smiles": smiles,
        "name": name,
        "source_species": source_species,
        "animal_dose_mgkg": animal_dose_mgkg,
    }
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _SESSION.post(f"{ENV_URL}/drug", json=payload, timeout=120, allow_redirects=True)
            if not resp.ok:
                detail = None
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"/drug failed with HTTP {resp.status_code}: {detail}")
            cfg = resp.json()
            validate_drug_config_or_raise(cfg)
            return cfg
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
            if attempt < max_attempts:
                wait_time = min(30, 2 ** (attempt - 1))
                debug_log(f"env_configure_drug attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"env_configure_drug failed after {max_attempts} attempts: {e}") from e

def env_step(action: dict) -> dict:
    """Execute step with retry logic."""
    ENV_URL = os.getenv("ENV_URL", DEFAULT_ENV_URL)
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _SESSION.post(f"{ENV_URL}/step", json={"action": action}, timeout=60, allow_redirects=True)
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
            if attempt < max_attempts:
                wait_time = min(30, 2 ** (attempt - 1))
                debug_log(f"env_step attempt {attempt} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"env_step failed after {max_attempts} attempts: {e}") from e


def env_close() -> None:
    """Stateless HTTP mode does not require explicit shutdown, but keep the hook for benchmark parity."""
    return None


def resolve_submission_drug(task_name: str) -> dict:
    task_default = DEFAULT_DRUGS.get(task_name, DEFAULT_DRUGS["phase_i_dosing"])
    return {
        "name": os.getenv("MEDICINE_NAME") or task_default["name"],
        "smiles": os.getenv("MEDICINE_SMILES") or task_default["smiles"],
        "source_species": os.getenv("SOURCE_SPECIES") or task_default["source_species"],
        "animal_dose_mgkg": float(os.getenv("ANIMAL_DOSE_MGKG", str(task_default["animal_dose_mgkg"]))),
    }


def normalize_env_response(result: dict) -> tuple[dict, float, bool]:
    """
    Normalize OpenEnv responses that may be either:
      A) flattened observation dict with reward/done fields
      B) {"observation": {...}, "reward": ..., "done": ...}
    """
    if isinstance(result, dict) and isinstance(result.get("observation"), dict):
        obs = result["observation"]
        reward = float(result.get("reward", obs.get("reward", 0.0)))
        done = bool(result.get("done", obs.get("done", False)))
        return obs, reward, done
    obs = dict(result) if isinstance(result, dict) else {}
    reward = float(obs.get("reward", 0.0))
    done = bool(obs.get("done", False))
    return obs, reward, done


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def validate_drug_config_or_raise(drug_cfg: dict) -> None:
    """
    Strict validation: /drug must provide complete, finite PK params.
    No fallback or synthetic PK values are permitted.
    """
    if not isinstance(drug_cfg, dict):
        raise RuntimeError("Invalid /drug response: expected object")
    params = drug_cfg.get("drug_params", {})
    required = {"ka", "F", "CL", "Vc", "Vp", "Q", "PPB", "fu"}
    missing = sorted(required.difference(params.keys() if isinstance(params, dict) else set()))
    if missing:
        raise RuntimeError(f"/drug configuration failed: missing drug_params keys {missing}")

    def _num(name: str) -> float:
        try:
            x = float(params[name])
        except Exception as exc:
            raise RuntimeError(f"/drug configuration failed: non-numeric {name}={params.get(name)!r}") from exc
        if not math.isfinite(x):
            raise RuntimeError(f"/drug configuration failed: non-finite {name}={x}")
        return x

    ka = _num("ka")
    F = _num("F")
    CL = _num("CL")
    Vc = _num("Vc")
    Vp = _num("Vp")
    Q = _num("Q")
    PPB = _num("PPB")
    fu = _num("fu")

    if not (0.0 < F <= 1.0):
        raise RuntimeError(f"/drug configuration failed: F out of bounds ({F})")
    if not (0.0 < fu <= 1.0):
        raise RuntimeError(f"/drug configuration failed: fu out of bounds ({fu})")
    if not (0.0 <= PPB <= 1.0):
        raise RuntimeError(f"/drug configuration failed: PPB out of bounds ({PPB})")
    if ka <= 0.0 or CL <= 0.0 or Vc <= 0.0 or Vp <= 0.0 or Q <= 0.0:
        raise RuntimeError(
            f"/drug configuration failed: expected ka/CL/Vc/Vp/Q > 0, got "
            f"ka={ka}, CL={CL}, Vc={Vc}, Vp={Vp}, Q={Q}"
        )


def derive_fragility_profile(drug_cfg: dict) -> dict:
    """
    Build a small risk profile used to temper early aggressiveness for fragile drugs.
    """
    admet = dict(drug_cfg.get("admet_summary", {}) or {})
    params = dict(drug_cfg.get("drug_params", {}) or {})
    cyp_inhibitions = list(admet.get("cyp_inhibitions", []) or [])
    clintox = float(admet.get("clintox_toxic_prob", 0.0) or 0.0)
    risk = float(admet.get("overall_risk_score", 0.0) or 0.0)
    f_oral = float(admet.get("F_oral", 0.5) or 0.5)
    ka = float(params.get("ka", 1.0) or 1.0)
    hed = float(drug_cfg.get("hed_mgkg", ANIMAL_DOSE_MGKG * 6.0 / 37.0))
    # Composite fragility score.
    low_f_penalty = 0.15 if f_oral < 0.3 else 0.0
    low_ka_penalty = 0.10 if ka < 0.5 else 0.0
    fragility = _clamp(
        0.55 * risk + 0.35 * clintox + 0.10 * min(1.0, len(cyp_inhibitions) / 3.0) + low_f_penalty + low_ka_penalty,
        0.0,
        1.0,
    )
    high_fragility = fragility >= 0.60 or hed >= 8.0
    first_step_cap_mult = 1.35 if high_fragility else 2.0
    return {
        "hed_mgkg": hed,
        "fragility": fragility,
        "high_fragility": high_fragility,
        "first_step_cap_mult": first_step_cap_mult,
        "cyp_inhibitions": cyp_inhibitions,
        "f_oral": f_oral,
        "ka": ka,
    }


def resolve_task_targets(drug_cfg: dict, hed: float) -> dict:
    task_targets = dict(drug_cfg.get("task_targets", {}) or {})
    return {
        "phase_i_dosing": float(task_targets.get("phase_i_dosing", hed * 1.5)),
        "allometric_scaling": float(task_targets.get("allometric_scaling", hed)),
        "combo_ddi": float(task_targets.get("combo_ddi", hed * 1.25)),
    }


def _organ_risk(obs: dict) -> float:
    hep = float(obs.get("hepatocyte_signal", 0.0))
    imm = float(obs.get("immune_signal", 0.0))
    ren = float(obs.get("renal_signal", 1.0))
    return max(hep, imm, 1.0 - ren)


def derive_aggressiveness_modifier(
    task_name: str,
    hed_ratio: float,
    risk: float,
    worsening_risk: float,
    fragility_profile: dict,
    safe_streak: int,
) -> float:
    fragility = float(fragility_profile.get("fragility", 0.0))
    f_oral = float(fragility_profile.get("f_oral", 0.5))
    ka = float(fragility_profile.get("ka", 1.0))
    mod = 1.0
    if fragility < 0.25 and risk < 0.25 and worsening_risk <= 0.0 and hed_ratio < 1.2 and safe_streak >= 2:
        mod *= 1.10
    if fragility > 0.55:
        mod *= 0.88
    if f_oral < 0.30:
        mod *= 0.92
    if ka < 0.50:
        mod *= 0.94
    if hed_ratio > 1.3:
        mod *= 0.90
    if hed_ratio > 1.8:
        mod *= 0.84
    if worsening_risk > 0.04:
        mod *= 0.86
    if task_name == "combo_ddi":
        return _clamp(mod, 0.70, 1.05)
    return _clamp(mod, 0.75, 1.20)


def choose_dynamic_cohort_size(
    task_name: str,
    dlt_count: int,
    risk: float,
    worsening_risk: float,
    hed_ratio: float,
    near_target: bool,
    boundary_detected: bool,
) -> int:
    if dlt_count >= 1:
        return 6
    if hed_ratio < 0.8:
        return 3
    if hed_ratio < 1.2:
        return 6 if worsening_risk > 0.05 else 3
    return 6


def choose_action(
    task_name: str,
    obs: dict,
    current_dose: float,
    prev_actions: List[dict],
    prev_observations: List[dict],
    hed: float,
    cyp_inhibitions: List[str],
    fragility_profile: dict,
    task_targets: dict,
) -> dict:
    """
    Deterministic safety-first controller with task-specific escalation policy.
    This avoids repeated flat actions and keeps behavior aligned to trial logic.
    """
    cfg = POLICY_CONFIG.get(task_name, POLICY_CONFIG["phase_i_dosing"])
    dlt_count = int(obs.get("dlt_count", 0))
    cohort_size = max(1, int(obs.get("cohort_size", 3)))
    dlt_rate = dlt_count / cohort_size
    hep = float(obs.get("hepatocyte_signal", 0.0))
    ren = float(obs.get("renal_signal", 1.0))
    imm = float(obs.get("immune_signal", 0.0))
    doc = str(obs.get("doctor_recommendation", "")).upper()
    risk = _organ_risk(obs)
    prev_risk = _organ_risk(prev_observations[-1]) if prev_observations else risk
    worsening_risk = risk - prev_risk
    prev_reward = float(prev_observations[-1].get("reward", 0.0)) if prev_observations else 0.0
    curr_reward = float(obs.get("reward", prev_reward))
    reward_delta = curr_reward - prev_reward
    safe_streak = 0
    for o in reversed(prev_observations):
        if int(o.get("dlt_count", 0)) == 0 and _organ_risk(o) <= cfg["max_safe_risk"]:
            safe_streak += 1
        else:
            break

    target = max(0.1, float(task_targets.get(task_name, hed)))
    ratio = current_dose / max(target, 1e-6)
    hed_ratio = current_dose / max(hed, 1e-6)
    near_target = ratio >= 0.8
    safe_above_hed = sum(
        1
        for o in prev_observations
        if float(o.get("dose_level", 0.0)) >= hed
        and int(o.get("dlt_count", 0)) == 0
        and _organ_risk(o) <= cfg["max_safe_risk"]
    )

    # Delayed/confirmed boundary detection to avoid early over-triggering.
    boundary_detected = (
        (hed_ratio >= 1.2)
        or (worsening_risk > 0.05)
        or (dlt_count >= 1)
    )

    dynamic_cohort = choose_dynamic_cohort_size(
        task_name=task_name,
        dlt_count=dlt_count,
        risk=risk,
        worsening_risk=worsening_risk,
        hed_ratio=hed_ratio,
        near_target=near_target,
        boundary_detected=boundary_detected,
    )

    # Early safe zone override: accelerate with 3-patient cohorts.
    if hed_ratio < 0.8 and dlt_count == 0:
        low, high = cfg["far_mult"]
        early_factor = high if safe_streak >= 1 else (low + high) / 2.0
        next_dose = min(50.0, max(0.1, current_dose * early_factor))
        if task_name in ("phase_i_dosing", "combo_ddi") and safe_above_hed == 0:
            next_dose = min(next_dose, hed * 1.05)
        return {"next_dose": round(next_dose, 4), "cohort_size": 3, "escalate": True}

    # Hard safety gates
    if dlt_count >= 2 or dlt_rate > 0.33:
        return {"next_dose": max(0.1, current_dose * 0.6), "cohort_size": 6, "escalate": False}
    if hep > 0.85 or ren < 0.45 or imm > 0.85:
        return {"next_dose": max(0.1, current_dose * 0.7), "cohort_size": 6, "escalate": False}
    if "DE-ESCALATE" in doc:
        return {"next_dose": max(0.1, current_dose * 0.75), "cohort_size": 6, "escalate": False}
    if "HOLD" in doc and (hep > 0.70 or ren < 0.65 or imm > 0.75):
        return {"next_dose": current_dose, "cohort_size": dynamic_cohort, "escalate": True}

    # Task-specific early termination for allometric scaling after first aligned move.
    if task_name == "allometric_scaling" and prev_actions and cfg.get("stop_after_first_aligned", False):
        first = float(prev_actions[0]["next_dose"])
        align_tol = float(cfg.get("align_tolerance", 0.12))
        if abs(first - target) / max(target, 1e-6) <= align_tol:
            return {"next_dose": round(current_dose, 4), "cohort_size": 3, "escalate": False}
        # If not perfectly aligned on step 1, settle toward HED once, then stop oscillation.
        if len(prev_actions) >= 1:
            settle = min(target, current_dose * 1.08)
            if hed_ratio >= 1.0 or worsening_risk > 0.02:
                settle = min(settle, current_dose)
            return {"next_dose": round(settle, 4), "cohort_size": 6 if worsening_risk > 0.03 else 3, "escalate": False}

    # DDI-aware modulation for combo task.
    ddi_sensitive = task_name == "combo_ddi" and ("CYP3A4" in set(cyp_inhibitions))
    if ddi_sensitive and (risk > 0.25 or worsening_risk > 0.04):
        return {"next_dose": max(0.1, current_dose * 0.9), "cohort_size": dynamic_cohort, "escalate": True}

    # Task-aware escalation aggressiveness by distance-to-target.
    if ratio < 0.45:
        low, high = cfg["far_mult"]
        if safe_streak >= 2:
            up_factor = high
        else:
            up_factor = (low + high) / 2.0
    elif ratio < 0.8:
        low, high = cfg["mid_mult"]
        up_factor = (low + high) / 2.0
    else:
        low, high = cfg["near_mult"]
        up_factor = (low + high) / 2.0

    # Drug-adaptive aggressiveness scaling (no drug-name special casing).
    up_factor *= derive_aggressiveness_modifier(
        task_name=task_name,
        hed_ratio=hed_ratio,
        risk=risk,
        worsening_risk=worsening_risk,
        fragility_profile=fragility_profile,
        safe_streak=safe_streak,
    )

    # Boundary-aware control: move into cautious refinement and avoid large jumps.
    if boundary_detected and task_name != "allometric_scaling":
        low, high = cfg["near_mult"]
        boundary_cap = min((low + high) / 2.0, 1.25)
        up_factor = min(up_factor, boundary_cap)

    # Mid-zone controlled escalation (0.8x-1.2x HED): moderate steps, not crawl and not jumps.
    if 0.8 <= hed_ratio < 1.2 and task_name != "allometric_scaling":
        low, high = cfg["near_mult"]
        up_factor = min(up_factor, (low + high) / 2.0)

    # HED-relative guardrails:
    # - phase_i_dosing: above ~1.75x HED, suppress far-band high-end catch-up
    # - combo_ddi: above ~1.25-1.5x HED, cap to mid/near behavior
    if task_name == "phase_i_dosing" and hed_ratio > 1.75:
        low, high = cfg["mid_mult"] if ratio < 0.8 else cfg["near_mult"]
        up_factor = min(up_factor, (low + high) / 2.0)
    if task_name == "combo_ddi" and hed_ratio > 1.35:
        low, high = cfg["near_mult"] if ratio >= 0.8 else cfg["mid_mult"]
        up_factor = min(up_factor, (low + high) / 2.0)

    # Overshoot brake: once uncertainty evidence is high (cohort=6 or rising risk),
    # prevent large jumps regardless of nominal task band.
    if task_name in ("phase_i_dosing", "combo_ddi"):
        if dynamic_cohort == 6 or worsening_risk > 0.03:
            up_factor = min(up_factor, 1.12 if hed_ratio < 1.2 else 1.08)

    # Reduce aggressiveness when approaching boundaries
    if risk > cfg["max_safe_risk"] or worsening_risk > 0.05:
        up_factor = min(up_factor, 1.12)
    if hep > 0.75 or ren < 0.65:
        return {"next_dose": current_dose, "cohort_size": dynamic_cohort, "escalate": True}
    if dlt_count == 0 and 0.45 < hep <= 0.75 and ren >= 0.65 and imm <= 0.75:
        up_factor = max(up_factor, 1.08)

    # Allometric task: first meaningful action should anchor near HED.
    if task_name == "allometric_scaling" and not prev_actions:
        anchored = max(0.1, min(50.0, target))
        # Fragility-aware cap to avoid immediate collapse on unstable profiles.
        cap = max(0.1, current_dose * float(fragility_profile.get("first_step_cap_mult", 2.0)))
        anchored = min(anchored, cap) if fragility_profile.get("high_fragility", False) else anchored
        return {"next_dose": round(anchored, 4), "cohort_size": 3, "escalate": True}

    next_dose = min(50.0, max(0.1, current_dose * up_factor))

    if task_name in ("phase_i_dosing", "combo_ddi"):
        if safe_above_hed == 0:
            next_dose = min(next_dose, hed * 1.05)
        elif safe_above_hed == 1:
            next_dose = min(next_dose, hed * 1.25)
        else:
            next_dose = min(next_dose, hed * 1.50)

    # Extra hard cap against single-step overshoot in boundary region.
    if boundary_detected and task_name != "allometric_scaling":
        next_dose = min(next_dose, current_dose * 1.25)
        # If risk is actively worsening near/above HED region, prefer hold with larger cohort.
        if worsening_risk > 0.04 and hed_ratio >= 1.2:
            next_dose = current_dose

    # Additional cross-drug crash prevention:
    # if risk is not improving and dose already above HED region, hold rather than jump.
    if task_name in ("phase_i_dosing", "combo_ddi"):
        if hed_ratio >= 1.25 and worsening_risk > 0.02 and dynamic_cohort == 6:
            next_dose = current_dose

    # Anti-oscillation / repetition guard: nudge if repeating same dose
    if len(prev_actions) >= 2:
        last = prev_actions[-1]["next_dose"]
        prev = prev_actions[-2]["next_dose"]
        if abs(last - current_dose) < 1e-6 and abs(prev - current_dose) < 1e-6:
            next_dose = min(50.0, max(0.1, current_dose * 1.12))
    if len(prev_actions) >= 3 and dlt_count == 0:
        if all(abs(a["next_dose"] - current_dose) < 1e-6 for a in prev_actions[-3:]):
            next_dose = min(50.0, max(0.1, current_dose * 1.15))

    # Late-episode under-target penalty avoidance: catch up if still far below.
    late_ratio = float(cfg.get("late_catchup_ratio", 0.5))
    if len(prev_actions) >= max(3, int(0.6 * MAX_STEPS)) and ratio < late_ratio and risk < cfg["max_safe_risk"]:
        next_dose = min(50.0, max(next_dose, current_dose * 1.35))
    if task_name == "combo_ddi" and len(prev_actions) >= max(4, int(0.7 * MAX_STEPS)):
        catchup_mult = float(cfg.get("late_catchup_mult", 1.2))
        if ratio < late_ratio and risk < 0.28 and dlt_count == 0 and hed_ratio <= 1.35:
            next_dose = min(50.0, max(next_dose, current_dose * catchup_mult))

    # Phase I refinement/stop near RP2D band to reduce overshoot.
    if task_name == "phase_i_dosing":
        refine_lo, refine_hi = cfg.get("refine_band", (0.9, 1.15))
        if refine_lo <= ratio <= refine_hi and risk < cfg["max_safe_risk"] and dlt_count <= 1:
            center_ratio = 0.98
            # Prefer stopping once plausibly in RP2D zone unless clearly still below center target.
            if ratio >= center_ratio and worsening_risk >= -0.01:
                return {"next_dose": round(current_dose, 4), "cohort_size": dynamic_cohort, "escalate": False}
            # Allow one small refinement only when still below center target and trend remains clean.
            if ratio < center_ratio and worsening_risk <= 0.02:
                next_dose = min(50.0, max(0.1, current_dose * 1.03))
            else:
                return {"next_dose": round(current_dose, 4), "cohort_size": dynamic_cohort, "escalate": False}

    return {"next_dose": round(next_dose, 4), "cohort_size": dynamic_cohort, "escalate": True}


def compute_terminal_score(
    task_name: str,
    rewards: List[float],
    actions: List[dict],
    observations: List[dict],
    hed: float,
    task_targets: dict,
) -> float:
    """
    Episode-quality score in [0,1] that is more informative than clipped reward sum.
    """
    if not rewards:
        return 0.0
    avg_reward = sum(rewards) / len(rewards)
    dlt_rates = [
        float(obs.get("dlt_count", 0)) / max(1.0, float(obs.get("cohort_size", 3)))
        for obs in observations
    ]
    safety = 1.0 - min(1.0, sum(dlt_rates) / max(1, len(dlt_rates)))

    if task_name == "allometric_scaling":
        first = actions[0]["next_dose"] if actions else 0.0
        target = max(0.1, float(task_targets.get("allometric_scaling", hed)))
        err = abs(first - target) / max(target, 1e-6)
        closeness = max(0.0, 1.0 - err)
        score = 0.75 * closeness + 0.25 * safety
        return float(min(1.0, max(0.0, score)))

    if task_name == "combo_ddi":
        max_dose = max((a["next_dose"] for a in actions), default=0.0)
        target = max(0.1, float(task_targets.get("combo_ddi", hed * 1.25)))
        efficacy = max(0.0, 1.0 - abs(max_dose - target) / max(target, 1e-6))
        risk_penalty = 0.0
        for i, obs in enumerate(observations):
            risk = _organ_risk(obs)
            if i < len(actions) and actions[i]["next_dose"] > (actions[i - 1]["next_dose"] if i > 0 else 0.0):
                if risk > 0.35:
                    risk_penalty += 0.08
        score = 0.45 * efficacy + 0.35 * safety + 0.20 * avg_reward - risk_penalty
        return float(min(1.0, max(0.0, score)))

    # phase_i_dosing
    max_safe_dose = 0.0
    for i, a in enumerate(actions):
        dlt_rate = dlt_rates[i] if i < len(dlt_rates) else 0.0
        if dlt_rate <= 0.33:
            max_safe_dose = max(max_safe_dose, float(a["next_dose"]))
    target = max(0.1, float(task_targets.get("phase_i_dosing", hed * 1.5)))
    closeness = max(0.0, 1.0 - abs(max_safe_dose - target) / max(target, 1e-6))
    late_underdose_penalty = 0.0
    if max_safe_dose < 0.7 * target and len(actions) >= max(6, int(0.75 * MAX_STEPS)):
        late_underdose_penalty = 0.20
    score = 0.55 * closeness + 0.25 * safety + 0.20 * avg_reward - late_underdose_penalty
    return float(min(1.0, max(0.0, score)))


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


ALL_TASK_NAMES: List[str] = ["phase_i_dosing", "allometric_scaling", "combo_ddi"]


def run_task(task_name: str, client: Optional[OpenAI]) -> float:
    """Run one episode for a single task and emit a [START]/[STEP]/[END] block."""
    rewards: List[float] = []
    actions_taken: List[dict] = []
    observations_seen: List[dict] = []
    dose_history: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    submission_drug = resolve_submission_drug(task_name)
    hed_anchor = max(0.1, float(submission_drug["animal_dose_mgkg"]) * 6.0 / 37.0)
    cyp_inhibitions: List[str] = []
    task_targets = {
        "phase_i_dosing": hed_anchor * 1.5,
        "allometric_scaling": hed_anchor,
        "combo_ddi": hed_anchor * 1.25,
    }
    fragility_profile = {
        "hed_mgkg": hed_anchor,
        "fragility": 0.0,
        "high_fragility": False,
        "first_step_cap_mult": 2.0,
        "cyp_inhibitions": [],
    }

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        if USE_LLM_POLICY and client is None:
            debug_log("HF_TOKEN/API_KEY missing; falling back to deterministic controller for this run.")

        # Configure drug with error handling
        try:
            drug_cfg_raw = env_configure_drug(
                smiles=str(submission_drug["smiles"]),
                name=str(submission_drug["name"]),
                source_species=str(submission_drug["source_species"]),
                animal_dose_mgkg=float(submission_drug["animal_dose_mgkg"]),
            )
            drug_cfg = drug_cfg_raw
            fragility_profile = derive_fragility_profile(drug_cfg)
            hed_anchor = float(drug_cfg.get("hed_mgkg", hed_anchor))
            task_targets = resolve_task_targets(drug_cfg, hed_anchor)
            cyp_inhibitions = list(fragility_profile.get("cyp_inhibitions", []) or [])
            debug_log(f"[{task_name}] /drug hed_mgkg={hed_anchor:.4f}")
            debug_log(f"[{task_name}] /drug admet_summary={json.dumps(drug_cfg.get('admet_summary', {}), sort_keys=True)}")
            debug_log(f"[{task_name}] /drug drug_params={json.dumps(drug_cfg.get('drug_params', {}), sort_keys=True)}")
            debug_log(
                f"[{task_name}] /drug derived fragility={fragility_profile['fragility']:.4f} "
                f"high_fragility={fragility_profile['high_fragility']} "
                f"first_step_cap_mult={fragility_profile['first_step_cap_mult']}"
            )
            debug_log(f"[{task_name}] /drug cyp_inhibitions={cyp_inhibitions}")
        except Exception as e:
            debug_log(f"[{task_name}] Drug configuration failed: {e}. Using defaults.")

        # Reset environment with error handling
        try:
            result = env_reset()
            obs, _, done = normalize_env_response(result)
            current_dose = float(obs.get("dose_level", 1.0))
            observations_seen.append(obs)
            dose_history.append(current_dose)
        except Exception as e:
            debug_log(f"[{task_name}] Environment reset failed: {e}. Aborting episode.")
            raise

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            dlt_rates = [
                float(o.get("dlt_count", 0)) / max(1.0, float(o.get("cohort_size", 3)))
                for o in observations_seen
            ]
            cumulative_dlt_rate = sum(dlt_rates) / max(1, len(dlt_rates))
            organ_risk_trend = (
                _organ_risk(observations_seen[-1]) - _organ_risk(observations_seen[-2])
                if len(observations_seen) >= 2
                else 0.0
            )

            action = choose_action(
                task_name,
                obs,
                current_dose,
                actions_taken,
                observations_seen,
                hed_anchor,
                cyp_inhibitions,
                fragility_profile,
                task_targets,
            )

            # Optional LLM policy mode; deterministic policy remains benchmark default.
            if USE_LLM_POLICY and client is not None:
                try:
                    message = get_model_message(
                        client=client,
                        obs=obs,
                        step=step,
                        task_name=task_name,
                        hed=hed_anchor,
                        dose_history=dose_history,
                        reward_history=rewards,
                        cumulative_dlt_rate=cumulative_dlt_rate,
                        organ_risk_trend=organ_risk_trend,
                    )
                    action = parse_action(message, current_dose)
                except Exception as e:
                    debug_log(f"[{task_name}] LLM policy failed at step {step}: {e}. Falling back to deterministic.")

            # Execute step with error handling
            try:
                result = env_step(action)
                obs, reward, done = normalize_env_response(result)
                observations_seen.append(obs)
                current_dose = float(obs.get("dose_level", current_dose))
                dose_history.append(current_dose)
                rewards.append(reward)
                actions_taken.append(action)
                steps_taken = step
                log_step(step=step, action=json.dumps(action),
                         reward=reward, done=done, error=None)
            except Exception as e:
                debug_log(f"[{task_name}] env_step failed at step {step}: {e}. Terminating episode.")
                break

            if done:
                break

        score = compute_terminal_score(
            task_name=task_name,
            rewards=rewards,
            actions=actions_taken,
            observations=observations_seen,
            hed=hed_anchor,
            task_targets=task_targets,
        )
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        debug_log(f"[{task_name}] Fatal error in run_task: {exc}")
        import traceback
        debug_log(traceback.format_exc())
    finally:
        try:
            env_close()
        except Exception as exc:
            debug_log(f"[{task_name}] env_close failed: {exc}")
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)

    return score


def main() -> None:
    """Run inference for all benchmark tasks so the validator can discover 3+ tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    # Honor TASK_NAME env var only if it explicitly selects a single task
    # AND the user opts out of multi-task mode. Default: run all 3 tasks.
    single_task_only = os.getenv("RUN_SINGLE_TASK", "0").lower() in ("1", "true", "yes")
    if single_task_only and TASK_NAME in ALL_TASK_NAMES:
        task_list = [TASK_NAME]
    else:
        task_list = list(ALL_TASK_NAMES)

    scores: dict = {}
    for task_name in task_list:
        try:
            scores[task_name] = run_task(task_name, client)
        except Exception as exc:
            debug_log(f"Task {task_name} aborted with unrecoverable error: {exc}")
            scores[task_name] = 0.0

    # Optional summary line on stderr; the validator parses [START]/[STEP]/[END] only.
    if scores:
        avg = sum(scores.values()) / len(scores)
        debug_log("FINAL SCORES " + ", ".join(f"{k}={v:.2f}" for k, v in scores.items()) + f", avg={avg:.2f}")


if __name__ == "__main__":
    main()
