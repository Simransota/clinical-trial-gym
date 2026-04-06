# server/rl_agent_environment.py

"""
Clinical Trial Gym — Main Environment.
Simulates Phase I drug dose escalation trials.
Wires together all agents and exposes step/reset/state API.

Fixes applied vs original:
  [Bug 5] _compute_reward: history[-2] was reading two steps ago because
          the current step was not yet in history when reward was computed.
          Fix: history is now appended BEFORE _compute_reward is called,
          and reward is recomputed/stored after. prev_dose is passed
          explicitly to avoid indexing confusion.
  [Bug 6] stopping reward branch could never fire because "escalate" was
          never saved in the history dict. Fix: action.escalate is now
          saved to history so the stopping component works correctly.
  [Bug 7] _grade_allometric: exponent written as (0.75 - 1.0) was opaque
          and could confuse maintainers. Rewritten as the explicit
          allometric formula with clear variable names.
"""

import random
import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RlAgentAction, RlAgentObservation
except ImportError:
    from models import RlAgentAction, RlAgentObservation

try:
    from .agents import (
        PatientAgent, HepatocyteAgent, ImmuneAgent,
        RenalAgent, MeasurementAgent, DoctorAgent,
    )
except ImportError:
    from agents import (
        PatientAgent, HepatocyteAgent, ImmuneAgent,
        RenalAgent, MeasurementAgent, DoctorAgent,
    )


# ── FDA trial rules ──────────────────────────────────────────────────────────
TRUE_RP2D = 12.0   # mg/kg — the correct answer; hidden from agent, used by grader

FDA_RULES = {
    "max_dlt_rate":   0.33,
    "max_steps":      10,
    "max_dose_mg_kg": 50.0,
}


class RlAgentEnvironment(Environment):
    """
    The Clinical Trial Gym environment.

    One episode = one complete Phase I dose escalation trial.

    The agent:
      - Sees: plasma concentration, DLT counts, organ signals, doctor advice
      - Does: choose next dose, cohort size, whether to escalate
      - Goal: find the RP2D as safely and quickly as possible
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state    = State(episode_id=str(uuid4()), step_count=0)
        self.cohort    = []
        self.current_dose = 1.0
        self.history   = []
        self.rp2d_dose = None
        self._done     = False
        self._use_llm  = True
        self.doctor    = DoctorAgent() if self._use_llm else None

    # ── reset() ─────────────────────────────────────────────────────────────
    def reset(self) -> RlAgentObservation:
        """Start a fresh trial episode."""
        self._state       = State(episode_id=str(uuid4()), step_count=0)
        self.current_dose = 1.0
        self.history      = []
        self.rp2d_dose    = None
        self._done        = False

        self.cohort = self._make_cohort(3)
        self._run_cohort(self.current_dose)

        return RlAgentObservation(
            phase="phase_i",
            cohort_size=len(self.cohort),
            dose_level=self.current_dose,
            plasma_conc=round(float(np.mean([p.blood_conc for p in self.cohort])), 3),
            dlt_count=0,
            dlt_grade=[0] * len(self.cohort),
            hepatocyte_signal=0.0,
            immune_signal=0.0,
            renal_signal=1.0,
            doctor_recommendation="Trial started. Begin dose escalation cautiously.",
            done=False,
            reward=0.0,
        )

    # ── step() ──────────────────────────────────────────────────────────────
    def step(self, action: RlAgentAction) -> RlAgentObservation:
        """
        Agent submits an action.
        Environment runs body simulation and returns new observation + reward.
        """
        self._state.step_count += 1

        # 1. Record previous dose BEFORE updating (used by reward function)
        prev_dose = self.current_dose

        # 2. Apply dose (clamped to safe range)
        self.current_dose = max(0.1, min(action.next_dose, FDA_RULES["max_dose_mg_kg"]))
        cohort_n          = max(3, min(action.cohort_size, 6))

        # 3. Make new cohort and run simulation
        self.cohort = self._make_cohort(cohort_n)
        self._run_cohort(self.current_dose)

        # 4. Read biological signals from each patient
        states      = [p.get_state() for p in self.cohort]
        dlt_grades  = [MeasurementAgent.grade_dlt(s) for s in states]
        hep_signals = [HepatocyteAgent.observe(s)     for s in states]
        imm_signals = [ImmuneAgent.observe(s)          for s in states]
        ren_signals = [RenalAgent.observe(s)            for s in states]
        cmax_vals   = [p.blood_conc                    for p in self.cohort]

        dlt_count = sum(1 for g in dlt_grades if g >= 3)
        avg_hep   = float(np.mean(hep_signals))
        avg_imm   = float(np.mean(imm_signals))
        avg_ren   = float(np.mean(ren_signals))
        avg_cmax  = float(np.mean(cmax_vals))

        # 5. FDA stopping rules
        dlt_rate      = dlt_count / len(self.cohort)
        fda_stop      = dlt_rate > FDA_RULES["max_dlt_rate"]
        max_steps_hit = self._state.step_count >= FDA_RULES["max_steps"]
        agent_stopped = not action.escalate

        if not fda_stop:
            self.rp2d_dose = self.current_dose

        self._done = fda_stop or agent_stopped or max_steps_hit

        # 6. Save to history BEFORE computing reward
        #    FIX [Bug 5]: history must be populated first so _compute_reward
        #    can read history[-1] as the current step and history[-2] as the
        #    actual previous step.
        #    FIX [Bug 6]: "escalate" is now saved so stopping reward fires.
        self.history.append({
            "step":        self._state.step_count,
            "dose":        self.current_dose,
            "dlt_count":   dlt_count,
            "dlt_rate":    dlt_rate,
            "cohort_size": len(self.cohort),
            "escalate":    action.escalate,   # FIX [Bug 6]
            "reward":      0.0,               # placeholder; filled below
        })

        # 7. Compute reward (now history[-1] is the current step)
        reward = self._compute_reward(
            dlt_count=dlt_count,
            cohort_size=len(self.cohort),
            fda_stop=fda_stop,
            avg_hep=avg_hep,
            avg_ren=avg_ren,
            prev_dose=prev_dose,              # FIX [Bug 5]: passed explicitly
        )
        self.history[-1]["reward"] = reward   # back-fill the placeholder

        # 8. Doctor recommendation
        if self._use_llm and self.doctor:
            rec = self.doctor.recommend(
                cmax=avg_cmax,
                dlt_count=dlt_count,
                cohort_size=len(self.cohort),
                cyp=avg_hep,
                gfr=avg_ren,
                immune=avg_imm,
                current_dose=self.current_dose,
            )
        else:
            if fda_stop:
                rec = f"DE-ESCALATE: {dlt_count}/{len(self.cohort)} DLTs exceeds FDA limit."
            elif avg_hep > 0.7:
                rec = "HOLD: Liver stress critically high, risk of enzyme saturation."
            elif avg_ren < 0.5:
                rec = "HOLD: Kidney function significantly impaired."
            elif dlt_count == 0 and avg_hep < 0.4:
                rec = "ESCALATE: Safety signals within acceptable limits."
            else:
                rec = "HOLD: Borderline safety signals, monitor before escalating."

        return RlAgentObservation(
            phase="phase_i",
            cohort_size=len(self.cohort),
            dose_level=round(self.current_dose, 3),
            plasma_conc=round(avg_cmax, 3),
            dlt_count=dlt_count,
            dlt_grade=dlt_grades,
            hepatocyte_signal=round(avg_hep, 3),
            immune_signal=round(avg_imm, 3),
            renal_signal=round(avg_ren, 3),
            doctor_recommendation=rec,
            done=self._done,
            reward=reward,
        )

    # ── state property ───────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    # ── graders ─────────────────────────────────────────────────────────────

    def grade_episode(self, task: str = "phase_i_dosing") -> float:
        """Score the full episode 0.0 to 1.0. Called by inference.py."""
        if task == "phase_i_dosing":
            return self._grade_phase_i()
        elif task == "allometric_scaling":
            return self._grade_allometric()
        elif task == "combo_ddi":
            return self._grade_combo_ddi()
        return self._grade_phase_i()

    def _grade_phase_i(self) -> float:
        if self.rp2d_dose is None:
            return 0.0
        error = abs(self.rp2d_dose - TRUE_RP2D) / TRUE_RP2D
        if   error <= 0.10: dose_score = 1.0
        elif error <= 0.25: dose_score = 0.8
        elif error <= 0.50: dose_score = 0.5
        elif error <= 0.75: dose_score = 0.3
        else:               dose_score = 0.1
        triggered = any(h["dlt_rate"] > FDA_RULES["max_dlt_rate"] for h in self.history)
        penalty   = 0.3 if triggered else 0.0
        bonus     = max(0.0, 0.1 * (1.0 - self._state.step_count / FDA_RULES["max_steps"]))
        return round(max(0.0, min(1.0, dose_score - penalty + bonus)), 3)

    def _grade_allometric(self) -> float:
        """
        FIX [Bug 7]: formula rewritten with explicit variable names.

        Allometric scaling from rat (0.25 kg) to human (70 kg):
          human_dose = rat_dose * (human_weight / rat_weight) ^ (exponent - 1)
        where exponent = 0.75 (standard allometric exponent for clearance).
        """
        if not self.history:
            return 0.0
        proposed = self.history[0]["dose"]

        rat_dose      = 8.0    # mg/kg in rat
        rat_weight    = 0.25   # kg
        human_weight  = 70.0   # kg
        allometric_exp = 0.75  # standard clearance scaling exponent

        # Human equivalent dose via allometric scaling
        human_equivalent = rat_dose * (human_weight / rat_weight) ** (allometric_exp - 1.0)

        error = abs(proposed - human_equivalent) / max(human_equivalent, 0.01)
        if   error <= 0.10: return 1.0
        elif error <= 0.25: return 0.8
        elif error <= 0.50: return 0.6
        elif error <= 1.00: return 0.3
        else:               return 0.1

    def _grade_combo_ddi(self) -> float:
        if not self.history:
            return 0.0
        safe_steps   = sum(1 for h in self.history if h["dlt_rate"] <= FDA_RULES["max_dlt_rate"])
        safety_score = safe_steps / max(len(self.history), 1)
        max_dose     = max((h["dose"] for h in self.history), default=0)
        efficacy     = min(1.0, max_dose / TRUE_RP2D)
        total_dlts   = sum(h["dlt_count"] for h in self.history)
        ddi_score    = max(0.0, 1.0 - total_dlts * 0.1)
        return round(min(1.0, 0.4 * efficacy + 0.4 * safety_score + 0.2 * ddi_score), 3)

    # ── private helpers ──────────────────────────────────────────────────────

    def _make_cohort(self, size: int):
        patients = []
        for _ in range(size):
            w = max(50, min(100, random.gauss(70, 10)))
            patients.append(PatientAgent(
                weight_kg=w,
                age=random.randint(25, 65),
                renal_factor=random.uniform(0.7, 1.0),
                hepatic_factor=random.uniform(0.7, 1.0),
            ))
        return patients

    def _run_cohort(self, dose: float, hours: float = 24.0):
        for p in self.cohort:
            p.reset()
            p.dose(dose)
            p.advance(hours=hours)

    def _compute_reward(
        self,
        dlt_count: int,
        cohort_size: int,
        fda_stop: bool,
        avg_hep: float,
        avg_ren: float,
        prev_dose: float,   # FIX [Bug 5]: explicitly passed, not inferred from history
    ) -> float:
        """
        Shaped reward function — observable signals only, no hidden RP2D.

        Components:
          40% — Safety        (no DLTs, no FDA violation)
          35% — Progress      (escalate when safe, de-escalate when not)
          15% — Stopping      (voluntary stop at right moment)
          10% — Organ health  (kidney + liver throughout)

        Called AFTER the current step has been appended to self.history,
        so history[-1] is the current step and history[-2] is the previous.
        """
        dlt_rate = dlt_count / max(cohort_size, 1)

        # ── Component 1: Safety (40%) ─────────────────────────────────────────
        if fda_stop:
            safety = 0.0
        elif dlt_rate == 0:
            safety = 1.0
        elif dlt_rate <= 0.167:   # 1 DLT in 6 — acceptable
            safety = 0.8
        elif dlt_rate <= 0.33:    # 2 DLTs in 6 — borderline
            safety = 0.4
        else:
            safety = 0.0

        # ── Component 2: Escalation progress (35%) ────────────────────────────
        # FIX [Bug 5]: use prev_dose (passed explicitly) instead of
        # history[-2] which was off-by-one in the original.
        if dlt_rate == 0 and avg_hep < 0.5 and avg_ren > 0.7:
            if self.current_dose > prev_dose:
                progress = min(1.0, self.current_dose / 20.0)
            else:
                progress = 0.3
        elif dlt_rate > 0 and self.current_dose <= prev_dose:
            # DLTs appeared AND agent de-escalated — correct behaviour
            progress = 0.7
        elif dlt_rate > 0:
            # DLTs appeared but agent kept pushing — penalise
            progress = 0.1
        else:
            progress = 0.5

        # ── Component 3: Stopping behaviour (15%) ─────────────────────────────
        # FIX [Bug 6]: history[-1]["escalate"] now exists (was always missing
        # before), so this branch can actually fire.
        if fda_stop:
            stopping = 0.0
        elif not self.history[-1].get("escalate", True):
            # Agent voluntarily stopped
            if dlt_rate > 0 and self.current_dose > 3.0:
                stopping = 1.0    # stopped right when DLTs appeared
            elif dlt_rate == 0 and self.current_dose > 8.0:
                stopping = 0.7    # conservative stop at meaningful dose
            else:
                stopping = 0.3    # stopped too early
        else:
            stopping = 0.5        # still escalating — neutral

        # ── Component 4: Organ health (10%) ──────────────────────────────────
        organ = avg_ren * 0.5 + (1.0 - avg_hep) * 0.5

        # ── Combine ───────────────────────────────────────────────────────────
        value = (
            0.40 * safety   +
            0.35 * progress +
            0.15 * stopping +
            0.10 * organ
        )

        # ── RP2D transition bonus ──────────────────────────────────────────────
        # Extra reward for the exact step where DLTs first appear after safe
        # escalation (the transition point that identifies the RP2D).
        if dlt_rate > 0 and len(self.history) >= 2:
            prev_dlt = self.history[-2].get("dlt_count", 0)
            if prev_dlt == 0:
                value = min(1.0, value + 0.15)

        return round(min(1.0, max(0.0, value)), 3)