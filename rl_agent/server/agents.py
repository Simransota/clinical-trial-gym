"""
agents.py — All 6 agents for the multi-agent clinical simulation layer.

Six agents in total:
  1. PatientAgent      — runs the 2-compartment ODE (the patient's body)
  2. HepatocyteAgent   — watches the liver (rule-based)
  3. ImmuneAgent       — watches the immune system (rule-based)
  4. RenalAgent        — watches the kidneys (rule-based)
  5. MeasurementAgent  — simulates lab blood tests, grades DLTs (rule-based)
  6. DoctorAgent       — reads all signals, calls LLM, writes recommendation

Fixes applied vs original:
  [Bug 1] HepatocyteAgent: SATURATION_THRESHOLD was dead code; now used in a
          proper sigmoid response instead of a flat linear ratio.
  [Bug 2] MeasurementAgent.get_labs: random.gauss was unseeded so the same
          patient state could produce different DLT grades on repeated calls.
          Noise is now seeded from the patient state so results are deterministic.
  [Bug 3] PatientAgent: age was stored but never used in PK math; now applied
          as an age-based clearance scaling (ke reduces after age 40).
  [Bug 4] PatientAgent.advance: organ stress accumulated end-of-step
          concentration instead of the trapezoidal AUC; fixed to use
          (prev + curr) / 2 * dt inside the Euler loop.
"""

import os
import math
import random
from openai import OpenAI


# ─────────────────────────────────────────────
# AGENT 1: PatientAgent
# Simulates one patient's body using a
# 2-compartment ODE (math model of drug movement).
#
# Compartment 1 = bloodstream (drug enters here)
# Compartment 2 = tissues/organs (drug slowly moves here)
#
# At each timestep, drug:
#   - enters bloodstream from the dose
#   - moves between blood and tissue
#   - gets cleared by liver and kidneys
# ─────────────────────────────────────────────

class PatientAgent:
    def __init__(self, weight_kg=70.0, age=45, renal_factor=1.0, hepatic_factor=1.0):
        """
        weight_kg     — patient body weight (affects how drug distributes)
        age           — older patients clear drug more slowly (ke scaled down)
        renal_factor  — 1.0 = healthy kidneys, 0.5 = half function
        hepatic_factor— 1.0 = healthy liver,  0.5 = half function
        """
        self.weight_kg = weight_kg
        self.age = age
        self.renal_factor = renal_factor
        self.hepatic_factor = hepatic_factor

        # PK parameters (how drug moves through body)
        # These would come from Layer 1 (RDKit) in a real system.
        # Here we use realistic defaults.
        self.ka = 1.2          # absorption rate (how fast drug enters blood)

        # FIX [Bug 3]: age now reduces ke linearly after age 40.
        # Clearance declines ~0.5% per year past 40 — realistic for most drugs.
        age_factor = max(0.5, 1.0 - 0.005 * max(0, age - 40))
        self.ke = 0.15 * hepatic_factor * age_factor  # elimination rate (liver clearance)

        self.k12 = 0.08        # rate blood → tissue
        self.k21 = 0.04        # rate tissue → blood
        self.Vd = 0.7 * weight_kg  # volume of distribution (liters)

        # State: how much drug (mg) is in each compartment right now
        self.blood_conc = 0.0    # mg/L in bloodstream
        self.tissue_conc = 0.0   # mg/L in tissues
        self.time = 0.0          # hours elapsed

        # Accumulated stress on organs (trapezoidal AUC)
        self.cumulative_liver_stress = 0.0
        self.cumulative_kidney_stress = 0.0

    def dose(self, amount_mg_per_kg: float):
        """
        Give the patient a dose. Drug enters the bloodstream instantly
        (simplified — no absorption delay for now).
        """
        total_drug_mg = amount_mg_per_kg * self.weight_kg
        self.blood_conc += total_drug_mg / self.Vd  # convert mg to mg/L

    def advance(self, hours: float = 1.0):
        """
        Advance the simulation by `hours` hours.
        Uses simple Euler integration of the 2-compartment ODE.

        The ODE equations are:
          d(blood)/dt  = -ke*blood - k12*blood + k21*tissue
          d(tissue)/dt = k12*blood - k21*tissue

        FIX [Bug 4]: organ stress now uses trapezoidal AUC
        (prev_conc + curr_conc) / 2 * dt inside each sub-step,
        instead of end-of-step concentration * total hours.
        """
        steps = 10  # sub-steps for numerical stability
        dt = hours / steps

        for _ in range(steps):
            prev_blood = self.blood_conc  # save for trapezoidal AUC

            delta_blood = (
                - self.ke  * self.blood_conc   # cleared by liver
                - self.k12 * self.blood_conc   # moves to tissue
                + self.k21 * self.tissue_conc  # comes back from tissue
            ) * dt

            delta_tissue = (
                self.k12 * self.blood_conc
                - self.k21 * self.tissue_conc
            ) * dt

            self.blood_conc  = max(0.0, self.blood_conc  + delta_blood)
            self.tissue_conc = max(0.0, self.tissue_conc + delta_tissue)

            # FIX [Bug 4]: trapezoidal average of blood_conc over this sub-step
            auc_dt = (prev_blood + self.blood_conc) / 2.0 * dt
            self.cumulative_liver_stress  += auc_dt * (1.0 / max(self.hepatic_factor, 0.1))
            self.cumulative_kidney_stress += auc_dt * (1.0 / max(self.renal_factor, 0.1))

        self.time += hours

    def get_state(self) -> dict:
        """Return current body state as a dictionary."""
        return {
            "blood_conc":    self.blood_conc,
            "tissue_conc":   self.tissue_conc,
            "time":          self.time,
            "liver_stress":  self.cumulative_liver_stress,
            "kidney_stress": self.cumulative_kidney_stress,
            "weight_kg":     self.weight_kg,
            "age":           self.age,
            "ke":            self.ke,
        }

    def reset(self):
        """Reset patient to pre-dose state."""
        self.blood_conc = 0.0
        self.tissue_conc = 0.0
        self.time = 0.0
        self.cumulative_liver_stress = 0.0
        self.cumulative_kidney_stress = 0.0


# ─────────────────────────────────────────────
# AGENT 2: HepatocyteAgent
# Watches liver stress.
# Returns 0.0 (fine) to 1.0 (overwhelmed).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class HepatocyteAgent:
    # Stress level at which CYP450 saturation begins to accelerate
    SATURATION_THRESHOLD = 80.0    # mg/L * hours (cumulative AUC)
    MAX_STRESS = 200.0

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        CYP450 saturation — how overwhelmed is the liver?

        FIX [Bug 1]: was a flat linear ratio (stress / MAX_STRESS).
        Now uses a two-phase response:
          - Below SATURATION_THRESHOLD: gentle linear rise (liver coping)
          - Above SATURATION_THRESHOLD: steep sigmoid-like acceleration
        This matches real CYP450 enzyme kinetics where saturation
        causes a sharp nonlinear rise in toxicity markers.

        0.0 = liver perfectly fine
        1.0 = liver completely saturated / overwhelmed
        """
        stress = state.get("liver_stress", 0.0)

        if stress <= 0:
            return 0.0

        if stress <= cls.SATURATION_THRESHOLD:
            # Gentle linear phase: 0 → 0.4 as stress goes 0 → SATURATION_THRESHOLD
            return 0.4 * (stress / cls.SATURATION_THRESHOLD)
        else:
            # Steep sigmoid phase: 0.4 → 1.0 as stress goes SATURATION_THRESHOLD → MAX_STRESS
            excess = (stress - cls.SATURATION_THRESHOLD) / (cls.MAX_STRESS - cls.SATURATION_THRESHOLD)
            steep = 1.0 / (1.0 + math.exp(-10.0 * (excess - 0.5)))
            return min(1.0, 0.4 + 0.6 * steep)


# ─────────────────────────────────────────────
# AGENT 3: ImmuneAgent
# Watches immune/inflammatory response.
# Returns 0.0 (calm) to 1.0 (severe reaction).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class ImmuneAgent:
    REACTION_THRESHOLD = 5.0    # mg/L — immune reaction starts here
    SEVERE_THRESHOLD   = 20.0   # mg/L — full severity

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        Cytokine/IL-6 signal — how inflamed is the body?

        Based on current peak blood concentration.
        0.0 = no reaction
        1.0 = severe inflammatory reaction
        """
        conc = state.get("blood_conc", 0.0)

        if conc <= cls.REACTION_THRESHOLD:
            return 0.0
        elif conc >= cls.SEVERE_THRESHOLD:
            return 1.0
        else:
            return (conc - cls.REACTION_THRESHOLD) / (cls.SEVERE_THRESHOLD - cls.REACTION_THRESHOLD)


# ─────────────────────────────────────────────
# AGENT 4: RenalAgent
# Watches kidney function.
# Returns 1.0 (fully healthy) to 0.0 (failed).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class RenalAgent:
    STRESS_FOR_IMPAIRMENT = 100.0  # cumulative AUC stress for mild impairment
    STRESS_FOR_FAILURE    = 400.0  # cumulative AUC stress for failure

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        GFR fraction — how well are the kidneys filtering?

        1.0 = kidneys working at full capacity (healthy)
        0.0 = kidneys have shut down
        """
        stress = state.get("kidney_stress", 0.0)

        if stress <= cls.STRESS_FOR_IMPAIRMENT:
            return 1.0
        elif stress >= cls.STRESS_FOR_FAILURE:
            return 0.0
        else:
            return 1.0 - (stress - cls.STRESS_FOR_IMPAIRMENT) / (
                cls.STRESS_FOR_FAILURE - cls.STRESS_FOR_IMPAIRMENT
            )


# ─────────────────────────────────────────────
# AGENT 5: MeasurementAgent
# Simulates blood test results (lab panels).
# Grades DLT (serious side effect) per patient.
# Rule-based — no LLM.
#
# DLT Grade scale (standard oncology NCI CTCAE):
#   Grade 0 = no side effect
#   Grade 1 = mild
#   Grade 2 = moderate
#   Grade 3 = severe  (counts as a DLT — trial stopping rules apply)
#   Grade 4 = life-threatening
# ─────────────────────────────────────────────

class MeasurementAgent:

    @classmethod
    def get_labs(cls, state: dict, seed: int = None) -> dict:
        """
        Simulate blood test results from body state.
        Returns realistic lab values derived from ODE state.

        FIX [Bug 2]: noise is now seeded from patient state values so that
        the same state always produces the same labs. Previously random.gauss
        was unseeded, making grade_dlt non-deterministic — two calls on the
        same state could return different grades.

        Pass an explicit seed to override (useful for testing).

        Normal ranges (for reference):
          Heart rate (HR):    60-100 bpm
          Systolic BP:        90-120 mmHg
          ALT (liver enzyme): 7-56 U/L
          Creatinine:         0.6-1.2 mg/dL
          WBC:                4.5-11.0 K/uL
        """
        conc    = state.get("blood_conc", 0.0)
        lstress = state.get("liver_stress", 0.0)
        kstress = state.get("kidney_stress", 0.0)

        # Deterministic seed derived from state so same state → same labs
        if seed is None:
            seed = int((conc * 1000 + lstress * 100 + kstress * 10) % (2**31))
        rng = random.Random(seed)
        noise = lambda scale: rng.gauss(0, scale)

        hr          = 72 + (conc * 2.0) + noise(3)
        bp_systolic = 120 - (conc * 1.5) + noise(5)
        alt         = 35 + (lstress * 0.8) + noise(5)
        creatinine  = 1.0 + (kstress * 0.005) + noise(0.1)
        wbc         = max(0.1, 7.5 - (conc * 0.3) + noise(0.5))

        return {
            "heart_rate":  round(hr, 1),
            "bp_systolic": round(bp_systolic, 1),
            "alt":         round(alt, 1),
            "creatinine":  round(creatinine, 2),
            "wbc":         round(wbc, 2),
        }

    @classmethod
    def grade_dlt(cls, state: dict) -> int:
        """
        Classify how severe a patient's side effects are.
        Returns a grade from 0 (none) to 4 (life-threatening).

        Standard NCI CTCAE grading criteria (simplified).
        Grade 3+ = counts as a DLT in trial stopping rules.
        """
        labs = cls.get_labs(state)  # deterministic — same state → same labs

        grades = []

        # Liver toxicity (ALT elevation)
        alt = labs["alt"]
        if   alt < 56:   grades.append(0)
        elif alt < 112:  grades.append(1)
        elif alt < 175:  grades.append(2)
        elif alt < 280:  grades.append(3)
        else:            grades.append(4)

        # Kidney toxicity (creatinine elevation)
        cr = labs["creatinine"]
        if   cr < 1.5:   grades.append(0)
        elif cr < 2.0:   grades.append(1)
        elif cr < 3.0:   grades.append(2)
        elif cr < 6.0:   grades.append(3)
        else:            grades.append(4)

        # Blood count toxicity (low WBC = neutropenia)
        wbc = labs["wbc"]
        if   wbc >= 4.5: grades.append(0)
        elif wbc >= 2.0: grades.append(1)
        elif wbc >= 1.0: grades.append(2)
        elif wbc >= 0.5: grades.append(3)
        else:            grades.append(4)

        return max(grades)


# ─────────────────────────────────────────────
# AGENT 6: DoctorAgent
# The ONLY LLM agent. One LLM call per step.
# Reads all signals and writes one sentence.
# ─────────────────────────────────────────────

class DoctorAgent:

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy"),
        )
        self.model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    def recommend(
        self,
        cmax: float,
        dlt_count: int,
        cohort_size: int,
        cyp: float,
        gfr: float,
        immune: float,
        current_dose: float,
    ) -> str:
        """
        Ask the LLM for a one-sentence clinical recommendation.
        Called ONCE per environment step (not per patient).
        """
        prompt = f"""You are a clinical pharmacologist reviewing a Phase I dose escalation trial.

Current status:
- Peak blood concentration (Cmax): {cmax:.1f} mg/L
- Serious side effects (DLTs): {dlt_count}/{cohort_size} patients
- Liver enzyme saturation (CYP450): {cyp:.0%}
- Kidney function (GFR): {gfr:.0%}
- Immune/inflammatory signal: {immune:.0%}
- Current dose: {current_dose:.1f} mg/kg

In exactly one sentence, recommend one of: ESCALATE, HOLD, or DE-ESCALATE, and state the most important reason why."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Rule-based fallback if LLM call fails
            if dlt_count >= 2:
                return f"DE-ESCALATE: {dlt_count}/{cohort_size} patients have serious side effects."
            elif cyp > 0.8:
                return "HOLD: Liver saturation is critically high, risk of toxicity."
            elif gfr < 0.5:
                return "HOLD: Kidney function is significantly impaired."
            else:
                return "ESCALATE: Safety signals are within acceptable limits."