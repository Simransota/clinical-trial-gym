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
          proper sigmoid so liver response curves sharply past the threshold
          instead of staying purely linear.
  [Bug 2] MeasurementAgent.get_labs: noise was unseeded — same state could
          produce different DLT grades on repeated calls. Labs are now
          deterministic: noise is seeded from a hash of the state values so
          the same physiological state always maps to the same lab result.
  [Bug 3] PatientAgent: age was stored but never wired into PK math. Now
          applied as a clearance modifier (ke reduced ~0.5% per year over 40).
  [Bug 4] PatientAgent.advance: organ stress was accumulated using end-of-step
          concentration, underestimating exposure for rapidly clearing drugs.
          Now accumulated inside the Euler loop using the trapezoidal rule
          (average of concentration before and after each sub-step).
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
        weight_kg      — patient body weight (affects how drug distributes)
        age            — affects hepatic clearance rate (older -> slower clearance)
        renal_factor   — 1.0 = healthy kidneys, 0.5 = half function
        hepatic_factor — 1.0 = healthy liver,   0.5 = half function
        """
        self.weight_kg     = weight_kg
        self.age           = age
        self.renal_factor  = renal_factor
        self.hepatic_factor = hepatic_factor

        # PK parameters (how drug moves through body).
        # These would come from Layer 1 (RDKit) in a real system.
        self.ka  = 1.2               # absorption rate
        self.k12 = 0.08              # rate blood -> tissue
        self.k21 = 0.04              # rate tissue -> blood
        self.Vd  = 0.7 * weight_kg  # volume of distribution (liters)

        # FIX [Bug 3]: age now reduces hepatic clearance by ~0.5% per year
        # above 40. A 60-year-old clears ~10% slower than a 40-year-old.
        # Floor at 0.5 so very elderly patients still clear the drug.
        age_clearance_factor = max(0.5, 1.0 - 0.005 * max(0, age - 40))

        # ke = elimination rate: scaled by liver health AND patient age
        self.ke = 0.15 * hepatic_factor * age_clearance_factor

        # Compartment state
        self.blood_conc  = 0.0   # mg/L in bloodstream
        self.tissue_conc = 0.0   # mg/L in tissues
        self.time        = 0.0   # hours elapsed

        # Accumulated organ stress (AUC-weighted)
        self.cumulative_liver_stress  = 0.0
        self.cumulative_kidney_stress = 0.0

    def dose(self, amount_mg_per_kg: float):
        """
        Give the patient a dose. Drug enters the bloodstream instantly
        (simplified — no absorption delay).
        """
        total_drug_mg = amount_mg_per_kg * self.weight_kg
        self.blood_conc += total_drug_mg / self.Vd

    def advance(self, hours: float = 1.0):
        """
        Advance the simulation by `hours` hours.
        Uses Euler integration of the 2-compartment ODE:

          d(blood)/dt  = -ke*blood - k12*blood + k21*tissue
          d(tissue)/dt =  k12*blood - k21*tissue

        FIX [Bug 4]: organ stress is now accumulated inside the Euler loop
        using the trapezoidal rule (average of concentration at the start and
        end of each sub-step). This correctly captures AUC rather than using
        only the end-of-step blood concentration.
        """
        steps = 10
        dt    = hours / steps

        liver_inv  = 1.0 / max(self.hepatic_factor, 0.1)
        kidney_inv = 1.0 / max(self.renal_factor,   0.1)

        for _ in range(steps):
            prev_blood = self.blood_conc   # save pre-step value for trapezoid

            delta_blood = (
                - self.ke  * self.blood_conc
                - self.k12 * self.blood_conc
                + self.k21 * self.tissue_conc
            ) * dt

            delta_tissue = (
                self.k12 * self.blood_conc
                - self.k21 * self.tissue_conc
            ) * dt

            self.blood_conc  = max(0.0, self.blood_conc  + delta_blood)
            self.tissue_conc = max(0.0, self.tissue_conc + delta_tissue)

            # Trapezoidal AUC: average of before and after concentration
            avg_conc = (prev_blood + self.blood_conc) * 0.5
            self.cumulative_liver_stress  += avg_conc * dt * liver_inv
            self.cumulative_kidney_stress += avg_conc * dt * kidney_inv

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
        """Reset patient to pre-dose state (PK parameters preserved)."""
        self.blood_conc  = 0.0
        self.tissue_conc = 0.0
        self.time        = 0.0
        self.cumulative_liver_stress  = 0.0
        self.cumulative_kidney_stress = 0.0


# ─────────────────────────────────────────────
# AGENT 2: HepatocyteAgent
# Watches liver stress.
# Returns 0.0 (fine) to 1.0 (overwhelmed).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class HepatocyteAgent:
    SATURATION_THRESHOLD = 80.0    # mg/L*h — below this, linear enzyme response
    MAX_STRESS           = 200.0   # mg/L*h — AUC at which saturation reaches 1.0

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        CYP450 saturation — how overwhelmed is the liver?

        FIX [Bug 1]: SATURATION_THRESHOLD is now used. Below it the response
        is linear (enzyme capacity ample). Above it a sigmoid accelerates
        saturation to model Michaelis-Menten kinetics qualitatively — enzyme
        reserves deplete sharply once the threshold is exceeded.

        Returns 0.0 (fine) to 1.0 (overwhelmed).
        """
        stress = state.get("liver_stress", 0.0)

        if stress <= cls.SATURATION_THRESHOLD:
            # Linear region: plenty of enzyme capacity remaining
            saturation = stress / cls.MAX_STRESS
        else:
            # Sigmoid acceleration past saturation threshold
            baseline  = cls.SATURATION_THRESHOLD / cls.MAX_STRESS
            remaining = 1.0 - baseline
            scale     = cls.MAX_STRESS - cls.SATURATION_THRESHOLD
            excess    = stress - cls.SATURATION_THRESHOLD
            # Map excess onto [-3, 3] for a clean sigmoid
            sigmoid_x = (excess / scale) * 6.0 - 3.0
            sigmoid_y = 1.0 / (1.0 + math.exp(-sigmoid_x))
            saturation = baseline + remaining * sigmoid_y

        return round(min(1.0, max(0.0, saturation)), 4)


# ─────────────────────────────────────────────
# AGENT 3: ImmuneAgent
# Watches immune/inflammatory response.
# Returns 0.0 (calm) to 1.0 (severe reaction).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class ImmuneAgent:
    REACTION_THRESHOLD = 5.0    # mg/L — below this: no immune reaction
    SEVERE_THRESHOLD   = 20.0   # mg/L — above this: maximal reaction

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        Cytokine/IL-6 signal — how inflamed is the body?
        Linear interpolation between reaction and severe thresholds.

        Returns 0.0 (no reaction) to 1.0 (severe inflammatory reaction).
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
    STRESS_FOR_IMPAIRMENT = 100.0
    STRESS_FOR_FAILURE    = 400.0

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        GFR fraction — how well are the kidneys filtering?

        Returns 1.0 (healthy) to 0.0 (failed).
        """
        stress = state.get("kidney_stress", 0.0)

        if stress <= cls.STRESS_FOR_IMPAIRMENT:
            return 1.0
        elif stress >= cls.STRESS_FOR_FAILURE:
            return 0.0
        else:
            return 1.0 - (stress - cls.STRESS_FOR_IMPAIRMENT) / (cls.STRESS_FOR_FAILURE - cls.STRESS_FOR_IMPAIRMENT)


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
#   Grade 3 = severe        <- counts as a DLT
#   Grade 4 = life-threatening
# ─────────────────────────────────────────────

class MeasurementAgent:

    @classmethod
    def get_labs(cls, state: dict) -> dict:
        """
        Simulate blood test results from body state.

        FIX [Bug 2]: noise is now deterministic — seeded from the state values
        so the same physiological state always produces the same lab numbers.
        Previously, unseeded random.gauss() caused grade_dlt() to return
        different grades on repeated calls with identical inputs.

        Normal ranges:
          Heart rate:  60-100 bpm
          Systolic BP: 90-120 mmHg
          ALT:         7-56 U/L
          Creatinine:  0.6-1.2 mg/dL
          WBC:         4.5-11.0 K/uL
        """
        conc    = state.get("blood_conc",    0.0)
        lstress = state.get("liver_stress",  0.0)
        kstress = state.get("kidney_stress", 0.0)

        # Deterministic seed: same physiology -> same labs, always
        seed = int(abs(conc * 1000 + lstress * 7 + kstress * 13)) % (2 ** 31)
        rng  = random.Random(seed)
        noise = lambda scale: rng.gauss(0, scale)

        hr          = 72  + (conc * 2.0)    + noise(3)
        bp_systolic = 120 - (conc * 1.5)    + noise(5)
        alt         = 35  + (lstress * 0.8) + noise(5)
        creatinine  = 1.0 + (kstress * 0.005) + noise(0.1)
        wbc         = max(0.1, 7.5 - (conc * 0.3) + noise(0.5))

        return {
            "heart_rate":  round(hr,         1),
            "bp_systolic": round(bp_systolic, 1),
            "alt":         round(alt,         1),
            "creatinine":  round(creatinine,  2),
            "wbc":         round(wbc,         2),
        }

    @classmethod
    def grade_dlt(cls, state: dict) -> int:
        """
        Classify side-effect severity: grade 0 (none) to 4 (life-threatening).
        Grade 3+ counts as a DLT.
        """
        labs   = cls.get_labs(state)
        grades = []

        alt = labs["alt"]
        if   alt < 56:  grades.append(0)
        elif alt < 112: grades.append(1)
        elif alt < 175: grades.append(2)
        elif alt < 280: grades.append(3)
        else:           grades.append(4)

        cr = labs["creatinine"]
        if   cr < 1.5: grades.append(0)
        elif cr < 2.0: grades.append(1)
        elif cr < 3.0: grades.append(2)
        elif cr < 6.0: grades.append(3)
        else:          grades.append(4)

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
            if dlt_count >= 2:
                return f"DE-ESCALATE: {dlt_count}/{cohort_size} patients have serious side effects."
            elif cyp > 0.8:
                return "HOLD: Liver saturation is critically high, risk of toxicity."
            elif gfr < 0.5:
                return "HOLD: Kidney function is significantly impaired."
            else:
                return "ESCALATE: Safety signals are within acceptable limits."