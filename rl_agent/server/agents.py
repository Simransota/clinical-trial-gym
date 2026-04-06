"""
agents.py — Step 3: All 5 agents for the multi-agent clinical simulation layer.

Five agents in total:
  1. PatientAgent      — runs the 2-compartment ODE (the patient's body)
  2. HepatocyteAgent   — watches the liver (rule-based)
  3. ImmuneAgent       — watches the immune system (rule-based)
  4. RenalAgent        — watches the kidneys (rule-based)
  5. MeasurementAgent  — simulates lab blood tests, grades DLTs (rule-based)
  6. DoctorAgent       — reads all signals, calls LLM, writes recommendation
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
        age           — affects clearance rate
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
        self.ke = 0.15 * hepatic_factor  # elimination rate (liver clearance)
        self.k12 = 0.08        # rate blood → tissue
        self.k21 = 0.04        # rate tissue → blood
        self.Vd = 0.7 * weight_kg  # volume of distribution (liters)

        # State: how much drug (mg) is in each compartment right now
        self.blood_conc = 0.0    # mg/L in bloodstream
        self.tissue_conc = 0.0   # mg/L in tissues
        self.time = 0.0          # hours elapsed

        # Accumulated stress on organs
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
        """
        steps = 10  # sub-steps for numerical stability
        dt = hours / steps

        for _ in range(steps):
            # How much drug moves between compartments this tiny timestep
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

        self.time += hours

        # Accumulate organ stress based on drug concentration
        self.cumulative_liver_stress  += self.blood_conc * hours * (1.0 / max(self.hepatic_factor, 0.1))
        self.cumulative_kidney_stress += self.blood_conc * hours * (1.0 / max(self.renal_factor, 0.1))

    def get_state(self) -> dict:
        """Return current body state as a dictionary."""
        return {
            "blood_conc":   self.blood_conc,       # mg/L in blood
            "tissue_conc":  self.tissue_conc,       # mg/L in tissue
            "time":         self.time,              # hours
            "liver_stress": self.cumulative_liver_stress,
            "kidney_stress":self.cumulative_kidney_stress,
            "weight_kg":    self.weight_kg,
            "ke":           self.ke,               # elimination rate
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
    # Threshold above which CYP450 enzymes get saturated
    SATURATION_THRESHOLD = 80.0   # mg/L * hours (cumulative exposure)
    MAX_STRESS = 200.0

    @classmethod
    def observe(cls, state: dict) -> float:
        """
        CYP450 saturation — how overwhelmed is the liver?

        Uses cumulative liver stress (total drug exposure over time).
        0.0 = liver perfectly fine
        1.0 = liver completely saturated / overwhelmed
        """
        stress = state.get("liver_stress", 0.0)
        # Sigmoid-like response — gradual increase then sharp rise
        saturation = stress / cls.MAX_STRESS
        # Cap at 1.0
        return min(1.0, max(0.0, saturation))


# ─────────────────────────────────────────────
# AGENT 3: ImmuneAgent
# Watches immune/inflammatory response.
# Returns 0.0 (calm) to 1.0 (severe reaction).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class ImmuneAgent:
    # Blood concentration above which immune reaction starts
    REACTION_THRESHOLD = 5.0    # mg/L
    SEVERE_THRESHOLD   = 20.0   # mg/L

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
            # Linear interpolation between thresholds
            return (conc - cls.REACTION_THRESHOLD) / (cls.SEVERE_THRESHOLD - cls.REACTION_THRESHOLD)


# ─────────────────────────────────────────────
# AGENT 4: RenalAgent
# Watches kidney function.
# Returns 1.0 (fully healthy) to 0.0 (failed).
# Rule-based — no LLM.
# ─────────────────────────────────────────────

class RenalAgent:
    STRESS_FOR_IMPAIRMENT = 100.0  # cumulative kidney stress for mild impairment
    STRESS_FOR_FAILURE    = 400.0  # cumulative kidney stress for failure

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
            # Linear decline
            return 1.0 - (stress - cls.STRESS_FOR_IMPAIRMENT) / (cls.STRESS_FOR_FAILURE - cls.STRESS_FOR_IMPAIRMENT)


# ─────────────────────────────────────────────
# AGENT 5: MeasurementAgent
# Simulates blood test results (lab panels).
# Grades DLT (serious side effect) per patient.
# Rule-based — no LLM.
#
# DLT Grade scale (standard oncology):
#   Grade 0 = no side effect
#   Grade 1 = mild
#   Grade 2 = moderate
#   Grade 3 = severe (counts as a DLT — trial stopping rules apply)
#   Grade 4 = life-threatening
#   Grade 5 = death
# ─────────────────────────────────────────────

class MeasurementAgent:

    @classmethod
    def get_labs(cls, state: dict) -> dict:
        """
        Simulate blood test results from body state.
        Returns realistic lab values derived from ODE state.

        Normal ranges (for reference):
          Heart rate (HR):    60-100 bpm
          Systolic BP:        90-120 mmHg
          ALT (liver enzyme): 7-56 U/L
          Creatinine:         0.6-1.2 mg/dL
        """
        conc   = state.get("blood_conc", 0.0)
        lstress = state.get("liver_stress", 0.0)
        kstress = state.get("kidney_stress", 0.0)

        # Add small random variation (realistic — lab tests aren't perfect)
        noise = lambda scale: random.gauss(0, scale)

        # Heart rate: rises with high drug concentrations
        hr = 72 + (conc * 2.0) + noise(3)

        # Blood pressure: slight drop with higher concentrations
        bp_systolic = 120 - (conc * 1.5) + noise(5)

        # ALT (liver enzyme): rises with liver stress
        # Normal = ~35 U/L. Grade 3 DLT = >5x normal = >175 U/L
        alt = 35 + (lstress * 0.8) + noise(5)

        # Creatinine (kidney waste): rises when kidneys struggle
        # Normal = ~1.0. Grade 3 DLT = >3x normal = >3.0 mg/dL
        creatinine = 1.0 + (kstress * 0.005) + noise(0.1)

        # White blood cell count: can drop with cytotoxic drugs
        # Normal = 4.5-11.0 K/uL. Grade 3 = <1.0 K/uL
        wbc = max(0.1, 7.5 - (conc * 0.3) + noise(0.5))

        return {
            "heart_rate":   round(hr, 1),
            "bp_systolic":  round(bp_systolic, 1),
            "alt":          round(alt, 1),       # liver enzyme
            "creatinine":   round(creatinine, 2), # kidney waste
            "wbc":          round(wbc, 2),        # white blood cells
        }

    @classmethod
    def grade_dlt(cls, state: dict) -> int:
        """
        Classify how severe a patient's side effects are.
        Returns a grade from 0 (none) to 4 (life-threatening).

        Standard NCI CTCAE grading criteria (simplified).
        Grade 3+ = counts as a DLT in your trial.
        """
        labs = cls.get_labs(state)

        grades = []

        # Liver toxicity (ALT elevation)
        alt = labs["alt"]
        if   alt < 56:   grades.append(0)   # normal
        elif alt < 112:  grades.append(1)   # 1-2x normal = Grade 1
        elif alt < 175:  grades.append(2)   # 2-5x normal = Grade 2
        elif alt < 280:  grades.append(3)   # 5-8x normal = Grade 3 (DLT!)
        else:            grades.append(4)   # >8x normal  = Grade 4

        # Kidney toxicity (creatinine elevation)
        cr = labs["creatinine"]
        if   cr < 1.5:   grades.append(0)
        elif cr < 2.0:   grades.append(1)
        elif cr < 3.0:   grades.append(2)
        elif cr < 6.0:   grades.append(3)   # Grade 3 (DLT!)
        else:            grades.append(4)

        # Blood count toxicity (low WBC = neutropenia)
        wbc = labs["wbc"]
        if   wbc >= 4.5: grades.append(0)
        elif wbc >= 2.0: grades.append(1)
        elif wbc >= 1.0: grades.append(2)
        elif wbc >= 0.5: grades.append(3)   # Grade 3 (DLT!)
        else:            grades.append(4)

        # Return the worst (highest) grade across all lab tests
        return max(grades)


# ─────────────────────────────────────────────
# AGENT 6: DoctorAgent
# The ONLY LLM agent. One LLM call per step.
# Reads all signals and writes one sentence.
# ─────────────────────────────────────────────

class DoctorAgent:

    def __init__(self):
        # Reads API credentials from environment variables
        # (as required by hackathon spec)
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy"),
        )
        self.model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    def recommend(
        self,
        cmax: float,          # peak blood concentration
        dlt_count: int,       # how many patients had serious side effects
        cohort_size: int,     # total patients in cohort
        cyp: float,           # liver saturation 0-1
        gfr: float,           # kidney function 0-1
        immune: float,        # immune reaction 0-1
        current_dose: float,  # current dose mg/kg
    ) -> str:
        """
        Ask the LLM for a one-sentence clinical recommendation.
        This is called ONCE per environment step (not per patient).
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
                temperature=0.3,  # low temperature = more consistent medical advice
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback if LLM call fails — rule-based default
            if dlt_count >= 2:
                return f"DE-ESCALATE: {dlt_count}/{cohort_size} patients have serious side effects."
            elif cyp > 0.8:
                return "HOLD: Liver saturation is critically high, risk of toxicity."
            elif gfr < 0.5:
                return "HOLD: Kidney function is significantly impaired."
            else:
                return "ESCALATE: Safety signals are within acceptable limits."
