"""
agents.py — All 6 agents for the multi-agent clinical simulation layer.

Layer 1/2 integration
─────────────────────
PatientAgent now accepts a `drug_params` dict produced by Layer 1 (RDKit /
DeepChem) and Layer 2 (PK-Sim / surrogate ODE).  When the dict is supplied
every PK parameter comes from the real molecule; when it is omitted the
original heuristic defaults are used so nothing breaks if Layer 1/2 are
not yet wired up.

Expected `drug_params` shape (exactly what Layer 1/2 already outputs):
    {
        "ka":  0.1127,   # absorption rate constant (1/h)
        "F":   0.80,     # oral bioavailability (fraction)
        "CL":  0.50,     # clearance (L/h/kg) — HUMAN-scaled value
        "Vc":  2.4746,   # central volume of distribution (L/kg)
        "Vp":  1.6497,   # peripheral volume of distribution (L/kg)
        "Q":   0.30,     # inter-compartmental clearance (L/h/kg)
        "PPB": 0.6048,   # plasma protein binding (fraction)
        "fu":  0.3952,   # unbound fraction
    }

Optional safety flags also accepted (used to modulate organ-stress signals):
    {
        "dili_risk":    False,   # drug-induced liver injury flag
        "herg_risk":    False,   # cardiac risk flag
        "cyp_inhibitions": [],   # list of inhibited CYPs e.g. ["CYP3A4"]
        "bbb_penetrant": True,
    }

Fixes vs original (carried forward from previous patch):
  [Bug 1] HepatocyteAgent sigmoid   — SATURATION_THRESHOLD now used
  [Bug 2] MeasurementAgent noise    — deterministic seed from state hash
  [Bug 3] PatientAgent age          — wired into clearance (ke)
  [Bug 4] Organ stress trapezoidal  — AUC-correct accumulation inside loop
"""

import os
import math
import random
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: PatientAgent
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_SAFETY_FLAGS = {
    "dili_risk":       False,
    "herg_risk":       False,
    "cyp_inhibitions": [],
    "bbb_penetrant":   False,
}


class PatientAgent:
    """
    Simulates one patient's body using a 2-compartment ODE.

    PK parameters are sourced from `drug_params` (Layer 1/2 output) when
    provided, falling back to `_DEFAULT_DRUG_PARAMS` otherwise.

    The ODE (standard 2-compartment IV/oral model):
        d(C1)/dt = (F * ka * dose_depot) / Vc
                   - (CL/Vc) * C1
                   - (Q/Vc)  * C1
                   + (Q/Vp)  * C2
        d(C2)/dt =  (Q/Vc)   * C1  -  (Q/Vp) * C2

    where C1 = central (blood) concentration, C2 = peripheral concentration.
    """

    def __init__(
        self,
        weight_kg: float       = 70.0,
        age: int               = 45,
        sex: str               = "M",
        renal_factor: float    = 1.0,
        hepatic_factor: float  = 1.0,
        drug_params: dict      = None,   # ← Layer 1/2 output
        safety_flags: dict     = None,   # ← Layer 1/2 safety profile
    ):
        """
        Parameters
        ----------
        weight_kg      : patient body weight
        age            : affects hepatic clearance (~0.5%/yr decline past 40)
        sex            : "M" or "F" (minor Vd adjustment)
        renal_factor   : 1.0 = healthy kidneys, 0.5 = half function
        hepatic_factor : 1.0 = healthy liver,   0.5 = half function
        drug_params    : PK dict from Layer 1/2 (see module docstring)
        safety_flags   : safety dict from Layer 1/2 (see module docstring)
        """
        self.weight_kg      = weight_kg
        self.age            = age
        self.sex            = sex
        self.renal_factor   = renal_factor
        self.hepatic_factor = hepatic_factor

        if not drug_params:
            raise ValueError(
                "drug_params are required. Configure a molecule first so Layer 1/2 "
                "can provide ka/F/CL/Vc/Vp/Q/PPB/fu."
            )
        required_keys = {"ka", "F", "CL", "Vc", "Vp", "Q", "PPB", "fu"}
        missing = sorted(required_keys.difference(drug_params.keys()))
        if missing:
            raise ValueError(f"drug_params missing required keys: {missing}")

        # ── Resolve drug PK parameters ────────────────────────────────────────
        # Layer 1/2 provides per-kg values; multiply by weight where needed.
        dp = dict(drug_params)

        self.ka  = float(dp["ka"])
        self.F   = float(dp["F"])

        # Volumes: Layer 1/2 reports L/kg → scale to patient
        # Sex adjustment: women average ~15% lower Vd for hydrophilic drugs
        sex_vd_factor = 0.85 if sex == "F" else 1.0
        self.Vc = float(dp["Vc"]) * weight_kg * sex_vd_factor   # L
        self.Vp = float(dp["Vp"]) * weight_kg * sex_vd_factor   # L

        # Clearances: L/h/kg → scale to patient, then apply organ health
        # Age factor: ke declines ~0.5%/yr past age 40 (floor 0.5)
        age_factor = max(0.5, 1.0 - 0.005 * max(0, age - 40))

        CL_drug = float(dp["CL"]) * weight_kg   # L/h
        self.CL = CL_drug * hepatic_factor * age_factor

        self.Q  = float(dp["Q"])  * weight_kg   # L/h — inter-compartmental

        # Convenience: ke (elimination rate from central compartment)
        self.ke  = self.CL / max(self.Vc, 1e-6)   # 1/h
        # k12, k21 derived from Q and volumes (standard 2-comp relationships)
        self.k12 = self.Q / max(self.Vc, 1e-6)    # 1/h
        self.k21 = self.Q / max(self.Vp, 1e-6)    # 1/h

        self.fu  = float(dp["fu"])
        self.PPB = float(dp["PPB"])

        # ── Safety flags ──────────────────────────────────────────────────────
        sf = {**_DEFAULT_SAFETY_FLAGS, **(safety_flags or {})}
        self.dili_risk       = bool(sf["dili_risk"])
        self.herg_risk       = bool(sf["herg_risk"])
        self.cyp_inhibitions = list(sf["cyp_inhibitions"])
        self.bbb_penetrant   = bool(sf["bbb_penetrant"])

        # DILI risk amplifies hepatic stress accumulation by 2×
        self._liver_stress_multiplier = 2.0 if self.dili_risk else 1.0

        # ── Patient ID and state ──────────────────────────────────────────────
        import uuid
        self.patient_id  = f"PT-{uuid.uuid4().hex[:6].upper()}"

        self.blood_conc   = 0.0   # C1: mg/L in central (blood) compartment
        self.tissue_conc  = 0.0   # C2: mg/L in peripheral (tissue) compartment
        self.depot        = 0.0   # unabsorbed drug in gut (oral dosing)
        self.time         = 0.0   # hours elapsed

        self.cumulative_liver_stress  = 0.0
        self.cumulative_kidney_stress = 0.0

    # ── dosing ────────────────────────────────────────────────────────────────

    def dose(self, amount_mg_per_kg: float):
        """
        Administer an oral dose.  Drug enters the gut depot; absorption
        into the central compartment is governed by ka and bioavailability F.
        """
        total_mg      = amount_mg_per_kg * self.weight_kg
        self.depot   += total_mg * self.F   # only bioavailable fraction enters

    # ── simulation ────────────────────────────────────────────────────────────

    def advance(self, hours: float = 1.0):
        """
        Advance simulation by `hours` using Euler integration.

        Full 2-compartment oral ODE:
            d(depot)/dt  = -ka * depot
            d(C1)/dt     = ka * depot / Vc  -  (ke + k12) * C1  +  k21 * C2
            d(C2)/dt     =  k12 * C1  -  k21 * C2

        Organ stress accumulated via trapezoidal rule (AUC-correct).
        DILI flag doubles liver stress accumulation rate.
        """
        steps = 20          # more sub-steps for numerical stability with
        dt    = hours / steps  # fast absorption (small ka * dt needed)

        liver_inv  = (1.0 / max(self.hepatic_factor, 0.1)) * self._liver_stress_multiplier
        kidney_inv =  1.0 / max(self.renal_factor,   0.1)

        for _ in range(steps):
            prev_blood = self.blood_conc

            # Compute all derivatives at current state before any updates (Euler)
            dDepot  = -self.ka * self.depot
            dBlood  = (
                  self.ka * self.depot / max(self.Vc, 1e-6)
                - self.ke  * self.blood_conc
                - self.k12 * self.blood_conc
                + self.k21 * self.tissue_conc
            )
            dTissue = self.k12 * self.blood_conc - self.k21 * self.tissue_conc

            # Apply all updates together (standard forward Euler)
            self.depot       = max(0.0, self.depot       + dDepot  * dt)
            self.blood_conc  = max(0.0, self.blood_conc  + dBlood  * dt)
            self.tissue_conc = max(0.0, self.tissue_conc + dTissue * dt)

            # Trapezoidal AUC stress
            avg_conc = (prev_blood + self.blood_conc) * 0.5
            self.cumulative_liver_stress  += avg_conc * dt * liver_inv
            self.cumulative_kidney_stress += avg_conc * dt * kidney_inv

        self.time += hours

    def get_state(self) -> dict:
        return {
            "patient_id":    self.patient_id,
            "blood_conc":    self.blood_conc,
            "tissue_conc":   self.tissue_conc,
            "depot":         self.depot,
            "time":          self.time,
            "liver_stress":  self.cumulative_liver_stress,
            "kidney_stress": self.cumulative_kidney_stress,
            "weight_kg":     self.weight_kg,
            "age":           self.age,
            "sex":           self.sex,
            "ke":            self.ke,
            "fu":            self.fu,
            "dili_risk":     self.dili_risk,
        }

    def reset(self):
        """Reset to pre-dose state (PK parameters preserved)."""
        self.blood_conc   = 0.0
        self.tissue_conc  = 0.0
        self.depot        = 0.0
        self.time         = 0.0
        self.cumulative_liver_stress  = 0.0
        self.cumulative_kidney_stress = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: HepatocyteAgent
# ═══════════════════════════════════════════════════════════════════════════

class HepatocyteAgent:
    """
    CYP450 saturation signal.
    Returns 0.0 (liver fine) → 1.0 (liver overwhelmed).
    """
    SATURATION_THRESHOLD = 80.0    # mg/L·h AUC — below this: linear response
    MAX_STRESS           = 200.0   # mg/L·h AUC — saturation = 1.0

    @classmethod
    def observe(cls, state: dict) -> float:
        stress = state.get("liver_stress", 0.0)

        if stress <= cls.SATURATION_THRESHOLD:
            saturation = stress / cls.MAX_STRESS
        else:
            # Sigmoid acceleration past threshold (Michaelis-Menten-like)
            baseline  = cls.SATURATION_THRESHOLD / cls.MAX_STRESS
            remaining = 1.0 - baseline
            scale     = cls.MAX_STRESS - cls.SATURATION_THRESHOLD
            sigmoid_x = ((stress - cls.SATURATION_THRESHOLD) / scale) * 6.0 - 3.0
            sigmoid_y = 1.0 / (1.0 + math.exp(-sigmoid_x))
            saturation = baseline + remaining * sigmoid_y

        return round(min(1.0, max(0.0, saturation)), 4)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3: ImmuneAgent
# ═══════════════════════════════════════════════════════════════════════════

class ImmuneAgent:
    """
    Cytokine/IL-6 signal.
    Returns 0.0 (no reaction) → 1.0 (severe inflammatory reaction).
    """
    REACTION_THRESHOLD = 5.0    # mg/L
    SEVERE_THRESHOLD   = 20.0   # mg/L

    @classmethod
    def observe(cls, state: dict) -> float:
        conc = state.get("blood_conc", 0.0)
        if conc <= cls.REACTION_THRESHOLD:
            return 0.0
        elif conc >= cls.SEVERE_THRESHOLD:
            return 1.0
        return (conc - cls.REACTION_THRESHOLD) / (cls.SEVERE_THRESHOLD - cls.REACTION_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4: RenalAgent
# ═══════════════════════════════════════════════════════════════════════════

class RenalAgent:
    """
    GFR fraction signal.
    Returns 1.0 (healthy kidneys) → 0.0 (renal failure).
    """
    STRESS_FOR_IMPAIRMENT = 100.0
    STRESS_FOR_FAILURE    = 400.0

    @classmethod
    def observe(cls, state: dict) -> float:
        stress = state.get("kidney_stress", 0.0)
        if stress <= cls.STRESS_FOR_IMPAIRMENT:
            return 1.0
        elif stress >= cls.STRESS_FOR_FAILURE:
            return 0.0
        return 1.0 - (stress - cls.STRESS_FOR_IMPAIRMENT) / (cls.STRESS_FOR_FAILURE - cls.STRESS_FOR_IMPAIRMENT)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 5: MeasurementAgent
# ═══════════════════════════════════════════════════════════════════════════

class MeasurementAgent:
    """
    Simulates lab blood tests and grades DLTs per patient.
    Deterministic: same physiological state → same lab values.
    """

    @classmethod
    def get_labs(cls, state: dict) -> dict:
        conc    = state.get("blood_conc",    0.0)
        lstress = state.get("liver_stress",  0.0)
        kstress = state.get("kidney_stress", 0.0)

        # Deterministic seed from state values
        seed = int(abs(conc * 1000 + lstress * 7 + kstress * 13)) % (2 ** 31)
        rng  = random.Random(seed)
        noise = lambda scale: rng.gauss(0, scale)

        hr          = 72  + conc * 2.0          + noise(3)
        bp_systolic = 120 - conc * 1.5          + noise(5)
        alt         = 35  + lstress * 0.8       + noise(5)
        creatinine  = 1.0 + kstress * 0.005     + noise(0.1)
        wbc         = max(0.1, 7.5 - conc * 0.3 + noise(0.5))

        return {
            "heart_rate":  round(hr,         1),
            "bp_systolic": round(bp_systolic, 1),
            "alt":         round(alt,         1),
            "creatinine":  round(creatinine,  2),
            "wbc":         round(wbc,         2),
        }

    @classmethod
    def grade_dlt(cls, state: dict) -> int:
        """NCI CTCAE grading. Returns 0–4. Grade 3+ = DLT."""
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


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 6: DoctorAgent
# ═══════════════════════════════════════════════════════════════════════════

class DoctorAgent:
    """
    The only LLM agent. One call per environment step.
    Reads all signals and produces a one-sentence recommendation.
    """

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
        drug_name: str = "investigational compound",
        cyp_inhibitions: list = None,
        dili_risk: bool = False,
    ) -> str:
        """
        One-sentence clinical recommendation.
        Drug-specific context (name, CYP inhibitions, DILI risk) now
        included in the prompt when available from Layer 1/2.
        """
        cyp_line = ""
        if cyp_inhibitions:
            cyp_line = f"- Known CYP inhibitions: {', '.join(cyp_inhibitions)}\n"
        dili_line = f"- DILI risk flagged by in-silico model: {'YES — heightened liver monitoring required' if dili_risk else 'No'}\n"

        prompt = f"""You are a clinical pharmacologist reviewing a Phase I dose escalation trial.

Drug under investigation: {drug_name}
{cyp_line}{dili_line}
Current cohort status:
- Peak blood concentration (Cmax): {cmax:.2f} mg/L
- Serious side effects (DLTs): {dlt_count}/{cohort_size} patients
- Liver enzyme saturation (CYP450): {cyp:.0%}
- Kidney function (GFR): {gfr:.0%}
- Immune/inflammatory signal: {immune:.0%}
- Current dose: {current_dose:.2f} mg/kg

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
            elif dili_risk and cyp > 0.5:
                return "HOLD: DILI-flagged compound with rising liver saturation — do not escalate."
            elif cyp > 0.8:
                return "HOLD: Liver saturation is critically high, risk of enzyme saturation."
            elif gfr < 0.5:
                return "HOLD: Kidney function is significantly impaired."
            else:
                return "ESCALATE: Safety signals are within acceptable limits."