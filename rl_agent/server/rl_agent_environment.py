# server/rl_agent_environment.py

"""
Clinical Trial Gym — Main Environment.
Wires together all agents and exposes step/reset/state API.

Layer 1/2 integration
─────────────────────
The environment now accepts a `drug_profile` dict at construction time.
This is the combined output of Layer 1 (RDKit/DeepChem) and Layer 2
(PK-Sim / surrogate ODE allometric scaling).

Expected `drug_profile` shape:
    {
        # ── Identity ──────────────────────────────────────────────────────
        "name":   "Aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",

        # ── PK parameters (human-scaled, from Layer 2 allometric output) ─
        "drug_params": {
            "ka":  0.1127,
            "F":   0.80,
            "CL":  0.50,    # L/h/kg  (human-scaled CL from Layer 2)
            "Vc":  2.4746,  # L/kg
            "Vp":  1.6497,  # L/kg
            "Q":   0.30,    # L/h/kg
            "PPB": 0.6048,
            "fu":  0.3952,
        },

        # ── Safety flags (from Layer 1 DeepChem/RDKit) ────────────────────
        "safety_flags": {
            "dili_risk":       False,
            "herg_risk":       False,
            "cyp_inhibitions": [],
            "bbb_penetrant":   True,
            "overall_risk_score": 0.0,
        },

        # ── Starting dose — human equivalent from allometric scaling ──────
        # Layer 2 already outputs this:  "Human Equivalent Dose: 1.62 mg/kg"
        "human_equivalent_dose": 1.62,   # mg/kg  (Layer 2 allometric output)
    }

When `drug_profile` is None the environment behaves exactly as before
(heuristic defaults, starting dose 1.0 mg/kg).

Bug fixes carried forward:
  [Bug 5] History appended before reward so history[-1] = current step
  [Bug 6] action.escalate saved in history so stopping reward fires
  [Bug 7] Allometric formula rewritten with explicit variable names
"""

import random
import numpy as np
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Default drug profile (aspirin-like) used when no molecule is configured.
# This lets the Phase 2 evaluator run episodes and invoke graders without
# needing to POST /drug first.
DEFAULT_DRUG_PROFILE = {
    "name": "default_compound",
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "drug_params": {
        "ka": 0.1127,
        "F": 0.80,
        "CL": 0.50,
        "Vc": 2.4746,
        "Vp": 1.6497,
        "Q": 0.30,
        "PPB": 0.6048,
        "fu": 0.3952,
    },
    "safety_flags": {
        "dili_risk": False,
        "herg_risk": False,
        "cyp_inhibitions": [],
        "bbb_penetrant": False,
        "overall_risk_score": 0.1,
    },
    "human_equivalent_dose": 1.62,
    "task_targets": {
        "phase_i_dosing": 1.62,
        "allometric_scaling": 1.62,
        "combo_ddi": 1.62,
    },
    "admet_summary": {
        "solubility": "moderate",
        "permeability": "high",
        "plasma_protein_binding": 0.6048,
    },
}

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
FDA_RULES = {
    "max_dlt_rate":   0.33,
    "max_steps":      10,
    "max_dose_mg_kg": 50.0,
}


class RlAgentEnvironment(Environment):
    """
    Clinical Trial Gym environment.

    One episode = one complete Phase I dose escalation trial.

    The agent:
      - Sees: plasma_conc, dlt_count, organ signals, doctor advice
      - Does: choose next_dose, cohort_size, escalate flag
      - Goal: find the RP2D as safely and quickly as possible

    Parameters
    ----------
    drug_profile : dict, optional
        Combined Layer 1/2 output (see module docstring).
        When supplied, PK parameters and starting dose come from the real
        molecule rather than heuristic defaults.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, drug_profile: dict = None):
        self._state    = State(episode_id=str(uuid4()), step_count=0)
        self.cohort       = []
        self.history      = []
        self.pk_traces    = []   # list[list[dict]] — per-step, per-patient time-series
        self.cohort_log   = []   # list[list[dict]] — per-step patient demographics + outcomes
        self.rp2d_dose    = None
        self._done        = False
        self._use_llm  = True
        self.doctor    = DoctorAgent() if self._use_llm else None

        # ── Unpack drug profile from Layer 1/2 ───────────────────────────────
        self.drug_name = "unconfigured"
        self.drug_params = {}
        self.safety_flags = {}
        self._start_dose = 0.0
        self._drug_configured = False
        self._drug_profile_hed = None
        self._configured_smiles = None
        self._task_targets = {
            "phase_i_dosing": None,
            "allometric_scaling": None,
            "combo_ddi": None,
        }
        self.configure_drug(drug_profile if drug_profile else DEFAULT_DRUG_PROFILE)

        self.current_dose = self._start_dose

    def _configure_from_action(self, action: RlAgentAction) -> None:
        drug_smiles = getattr(action, "drug_smiles", None)
        if not drug_smiles:
            return

        should_reconfigure = (
            not self._drug_configured
            or (
                self._state.step_count == 0
                and drug_smiles != self._configured_smiles
            )
        )
        if not should_reconfigure:
            return

        try:
            from ..drug_profile_builder import DrugProfileBuilder
        except ImportError:
            from rl_agent.drug_profile_builder import DrugProfileBuilder

        builder = DrugProfileBuilder(
            smiles=drug_smiles,
            name=getattr(action, "drug_name", None) or "investigational compound",
            source_species=getattr(action, "source_species", None) or "rat",
            animal_dose_mgkg=float(getattr(action, "animal_dose_mgkg", None) or 8.0),
        )
        self.configure_drug(builder.build())

    # ── reset() ─────────────────────────────────────────────────────────────
    def reset(self) -> RlAgentObservation:
        """Start a fresh trial episode."""
        self._state       = State(episode_id=str(uuid4()), step_count=0)
        self.current_dose = self._start_dose
        self.history      = []
        self.pk_traces    = []
        self.cohort_log   = []
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
            doctor_recommendation=(
                f"Trial started for {self.drug_name}. "
                f"Starting dose {self.current_dose:.3f} mg/kg "
                f"(1/10 of allometric HED). Begin escalation cautiously."
            ),
            done=False,
            reward=0.0,
        )

    # ── step() ──────────────────────────────────────────────────────────────
    def step(self, action: RlAgentAction) -> RlAgentObservation:
        self._configure_from_action(action)
        self._state.step_count += 1

        # 1. Record previous dose before updating (used by reward)
        prev_dose = self.current_dose

        # 2. Apply action (clamped)
        self.current_dose = max(0.1, min(action.next_dose, FDA_RULES["max_dose_mg_kg"]))
        cohort_n          = max(3, min(action.cohort_size, 6))

        # 3. Simulate cohort
        self.cohort = self._make_cohort(cohort_n)
        self._run_cohort(self.current_dose)

        # 4. Collect signals
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

        # 4b. Capture PK traces for Layer 5 analysis
        # Re-simulate each patient to get the full time-series (24 timepoints)
        step_traces = []
        for p in self.cohort:
            trace = self._get_pk_trace(p, self.current_dose)
            step_traces.append(trace)
        self.pk_traces.append(step_traces)

        # 4c. Capture patient demographics + outcomes for Layer 5
        step_cohort = []
        for i, p in enumerate(self.cohort):
            s = states[i]
            step_cohort.append({
                "patient_id":    s.get("patient_id", f"PT-{i}"),
                "age":           s["age"],
                "sex":           s["sex"],
                "weight_kg":     s["weight_kg"],
                "ke":            s["ke"],
                "fu":            s["fu"],
                "renal_factor":  p.renal_factor,
                "hepatic_factor":p.hepatic_factor,
                "blood_conc":    s["blood_conc"],
                "liver_stress":  s["liver_stress"],
                "kidney_stress": s["kidney_stress"],
                "hep_signal":    hep_signals[i],
                "imm_signal":    imm_signals[i],
                "ren_signal":    ren_signals[i],
                "dlt_grade":     dlt_grades[i],
                "is_dlt":        dlt_grades[i] >= 3,
            })
        self.cohort_log.append(step_cohort)

        # 5. FDA stopping rules
        dlt_rate      = dlt_count / len(self.cohort)
        fda_stop      = dlt_rate > FDA_RULES["max_dlt_rate"]
        max_steps_hit = self._state.step_count >= FDA_RULES["max_steps"]
        agent_stopped = not action.escalate

        # Track the highest dose that did not trigger FDA stopping: this is the
        # running estimate of RP2D (highest tolerated dose seen so far).
        if not fda_stop:
            if self.rp2d_dose is None or self.current_dose > self.rp2d_dose:
                self.rp2d_dose = self.current_dose

        self._done = fda_stop or agent_stopped or max_steps_hit

        # 6. Append to history BEFORE computing reward (fixes off-by-one)
        self.history.append({
            "step":        self._state.step_count,
            "dose":        self.current_dose,
            "dlt_count":   dlt_count,
            "dlt_rate":    dlt_rate,
            "cohort_size": len(self.cohort),
            "escalate":    action.escalate,   # needed by stopping reward
            "reward":      0.0,               # back-filled below
        })

        # 7. Compute reward
        reward = self._compute_reward(
            dlt_count=dlt_count,
            cohort_size=len(self.cohort),
            fda_stop=fda_stop,
            avg_hep=avg_hep,
            avg_ren=avg_ren,
            prev_dose=prev_dose,
        )
        self.history[-1]["reward"] = reward

        # 8. Doctor recommendation (with drug-specific context from Layer 1/2)
        if self._use_llm and self.doctor:
            rec = self.doctor.recommend(
                cmax=avg_cmax,
                dlt_count=dlt_count,
                cohort_size=len(self.cohort),
                cyp=avg_hep,
                gfr=avg_ren,
                immune=avg_imm,
                current_dose=self.current_dose,
                drug_name=self.drug_name,
                cyp_inhibitions=self.safety_flags.get("cyp_inhibitions", []),
                dili_risk=self.safety_flags.get("dili_risk", False),
            )
        else:
            rec = self._rule_based_rec(dlt_count, len(self.cohort), fda_stop, avg_hep, avg_ren)

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
        target = float(self._task_targets.get("phase_i_dosing") or self._drug_profile_hed or 1.0)
        error = abs(self.rp2d_dose - target) / max(target, 1e-6)
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
        Score how well the agent's first dose matched the allometric HED.
        If Layer 1/2 provided a human_equivalent_dose we use that as ground
        truth; otherwise fall back to the original formula.
        """
        if not self.history:
            return 0.0
        proposed = self.history[0]["dose"]

        # Use Layer 2 HED as ground truth when available
        if hasattr(self, "_drug_profile_hed") and self._drug_profile_hed:
            human_equivalent = self._drug_profile_hed
        else:
            # Original allometric formula (explicit variable names)
            rat_dose       = 8.0
            rat_weight_kg  = 0.25
            human_weight_kg = 70.0
            allometric_exp  = 0.75
            human_equivalent = rat_dose * (human_weight_kg / rat_weight_kg) ** (allometric_exp - 1.0)

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
        target = float(self._task_targets.get("combo_ddi") or self._drug_profile_hed or 1.0)
        efficacy = max(0.0, 1.0 - abs(max_dose - target) / max(target, 1e-6))
        total_dlts   = sum(h["dlt_count"] for h in self.history)
        ddi_score    = max(0.0, 1.0 - total_dlts * 0.1)
        return round(min(1.0, 0.4 * efficacy + 0.4 * safety_score + 0.2 * ddi_score), 3)

    # ── private helpers ──────────────────────────────────────────────────────

    def _make_cohort(self, size: int):
        """
        Create `size` PatientAgents with randomised demographics.
        Drug PK parameters and safety flags from Layer 1/2 are passed
        to every patient so their ODE uses the real molecule's properties.
        """
        patients = []
        for _ in range(size):
            w   = max(50.0, min(100.0, random.gauss(70, 10)))
            age = random.randint(25, 75)
            sex = random.choice(["M", "F"])
            patients.append(PatientAgent(
                weight_kg      = w,
                age            = age,
                sex            = sex,
                renal_factor   = random.uniform(0.7, 1.0),
                hepatic_factor = random.uniform(0.7, 1.0),
                drug_params    = self.drug_params,   # ← Layer 1/2 required
                safety_flags   = self.safety_flags,  # ← Layer 1/2
            ))
        return patients

    def _run_cohort(self, dose: float, hours: float = 24.0):
        for p in self.cohort:
            p.reset()
            p.dose(dose)
            p.advance(hours=hours)

    def _get_pk_trace(self, patient, dose: float, hours: float = 24.0, n_points: int = 48) -> dict:
        """
        Re-simulate a single patient at fine time resolution to get the full
        concentration-time curve for plotting. Returns a dict with time,
        blood_conc, tissue_conc arrays — ready for Layer 5 PK curve plots.
        """
        # Clone a fresh patient with same parameters (PatientAgent already imported at module level)
        p = PatientAgent(
            weight_kg      = patient.weight_kg,
            age            = patient.age,
            sex            = patient.sex,
            renal_factor   = patient.renal_factor,
            hepatic_factor = patient.hepatic_factor,
            drug_params    = self.drug_params,
            safety_flags   = self.safety_flags,
        )
        p.dose(dose)
        dt = hours / n_points
        times, blood, tissue = [], [], []
        for i in range(n_points):
            times.append(round(i * dt, 3))
            blood.append(round(p.blood_conc, 4))
            tissue.append(round(p.tissue_conc, 4))
            p.advance(hours=dt)
        # Add final timepoint
        times.append(hours)
        blood.append(round(p.blood_conc, 4))
        tissue.append(round(p.tissue_conc, 4))
        return {
            "patient_id":    getattr(patient, "patient_id", "unknown"),
            "age":           patient.age,
            "sex":           patient.sex,
            "weight_kg":     round(patient.weight_kg, 1),
            "time_h":        times,
            "blood_conc":    blood,
            "tissue_conc":   tissue,
            "cmax":          round(max(blood), 4),
            "tmax":          round(times[blood.index(max(blood))], 2),
            "auc":           round(sum((blood[i] + blood[i+1]) / 2 * (times[i+1] - times[i])
                                       for i in range(len(times)-1)), 4),
        }

    def configure_drug(self, drug_profile: dict) -> None:
        """
        Reconfigure the environment with a new drug profile from Layer 1+2.

        Call this before reset() to apply the new molecule to the next episode.
        Called automatically by the POST /drug endpoint.

        Parameters
        ----------
        drug_profile : dict
            Output of DrugProfileBuilder.build() — contains drug_params,
            safety_flags, human_equivalent_dose, name, and admet_summary.
        """
        required_top = {"name", "drug_params", "safety_flags", "human_equivalent_dose"}
        missing_top = sorted(required_top.difference(drug_profile.keys()))
        if missing_top:
            raise ValueError(f"drug_profile missing required fields: {missing_top}")

        required_pk = {"ka", "F", "CL", "Vc", "Vp", "Q", "PPB", "fu"}
        params = drug_profile["drug_params"]
        missing_pk = sorted(required_pk.difference(params.keys()))
        if missing_pk:
            raise ValueError(f"drug_params missing required fields: {missing_pk}")

        hed = float(drug_profile["human_equivalent_dose"])
        if hed <= 0:
            raise ValueError("human_equivalent_dose must be > 0")

        self.drug_name    = str(drug_profile["name"])
        self._configured_smiles = str(drug_profile.get("smiles") or "")
        self.drug_params  = dict(params)
        self.safety_flags = dict(drug_profile["safety_flags"])
        self._drug_profile_hed = hed
        self._task_targets = dict(drug_profile.get("task_targets", {}))
        self._start_dose  = round(max(0.1, hed / 10.0), 3)
        self.current_dose = self._start_dose
        self._drug_configured = True

    def get_episode_data(self) -> dict:
        """
        Return all data collected during the episode for Layer 5 analysis.
        Call this after the episode ends.

        Returns
        -------
        dict with keys:
            drug_name       : str
            drug_params     : dict  (Layer 1/2 PK params)
            safety_flags    : dict  (Layer 1/2 safety flags)
            start_dose      : float
            history         : list[dict]  (one entry per step)
            pk_traces       : list[list[dict]]  (step → patient → time-series)
            cohort_log      : list[list[dict]]  (step → patient → demographics)
            final_score     : dict  (all three task scores)
            rp2d_dose       : float | None
            steps_taken     : int
        """
        return {
            "drug_name":    self.drug_name,
            "drug_params":  self.drug_params,
            "safety_flags": self.safety_flags,
            "task_targets": dict(self._task_targets),
            "safety_limits": {
                "dlt_rate_limit": FDA_RULES["max_dlt_rate"],
                "max_steps": FDA_RULES["max_steps"],
                "max_dose_mgkg": FDA_RULES["max_dose_mg_kg"],
            },
            "start_dose":   self._start_dose,
            "history":      self.history,
            "pk_traces":    self.pk_traces,
            "cohort_log":   self.cohort_log,
            "final_score": {
                "phase_i_dosing":     self._grade_phase_i(),
                "allometric_scaling": self._grade_allometric(),
                "combo_ddi":          self._grade_combo_ddi(),
            },
            "rp2d_dose":   self.rp2d_dose,
            "steps_taken": self._state.step_count,
        }

    def _rule_based_rec(self, dlt_count, cohort_size, fda_stop, avg_hep, avg_ren) -> str:
        dili = self.safety_flags.get("dili_risk", False)
        if fda_stop:
            return f"DE-ESCALATE: {dlt_count}/{cohort_size} DLTs exceeds FDA limit."
        elif dili and avg_hep > 0.5:
            return f"HOLD: DILI-flagged compound ({self.drug_name}) — liver saturation rising."
        elif avg_hep > 0.7:
            return "HOLD: Liver stress critically high."
        elif avg_ren < 0.5:
            return "HOLD: Kidney function significantly impaired."
        elif dlt_count == 0 and avg_hep < 0.4:
            return "ESCALATE: Safety signals within acceptable limits."
        return "HOLD: Borderline safety signals, monitor before escalating."

    def _compute_reward(
        self,
        dlt_count: int,
        cohort_size: int,
        fda_stop: bool,
        avg_hep: float,
        avg_ren: float,
        prev_dose: float,
    ) -> float:
        dlt_rate = dlt_count / max(cohort_size, 1)
        phase_target = float(self._task_targets.get("phase_i_dosing") or self._drug_profile_hed or self.current_dose)
        target_error = abs(self.current_dose - phase_target) / max(phase_target, 1e-6)
        target_band = 0.15

        # Safety (40%)
        if fda_stop:
            safety = 0.0
        elif dlt_rate == 0:
            safety = 1.0
        elif dlt_rate <= 0.167:
            safety = 0.8
        elif dlt_rate <= 0.33:
            safety = 0.4
        else:
            safety = 0.0

        # Progress (35%) — reward movement toward a molecule-specific target band.
        if dlt_rate == 0 and avg_hep < 0.5 and avg_ren > 0.7:
            progress = max(0.0, 1.0 - target_error)
            if self.current_dose > prev_dose and self.current_dose <= phase_target * (1.0 + target_band):
                progress = min(1.0, progress + 0.1)
        elif dlt_rate > 0 and self.current_dose <= prev_dose:
            progress = 0.7 * max(0.0, 1.0 - target_error)
        elif dlt_rate > 0:
            progress = 0.1
        else:
            progress = 0.5

        # Stopping (15%) — history[-1]["escalate"] now reliably present
        if fda_stop:
            stopping = 0.0
        elif not self.history[-1].get("escalate", True):
            if target_error <= target_band and dlt_rate <= FDA_RULES["max_dlt_rate"]:
                stopping = 1.0
            elif dlt_rate > 0 and self.current_dose > self._drug_profile_hed:
                stopping = 1.0
            elif dlt_rate == 0 and self.current_dose > phase_target * 0.9:
                stopping = 0.7
            else:
                stopping = 0.3
        else:
            stopping = 0.5

        # Organ health (10%)
        organ = avg_ren * 0.5 + (1.0 - avg_hep) * 0.5

        value = 0.40 * safety + 0.35 * progress + 0.15 * stopping + 0.10 * organ

        # RP2D transition bonus
        if dlt_rate > 0 and len(self.history) >= 2:
            if self.history[-2].get("dlt_count", 0) == 0:
                value = min(1.0, value + 0.15)

        return round(min(1.0, max(0.0, value)), 3)
