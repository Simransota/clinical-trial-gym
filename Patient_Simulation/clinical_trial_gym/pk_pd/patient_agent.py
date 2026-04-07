"""
PatientAgent: Simulates a single patient in a clinical trial cohort.

This is where Layer 1 (drug properties) and Layer 2 (PK/PD simulation) connect.

Each patient has:
  - An individual PK/PD profile (variability around population mean)
  - A BioGears-calibrated surrogate ODE as its internal state machine
  - DLT (Dose-Limiting Toxicity) grading per CTCAE criteria
  - An observation vector that feeds upward to the RL trial-designer agent

Population variability:
  Real patients differ in CL, Vd, PPB, age, sex, organ function.
  We model inter-individual variability (IIV) via log-normal distributions
  around population mean PK parameters (standard in pharmacometrics).

  η_i ~ N(0, ω²)
  CL_i = CL_pop × exp(η_CL_i)   ← typical for hepatic clearance
  Vd_i = Vd_pop × exp(η_Vd_i)   ← volume scales with lean body mass

CTCAE toxicity grading:
  Grade 0: No toxicity
  Grade 1: Mild (manageable, no dose change)
  Grade 2: Moderate (may need dose hold)
  Grade 3: Severe (DLT — triggers safety stopping rules)
  Grade 4: Life-threatening (DLT — immediate stopping rule)
  Grade 5: Fatal

DLT is defined here as Grade ≥ 3 toxicity in the first 28 days.

References:
  - CTCAE v5.0 (NCI, 2017)
  - Beal & Sheiner NONMEM User's Guide (1992) — IIV model
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from clinical_trial_gym.drug.properties import DrugProfile
from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE, PKPDState


# ---------------------------------------------------------------------------
# Toxicity grading
# ---------------------------------------------------------------------------

class CTCAEGrade(IntEnum):
    NONE           = 0
    MILD           = 1
    MODERATE       = 2
    SEVERE         = 3   # DLT threshold (Grade 3+)
    LIFE_THREATENING = 4
    FATAL          = 5

    @property
    def is_dlt(self) -> bool:
        return self >= CTCAEGrade.SEVERE


# ---------------------------------------------------------------------------
# Patient covariate profile
# ---------------------------------------------------------------------------

@dataclass
class PatientCovariates:
    """
    Demographic and clinical characteristics of a patient.

    These modulate individual PK parameters via known covariate relationships
    (e.g., renal impairment reduces CL; age reduces hepatic metabolism).
    """
    patient_id: str
    age: float          # years
    weight_kg: float    # body weight
    sex: str            # "M" | "F"
    renal_function: float   # eGFR fraction of normal [0, 1]; 1.0 = normal
    hepatic_function: float # Child-Pugh score fraction [0, 1]; 1.0 = normal
    ecog_score: int         # performance status [0, 4]

    def cl_adjustment(self) -> float:
        """
        Covariate-based clearance adjustment factor.
        CL_individual = CL_population × cl_adjustment()
        """
        # Age effect: CL decreases ~1% per year above 40
        age_factor = max(0.4, 1.0 - 0.01 * max(0, self.age - 40))
        # Renal function: fraction of normal GFR
        renal_factor = 0.3 + 0.7 * self.renal_function
        # Hepatic function
        hepatic_factor = 0.4 + 0.6 * self.hepatic_function
        # Sex: women have ~15% lower CL for most hepatically cleared drugs
        sex_factor = 0.87 if self.sex == "F" else 1.0
        return float(age_factor * renal_factor * hepatic_factor * sex_factor)

    def vd_adjustment(self) -> float:
        """Body-weight-normalized Vd adjustment."""
        # Volume scales approximately linearly with lean body mass
        ref_weight = 70.0
        return float(np.sqrt(self.weight_kg / ref_weight))  # ~LBM proxy


@dataclass
class PopulationVariability:
    """
    Inter-individual variability parameters (IIV).
    η values are log-normal: param_i = param_pop * exp(η_i), η ~ N(0, ω²)

    Typical ω values from NONMEM literature:
      CL: 0.3-0.5 (CV 30-50%)
      Vd: 0.2-0.4
      ka: 0.4-0.7 (highly variable)
    """
    omega_CL: float = 0.35   # IIV for clearance
    omega_Vd: float = 0.25   # IIV for volume of distribution
    omega_ka: float = 0.50   # IIV for absorption rate
    omega_EC50: float = 0.40 # IIV for pharmacodynamic EC50


# ---------------------------------------------------------------------------
# Patient Agent
# ---------------------------------------------------------------------------

class PatientAgent:
    """
    Simulates a single patient in a clinical trial.

    Parameters
    ----------
    drug_profile : DrugProfile
        Full drug characterization from Layer 1 (MolecularPropertyExtractor).
    covariates : PatientCovariates, optional
        Patient demographics/clinical characteristics. If None, a
        "typical" patient is used (70 kg, 45 yo male, normal organ function).
    iiv : PopulationVariability, optional
        IIV parameters for sampling individual PK parameters.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.
    biogears_calibration : dict, optional
        BioGears-derived calibration factors for the surrogate ODE.
    dlt_window_days : int
        Observation window for DLT assessment. Default: 28 days (Phase I).
    """

    def __init__(
        self,
        drug_profile: DrugProfile,
        covariates: Optional[PatientCovariates] = None,
        iiv: Optional[PopulationVariability] = None,
        rng: Optional[np.random.Generator] = None,
        biogears_calibration: Optional[dict] = None,
        dlt_window_days: int = 28,
    ):
        self.drug_profile = drug_profile
        self.covariates = covariates or _default_covariates()
        self.iiv = iiv or PopulationVariability()
        self.rng = rng or np.random.default_rng()
        self.biogears_calibration = biogears_calibration
        self.dlt_window_h = dlt_window_days * 24

        # Sample individual PK parameters
        self._individual_pkpd = self._sample_individual_pkpd()

        # Initialize surrogate ODE with individual params
        self.ode = SurrogateODE(
            pkpd_params=self._individual_pkpd["pk"],
            pd_params=self._individual_pkpd["pd"],
            body_weight_kg=self.covariates.weight_kg,
            biogears_calibration=biogears_calibration,
        )

        # Trial state
        self.dose_history: List[Dict] = []
        self.dlt_events: List[Dict] = []
        self.current_grade: CTCAEGrade = CTCAEGrade.NONE
        self.peak_grade: CTCAEGrade = CTCAEGrade.NONE
        self.elapsed_h: float = 0.0
        self.is_active: bool = True   # False if withdrawn or fatal event

    # ------------------------------------------------------------------
    # PK parameter sampling
    # ------------------------------------------------------------------

    def _sample_individual_pkpd(self) -> dict:
        """
        Sample individual PK/PD parameters using IIV model.
        Covariate effects are applied on top of population values.
        """
        pop_pk = dict(self.drug_profile.pkpd_params)
        iiv = self.iiv
        cov = self.covariates
        rng = self.rng

        # Log-normal IIV sampling
        eta_CL  = rng.normal(0, iiv.omega_CL)
        eta_Vd  = rng.normal(0, iiv.omega_Vd)
        eta_ka  = rng.normal(0, iiv.omega_ka)
        eta_EC50 = rng.normal(0, iiv.omega_EC50)

        # Individual PK params
        cl_adj = cov.cl_adjustment()
        vd_adj = cov.vd_adjustment()

        # All PK params come from ADMETProperties.to_pkpd_params() — no fallbacks.
        # to_pkpd_params() is guaranteed to return all required keys because
        # every ADMET prediction comes from a trained DeepChem model.
        ind_pk = {
            "ka":  pop_pk["ka"]  * np.exp(eta_ka),
            "F":   pop_pk["F"],   # not IIV-modulated in v1
            "CL":  pop_pk["CL"]  * np.exp(eta_CL) * cl_adj,
            "Vc":  pop_pk["Vc"]  * np.exp(eta_Vd) * vd_adj,
            "Vp":  pop_pk["Vp"]  * np.exp(eta_Vd) * vd_adj,
            "Q":   pop_pk["Q"],
            "PPB": pop_pk["PPB"],
            "fu":  pop_pk["fu"],
        }
        # Clamp to physiologically plausible ranges
        ind_pk["CL"]  = float(np.clip(ind_pk["CL"],  0.01, 50.0))
        ind_pk["Vc"]  = float(np.clip(ind_pk["Vc"],  0.05, 20.0))
        ind_pk["Vp"]  = float(np.clip(ind_pk["Vp"],  0.05, 20.0))
        ind_pk["ka"]  = float(np.clip(ind_pk["ka"],  0.01, 10.0))
        ind_pk["F"]   = float(np.clip(ind_pk["F"],   0.01, 1.0))

        # Individual PD params (EC50 has most IIV).
        # Population values come from ADMETProperties.to_pd_params() — these
        # are drug-specific and molecularly derived, not hardcoded constants.
        pop_pd = self.drug_profile.admet.to_pd_params()
        ind_pd = {
            "Emax": pop_pd["Emax"],
            "EC50": pop_pd["EC50"] * np.exp(eta_EC50),
            "n":    pop_pd["n"],
            "MTC":  pop_pd["MTC"],
            "MEC":  pop_pd["MEC"],
        }
        ind_pd["EC50"] = float(np.clip(ind_pd["EC50"], 0.01, 50.0))

        return {"pk": ind_pk, "pd": ind_pd}

    # ------------------------------------------------------------------
    # Dosing and simulation
    # ------------------------------------------------------------------

    def administer(self, dose_mgkg: float, time_h: float = None, route: str = "oral"):
        """
        Administer a dose to this patient.

        Parameters
        ----------
        dose_mgkg : float
            Dose in mg/kg.
        time_h : float, optional
            Time of administration. If None, uses current elapsed time.
        route : str
            'oral' or 'iv_bolus'.
        """
        if not self.is_active:
            return

        t = time_h if time_h is not None else self.elapsed_h
        self.ode.administer_dose(dose_mgkg=dose_mgkg, time_h=t, route=route)
        self.dose_history.append({
            "time_h": t, "dose_mgkg": dose_mgkg, "route": route
        })

    def step(self, duration_h: float = 24.0) -> PKPDState:
        """
        Advance simulation by duration_h hours.

        Parameters
        ----------
        duration_h : float
            Simulation step size in hours. Default: 24h (one day).

        Returns
        -------
        PKPDState
            State at the end of this step.
        """
        if not self.is_active:
            return self.ode.current_state

        states = self.ode.simulate(
            duration_h=duration_h,
            dt_h=1.0,   # 1h resolution inside the step
            t_start=self.elapsed_h,
        )
        self.elapsed_h += duration_h

        # Assess toxicity at this step
        self._assess_toxicity(states)

        return self.ode.current_state

    # ------------------------------------------------------------------
    # Toxicity assessment (CTCAE-based)
    # ------------------------------------------------------------------

    def _assess_toxicity(self, states: list):
        """
        Map PK/PD state to CTCAE toxicity grade.

        Grading thresholds are set relative to the drug's MTC
        (minimum toxic concentration) from the PD model.
        In v2, biological sub-agents (hepatocyte, immune) will replace this.
        """
        if not states:
            return

        MTC = self.ode.pd["MTC"]

        for state in states:
            tox_ratio = state.Cc / (MTC + 1e-10)

            # Grading thresholds (multiples of MTC)
            if tox_ratio < 0.5:
                grade = CTCAEGrade.NONE
            elif tox_ratio < 0.8:
                grade = CTCAEGrade.MILD
            elif tox_ratio < 1.0:
                grade = CTCAEGrade.MODERATE
            elif tox_ratio < 1.5:
                grade = CTCAEGrade.SEVERE
            elif tox_ratio < 2.5:
                grade = CTCAEGrade.LIFE_THREATENING
            else:
                grade = CTCAEGrade.FATAL

            self.current_grade = grade
            if grade > self.peak_grade:
                self.peak_grade = grade

            # Record DLT events
            if grade.is_dlt and self.elapsed_h <= self.dlt_window_h:
                self.dlt_events.append({
                    "time_h": state.t,
                    "grade": grade,
                    "Cc": state.Cc,
                    "tox_score": state.toxicity_score,
                })

            # Fatal event: withdraw patient
            if grade == CTCAEGrade.FATAL:
                self.is_active = False
                break

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------

    @property
    def observation(self) -> np.ndarray:
        """
        Full observation vector for this patient.

        Concatenates:
          - PK/PD state (9 features)
          - Patient covariates (5 features)
          - DLT status (3 features)

        Shape: (17,)
        """
        pk_obs = self.ode.current_state.to_array()   # shape (9,)

        cov_obs = np.array([
            self.covariates.age / 100.0,
            self.covariates.weight_kg / 100.0,
            float(self.covariates.sex == "F"),
            self.covariates.renal_function,
            self.covariates.hepatic_function,
        ], dtype=np.float32)   # shape (5,)

        dlt_obs = np.array([
            float(self.peak_grade.is_dlt),
            float(self.current_grade) / 5.0,   # normalized grade
            len(self.dlt_events) / 10.0,        # normalized DLT count
        ], dtype=np.float32)   # shape (3,)

        return np.concatenate([pk_obs, cov_obs, dlt_obs])   # shape (17,)

    OBS_DIM: int = 17

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

    @property
    def has_dlt(self) -> bool:
        """True if patient experienced a DLT in the assessment window."""
        return any(
            e["time_h"] <= self.dlt_window_h
            for e in self.dlt_events
        )

    @property
    def pk_summary(self) -> dict:
        return self.ode.get_summary_stats()

    def reset(self, resample_iiv: bool = True):
        """Reset for a new episode. Optionally resample individual params."""
        if resample_iiv:
            self._individual_pkpd = self._sample_individual_pkpd()
            self.ode = SurrogateODE(
                pkpd_params=self._individual_pkpd["pk"],
                pd_params=self._individual_pkpd["pd"],
                body_weight_kg=self.covariates.weight_kg,
                biogears_calibration=self.biogears_calibration,
            )
        else:
            self.ode.reset()

        self.dose_history.clear()
        self.dlt_events.clear()
        self.current_grade = CTCAEGrade.NONE
        self.peak_grade = CTCAEGrade.NONE
        self.elapsed_h = 0.0
        self.is_active = True

    def __repr__(self) -> str:
        return (
            f"PatientAgent(id={self.covariates.patient_id}, "
            f"age={self.covariates.age:.0f}, sex={self.covariates.sex}, "
            f"peak_grade={self.peak_grade.name}, "
            f"active={self.is_active})"
        )


# ---------------------------------------------------------------------------
# Population sampler
# ---------------------------------------------------------------------------

class PatientPopulation:
    """
    Generates a cohort of PatientAgents with realistic demographic variability.

    Parameters
    ----------
    drug_profile : DrugProfile
        Shared drug profile for all patients.
    n_patients : int
        Cohort size.
    age_range : tuple
        (min_age, max_age) in years. Default: (18, 75).
    weight_range : tuple
        (min_weight, max_weight) in kg. Default: (45, 120).
    female_fraction : float
        Fraction of female patients. Default: 0.5.
    rng_seed : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        drug_profile: DrugProfile,
        n_patients: int = 6,
        age_range: Tuple[float, float] = (18.0, 75.0),
        weight_range: Tuple[float, float] = (45.0, 120.0),
        female_fraction: float = 0.5,
        rng_seed: Optional[int] = None,
    ):
        self.drug_profile = drug_profile
        self.n_patients = n_patients
        self.age_range = age_range
        self.weight_range = weight_range
        self.female_fraction = female_fraction
        self.rng = np.random.default_rng(rng_seed)

    def sample(self) -> List[PatientAgent]:
        """Generate a cohort of n_patients patients."""
        agents = []
        for i in range(self.n_patients):
            cov = PatientCovariates(
                patient_id=f"PT-{uuid.uuid4().hex[:6].upper()}",
                age=float(self.rng.uniform(*self.age_range)),
                weight_kg=float(self.rng.uniform(*self.weight_range)),
                sex="F" if self.rng.random() < self.female_fraction else "M",
                renal_function=float(np.clip(self.rng.normal(1.0, 0.15), 0.3, 1.0)),
                hepatic_function=float(np.clip(self.rng.normal(1.0, 0.10), 0.4, 1.0)),
                ecog_score=int(self.rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])),
            )
            agent = PatientAgent(
                drug_profile=self.drug_profile,
                covariates=cov,
                rng=np.random.default_rng(self.rng.integers(0, 2**31)),
            )
            agents.append(agent)
        return agents


def _default_covariates() -> PatientCovariates:
    return PatientCovariates(
        patient_id="PT-DEFAULT",
        age=45.0,
        weight_kg=70.0,
        sex="M",
        renal_function=1.0,
        hepatic_function=1.0,
        ecog_score=0,
    )