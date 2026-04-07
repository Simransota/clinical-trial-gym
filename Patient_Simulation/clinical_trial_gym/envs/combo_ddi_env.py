"""
Task 3: ComboDDIEnv — Combination Drug Scheduling with CYP450 Interaction.

The agent must schedule two drugs simultaneously while managing CYP450 enzyme
competition.  Drug A inhibits CYP3A4 (perpetrator); Drug B is cleared by
CYP3A4 (victim substrate).  When Drug A is present, Drug B's clearance is
reduced, raising its AUC/Cmax and risking toxicity.

Scientific basis
----------------
Competitive inhibition model (FDA M12 DDI Guidance, 2024):
    CL_B_eff = CL_B × [fm_3A4 / (1 + [A]_free / Ki_A)  +  (1 − fm_3A4)]

where:
    [A]_free = Cc_A × fu_A        (free inhibitor concentration)
    Ki_A     = CYP3A4 inhibition constant of Drug A (from ADMET predictions)
    fm_3A4   = fraction of Drug B cleared by CYP3A4 (estimated from ADMET)

This produces the classic "victim drug AUC ratio" = 1 + fm × [A]_free/Ki.

A good policy:
    1. Separate doses temporally (give B before A absorbs, or long after)
    2. Reduce Drug B dose to compensate for expected interaction
    3. Avoid large Drug A doses that saturate CYP3A4

DDI severity categories (FDA Guidance):
    Weak:     AUC ratio 1.25–2×
    Moderate: AUC ratio 2–5×
    Strong:   AUC ratio ≥ 5×

Action space
------------
Box(3):
    [0] dose_A normalized ∈ [0, 1]  → actual dose in [0, max_dose_A]
    [1] dose_B normalized ∈ [0, 1]  → actual dose in [0, max_dose_B]
    [2] timing_offset_norm ∈ [0, 1] → offset_h ∈ [−24, +24]
        0.0 = give B 24h before A (avoid interaction)
        0.5 = simultaneous
        1.0 = give B 24h after A  (maximum interaction)

Observation space
-----------------
Box(84,):
    drug_A_features (29) + drug_B_features (29) +
    pk_state_A (9) + pk_state_B (9) + interaction_state (6) + step_info (2)

Interaction state (6):
    [0] current_inhibition_factor  (1 + [A]_free / Ki_A)
    [1] ddi_auc_ratio              (AUC_B_actual / AUC_B_alone)
    [2] combined_dlt_rate          (fraction of patients with any DLT)
    [3] cyp_saturation             ([A]_free / Ki_A clipped to [0,5])
    [4] effect_A_normalized
    [5] effect_B_normalized

Reward
------
Per step (daily):
    R_combined_efficacy = α × (effect_A + effect_B) / 2
    R_safety            = −β × DLT_rate × severity_weight
    R_ddi_penalty       = −γ × max(0, ddi_ratio − 1.25)   (interaction cost)
    R_cost              = −δ × (dose_A + dose_B) / (max_A + max_B)

Episode terminal:
    R_terminal = +8 if mean_dlt_rate < 0.20 and mean_combined_effect > 0.5
               = +3 if mean_dlt_rate < 0.33  (safe but lower efficacy)
               = −3 if any fatal event
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from clinical_trial_gym.drug.molecule   import DrugMolecule
from clinical_trial_gym.drug.admet      import ADMETPredictor
from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
from clinical_trial_gym.pk_pd.patient_agent import PatientAgent, PatientPopulation
from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
from clinical_trial_gym.science import derive_combo_ddi_priors, require_finite_keys


# ---------------------------------------------------------------------------
# CYP fm estimator — fraction of victim drug cleared by CYP3A4
# ---------------------------------------------------------------------------

def _estimate_fm_3a4(pkpd_params: dict, admet) -> float:
    """
    Estimate fraction of clearance via CYP3A4 (fm_3A4) for a drug.

    Proxy model based on:
        - predicted unbound fraction and lipophilicity (renal vs hepatic split)
        - relative CYP inhibition probabilities across major isoforms
        - hepatic clearance share assigned to CYP3A4

    For the DDI model, fm_3A4 of the VICTIM (Drug B) is what matters.
    """
    require_finite_keys(pkpd_params, ("fu",), context="combo_ddi victim pkpd_params")
    logd = admet._require_finite("predicted_logD", admet.predicted_logD)
    fu = max(float(pkpd_params["fu"]), 0.005)
    renal_fraction = float(np.clip(fu * np.exp(-0.35 * (logd ** 2)), 0.01, 0.95))
    hepatic_fraction = 1.0 - renal_fraction
    cyp_probs = np.array([
        admet._require_finite("_cyp3a4_prob", getattr(admet, "_cyp3a4_prob", np.nan)),
        admet._require_finite("_cyp2d6_prob", getattr(admet, "_cyp2d6_prob", np.nan)),
        admet._require_finite("_cyp2c9_prob", getattr(admet, "_cyp2c9_prob", np.nan)),
        admet._require_finite("_cyp2c19_prob", getattr(admet, "_cyp2c19_prob", np.nan)),
        admet._require_finite("_cyp1a2_prob", getattr(admet, "_cyp1a2_prob", np.nan)),
    ], dtype=np.float64)
    cyp_share_3a4 = float(cyp_probs[0] / (np.sum(cyp_probs) + 1e-8))
    return float(np.clip(hepatic_fraction * cyp_share_3a4, 0.0, 0.99))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ComboDDIEnv(gym.Env):
    """
    Gymnasium environment for combination drug scheduling with CYP450 DDI.

    Parameters
    ----------
    smiles_A : str
        SMILES of Drug A (CYP3A4 inhibitor / perpetrator).
    smiles_B : str
        SMILES of Drug B (CYP3A4 substrate / victim).
    drug_name_A : str
    drug_name_B : str
    trial_days : int
        Length of simulated trial in days. Default: 28.
    cohort_size : int
        Patients per cohort. Default: 6.
    rng_seed : int, optional
    admet_predictor : ADMETPredictor, optional
        Shared predictor; avoids re-training per instance.
    """

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 84   # 29+29+9+9+6+2

    def __init__(
        self,
        smiles_A: str,
        smiles_B: str,
        drug_name_A: str = "drug_A",
        drug_name_B: str = "drug_B",
        trial_days: int = 28,
        cohort_size: int = 6,
        rng_seed: Optional[int] = None,
        admet_predictor: Optional[ADMETPredictor] = None,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Drug profiles
        # ----------------------------------------------------------------
        predictor = admet_predictor or ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)

        self._mol_A    = DrugMolecule(smiles_A, name=drug_name_A)
        self._mol_B    = DrugMolecule(smiles_B, name=drug_name_B)
        self._profile_A = extractor.extract(self._mol_A)
        self._profile_B = extractor.extract(self._mol_B)

        self._pkpd_A = self._profile_A.pkpd_params
        self._pkpd_B = self._profile_B.pkpd_params
        self._pd_A   = self._profile_A.admet.to_pd_params()
        self._pd_B   = self._profile_B.admet.to_pd_params()

        # ----------------------------------------------------------------
        # CYP450 DDI parameters
        # ----------------------------------------------------------------
        # Ki of Drug A for CYP3A4 (perpetrator inhibition constant)
        ki_dict_A      = self._profile_A.admet.cyp_ki_values()
        self._ki_A = float(ki_dict_A["CYP3A4"])   # µM
        self._fu_A = float(self._pkpd_A["fu"])

        # fm_3A4 of Drug B (victim clearance fraction via CYP3A4)
        self._fm_3A4_B = _estimate_fm_3a4(self._pkpd_B, self._profile_B.admet)
        self._priors = derive_combo_ddi_priors(
            self._profile_A, self._profile_B, self._fm_3A4_B
        )

        # ----------------------------------------------------------------
        # Max doses (derived from drug PD)
        # ----------------------------------------------------------------
        def _max_dose(pkpd: dict, pd: dict) -> float:
            require_finite_keys(pd, ("MTC",), context="combo_ddi pd_params")
            require_finite_keys(pkpd, ("Vc", "F"), context="combo_ddi pkpd_params")
            MTC = float(pd["MTC"])
            Vc = max(float(pkpd["Vc"]), 0.05)
            F = max(float(pkpd["F"]), 0.05)
            return float(np.clip(MTC * Vc / F * 2.0, 1.0, 100.0))

        self._max_dose_A = _max_dose(self._pkpd_A, self._pd_A)
        self._max_dose_B = _max_dose(self._pkpd_B, self._pd_B)

        # ----------------------------------------------------------------
        # Baseline AUC of Drug B (without DDI) for ratio computation
        # ----------------------------------------------------------------
        self._baseline_auc_B = self._simulate_auc_alone(self._pkpd_B, self._pd_B,
                                                          self._max_dose_B * 0.5)

        # ----------------------------------------------------------------
        # Trial parameters
        # ----------------------------------------------------------------
        self.trial_days  = trial_days
        self.cohort_size = cohort_size
        self._rng        = np.random.default_rng(rng_seed)

        # ----------------------------------------------------------------
        # Gymnasium spaces
        # ----------------------------------------------------------------
        self.action_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )

        # ----------------------------------------------------------------
        # Episode state
        # ----------------------------------------------------------------
        self._step_idx        = 0
        self._done            = False
        self._cum_dlt_A       = 0
        self._cum_dlt_B       = 0
        self._cum_dlt_any     = 0
        self._patients_A: list = []
        self._patients_B: list = []
        self._ddi_ratios: list = []
        self._effects_A: list = []
        self._effects_B: list = []
        self._seen_dlt_patient_ids: set[str] = set()
        self._ode_A: Optional[SurrogateODE] = None
        self._ode_B: Optional[SurrogateODE] = None

        # Last-step interaction state (for observation)
        self._last_inhibition_factor = 1.0
        self._last_ddi_auc_ratio     = 1.0
        self._last_dlt_rate          = 0.0
        self._last_effect_A          = 0.0
        self._last_effect_B          = 0.0
        self._last_cyp_sat           = 0.0

        # Latest PK states
        self._pk_state_A = np.zeros(9, dtype=np.float32)
        self._pk_state_B = np.zeros(9, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_idx      = 0
        self._done          = False
        self._cum_dlt_A     = 0
        self._cum_dlt_B     = 0
        self._cum_dlt_any   = 0
        self._ddi_ratios    = []
        self._effects_A     = []
        self._effects_B     = []
        self._seen_dlt_patient_ids = set()

        # Sample cohort of patients for both drugs
        pop_A = PatientPopulation(
            drug_profile=self._profile_A,
            n_patients=self.cohort_size,
            rng_seed=int(self._rng.integers(0, 2**31)),
        )
        pop_B = PatientPopulation(
            drug_profile=self._profile_B,
            n_patients=self.cohort_size,
            rng_seed=int(self._rng.integers(0, 2**31)),
        )
        self._patients_A = pop_A.sample()
        self._patients_B = pop_B.sample()

        # Dedicated ODE instances for population-mean PK (DDI tracking)
        self._ode_A = SurrogateODE(
            pkpd_params=self._pkpd_A, pd_params=self._pd_A,
        )
        self._ode_B = SurrogateODE(
            pkpd_params=self._pkpd_B, pd_params=self._pd_B,
        )

        self._last_inhibition_factor = 1.0
        self._last_ddi_auc_ratio     = 1.0
        self._last_dlt_rate          = 0.0
        self._last_effect_A          = 0.0
        self._last_effect_B          = 0.0
        self._last_cyp_sat           = 0.0
        self._pk_state_A = np.zeros(9, dtype=np.float32)
        self._pk_state_B = np.zeros(9, dtype=np.float32)

        return self._build_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            warnings.warn("step() called after episode ended.")
            return self._build_obs(), 0.0, True, False, self._info()

        # ----------------------------------------------------------------
        # Decode action
        # ----------------------------------------------------------------
        dose_A = float(np.clip(action[0], 0.0, 1.0)) * self._max_dose_A
        dose_B = float(np.clip(action[1], 0.0, 1.0)) * self._max_dose_B
        # timing_offset ∈ [-24, +24] hours (positive = B after A)
        timing_offset_h = (float(np.clip(action[2], 0.0, 1.0)) * 48.0) - 24.0

        t_now = self._step_idx * 24.0  # hours since trial start

        # ----------------------------------------------------------------
        # Schedule doses
        # ----------------------------------------------------------------
        t_dose_A = t_now
        t_dose_B = float(np.clip(t_now + timing_offset_h, 0.0, t_now + 24.0))

        # ----------------------------------------------------------------
        # Simulate 1 day with DDI coupling
        # ----------------------------------------------------------------
        (dlt_any, effect_A, effect_B,
         inhibition_factor, ddi_auc_ratio, cyp_sat) = self._simulate_day(
            dose_A, dose_B, t_dose_A, t_dose_B
        )

        # Update running statistics
        self._last_inhibition_factor = inhibition_factor
        self._last_ddi_auc_ratio     = ddi_auc_ratio
        self._last_dlt_rate          = dlt_any / max(self.cohort_size, 1)
        self._last_effect_A          = effect_A
        self._last_effect_B          = effect_B
        self._last_cyp_sat           = cyp_sat
        self._effects_A.append(effect_A)
        self._effects_B.append(effect_B)
        self._ddi_ratios.append(ddi_auc_ratio)

        # ----------------------------------------------------------------
        # Reward
        # ----------------------------------------------------------------
        reward = self._compute_reward(
            dose_A, dose_B, effect_A, effect_B,
            self._last_dlt_rate, ddi_auc_ratio
        )

        # ----------------------------------------------------------------
        # Terminal
        # ----------------------------------------------------------------
        self._step_idx += 1
        truncated  = False
        terminated = False

        if self._step_idx >= self.trial_days:
            terminated = True
            self._done = True
            mean_eff_A = float(np.mean(self._effects_A)) if self._effects_A else 0.0
            mean_eff_B = float(np.mean(self._effects_B)) if self._effects_B else 0.0
            mean_dlt   = self._last_dlt_rate
            mean_ddi   = float(np.mean(self._ddi_ratios)) if self._ddi_ratios else 1.0

            # Terminal bonus: safe combination with good efficacy
            if mean_dlt < 0.20 and (mean_eff_A + mean_eff_B) / 2 > 0.4:
                reward += 8.0
            elif mean_dlt < 0.33:
                reward += 3.0
            if any(not p.is_active for p in self._patients_A + self._patients_B):
                reward -= 3.0  # fatal event penalty

        obs = self._build_obs()
        return obs, float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # One-day simulation with DDI coupling
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        dose_A: float,
        dose_B: float,
        t_dose_A: float,
        t_dose_B: float,
    ) -> Tuple[int, float, float, float, float, float]:
        """
        Simulate one 24-hour period for both drugs with CYP DDI.

        Algorithm:
        1. Administer Drug A at t_dose_A.
        2. Administer Drug B at t_dose_B.
        3. Simulate both drugs simultaneously in hourly increments.
        4. At each hour: compute [A]_free, update CL_B_eff via inhibition model,
           advance both ODEs.

        Returns
        -------
        dlt_any : int       (patients with new DLT this day)
        effect_A : float
        effect_B : float
        inhibition_factor : float   (mean over 24h)
        ddi_auc_ratio : float       (AUC_B_actual / AUC_B_baseline)
        cyp_saturation : float      ([A]_free / Ki_A, mean over 24h)
        """
        t_start = self._step_idx * 24.0
        t_end   = t_start + 24.0

        effects_A, effects_B = [], []
        inhibition_factors = []
        cyp_sats = []

        # Administer doses to all patients
        for pt_A in self._patients_A:
            if pt_A.is_active:
                pt_A.administer(dose_A, time_h=t_dose_A, route="oral")
        for pt_B in self._patients_B:
            if pt_B.is_active:
                # Give B with the scheduled timing offset
                t_B = float(np.clip(t_dose_B, 0.0, t_start + 23.0))
                pt_B.administer(dose_B, time_h=t_B, route="oral")

        # Also track population-mean ODEs for DDI computation
        self._ode_A.administer_dose(dose_A, time_h=t_dose_A, route="oral")
        self._ode_B.administer_dose(dose_B, time_h=t_dose_B, route="oral")

        # Hourly simulation with DDI coupling
        for h in range(24):
            t = t_start + h
            t_next = t + 1.0

            # Get Drug A free concentration (for inhibition computation)
            state_A = self._ode_A.current_state
            cc_A_free = float(state_A.Cc) * self._fu_A

            # Compute inhibition factor: 1 + [A]_free / Ki_A
            # Ki is in µM; Cc is in mg/L.
            # Convert Cc (mg/L) to µM: µM = (mg/L × 1000) / MW
            MW_A = float(self._mol_A.molecular_weight)
            cc_A_um = (cc_A_free * 1000.0) / max(MW_A, 1.0)  # µM
            inhib_factor = 1.0 + cc_A_um / (self._ki_A + 1e-8)
            cyp_sat = cc_A_um / (self._ki_A + 1e-8)

            inhibition_factors.append(inhib_factor)
            cyp_sats.append(cyp_sat)

            # Update Drug B's effective clearance
            cl_b_eff_factor = 1.0 + self._fm_3A4_B * (inhib_factor - 1.0)
            self._ode_B.set_cyp_inhibition_factor(cl_b_eff_factor)
            for pt_B in self._patients_B:
                if pt_B.is_active:
                    pt_B.ode.set_cyp_inhibition_factor(cl_b_eff_factor)

            # Advance both population ODEs by 1 hour
            states_A = self._ode_A.simulate(duration_h=1.0, dt_h=0.5, t_start=t)
            states_B = self._ode_B.simulate(duration_h=1.0, dt_h=0.5, t_start=t)

            if states_A:
                self._pk_state_A = states_A[-1].to_array()
                effects_A.append(states_A[-1].effect)
            if states_B:
                self._pk_state_B = states_B[-1].to_array()
                effects_B.append(states_B[-1].effect)

        # Advance patient agents by full 24h
        for pt_A in self._patients_A:
            if pt_A.is_active:
                pt_A.step(duration_h=24.0)
        for pt_B in self._patients_B:
            if pt_B.is_active:
                pt_B.step(duration_h=24.0)

        # Count DLTs (first occurrence only)
        dlt_count = 0
        for patient in self._patients_A + self._patients_B:
            patient_id = patient.covariates.patient_id
            if patient.has_dlt and patient_id not in self._seen_dlt_patient_ids:
                self._seen_dlt_patient_ids.add(patient_id)
                dlt_count += 1

        # Compute DDI AUC ratio
        auc_B_actual = float(self._ode_B.current_state.cumulative_AUC)
        ddi_ratio = float(np.clip(
            auc_B_actual / (self._baseline_auc_B + 1e-8), 0.5, 20.0
        ))

        mean_eff_A = float(np.mean(effects_A)) if effects_A else 0.0
        mean_eff_B = float(np.mean(effects_B)) if effects_B else 0.0
        mean_inhib = float(np.mean(inhibition_factors)) if inhibition_factors else 1.0
        mean_cyp   = float(np.mean(cyp_sats)) if cyp_sats else 0.0

        return dlt_count, mean_eff_A, mean_eff_B, mean_inhib, ddi_ratio, mean_cyp

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        dose_A: float,
        dose_B: float,
        effect_A: float,
        effect_B: float,
        dlt_rate: float,
        ddi_ratio: float,
    ) -> float:
        """
        Multi-component reward for combination drug scheduling.

        R_efficacy : combined PD effect of both drugs (want high)
        R_safety   : DLT penalty (want low)
        R_ddi      : DDI penalty — interaction is unavoidable but should be managed
        R_cost     : dose cost (want minimum effective dose)
        """
        # --- Combined efficacy ---
        R_efficacy = self._priors.efficacy_weight * (effect_A + effect_B) / 2.0

        # --- Safety ---
        R_safety = -self._priors.safety_weight * dlt_rate

        # --- DDI severity penalty ---
        # Penalise interaction ratio > 1.25 (weak DDI threshold, FDA M12)
        ddi_excess = float(max(0.0, ddi_ratio - self._priors.weak_threshold))
        severity_multiplier = (
            3.0 if ddi_ratio >= self._priors.strong_threshold
            else 2.0 if ddi_ratio >= self._priors.moderate_threshold
            else 1.0
        )
        R_ddi = -self._priors.interaction_weight * severity_multiplier * ddi_excess

        # --- Dose cost ---
        dose_A_norm = dose_A / (self._max_dose_A + 1e-8)
        dose_B_norm = dose_B / (self._max_dose_B + 1e-8)
        R_cost = -self._priors.cost_weight * (dose_A_norm + dose_B_norm) / 2.0

        return float(R_efficacy + R_safety + R_ddi + R_cost)

    # ------------------------------------------------------------------
    # PK helpers
    # ------------------------------------------------------------------

    def _simulate_auc_alone(self, pkpd: dict, pd: dict, dose: float) -> float:
        """Simulate AUC of a drug in isolation (no DDI)."""
        ode = SurrogateODE(pkpd_params=pkpd, pd_params=pd)
        ode.administer_dose(dose, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=168.0, dt_h=1.0)
        return float(states[-1].cumulative_AUC) if states else 1.0

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        feat_A = self._profile_A.observation_vector   # (29,)
        feat_B = self._profile_B.observation_vector   # (29,)
        pk_A   = self._pk_state_A                      # (9,)
        pk_B   = self._pk_state_B                      # (9,)

        interaction = np.array([
            float(np.clip(self._last_inhibition_factor / 10.0, 0, 1)),   # [0]
            float(np.clip(self._last_ddi_auc_ratio / 5.0, 0, 1)),        # [1]
            float(self._last_dlt_rate),                                   # [2]
            float(np.clip(self._last_cyp_sat / 5.0, 0, 1)),              # [3]
            float(self._last_effect_A),                                   # [4]
            float(self._last_effect_B),                                   # [5]
        ], dtype=np.float32)

        step_info = np.array([
            float(self._step_idx / max(self.trial_days, 1)),   # [0]
            float(np.mean(self._ddi_ratios) / self._priors.strong_threshold)
            if self._ddi_ratios else 0.0,                      # [1]
        ], dtype=np.float32)

        obs = np.concatenate([feat_A, feat_B, pk_A, pk_B, interaction, step_info])
        if not np.all(np.isfinite(obs)):
            raise ValueError("Combo DDI observation contains non-finite values")
        return obs.astype(np.float32)

    def _info(self) -> dict:
        return {
            "step":             self._step_idx,
            "ki_A_um":          self._ki_A,
            "fm_3A4_B":         self._fm_3A4_B,
            "inhibition_factor": self._last_inhibition_factor,
            "ddi_auc_ratio":    self._last_ddi_auc_ratio,
            "dlt_rate":         self._last_dlt_rate,
            "effect_A":         self._last_effect_A,
            "effect_B":         self._last_effect_B,
            "drug_A":           self._mol_A.name,
            "drug_B":           self._mol_B.name,
        }

    def render(self, mode="human"):
        print(
            f"[ComboDDI] A={self._mol_A.name} B={self._mol_B.name}  "
            f"Day={self._step_idx}/{self.trial_days}  "
            f"DDI_ratio={self._last_ddi_auc_ratio:.2f}×  "
            f"InhibFactor={self._last_inhibition_factor:.2f}  "
            f"DLT={self._last_dlt_rate:.2%}  "
            f"Effect=({self._last_effect_A:.2f}, {self._last_effect_B:.2f})"
        )
