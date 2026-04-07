"""
Task 1: PhaseIDoseEscalationEnv — Phase I Dose-Finding Trial.

The RL agent acts as the trial designer and must find the Recommended Phase 2
Dose (RP2D) by escalating through cohorts, reading DLT signals, and stopping
at the right dose.

Scientific basis
----------------
Phase I oncology design: RL-tunable cohorts within a drug-specific 3-6 patient
range, with target toxicity interval derived from molecular risk and model
confidence.
DLT window: 28 days (standard Phase I observation period).
RP2D definition: highest dose with DLT rate < 33% (equivalent to MTD).

Reference: Optimal dose escalation methods using deep reinforcement learning
in phase I oncology trials (PubMed 36717962, 2023).

FDA Project Optimus (2024): maximize efficacy at minimum tolerated exposure.

Action space
------------
MultiDiscrete([n_dose_levels, n_cohort_options]):
    - dim 0: dose action — 0=de-escalate, 1=hold, 2=escalate
    - dim 1: cohort size index into drug-specific cohort options

Observation space
-----------------
Box(39,) = drug_features(29) + trial_state(10)

Trial state components:
    [0] current_dose_normalized
    [1] current_dose_level / n_levels
    [2] cohort_index / max_cohorts
    [3] dlt_rate_current_cohort
    [4] dlt_rate_previous_cohort
    [5] mean_pk_effect_current
    [6] overall_risk_score (from drug profile)
    [7] n_patients_enrolled / max_patients
    [8] forced_deescalation_flag  (FDA safety wrapper triggered)
    [9] below_mec_fraction        (fraction of patients below efficacy threshold)

Reward
------
Per-cohort step:
    R = R_safety + R_efficacy + R_cost + R_speed
    R_safety   = utility around a drug-specific target DLT interval
    R_efficacy = mean_effect × exposure_coverage
    R_cost     = enrollment penalty scaled by prediction uncertainty
    R_speed    = small per-step convergence penalty

Terminal:
    RP2D found → +10 − 5 × DLT_rate_at_RP2D  (cleaner RP2D = bigger bonus)
    Stopped (safety) → 0
    Max steps exceeded → −2

Hard safety constraint (FDA irremovable wrapper)
-------------------------------------------------
    If ≥ 2/3 DLTs in a 3-patient cohort → forced de-escalation
    If ≥ 3/6 DLTs in a 6-patient cohort → trial stopped
    Agent's dose action is overridden if it violates the above.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from clinical_trial_gym.drug.molecule    import DrugMolecule
from clinical_trial_gym.drug.admet       import ADMETPredictor
from clinical_trial_gym.drug.properties  import MolecularPropertyExtractor
from clinical_trial_gym.pk_pd.patient_agent import PatientAgent, PatientPopulation
from clinical_trial_gym.science import derive_phase_i_priors, require_finite_keys


# ---------------------------------------------------------------------------
# Dose ladder builder — fully derived from drug PK/PD (no hardcoding)
# ---------------------------------------------------------------------------

def build_dose_levels(
    pkpd_params: dict,
    pd_params: dict,
    n_levels: int = 8,
    safety_margin: float = 0.1,
) -> np.ndarray:
    """
    Construct a drug-specific dose ladder from first principles.

    Starting dose:
        minimum dose to reach 10% of MEC in plasma after oral absorption.
        D_start = MEC × 0.1 × Vc / F    (conservative FDA guidance approach)

    Maximum dose:
        3× the estimated MTD (where Cmax would hit MTC):
        D_max = MTC × Vc / F × 3

    Levels: log-uniform between D_start and D_max.

    Parameters
    ----------
    pkpd_params : dict
        PK parameters from ADMETProperties.to_pkpd_params().
    pd_params : dict
        PD parameters from ADMETProperties.to_pd_params().
    n_levels : int
        Number of dose levels (default 8).
    safety_margin : float
        Scale factor on D_start (smaller = more conservative start).

    Returns
    -------
    np.ndarray of shape (n_levels,) in mg/kg, ascending.
    """
    require_finite_keys(pkpd_params, ("Vc", "F"), context="phase_i pkpd_params")
    require_finite_keys(pd_params, ("MEC", "MTC"), context="phase_i pd_params")
    Vc = max(float(pkpd_params["Vc"]), 0.05)
    F = max(float(pkpd_params["F"]), 0.05)
    MEC = max(float(pd_params["MEC"]), 0.01)
    MTC = max(float(pd_params["MTC"]), MEC * 2.0)

    # Starting dose: achieve 10% × MEC in plasma (conservative first-in-human)
    D_start = float(np.clip(MEC * safety_margin * Vc / F, 0.005, 5.0))
    # Maximum dose: 3× dose that would hit MTC at Cmax
    D_max   = float(np.clip(MTC * Vc / F * 3.0, D_start * 2, 200.0))

    levels = np.logspace(
        np.log10(D_start), np.log10(D_max), num=n_levels
    )
    return levels.astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PhaseIDoseEscalationEnv(gym.Env):
    """
    Gymnasium environment for Phase I dose-escalation trial design.

    Parameters
    ----------
    smiles : str
        SMILES string of the drug to trial. All parameters are derived
        from this molecule — nothing is hardcoded.
    drug_name : str, optional
        Human-readable name for logging.
    n_dose_levels : int
        Number of dose levels in the trial. Default: 8.
    max_cohorts : int
        Maximum number of cohorts before forced termination. Default: 12.
    dlt_window_days : int
        Days to observe each patient for DLTs. Default: 28.
    rng_seed : int, optional
        Random seed for reproducibility.
    admet_predictor : ADMETPredictor, optional
        Shared ADMET model (avoids re-training per env instance).
    """

    metadata = {"render_modes": ["human"]}

    # Observation = drug_features (29) + trial_state (10)
    OBS_DIM = 39
    # DLT rate at which a dose level is considered the MTD
    MTD_DLT_THRESHOLD = 0.33

    def __init__(
        self,
        smiles: str,
        drug_name: str = "drug",
        n_dose_levels: int = 8,
        max_cohorts: int = 12,
        dlt_window_days: int = 28,
        rng_seed: Optional[int] = None,
        admet_predictor: Optional[ADMETPredictor] = None,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Build drug profile (Layer 1)
        # ----------------------------------------------------------------
        predictor = admet_predictor or ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        self._mol     = DrugMolecule(smiles, name=drug_name)
        self._profile = extractor.extract(self._mol)

        # Drug-specific PK/PD
        self._pkpd   = self._profile.pkpd_params
        self._pd     = self._profile.admet.to_pd_params()
        self._priors = derive_phase_i_priors(self._profile)

        # ----------------------------------------------------------------
        # Dose levels derived from drug properties
        # ----------------------------------------------------------------
        self.n_dose_levels   = n_dose_levels
        self._dose_levels    = build_dose_levels(self._pkpd, self._pd, n_dose_levels)

        # ----------------------------------------------------------------
        # Trial parameters
        # ----------------------------------------------------------------
        self.max_cohorts      = max_cohorts
        self.dlt_window_days  = dlt_window_days
        self._cohort_options = self._priors.cohort_options
        self._min_cohort_size = min(self._cohort_options)
        self._max_cohort_size = max(self._cohort_options)

        self._rng = np.random.default_rng(rng_seed)

        # ----------------------------------------------------------------
        # Gymnasium spaces
        # ----------------------------------------------------------------
        # Action: (dose_action, cohort_size_action)
        #   dose_action    : 0=de-escalate, 1=hold, 2=escalate
        #   cohort_size_act: 0=3pts, 1=4pts, 2=5pts, 3=6pts
        self.action_space = spaces.MultiDiscrete([3, len(self._cohort_options)])

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        # Bounded components
        obs_low[29:39]  = 0.0
        obs_high[29:39] = 1.0
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # ----------------------------------------------------------------
        # Episode state (initialised in reset())
        # ----------------------------------------------------------------
        self._current_level: int = 0
        self._cohort_idx:    int = 0
        self._prev_dlt_rate: float = 0.0
        self._prev_effect:   float = 0.0
        self._total_enrolled: int = 0
        self._dlt_history:   List[float] = []   # per-cohort DLT rates
        self._dose_history:  List[float] = []   # per-cohort doses
        self._rp2d_level:    Optional[int]  = None
        self._done:          bool = False
        self._forced_deescalation: bool = False

        # Patient population sampler (shared, reseeded per reset)
        self._pop_sampler = PatientPopulation(
            drug_profile=self._profile,
            n_patients=self._max_cohort_size,
            rng_seed=int(self._rng.integers(0, 2**31)),
        )

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

        self._current_level   = 0
        self._cohort_idx      = 0
        self._prev_dlt_rate   = 0.0
        self._prev_effect     = 0.0
        self._total_enrolled  = 0
        self._dlt_history     = []
        self._dose_history    = []
        self._rp2d_level      = None
        self._done            = False
        self._forced_deescalation = False

        obs = self._build_obs(dlt_rate=0.0, mean_effect=0.0, below_mec_frac=0.0)
        return obs, self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            warnings.warn("step() called after episode ended. Call reset().")
            obs = self._build_obs(0.0, 0.0, 0.0)
            return obs, 0.0, True, False, self._info()

        # ----------------------------------------------------------------
        # Decode action
        # ----------------------------------------------------------------
        dose_act, cohort_act = int(action[0]), int(action[1])
        cohort_size = self._cohort_options[cohort_act]

        # Apply dose action (clipped to valid level range)
        new_level = self._current_level + (dose_act - 1)  # -1/0/+1
        new_level = int(np.clip(new_level, 0, self.n_dose_levels - 1))

        # FDA safety wrapper — override agent if escalation skips levels
        if new_level > self._current_level + 1:
            new_level = self._current_level + 1   # max 1 level at a time

        self._current_level = new_level
        dose_mgkg = float(self._dose_levels[self._current_level])

        # ----------------------------------------------------------------
        # Simulate cohort
        # ----------------------------------------------------------------
        dlt_count, mean_effect, below_mec_frac = self._run_cohort(
            dose_mgkg, cohort_size
        )
        dlt_rate = dlt_count / cohort_size

        # ----------------------------------------------------------------
        # FDA irremovable safety wrapper
        # ----------------------------------------------------------------
        self._forced_deescalation = False
        forced_stop = False

        if cohort_size == 3 and dlt_count >= 2:
            # 3+3 rule: ≥2/3 DLTs → must de-escalate
            self._forced_deescalation = True
            if self._current_level > 0:
                self._current_level -= 1

        if cohort_size >= 6 and dlt_count >= 3:
            # ≥3/6 DLTs in expanded cohort → stop trial
            forced_stop = True

        # ----------------------------------------------------------------
        # Track cohort history
        # ----------------------------------------------------------------
        self._dlt_history.append(dlt_rate)
        self._dose_history.append(dose_mgkg)
        self._total_enrolled += cohort_size
        self._cohort_idx     += 1

        # ----------------------------------------------------------------
        # Compute reward
        # ----------------------------------------------------------------
        reward = self._compute_reward(
            dlt_rate, mean_effect, cohort_size, below_mec_frac
        )

        # ----------------------------------------------------------------
        # Terminal condition
        # ----------------------------------------------------------------
        truncated = False

        if forced_stop:
            # Trial stopped by safety board
            self._done = True
            self._rp2d_level = max(self._current_level - 1, 0)
            reward += 0.0   # no bonus — forced stop is a failure

        elif dlt_rate >= self._priors.target_dlt_upper:
            # MTD exceeded — previous dose is RP2D
            self._done = True
            self._rp2d_level = max(self._current_level - 1, 0)
            # Small terminal reward: we found the boundary
            reward += 3.0 * max(0.0, 1.0 - dlt_rate / max(self._priors.target_dlt_upper, 1e-8))

        elif self._current_level == self.n_dose_levels - 1 and dlt_rate <= self._priors.target_dlt_upper:
            # Reached max dose without exceeding DLT → max level is RP2D
            self._done = True
            self._rp2d_level = self._current_level
            reward += 10.0 * max(0.0, 1.0 - dlt_rate / max(self._priors.target_dlt_upper, 1e-8))

        elif self._cohort_idx >= self.max_cohorts:
            # Max cohorts reached → best dose so far is RP2D
            truncated = True
            self._done = True
            self._rp2d_level = self._current_level
            reward -= 2.0   # penalty for not converging

        self._prev_dlt_rate = dlt_rate
        self._prev_effect   = mean_effect

        obs = self._build_obs(dlt_rate, mean_effect, below_mec_frac)
        terminated = self._done and not truncated
        return obs, float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Cohort simulation
    # ------------------------------------------------------------------

    def _run_cohort(
        self, dose_mgkg: float, cohort_size: int
    ) -> Tuple[int, float, float]:
        """
        Simulate a cohort of patients at the given dose.

        Returns
        -------
        dlt_count : int
        mean_effect : float   (mean PD effect across cohort, 0–1)
        below_mec_frac : float (fraction of patients below MEC)
        """
        # Sample cohort
        self._pop_sampler.n_patients = cohort_size
        self._pop_sampler.rng = np.random.default_rng(
            int(self._rng.integers(0, 2**31))
        )
        patients = self._pop_sampler.sample()

        dlt_count     = 0
        effects       = []
        below_mec     = 0

        for patient in patients:
            # Administer dose on day 1 of DLT window
            patient.administer(dose_mgkg, time_h=0.0, route="oral")
            # Simulate entire DLT window (28 days = 672 hours)
            for day in range(self.dlt_window_days):
                patient.step(duration_h=24.0)
                if not patient.is_active:
                    break

            if patient.has_dlt:
                dlt_count += 1

            summary = patient.pk_summary
            effects.append(float(summary.get("mean_effect", 0.0)))

            MEC = self._pd.get("MEC", 0.2)
            cmax = float(summary.get("Cmax", 0.0))
            if cmax < MEC:
                below_mec += 1

        mean_effect    = float(np.mean(effects)) if effects else 0.0
        below_mec_frac = float(below_mec / max(cohort_size, 1))
        return dlt_count, mean_effect, below_mec_frac

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        dlt_rate: float,
        mean_effect: float,
        cohort_size: int,
        below_mec_frac: float,
    ) -> float:
        """
        Multi-component reward reflecting clinical trial quality.

        Components (all in [-∞, +∞]):
            R_safety   : penalise DLTs; reward safe cohorts
            R_efficacy : reward pharmacological effect above MEC
            R_cost     : penalise large cohorts (trial cost)
            R_speed    : constant step penalty (efficiency)

        Weights grounded in FDA Project Optimus (2024) framework:
            safety is the primary criterion;
            efficacy guides dose selection within the safe range.
        """
        # --- Safety ---
        distance_to_target = abs(dlt_rate - self._priors.target_dlt)
        if dlt_rate <= self._priors.target_dlt_upper:
            R_safety = 0.5 - self._priors.safety_weight * distance_to_target
        else:
            excess = dlt_rate - self._priors.target_dlt_upper
            R_safety = -self._priors.safety_weight * (0.5 + 2.0 * excess)

        # --- Efficacy ---
        # Reward effect only when patients are above MEC
        efficacy_fraction = 1.0 - below_mec_frac
        R_efficacy = self._priors.efficacy_weight * mean_effect * efficacy_fraction

        # --- Cost (per patient enrolled) ---
        R_cost = -self._priors.cost_weight * cohort_size

        # --- Speed (constant step penalty to encourage convergence) ---
        R_speed = -self._priors.speed_weight

        return float(R_safety + R_efficacy + R_cost + R_speed)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        dlt_rate: float,
        mean_effect: float,
        below_mec_frac: float,
    ) -> np.ndarray:
        # Drug features: 29-dim float32 from MolecularPropertyExtractor
        drug_feat = self._profile.observation_vector   # (29,)

        # Trial state: 10-dim float32
        max_dose = float(self._dose_levels[-1])
        cur_dose = float(self._dose_levels[self._current_level])
        trial_state = np.array([
            cur_dose / (max_dose + 1e-8),                             # [0]
            self._current_level / (self.n_dose_levels - 1 + 1e-8),    # [1]
            self._cohort_idx / (self.max_cohorts + 1e-8),             # [2]
            float(dlt_rate),                                          # [3]
            float(self._prev_dlt_rate),                               # [4]
            float(mean_effect),                                       # [5]
            float(self._profile.safety_flags["overall_risk_score"]),            # [6]
            self._total_enrolled / (self.max_cohorts * self._max_cohort_size + 1e-8),  # [7]
            float(self._forced_deescalation),                        # [8]
            float(below_mec_frac),                                   # [9]
        ], dtype=np.float32)

        obs = np.concatenate([drug_feat, trial_state])
        if not np.all(np.isfinite(obs)):
            raise ValueError("Phase I observation contains non-finite values")
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _info(self) -> dict:
        return {
            "current_dose_mgkg": float(self._dose_levels[self._current_level]),
            "current_level":     self._current_level,
            "cohort_index":      self._cohort_idx,
            "total_enrolled":    self._total_enrolled,
            "rp2d_level":        self._rp2d_level,
            "rp2d_dose_mgkg":    (
                float(self._dose_levels[self._rp2d_level])
                if self._rp2d_level is not None else None
            ),
            "dlt_history":   list(self._dlt_history),
            "dose_history":  list(self._dose_history),
            "dose_levels":   self._dose_levels.tolist(),
            "cohort_options": list(self._cohort_options),
            "target_dlt_interval": [
                self._priors.target_dlt_lower,
                self._priors.target_dlt_upper,
            ],
            "drug_name":     self._mol.name,
        }

    def render(self, mode: str = "human"):
        print(
            f"[PhaseI] Drug={self._mol.name}  "
            f"Cohort={self._cohort_idx}  "
            f"Level={self._current_level}  "
            f"Dose={self._dose_levels[self._current_level]:.3f} mg/kg  "
            f"RP2D={self._rp2d_level}"
        )

    @property
    def dose_levels(self) -> np.ndarray:
        return self._dose_levels.copy()

    @property
    def drug_profile(self):
        return self._profile
