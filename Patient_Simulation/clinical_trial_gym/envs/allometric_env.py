"""
Task 2: AllometricScalingEnv — Species-Bridging Dose Prediction.

The agent sees rat preclinical PK data and must propose the correct human
equivalent dose (HED) and target exposure range for the Phase I starting dose.

This tests whether the agent has internalised allometric scaling laws:
    CL_human = CL_rat × (BW_human / BW_rat)^0.75
    HED      = Animal_dose × (Km_animal / Km_human)   [FDA Km-factor method]

Scientific basis
----------------
FDA Guidance: Estimating the Maximum Safe Starting Dose in Initial Clinical
Trials for Therapeutics in Adult Healthy Volunteers (2005).
Boxenbaum H (1982): allometric exponents (CL ∝ BW^0.75, Vd ∝ BW^1.0).
ICH M12 DDI Guidance (2024): fm and clearance pathway fractions.

Task structure
--------------
Multi-step adaptive dosing (5 cycles):
    Cycle 0: Agent proposes initial HED from rat data alone.
    Cycle 1–4: Agent observes human PK response from a 3-patient minicohor
               and refines the dose.

Reward is based on target exposure attainment:
    Target AUC  = allometrically predicted human AUC at HED
    Target Cmax = below MTC (safety)
    R = −|log(AUC_actual / AUC_target)|  − penalty_for_Cmax_overage

Action space
------------
Box(1): proposed dose in mg/kg, normalized to [0, 1] relative to max dose.

Observation space
-----------------
Box(55,) = drug_features(29) + rat_pk_state(8) + scaling_context(6) + cycle_state(12)

Scaling context (6 features):
    [0] log10(BW_ratio)  = log10(70/0.25) for rat→human
    [1] CL_exponent      = 0.75 (standard allometric)
    [2] Vd_exponent      = 1.00
    [3] km_ratio         = 6/37  (rat Km / human Km)
    [4] AUC_scaling_factor   (predicted AUC ratio after scaling)
    [5] Cmax_scaling_factor  (predicted Cmax ratio after scaling)

Cycle state (12 features, updated each cycle):
    [0–2]  previous proposed dose (normalized), actual AUC, actual Cmax
    [3–5]  AUC target (normalized), AUC error, Cmax safety margin
    [6–8]  cycle index / max_cycles, n_patients_dosed, safety_violations
    [9–11] mean_effect, dlt_rate, below_mec_frac
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from clinical_trial_gym.drug.molecule    import DrugMolecule
from clinical_trial_gym.drug.admet       import ADMETPredictor
from clinical_trial_gym.drug.properties  import MolecularPropertyExtractor
from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
from clinical_trial_gym.pk_pd.patient_agent import PatientAgent, PatientPopulation
from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
from clinical_trial_gym.science import require_finite_keys


class AllometricScalingEnv(gym.Env):
    """
    Gymnasium environment for allometric dose-scaling from rat to human.

    The agent must discover the correct human-equivalent dose by combining
    knowledge of allometric scaling laws with observed human PK responses.

    Parameters
    ----------
    smiles : str
        SMILES string. All PK/PD params derived from this molecule.
    drug_name : str
    source_species : str
        Animal species of the preclinical study. Default: 'rat'.
    rat_dose_mgkg : float, optional
        Rat dose used in preclinical study. If None, computed from drug PK.
    max_human_dose_mgkg : float, optional
        Upper bound for agent's action space. If None, derived from MTC.
    max_cycles : int
        Number of adaptive refinement cycles. Default: 5.
    cohort_size : int
        Patients per mini-cohort observation. Default: 3.
    rng_seed : int, optional
    admet_predictor : ADMETPredictor, optional
    """

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 55   # 29 + 8 + 6 + 12

    def __init__(
        self,
        smiles: str,
        drug_name: str = "drug",
        source_species: str = "rat",
        rat_dose_mgkg: Optional[float] = None,
        max_human_dose_mgkg: Optional[float] = None,
        max_cycles: int = 5,
        cohort_size: int = 3,
        rng_seed: Optional[int] = None,
        admet_predictor: Optional[ADMETPredictor] = None,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Drug profile
        # ----------------------------------------------------------------
        predictor = admet_predictor or ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        self._mol     = DrugMolecule(smiles, name=drug_name)
        self._profile = extractor.extract(self._mol)

        self._pkpd = self._profile.pkpd_params
        self._pd   = self._profile.admet.to_pd_params()

        # ----------------------------------------------------------------
        # Allometric scaler
        # ----------------------------------------------------------------
        self._scaler = AllometricScaler(
            source_species=source_species,
            target_species="human",
            method="simple",
        )
        self._source = source_species

        # ----------------------------------------------------------------
        # Rat dose (preclinical reference)
        # ----------------------------------------------------------------
        if rat_dose_mgkg is None:
            # Use dose that achieves ~EC50 in rat (target exposure level)
            require_finite_keys(self._pd, ("EC50",), context="allometric pd_params")
            require_finite_keys(self._pkpd, ("Vc", "F"), context="allometric pkpd_params")
            EC50 = float(self._pd["EC50"])
            Vc = float(self._pkpd["Vc"])
            F = max(float(self._pkpd["F"]), 0.05)
            rat_dose_mgkg = float(np.clip(EC50 * Vc / F, 0.1, 50.0))
        self._rat_dose = rat_dose_mgkg

        # ----------------------------------------------------------------
        # True HED (ground truth target for reward)
        # ----------------------------------------------------------------
        self._true_hed = self._scaler.scale_dose(self._rat_dose)

        # Compute target human AUC from scaled PK params
        human_params_raw = self._scaler.scale(self._pkpd)
        human_params = {k: v for k, v in human_params_raw.items() if not k.startswith("_")}
        # Quick ODE simulation at HED to get target AUC
        self._target_auc  = self._simulate_auc(human_params, self._true_hed)
        self._target_cmax = self._simulate_cmax(human_params, self._true_hed)

        # ----------------------------------------------------------------
        # Max dose for action space
        # ----------------------------------------------------------------
        if max_human_dose_mgkg is None:
            require_finite_keys(self._pd, ("MTC",), context="allometric pd_params")
            require_finite_keys(human_params, ("Vc", "F"), context="allometric scaled human params")
            MTC = float(self._pd["MTC"])
            Vc = max(float(human_params["Vc"]), 0.05)
            F = max(float(human_params["F"]), 0.05)
            max_human_dose_mgkg = float(np.clip(MTC * Vc / F * 5.0, self._true_hed * 3, 200.0))
        self._max_dose = max_human_dose_mgkg
        self._human_pk = human_params

        # ----------------------------------------------------------------
        # Trial params
        # ----------------------------------------------------------------
        self.max_cycles  = max_cycles
        self.cohort_size = cohort_size
        self._rng        = np.random.default_rng(rng_seed)

        # ----------------------------------------------------------------
        # Gymnasium spaces
        # ----------------------------------------------------------------
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32
        )

        # ----------------------------------------------------------------
        # Precompute static scaling context (6 features)
        # ----------------------------------------------------------------
        from clinical_trial_gym.pk_pd.allometric_scaler import SPECIES_DB
        src  = SPECIES_DB[source_species]
        tgt  = SPECIES_DB["human"]
        bw_ratio = tgt.body_weight_kg / src.body_weight_kg
        _KM = {"mouse": 3, "rat": 6, "monkey": 12, "dog": 20, "human": 37}
        km_ratio = _KM.get(source_species, 6) / 37.0
        auc_scale = float(np.clip(
            (bw_ratio ** 0.25) * km_ratio, 0.01, 50.0))
        cmax_scale = float(np.clip(
            bw_ratio ** (1.0 - 0.75), 0.1, 20.0))
        self._scaling_context = np.array([
            float(np.log10(bw_ratio + 1e-8)),
            0.75,   # CL exponent
            1.00,   # Vd exponent
            float(km_ratio),
            float(np.clip(auc_scale, 0, 10)),
            float(np.clip(cmax_scale, 0, 10)),
        ], dtype=np.float32)

        # ----------------------------------------------------------------
        # Rat PK state (8 features) — simulated at rat_dose, static per episode
        # ----------------------------------------------------------------
        self._rat_pk_obs = self._compute_rat_pk_obs()

        # Episode state
        self._cycle        = 0
        self._proposed_dose = self._true_hed  # initialised, overwritten in step
        self._last_auc     = 0.0
        self._last_cmax    = 0.0
        self._last_effect  = 0.0
        self._last_dlt_rate = 0.0
        self._last_below_mec = 0.0
        self._safety_violations = 0
        self._done         = False

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

        self._cycle             = 0
        self._proposed_dose     = 0.0
        self._last_auc          = 0.0
        self._last_cmax         = 0.0
        self._last_effect       = 0.0
        self._last_dlt_rate     = 0.0
        self._last_below_mec    = 0.0
        self._safety_violations = 0
        self._done              = False

        return self._build_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Decode: action in [0,1] → dose in [0, max_dose]
        proposed_dose = float(np.clip(action[0], 0.0, 1.0)) * self._max_dose
        self._proposed_dose = proposed_dose

        # Simulate 3-patient mini-cohort at proposed dose
        auc, cmax, mean_effect, dlt_rate, below_mec_frac = self._observe_human_pk(
            proposed_dose
        )
        self._last_auc       = auc
        self._last_cmax      = cmax
        self._last_effect    = mean_effect
        self._last_dlt_rate  = dlt_rate
        self._last_below_mec = below_mec_frac

        if cmax > self._pd["MTC"]:
            self._safety_violations += 1

        # ----------------------------------------------------------------
        # Reward: log-ratio of AUC to target (scale-invariant)
        # ----------------------------------------------------------------
        reward = self._compute_reward(proposed_dose, auc, cmax, dlt_rate, below_mec_frac)

        self._cycle += 1
        terminated = self._cycle >= self.max_cycles
        truncated  = False
        if terminated:
            self._done = True
            # Terminal bonus if final dose is within ±30% of true HED
            if abs(proposed_dose - self._true_hed) / (self._true_hed + 1e-6) <= 0.30:
                reward += 5.0

        obs = self._build_obs()
        return obs, float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        proposed_dose: float,
        auc: float,
        cmax: float,
        dlt_rate: float,
        below_mec_frac: float,
    ) -> float:
        """
        Reward based on three criteria:

        1. AUC accuracy: |log(AUC_actual / AUC_target)| — scale-invariant
           Log-ratio penalises both under- and over-exposure equally.

        2. Safety: penalty if Cmax exceeds MTC (toxicity threshold).

        3. Efficacy: reward for patients above MEC.
        """
        # --- AUC accuracy (primary signal) ---
        log_ratio = abs(
            np.log(max(auc, 1e-8) + 1e-8)
            - np.log(max(self._target_auc, 1e-8) + 1e-8)
        )
        R_auc = -float(np.clip(log_ratio, 0, 5))

        # --- Safety ---
        MTC = self._pd["MTC"]
        cmax_ratio = cmax / (MTC + 1e-8)
        R_safety = -2.0 * float(np.clip(cmax_ratio - 1.0, 0, 3)) - 1.5 * dlt_rate

        # --- Efficacy ---
        mean_effect = self._last_effect
        R_efficacy = 0.5 * mean_effect if mean_effect else 0.0
        R_efficacy *= (1.0 - below_mec_frac)

        return float(R_auc + R_safety + R_efficacy)

    # ------------------------------------------------------------------
    # PK simulation helpers
    # ------------------------------------------------------------------

    def _simulate_auc(self, params: dict, dose: float) -> float:
        """Simulate AUC at given dose with given PK params."""
        ode = SurrogateODE(pkpd_params=params, pd_params=self._pd)
        ode.administer_dose(dose, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=168.0, dt_h=1.0)  # 7 days
        return float(states[-1].cumulative_AUC) if states else 0.0

    def _simulate_cmax(self, params: dict, dose: float) -> float:
        """Simulate Cmax at given dose."""
        ode = SurrogateODE(pkpd_params=params, pd_params=self._pd)
        ode.administer_dose(dose, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=48.0, dt_h=0.5)
        return float(states[-1].Cmax) if states else 0.0

    def _compute_rat_pk_obs(self) -> np.ndarray:
        """Simulate rat PK at rat_dose; return 8-feature obs array."""
        ode = SurrogateODE(pkpd_params=self._pkpd, pd_params=self._pd)
        ode.administer_dose(self._rat_dose, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=72.0, dt_h=1.0)
        if not states:
            raise ValueError("Rat PK simulation returned no states")
        stats = ode.get_summary_stats()
        return np.array([
            float(np.clip(stats["AUC"] / 1000.0, 0, 1)),
            float(np.clip(stats["Cmax"] / 50.0, 0, 1)),
            float(np.clip(stats["Tmax_h"] / 48.0, 0, 1)),
            float(np.clip(stats["mean_effect"], 0, 1)),
            float(np.clip(self._rat_dose / 50.0, 0, 1)),
            float(self._pkpd["CL"] / 20.0),
            float(self._pkpd["Vc"] / 10.0),
            float(self._pkpd["F"]),
        ], dtype=np.float32)

    def _observe_human_pk(
        self, dose: float
    ) -> Tuple[float, float, float, float, float]:
        """Simulate 3 patients at dose; return mean AUC, Cmax, effect, DLT rate, below-MEC frac."""
        pop = PatientPopulation(
            drug_profile=self._profile,
            n_patients=self.cohort_size,
            rng_seed=int(self._rng.integers(0, 2**31)),
        )
        # Override with allometrically scaled PK params
        patients = pop.sample()

        aucs, cmaxes, effects = [], [], []
        dlt_count = 0
        below_mec = 0

        for patient in patients:
            # Patch patient ODE with human-scaled PK (allometric transfer)
            patient.ode.pk.update(self._human_pk)
            patient.administer(dose, time_h=0.0, route="oral")
            for _ in range(28):
                patient.step(24.0)
                if not patient.is_active:
                    break
            stats = patient.pk_summary
            aucs.append(stats.get("AUC", 0.0))
            cmaxes.append(stats.get("Cmax", 0.0) if stats else 0.0)
            effects.append(stats.get("mean_effect", 0.0))
            if patient.has_dlt:
                dlt_count += 1
            MEC = self._pd["MEC"]
            if stats.get("Cmax", 0.0) < MEC:
                below_mec += 1

        n = max(len(patients), 1)
        return (
            float(np.mean(aucs)),
            float(np.mean(cmaxes)),
            float(np.mean(effects)),
            dlt_count / n,
            below_mec / n,
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        drug_feat = self._profile.observation_vector   # (29,)
        rat_pk    = self._rat_pk_obs                   # (8,)
        scaling   = self._scaling_context              # (6,)

        cycle_state = np.array([
            float(self._proposed_dose / (self._max_dose + 1e-8)),         # [0]
            float(np.clip(self._last_auc / (self._target_auc + 1e-8), 0, 5)),  # [1]
            float(np.clip(self._last_cmax / (self._pd["MTC"] + 1e-8), 0, 3)),  # [2]
            float(np.clip(self._target_auc / 1000.0, 0, 1)),             # [3]
            float(np.clip(abs(self._last_auc - self._target_auc) / (self._target_auc + 1e-8), 0, 5)),  # [4]
            float(np.clip(1.0 - self._last_cmax / (self._pd["MTC"] + 1e-8), 0, 1)),  # [5]
            float(self._cycle / (self.max_cycles + 1e-8)),                # [6]
            float(self.cohort_size * self._cycle / 30.0),                 # [7]
            float(self._safety_violations / (self._cycle + 1e-8)),        # [8]
            float(self._last_effect),                                     # [9]
            float(self._last_dlt_rate),                                   # [10]
            float(self._last_below_mec),                                  # [11]
        ], dtype=np.float32)

        obs = np.concatenate([drug_feat, rat_pk, scaling, cycle_state])
        if not np.all(np.isfinite(obs)):
            raise ValueError("Allometric observation contains non-finite values")
        return obs.astype(np.float32)

    def _info(self) -> dict:
        return {
            "cycle":          self._cycle,
            "true_hed":       self._true_hed,
            "proposed_dose":  self._proposed_dose,
            "target_auc":     self._target_auc,
            "last_auc":       self._last_auc,
            "dose_error_pct": abs(self._proposed_dose - self._true_hed)
                              / (self._true_hed + 1e-8) * 100,
            "drug_name":      self._mol.name,
        }

    def render(self, mode="human"):
        print(
            f"[Allometric] Drug={self._mol.name}  "
            f"Cycle={self._cycle}/{self.max_cycles}  "
            f"Proposed={self._proposed_dose:.3f}  "
            f"TrueHED={self._true_hed:.3f}  "
            f"AUC_err={abs(self._last_auc - self._target_auc):.2f}"
        )
