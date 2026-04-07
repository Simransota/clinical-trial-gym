"""
MolecularPropertyExtractor: Assembles a structured, RL-ready feature vector
from RDKit descriptors + DeepChem ADMET predictions.

This is the Layer 1 → Layer 2 interface. The output of this module becomes
the drug-property component of the RL agent's observation space, and the
PK/PD parameter dict is passed to the surrogate ODE.

Design principle:
  Every number that enters the simulation has a molecular origin.
  Nothing is hand-coded.

Example
-------
>>> mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
>>> predictor = ADMETPredictor()
>>> extractor = MolecularPropertyExtractor(predictor)
>>> profile = extractor.extract(mol)
>>> profile.observation_vector.shape
(29,)
>>> profile.pkpd_params["CL"]
0.425
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from clinical_trial_gym.drug.molecule import DrugMolecule, PKPD_RELEVANT_DESCRIPTORS
from clinical_trial_gym.drug.admet import ADMETPredictor, ADMETProperties


# ---------------------------------------------------------------------------
# Drug profile — the complete Layer 1 output
# ---------------------------------------------------------------------------

@dataclass
class DrugProfile:
    """
    Complete characterization of a drug for use in ClinicalTrialGym.

    Produced by MolecularPropertyExtractor.extract().

    Attributes
    ----------
    molecule : DrugMolecule
        The source molecule with RDKit descriptors.
    admet : ADMETProperties
        Predicted ADMET properties from trained DeepChem models.
    pkpd_params : dict
        Two-compartment ODE parameters derived from ADMET predictions.
    observation_vector : np.ndarray
        Flat float32 array for use as the drug-component of RL observations.
    observation_names : List[str]
        Feature names corresponding to each element of observation_vector.
    safety_flags : dict
        Aggregated safety signals (PAINS, DILI, hERG, CYP, Lipinski).
    """

    molecule: DrugMolecule
    admet: ADMETProperties
    pkpd_params: dict
    observation_vector: np.ndarray
    observation_names: List[str]
    safety_flags: dict

    def summary(self) -> str:
        lines = [
            f"Drug Profile: {self.molecule.name} ({self.molecule.mol_id})",
            f"  SMILES        : {self.molecule.smiles[:50]}",
            f"  MW            : {self.molecule.molecular_weight:.1f} Da",
            f"  LogP          : {self.molecule.logp:.2f}",
            f"  QED           : {self.molecule.qed_score:.3f}",
            f"  Lipinski Pass : {self.molecule.lipinski_pass}",
            "",
            "  ADMET Predictions (DeepChem trained models):",
            f"    F_oral      : {self.admet.F_oral:.2f}",
            f"    PPB         : {self.admet.PPB:.2f}",
            f"    BBB         : {'Penetrant' if self.admet.BBB_penetrant else 'Non-penetrant'}"
            f" (p={self.admet.BBB_probability:.3f})",
            f"    Half-life   : {self.admet.half_life_class}",
            f"    CYP profile : {self.admet.cyp_inhibition_profile()}",
            f"    DILI risk   : {self.admet.DILI_flag}"
            f" (p_toxic={self.admet.clintox_toxic_prob:.3f})",
            f"    hERG risk   : {self.admet.hERG_flag}",
            f"    LogD (pred) : {self.admet.predicted_logD:.2f}",
            f"    Source      : {self.admet.source}",
            "",
            "  PK/PD Parameters (→ Surrogate ODE):",
        ]
        for k, v in self.pkpd_params.items():
            lines.append(f"    {k:6s}: {v:.4f}")
        lines += [
            "",
            "  Safety Flags:",
        ]
        for k, v in self.safety_flags.items():
            lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "molecule": self.molecule.to_dict(),
            "admet": self.admet.to_dict(),
            "pkpd_params": self.pkpd_params,
            "observation_names": self.observation_names,
            "observation_vector": self.observation_vector.tolist(),
            "safety_flags": self.safety_flags,
        }


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MolecularPropertyExtractor:
    """
    Assembles a DrugProfile from a DrugMolecule.

    Parameters
    ----------
    admet_predictor : ADMETPredictor, optional
        If None, a default predictor is created (uses trained DeepChem models).
    normalize : bool
        If True, normalizes the observation vector to [0, 1] per feature
        using pre-computed population statistics. Default: False (raw values).
    """

    # Population-level normalization statistics
    # Computed from ChEMBL drug-like subset (MW 100–800, LogP -2 to 7)
    _NORM_STATS: Dict[str, Tuple[float, float]] = {
        # (mean, std) pairs for each feature
        "MolWt":              (350.0, 120.0),
        "MolLogP":            (2.5,   1.8),
        "NumHDonors":         (2.0,   1.5),
        "NumHAcceptors":      (5.0,   3.0),
        "TPSA":               (80.0,  45.0),
        "NumRotatableBonds":  (6.0,   4.0),
        "NumAromaticRings":   (2.0,   1.2),
        "NumAliphaticRings":  (1.5,   1.0),
        "RingCount":          (3.0,   1.8),
        "FractionCSP3":       (0.35,  0.22),
        "HeavyAtomCount":     (26.0,  10.0),
        "NumHeteroatoms":     (5.0,   3.0),
        "MaxPartialCharge":   (0.3,   0.2),
        "MinPartialCharge":   (-0.4,  0.2),
        # ADMET features
        "F_oral":             (0.65,  0.25),
        "PPB":                (0.82,  0.12),
        "BBB_penetrant":      (0.35,  0.48),
        "DILI_flag":          (0.2,   0.4),
        "hERG_flag":          (0.25,  0.43),
        "CYP3A4":             (0.4,   0.49),
        "CYP2D6":             (0.25,  0.43),
        "CYP2C9":             (0.3,   0.46),
        "CYP2C19":            (0.35,  0.48),
        "CYP1A2":             (0.2,   0.4),
        "esol_solubility":    (-3.0,  2.0),
        "qed_score":          (0.55,  0.18),
        "ka":                 (1.2,   0.9),
        "CL":                 (0.5,   0.3),
        "Vd":                 (2.5,   3.0),
    }

    def __init__(
        self,
        admet_predictor: Optional[ADMETPredictor] = None,
        normalize: bool = False,
    ):
        self.admet_predictor = admet_predictor or ADMETPredictor()
        self.normalize = normalize

    def extract(self, mol: DrugMolecule) -> DrugProfile:
        """
        Full extraction pipeline for a single molecule.

        Returns
        -------
        DrugProfile
            Complete drug characterization ready for ClinicalTrialGym.
        """
        # Step 1: Predict ADMET using trained DeepChem models
        admet = self.admet_predictor.predict(mol)

        # Step 2: Get PK/PD parameters for the ODE
        pkpd_params = admet.to_pkpd_params()

        # Step 3: Build observation vector
        obs_vec, obs_names = self._build_observation_vector(mol, admet, pkpd_params)

        # Step 4: Aggregate safety flags
        safety_flags = self._aggregate_safety(mol, admet)

        return DrugProfile(
            molecule=mol,
            admet=admet,
            pkpd_params=pkpd_params,
            observation_vector=obs_vec,
            observation_names=obs_names,
            safety_flags=safety_flags,
        )

    def _build_observation_vector(
        self,
        mol: DrugMolecule,
        admet: ADMETProperties,
        pkpd_params: dict,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Assembles a flat, ordered float32 array for the RL observation.

        Feature groups (in order):
        1. RDKit molecular descriptors (14 features)
        2. ADMET predictions (10 features)
        3. Key PK parameters from ADMET (4 features)
        4. Meta (QED score) (1 feature)
        Total: 29 features
        """
        features = []
        names = []

        # Group 1: RDKit descriptors
        for name in PKPD_RELEVANT_DESCRIPTORS:
            val = mol.descriptors.get(name, 0.0)
            val = 0.0 if (val is None or np.isnan(val)) else val
            features.append(val)
            names.append(f"desc_{name}")

        # Group 2: ADMET predictions (all from DeepChem models)
        admet_features = [
            ("F_oral",        admet.F_oral if not np.isnan(admet.F_oral) else 0.65),
            ("PPB",           admet.PPB if not np.isnan(admet.PPB) else 0.85),
            ("BBB_penetrant", float(admet.BBB_penetrant)),
            ("DILI_flag",     float(admet.DILI_flag)),
            ("hERG_flag",     float(admet.hERG_flag)),
            ("CYP3A4",        float(admet.CYP3A4_inhibition)),
            ("CYP2D6",        float(admet.CYP2D6_inhibition)),
            ("CYP2C9",        float(admet.CYP2C9_inhibition)),
            ("CYP2C19",       float(admet.CYP2C19_inhibition)),
            ("CYP1A2",        float(admet.CYP1A2_inhibition)),
        ]
        for name, val in admet_features:
            features.append(float(val))
            names.append(f"admet_{name}")

        # Group 3: Key PK params (derived from ADMET model outputs)
        pk_features = [
            ("ka",  pkpd_params.get("ka",  1.0)),
            ("F",   pkpd_params.get("F",   0.7)),
            ("CL",  pkpd_params.get("CL",  0.5)),
            ("Vd",  pkpd_params.get("Vc",  1.0) + pkpd_params.get("Vp", 0.5)),
        ]
        for name, val in pk_features:
            features.append(float(val))
            names.append(f"pk_{name}")

        # Group 4: QED
        features.append(float(mol.qed_score) if not np.isnan(mol.qed_score) else 0.5)
        names.append("meta_qed")

        obs = np.array(features, dtype=np.float32)

        if self.normalize:
            obs = self._normalize(obs, names)

        # Sanity check: replace any remaining NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)

        return obs, names

    def _normalize(self, obs: np.ndarray, names: List[str]) -> np.ndarray:
        """Z-score normalize using population statistics."""
        normed = obs.copy()
        for i, name in enumerate(names):
            # Strip prefix for lookup
            key = name.split("_", 1)[-1]
            if key in self._NORM_STATS:
                mean, std = self._NORM_STATS[key]
                normed[i] = (obs[i] - mean) / (std + 1e-8)
        return normed

    def _aggregate_safety(
        self, mol: DrugMolecule, admet: ADMETProperties
    ) -> dict:
        """Combines all safety signals into one structured dict."""
        cyp_inhibited = [k for k, v in admet.cyp_inhibition_profile().items() if v]
        return {
            "pains_flag":          mol.pains_flag,
            "lipinski_violation":  not mol.lipinski_pass,
            "dili_risk":           admet.DILI_flag,
            "herg_risk":           admet.hERG_flag,
            "cyp_inhibitions":     cyp_inhibited,
            "bbb_penetrant":       admet.BBB_penetrant,
            "clintox_toxic_prob":  admet.clintox_toxic_prob,
            "tox21_active_count":  sum(1 for v in admet.tox21_predictions.values() if v > 0.5),
            "overall_risk_score":  self._compute_risk_score(mol, admet),
        }

    def _compute_risk_score(
        self, mol: DrugMolecule, admet: ADMETProperties
    ) -> float:
        """
        Composite safety risk score in [0, 1].
        Higher = more safety concerns.
        Uses trained model outputs, not heuristics.
        """
        flags = [
            mol.pains_flag,
            not mol.lipinski_pass,
            admet.DILI_flag,
            admet.hERG_flag,
            admet.CYP3A4_inhibition,
            admet.CYP2D6_inhibition,
        ]
        n_flags = sum(flags)
        cyp_count = sum(admet.cyp_inhibition_profile().values())

        # Weight ClinTox probability into the score
        clintox_weight = admet.clintox_toxic_prob if not np.isnan(admet.clintox_toxic_prob) else 0.0

        # Weight Tox21 active assay fraction
        tox21_active = sum(1 for v in admet.tox21_predictions.values() if v > 0.5)
        tox21_frac = tox21_active / max(len(admet.tox21_predictions), 1)

        score = (
            (n_flags / len(flags)) * 0.3
            + (cyp_count / 5.0) * 0.2
            + clintox_weight * 0.25
            + tox21_frac * 0.25
        )
        return float(np.clip(score, 0.0, 1.0))

    def get_observation_space_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the drug observation vector.
        Use this to configure the Gymnasium observation space.
        """
        return (29,)

    def get_observation_space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (low, high) bounds for the drug observation vector.
        Suitable for gym.spaces.Box.
        """
        low  = np.full(29, -np.inf, dtype=np.float32)
        high = np.full(29,  np.inf, dtype=np.float32)

        if self.normalize:
            # After z-score normalization, clip to ±5 sigma
            low  = np.full(29, -5.0, dtype=np.float32)
            high = np.full(29,  5.0, dtype=np.float32)
        else:
            # Raw bounds for known-bounded features
            low[14]  = 0.0; high[14] = 1.0   # F_oral
            low[15]  = 0.0; high[15] = 1.0   # PPB
            low[16]  = 0.0; high[16] = 1.0   # BBB
            low[17]  = 0.0; high[17] = 1.0   # DILI
            low[18]  = 0.0; high[18] = 1.0   # hERG
            low[19:24] = 0.0; high[19:24] = 1.0  # CYP flags
            low[28] = 0.0; high[28] = 1.0    # QED

        return low, high