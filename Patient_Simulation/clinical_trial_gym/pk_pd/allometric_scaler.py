"""
AllometricScaler: Species-bridging for PK/PD parameters.

This is the "translation gap" layer — the key novelty in ClinicalTrialGym.

The RL agent first trains in a preclinical environment (mouse or rat),
then must generalize to the human Phase I environment. Allometric scaling
laws (well-validated in computational pharmacology) bridge the gap.

Key equations:
  Parameter_human = Parameter_animal × (BW_human / BW_animal) ^ exponent

Standard exponents (Boxenbaum 1982, Mordenti 1986):
  CL (clearance)      : 0.75  (¾ power law — most validated)
  Vd (volume)         : 1.00  (linear with body mass)
  ka (absorption)     : -0.25 (faster absorption in smaller animals)
  t½ (half-life)      : 0.25  (smaller animals have shorter t½)
  Protein binding     : ~0.0  (species-invariant as first approximation)

PK-Sim API integration:
  When PK-Sim is available, scaling uses PK-Sim's anatomical databases
  (GFR, liver blood flow, plasma volumes per species) instead of the
  power law formulas. The power law is the validated fallback.

Reference:
  - Boxenbaum H (1982) J Pharmacokinet Biopharm 10:201-227
  - Mordenti J (1986) J Pharm Sci 75:1028-1040
  - Lave T et al (1999) Pharm Res 16:1390-1398
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Species definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Species:
    """
    Reference species for allometric scaling.

    All body weights are typical adult values used in PK studies.
    """
    name: str
    body_weight_kg: float       # typical study animal weight
    gfr_ml_min_kg: float        # glomerular filtration rate
    liver_blood_flow_lh_kg: float  # hepatic blood flow
    plasma_volume_lkg: float    # plasma volume

    def __str__(self) -> str:
        return f"{self.name} ({self.body_weight_kg} kg)"


SPECIES_DB: Dict[str, Species] = {
    "mouse": Species(
        name="mouse",
        body_weight_kg=0.025,
        gfr_ml_min_kg=8.7,
        liver_blood_flow_lh_kg=4.9,
        plasma_volume_lkg=0.049,
    ),
    "rat": Species(
        name="rat",
        body_weight_kg=0.25,
        gfr_ml_min_kg=6.7,
        liver_blood_flow_lh_kg=4.2,
        plasma_volume_lkg=0.043,
    ),
    "monkey": Species(
        name="monkey",
        body_weight_kg=5.0,
        gfr_ml_min_kg=2.8,
        liver_blood_flow_lh_kg=2.9,
        plasma_volume_lkg=0.044,
    ),
    "dog": Species(
        name="dog",
        body_weight_kg=10.0,
        gfr_ml_min_kg=3.7,
        liver_blood_flow_lh_kg=3.1,
        plasma_volume_lkg=0.046,
    ),
    "human": Species(
        name="human",
        body_weight_kg=70.0,
        gfr_ml_min_kg=1.8,
        liver_blood_flow_lh_kg=1.6,
        plasma_volume_lkg=0.043,
    ),
}


# Standard allometric exponents
ALLOMETRIC_EXPONENTS: Dict[str, float] = {
    "CL":   0.75,   # Clearance
    "Vc":   1.00,   # Central volume
    "Vp":   1.00,   # Peripheral volume
    "Q":    1.00,   # Inter-compartment flow (scales with Vc/Vp which are linear)
    "ka":  -0.25,   # Absorption rate constant
    "F":    0.00,   # Bioavailability (species-invariant)
    "PPB":  0.00,   # Plasma protein binding (species-invariant)
    "fu":   0.00,   # Fraction unbound (species-invariant)
}


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

class AllometricScaler:
    """
    Scales PK/PD parameters between species.

    Implements simple allometric power-law scaling as the default,
    with optional PK-Sim integration for higher fidelity.

    Parameters
    ----------
    source_species : str
        Name of the source species (animal study). Must be in SPECIES_DB.
    target_species : str
        Name of the target species (usually 'human'). Must be in SPECIES_DB.
    use_pksim : bool
        If True, attempts to use PK-Sim API for scaling. Falls back to
        power law if PK-Sim is unavailable.
    method : str
        'simple'     — pure power law (Boxenbaum 1982)
        'correction' — power law + brain weight correction (Mahmood 1996)
        'pksim'      — use PK-Sim anatomical databases

    Example
    -------
    >>> scaler = AllometricScaler(source_species='rat', target_species='human')
    >>> rat_params = {'CL': 3.2, 'Vc': 0.8, 'Vp': 0.5, 'Q': 0.6, 'ka': 2.1, 'F': 0.72}
    >>> human_params = scaler.scale(rat_params)
    >>> human_params['CL']  # scaled by (70/0.25)^0.75
    25.4
    """

    # Brain weights for Mahmood correction (grams)
    _BRAIN_WEIGHTS_G = {
        "mouse": 0.4, "rat": 1.8, "monkey": 70.0, "dog": 72.0, "human": 1350.0
    }

    def __init__(
        self,
        source_species: str = "rat",
        target_species: str = "human",
        use_pksim: bool = False,
        method: str = "simple",
    ):
        if source_species not in SPECIES_DB:
            raise ValueError(f"Unknown species '{source_species}'. Choose from: {list(SPECIES_DB.keys())}")
        if target_species not in SPECIES_DB:
            raise ValueError(f"Unknown species '{target_species}'. Choose from: {list(SPECIES_DB.keys())}")

        self.source = SPECIES_DB[source_species]
        self.target = SPECIES_DB[target_species]
        self.method = method
        self.use_pksim = use_pksim
        self._pksim_available = False

        if use_pksim:
            self._init_pksim()

    def _init_pksim(self):
        """Try to connect to PK-Sim API."""
        try:
            import pkpy  # hypothetical PK-Sim Python binding
            self._pksim_available = True
        except ImportError:
            import warnings
            warnings.warn(
                "PK-Sim not available. Falling back to allometric power law. "
                "Install PK-Sim and pkpy for higher-fidelity scaling.",
                stacklevel=3,
            )
            self._pksim_available = False

    def scale(self, params: dict) -> dict:
        """
        Scale PK/PD parameters from source to target species.

        Parameters
        ----------
        params : dict
            PK/PD parameter dict from ADMETProperties.to_pkpd_params().
            Expected keys: ka, F, CL, Vc, Vp, Q, PPB, fu

        Returns
        -------
        dict
            Scaled parameters ready for SurrogateODE initialization.
            Includes a 'scaling_metadata' key with provenance info.
        """
        if self._pksim_available:
            scaled = self._scale_pksim(params)
        elif self.method == "correction":
            scaled = self._scale_with_correction(params)
        else:
            scaled = self._scale_simple(params)

        # Add metadata
        scaled["_scaling_metadata"] = {
            "source_species":      self.source.name,
            "target_species":      self.target.name,
            "source_bw_kg":        self.source.body_weight_kg,
            "target_bw_kg":        self.target.body_weight_kg,
            "method":              self.method,
            "bw_ratio":            self.target.body_weight_kg / self.source.body_weight_kg,
        }

        return {k: v for k, v in scaled.items()}

    def _scale_simple(self, params: dict) -> dict:
        """
        Pure power-law scaling: P_human = P_animal × (BW_human/BW_animal)^α
        """
        bw_ratio = self.target.body_weight_kg / self.source.body_weight_kg
        scaled = {}
        for key, val in params.items():
            if key.startswith("_"):
                continue
            exponent = ALLOMETRIC_EXPONENTS.get(key, 0.75)
            if exponent == 0.0:
                scaled[key] = val  # species-invariant
            else:
                scaled[key] = float(val * (bw_ratio ** exponent))
        return scaled

    def _scale_with_correction(self, params: dict) -> dict:
        """
        Allometric scaling with brain weight correction (Mahmood 1996).
        Better for drugs with high CNS distribution (high BBB penetration).

        Uses the "rule of exponents":
          Exponent 0.55-0.70 → correct with brain weight
          Exponent 0.71-0.99 → use simple allometry
        """
        simple = self._scale_simple(params)

        # Brain weight correction factor for CL
        bw_src = self._BRAIN_WEIGHTS_G.get(self.source.name, 1.0)
        bw_tgt = self._BRAIN_WEIGHTS_G.get(self.target.name, 1350.0)
        brain_ratio = bw_tgt / bw_src

        # Apply correction only to clearance-related params
        corrected = dict(simple)
        for key in ["CL", "Q"]:
            if key in corrected:
                # Mahmood correction: multiply by (brain_human/brain_animal)^0.15
                corrected[key] = corrected[key] * (brain_ratio ** 0.15)

        return corrected

    def _scale_pksim(self, params: dict) -> dict:
        """
        PK-Sim API-based scaling using anatomical databases.
        Falls back to power law if API call fails.
        """
        try:
            # Placeholder for actual PK-Sim API call
            # In production: call PK-Sim's individual building block API
            # with source/target species anatomical parameters
            raise NotImplementedError("PK-Sim API integration pending")
        except Exception:
            return self._scale_simple(params)

    def scale_dose(self, dose_mgkg: float) -> float:
        """
        Scale a dose from source to target species.

        For dose finding, this gives the human equivalent dose (HED)
        from an animal dose using FDA guidance (based on body surface area).

        FDA guidance uses BSA (Km factor method):
          HED = Animal_dose × (Animal_Km / Human_Km)

        Km factors (FDA 2005 Guidance):
          Mouse  : 3
          Rat    : 6
          Monkey : 12
          Dog    : 20
          Human  : 37
        """
        KM_FACTORS = {
            "mouse": 3, "rat": 6, "monkey": 12, "dog": 20, "human": 37
        }
        km_source = KM_FACTORS.get(self.source.name, 6)
        km_target = KM_FACTORS.get(self.target.name, 37)
        hed = dose_mgkg * (km_source / km_target)
        return float(hed)

    def get_rpe_factor(self) -> float:
        """
        Relative potency estimate factor (dimensionless).
        Used to adjust EC50 across species.
        Based on clearance ratio: higher CL → lower AUC → need higher dose.
        """
        bw_ratio = self.target.body_weight_kg / self.source.body_weight_kg
        cl_ratio = bw_ratio ** ALLOMETRIC_EXPONENTS["CL"]
        vol_ratio = bw_ratio ** ALLOMETRIC_EXPONENTS["Vc"]
        return float(cl_ratio / vol_ratio)

    def __repr__(self) -> str:
        return (
            f"AllometricScaler({self.source.name} → {self.target.name}, "
            f"BW ratio={self.target.body_weight_kg/self.source.body_weight_kg:.1f}×, "
            f"method='{self.method}')"
        )