from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from clinical_trial_gym.drug.properties import DrugProfile


def require_finite_keys(
    params: Mapping[str, float],
    keys: Iterable[str],
    *,
    context: str,
) -> None:
    """Fail loudly when simulation-critical parameters are missing."""
    for key in keys:
        if key not in params:
            raise KeyError(f"{context} missing required key '{key}'")
        value = float(params[key])
        if not np.isfinite(value):
            raise ValueError(f"{context} key '{key}' must be finite, got {value!r}")


@dataclass(frozen=True)
class PhaseITrialPriors:
    cohort_options: tuple[int, ...]
    target_dlt: float
    target_dlt_lower: float
    target_dlt_upper: float
    safety_weight: float
    efficacy_weight: float
    cost_weight: float
    speed_weight: float


@dataclass(frozen=True)
class DDIPriors:
    fm_victim: float
    weak_threshold: float
    moderate_threshold: float
    strong_threshold: float
    safety_weight: float
    efficacy_weight: float
    interaction_weight: float
    cost_weight: float


def derive_phase_i_priors(profile: DrugProfile) -> PhaseITrialPriors:
    """
    Derive dose-finding priors from drug risk and prediction certainty.

    The constants here are scientific design priors rather than hidden
    fallbacks: they encode common early oncology design targets and are
    adjusted per molecule via risk score, therapeutic index, and model
    confidence.
    """
    admet = profile.admet
    pd = admet.to_pd_params()
    therapeutic_index = float(pd["MTC"] / pd["EC50"])
    confidence = float(np.clip(admet.prediction_confidence, 0.05, 1.0))
    risk = float(np.clip(profile.safety_flags["overall_risk_score"], 0.0, 1.0))

    target_dlt = float(np.clip(0.30 - 0.08 * risk - 0.04 * (1.0 - confidence), 0.16, 0.30))
    lower = float(max(0.10, target_dlt - 0.05))
    upper = float(min(0.33, target_dlt + 0.05))

    min_cohort = 3
    max_cohort = 4 if (risk > 0.75 or confidence < 0.55) else 5 if risk > 0.50 else 6

    return PhaseITrialPriors(
        cohort_options=tuple(range(min_cohort, max_cohort + 1)),
        target_dlt=target_dlt,
        target_dlt_lower=lower,
        target_dlt_upper=upper,
        safety_weight=float(2.2 + 1.4 * risk + 0.4 * (1.0 - confidence)),
        efficacy_weight=float(0.7 + 0.4 * therapeutic_index / (therapeutic_index + 10.0)),
        cost_weight=float(0.02 + 0.03 * (1.0 - confidence)),
        speed_weight=float(0.04 + 0.04 * (1.0 - confidence)),
    )


def derive_combo_ddi_priors(
    perpetrator_profile: DrugProfile,
    victim_profile: DrugProfile,
    fm_victim: float,
) -> DDIPriors:
    """
    Derive DDI reward weights from both molecules and the victim's fm.
    """
    risk_a = float(np.clip(perpetrator_profile.safety_flags["overall_risk_score"], 0.0, 1.0))
    risk_b = float(np.clip(victim_profile.safety_flags["overall_risk_score"], 0.0, 1.0))
    confidence = float(
        np.clip(
            min(
                perpetrator_profile.admet.prediction_confidence,
                victim_profile.admet.prediction_confidence,
            ),
            0.05,
            1.0,
        )
    )
    combo_risk = max(risk_a, risk_b)
    interaction_weight = 0.25 + 0.75 * float(np.clip(fm_victim, 0.0, 1.0))
    return DDIPriors(
        fm_victim=float(np.clip(fm_victim, 0.0, 0.99)),
        weak_threshold=1.25,
        moderate_threshold=2.0,
        strong_threshold=5.0,
        safety_weight=float(2.0 + 1.2 * combo_risk + 0.5 * (1.0 - confidence)),
        efficacy_weight=float(0.6 + 0.2 * confidence),
        interaction_weight=float(interaction_weight),
        cost_weight=float(0.03 + 0.02 * combo_risk),
    )
