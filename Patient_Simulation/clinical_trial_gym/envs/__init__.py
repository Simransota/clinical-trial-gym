"""
Layer 3: Gymnasium RL environments for ClinicalTrialGym.

Task 1 — phase_i_dosing   (Easy)
    PhaseIDoseEscalationEnv: agent runs Phase I trial and finds the RP2D.

Task 2 — allometric_scaling  (Medium)
    AllometricScalingEnv: given rat PK, propose correct human equivalent dose.

Task 3 — combo_ddi  (Hard)
    ComboDDIEnv: schedule two drugs while managing CYP450 enzyme competition.
"""

from clinical_trial_gym.envs.phase_i_env      import PhaseIDoseEscalationEnv
from clinical_trial_gym.envs.allometric_env   import AllometricScalingEnv
from clinical_trial_gym.envs.combo_ddi_env    import ComboDDIEnv

__all__ = [
    "PhaseIDoseEscalationEnv",
    "AllometricScalingEnv",
    "ComboDDIEnv",
]
