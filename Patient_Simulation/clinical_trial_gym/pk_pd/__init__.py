"""Layer 2: PK/PD simulation (surrogate ODE, patient agents, allometric scaling)."""
from clinical_trial_gym.pk_pd.surrogate_ode     import SurrogateODE, PKPDState, DoseEvent
from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler, SPECIES_DB, ALLOMETRIC_EXPONENTS
from clinical_trial_gym.pk_pd.patient_agent     import (
    PatientAgent, PatientPopulation, PatientCovariates, PopulationVariability, CTCAEGrade,
)

__all__ = [
    "SurrogateODE", "PKPDState", "DoseEvent",
    "AllometricScaler", "SPECIES_DB", "ALLOMETRIC_EXPONENTS",
    "PatientAgent", "PatientPopulation", "PatientCovariates",
    "PopulationVariability", "CTCAEGrade",
]
