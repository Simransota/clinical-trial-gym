"""
ClinicalTrialGym — Gymnasium-compliant RL environments for clinical trial optimization.

Full pipeline:
    Preclinical (animal PK/PD)
        ↓  allometric scaling
    Phase I  (dose-finding, safety first)
        ↓  RP2D handoff
    Phase II (efficacy signal, adaptive)
        ↓  go/no-go gate
    Phase III (RCT, superiority test)
        ↓  approval probability

Layers:
    Layer 1: RDKit/DeepChem — SMILES → molecular features + ADMET predictions
    Layer 2: Surrogate ODE  — two-compartment PK/PD model (BioGears-calibrated)
    Layer 3: Gymnasium envs — RL task environments
"""

__version__ = "0.1.0"
__author__  = "ClinicalTrialGym"

from clinical_trial_gym.drug.molecule    import DrugMolecule
from clinical_trial_gym.drug.admet       import ADMETPredictor, ADMETProperties
from clinical_trial_gym.drug.properties  import MolecularPropertyExtractor, DrugProfile
from clinical_trial_gym.pk_pd.surrogate_ode    import SurrogateODE, PKPDState
from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
from clinical_trial_gym.pk_pd.patient_agent    import PatientAgent, PatientPopulation

__all__ = [
    "DrugMolecule",
    "ADMETPredictor",
    "ADMETProperties",
    "MolecularPropertyExtractor",
    "DrugProfile",
    "SurrogateODE",
    "PKPDState",
    "AllometricScaler",
    "PatientAgent",
    "PatientPopulation",
]
