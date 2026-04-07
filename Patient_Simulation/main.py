"""
ClinicalTrialGym: Layer 1 + Layer 2 demonstration.

This script demonstrates the full pipeline:
  SMILES → RDKit descriptors → DeepChem ADMET models → PK/PD parameters
  → Surrogate ODE → Patient simulation → DLT assessment

Every number entering the simulation is derived from the molecule's
structure via trained models. Nothing is hardcoded.

Run:
    python main.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import warnings
import sys
warnings.filterwarnings("ignore")

import rdkit.RDLogger as RDLogger
RDLogger.DisableLog('rdApp.*')

# Suppress absl warnings
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# Suppress the weird cleanup Exception output
def dummy_excepthook(*args, **kwargs):
    pass
sys.unraisablehook = dummy_excepthook

from clinical_trial_gym.drug.molecule import DrugMolecule
from clinical_trial_gym.drug.admet import ADMETPredictor
from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
from clinical_trial_gym.pk_pd.patient_agent import PatientAgent, PatientPopulation

# ── Layer 1: Drug Property Pipeline ──────────────────────────────────────────

# Start from a SMILES string — nothing is hardcoded
mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
print(mol)
# DrugMolecule(name='Aspirin', MW=180.2, LogP=1.31, QED=0.554, Lipinski=✓)

# Predict ADMET properties using trained DeepChem models
# First run trains models on MoleculeNet datasets; subsequent runs use cache.
predictor = ADMETPredictor()
extractor = MolecularPropertyExtractor(predictor)
profile = extractor.extract(mol)
print(profile.summary())

# The RL observation vector (drug component)
print(f"\nObservation vector shape: {profile.observation_vector.shape}")  # (29,)

# PK/PD parameters for the ODE — all derived from ADMET model outputs
print(f"\nPK/PD params (from trained DeepChem models):")
for k, v in profile.pkpd_params.items():
    print(f"  {k}: {v:.4f}")


# ── Allometric scaling: rat preclinical → human ───────────────────────────────

scaler = AllometricScaler(source_species="rat", target_species="human")

# Scale PK params
rat_params = profile.pkpd_params
human_params = scaler.scale(rat_params)
print(f"\nRat CL: {rat_params['CL']:.3f} L/h/kg")
print(f"Human CL (scaled): {human_params['CL']:.3f} L/h/kg")

# Scale dose (FDA Km method)
rat_dose_mgkg = 10.0
hed = scaler.scale_dose(rat_dose_mgkg)
print(f"Human Equivalent Dose: {hed:.2f} mg/kg")


# ── Layer 2: Patient Simulation ───────────────────────────────────────────────

# Simulate a 6-patient Phase I cohort
pop = PatientPopulation(profile, n_patients=6, rng_seed=42)
cohort = pop.sample()

for patient in cohort:
    patient.administer(dose_mgkg=hed, time_h=0.0, route="oral")
    patient.step(duration_h=24.0)   # Day 1

# Check for DLTs
dlts = [p for p in cohort if p.has_dlt]
print(f"\nDLTs in cohort: {len(dlts)}/{len(cohort)}")

# Get RL observation for each patient
for patient in cohort:
    obs = patient.observation   # shape (17,)
    print(f"  {patient} → peak grade: {patient.peak_grade.name}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── PK Summary per patient ──")
for patient in cohort:
    stats = patient.pk_summary
    if stats:
        print(
            f"  {patient.covariates.patient_id}: "
            f"AUC={stats['AUC']:.2f} mg·h/L, "
            f"Cmax={stats['Cmax']:.3f} mg/L, "
            f"Tmax={stats['Tmax_h']:.1f}h"
        )