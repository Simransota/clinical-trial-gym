# ClinicalTrialGym

A Gymnasium-compliant RL environment for full drug trial lifecycle optimization.

> **Goal**: Before real clinical testing begins, the system learns the optimal cohort size, dosing schedule, and trial structure — so testing is done on the right patients, at the right dose, within budget, with predictable outcomes.

---

## Architecture

```
SMILES string (e.g. "CC(=O)Oc1ccccc1C(=O)O")
        │
        ▼
┌─────────────────────────────────────┐
│  LAYER 1: Molecular Property Pipeline│
│                                     │
│  DrugMolecule (RDKit)               │
│    ├── MW, LogP, TPSA, QED          │
│    └── Lipinski / PAINS flags       │
│                                     │
│  ADMETPredictor (DeepChem/heuristic)│
│    ├── F_oral, PPB, BBB             │
│    ├── CYP inhibition profile       │
│    └── DILI, hERG risk flags        │
│                                     │
│  MolecularPropertyExtractor         │
│    ├── observation_vector [29]      │  ← drug component of RL obs
│    ├── pkpd_params (ka,F,CL,Vc...) │  → into Layer 2
│    └── safety_flags                 │
└─────────────────────────────────────┘
        │  pkpd_params
        ▼
┌─────────────────────────────────────┐
│  LAYER 2: Surrogate PK/PD Simulation│
│                                     │
│  AllometricScaler                   │
│    rat/mouse → human via BW^α       │  ← "translation gap"
│    FDA Km-factor dose scaling       │
│                                     │
│  SurrogateODE (two-compartment)     │
│    ├── Calibrated to BioGears data  │
│    ├── Depot → Central → Peripheral │
│    ├── Emax/Hill PD model           │
│    └── AUC, Cmax, toxicity tracking │
│                                     │
│  PatientAgent                       │
│    ├── IIV via log-normal sampling  │
│    ├── Covariate adjustments        │
│    │   (age, sex, organ function)   │
│    ├── CTCAE toxicity grading       │
│    └── observation_vector [17]      │  → into RL environment
│                                     │
│  PatientPopulation                  │
│    └── cohort of N PatientAgents    │
└─────────────────────────────────────┘
        │
        ▼  (Layer 3 — coming next)
┌─────────────────────────────────────┐
│  Gymnasium Environment              │
│    ├── FDA safety wrapper           │
│    ├── Multi-objective Pareto reward│
│    │   [efficacy, safety, cost,     │
│    │    trial_speed]                │
│    └── Phase pipeline               │
│        Preclinical → I → II → III   │
└─────────────────────────────────────┘
```

---

## Installation

```bash
# Clone and install
git clone https://github.com/yourlab/ClinicalTrialGym.git
cd ClinicalTrialGym
pip install -e ".[dev]"

# RDKit (required for Layer 1):
conda install -c conda-forge rdkit

# DeepChem (optional — falls back to validated heuristics):
pip install deepchem
```

---

## Quickstart

```python
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

# Predict ADMET properties
predictor = ADMETPredictor()
extractor = MolecularPropertyExtractor(predictor)
profile = extractor.extract(mol)
print(profile.summary())

# The RL observation vector (drug component)
print(profile.observation_vector.shape)  # (29,)

# PK/PD parameters for the ODE
print(profile.pkpd_params)
# {'ka': 1.24, 'F': 0.80, 'CL': 0.41, 'Vc': 0.48, 'Vp': 0.32, ...}


# ── Allometric scaling: rat preclinical → human ───────────────────────────────

scaler = AllometricScaler(source_species="rat", target_species="human")

# Scale PK params
rat_params = profile.pkpd_params
human_params = scaler.scale(rat_params)
print(f"Rat CL: {rat_params['CL']:.3f} L/h/kg")
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
print(f"DLTs in cohort: {len(dlts)}/{len(cohort)}")

# Get RL observation for each patient
for patient in cohort:
    obs = patient.observation   # shape (17,)
    print(f"  {patient} → peak grade: {patient.peak_grade.name}")
```

---

## Running Tests

```bash
pytest tests/test_layer1_layer2.py -v
```

Expected output (without DeepChem, heuristic mode):
```
tests/test_layer1_layer2.py::TestDrugMolecule::test_valid_smiles_parses PASSED
tests/test_layer1_layer2.py::TestDrugMolecule::test_aspirin_molecular_weight PASSED
...
tests/test_layer1_layer2.py::TestEndToEndPipeline::test_allometric_transfer PASSED
42 passed in 3.2s
```

---

## Project Structure

```
ClinicalTrialGym/
├── clinical_trial_gym/
│   ├── drug/
│   │   ├── molecule.py       # DrugMolecule: SMILES → RDKit descriptors
│   │   ├── admet.py          # ADMETPredictor: DeepChem / heuristic models
│   │   └── properties.py     # MolecularPropertyExtractor: Layer 1→2 interface
│   ├── pk_pd/
│   │   ├── surrogate_ode.py  # SurrogateODE: 2-compartment PK/PD model
│   │   ├── allometric_scaler.py  # AllometricScaler: species bridging
│   │   └── patient_agent.py  # PatientAgent + PatientPopulation
│   ├── envs/                 # [Layer 3] Gymnasium environments
│   └── wrappers/             # [Layer 3] FDA safety + Pareto reward wrappers
├── tests/
│   └── test_layer1_layer2.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## What Each Number Means (No Hardcoding)

| Parameter | Source | Meaning |
|-----------|--------|---------|
| `CL` | RDKit LogP + ADMET | Hepatic/renal clearance (L/h/kg) |
| `Vc` | ADMET Vd estimate | Central compartment volume (L/kg) |
| `ka` | Caco-2 permeability | Oral absorption rate constant (1/h) |
| `F` | Veber/Lipinski rules | Oral bioavailability fraction |
| `PPB` | Austin (2002) model | Plasma protein binding fraction |
| `EC50` | Population mean + IIV | Half-maximal effective concentration |
| `MTC` | PD model prior | Minimum toxic concentration (mg/L) |

---

## Roadmap

- **v0.1** (current): Layer 1 (RDKit/ADMET) + Layer 2 (surrogate ODE + allometric scaling)
- **v0.2**: Layer 3 — Gymnasium environments (Preclinical, Phase I, II, III)
- **v0.3**: FDA safety wrapper + multi-objective Pareto reward
- **v0.4**: Biological sub-agents (Layer 3 LangGraph orchestration)
- **v1.0**: BioGears calibration pipeline + full paper experiments

---

## Citation

If you use ClinicalTrialGym in your research:

```bibtex
@software{clinicaltrialgym2025,
  title  = {ClinicalTrialGym: RL Environment for Drug Trial Lifecycle Optimization},
  year   = {2025},
  note   = {Preprint forthcoming. Nature Digital Medicine target.}
}
```
