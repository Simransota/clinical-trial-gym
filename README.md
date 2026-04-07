# Clinical Trial Gym

A reinforcement learning environment that simulates Phase I clinical drug trials. An AI agent learns to find the correct therapeutic dose for a new drug — high enough to be effective, low enough to be safe — by running simulated trials on virtual patients whose biology is grounded in real molecular chemistry.

The system is built in four layers. Layer 1 reads a molecule's SMILES string and computes physical properties. Layer 2 translates those properties into a pharmacokinetic model and scales it from animal to human. Layers 3 and 4 run the multi-agent simulation and orchestrate the full trial. The AI agent never sees hardcoded drug numbers — every parameter flows from the molecule itself.

---

## Table of Contents

- [The Science Behind It](#the-science-behind-it)
- [Architecture Overview](#architecture-overview)
- [Layer 1 — Molecular Properties (RDKit + DeepChem)](#layer-1--molecular-properties-rdkit--deepchem)
- [Layer 2 — PK/PD Simulation (Surrogate ODE + Allometric Scaling)](#layer-2--pkpd-simulation-surrogate-ode--allometric-scaling)
- [Layer 3 — Multi-Agent Biology Simulation](#layer-3--multi-agent-biology-simulation)
- [Layer 4 — RL Environment Orchestrator](#layer-4--rl-environment-orchestrator)
- [The Three Tasks](#the-three-tasks)
- [Reward Function](#reward-function)
- [Project Structure](#project-structure)
- [File-by-File Reference](#file-by-file-reference)
- [Setup and Running](#setup-and-running)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Key Concepts Glossary](#key-concepts-glossary)

---

## The Science Behind It

### What is a Phase I clinical trial?

When a new drug is developed, it cannot go straight to patients. The drug first goes through animal testing (preclinical), then a series of human trials. Phase I is the first human study. Its goal is not to prove the drug works — it is to find out how much is safe to give. Patients receive increasing doses until side effects appear. The dose just below the toxic threshold is called the **RP2D** (Recommended Phase II Dose).

### Why is this hard?

Finding the RP2D requires careful dose escalation across cohorts of 3–6 patients. Too slow and patients spend weeks at doses too low to matter. Too fast and patients develop severe toxicity. The FDA has strict stopping rules: if more than 1 in 3 patients in a cohort develops a serious side effect (called a **DLT**, Dose Limiting Toxicity), the trial must stop.

### What does the AI agent do?

The RL agent acts as the trial designer. At each step it decides:
1. What dose to give the next cohort
2. How many patients to enroll (3 or 6)
3. Whether to keep escalating or stop

It reads back signals from the simulated patients — plasma drug concentrations, liver stress, kidney function, immune response, lab values — and a clinical recommendation from a doctor agent. It learns over many simulated trials to escalate efficiently while keeping patients safe.

### Why simulate from a molecule?

Traditional trial simulation uses hand-picked numbers for drug behavior. This system starts from a SMILES string — the chemical structure of the drug — and computes everything from there. This means the agent encounters drugs with genuinely different properties (lipophilic vs hydrophilic, hepatotoxic vs renally cleared, CYP-inhibiting vs not) rather than variations on one made-up drug. It opens the door to a publishable research question: can an RL agent learn which molecular features predict good trial outcomes?

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RL AGENT (PPO / LLM)                        │
│              Decides: next_dose, cohort_size, escalate              │
└────────────────────────────┬────────────────────────────────────────┘
                             │ observation vector
┌────────────────────────────▼────────────────────────────────────────┐
│                    LAYER 4 — RL Environment                         │
│          RlAgentEnvironment: reset() / step() / grade()             │
│          FDA stopping rules · Reward computation · Episode history  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ cohort signals
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 3 — Multi-Agent Biology                      │
│  PatientAgent × N  ·  HepatocyteAgent  ·  ImmuneAgent              │
│  RenalAgent        ·  MeasurementAgent ·  DoctorAgent (LLM)        │
└────────────────────────────┬────────────────────────────────────────┘
                             │ PK parameters (human-scaled)
┌────────────────────────────▼────────────────────────────────────────┐
│                   LAYER 2 — PK/PD Simulation                        │
│  SurrogateODE (2-compartment) · AllometricScaler (rat → human)     │
│  PatientPopulation (IIV sampling) · Covariate effects               │
└────────────────────────────┬────────────────────────────────────────┘
                             │ ADMET predictions → PK parameters
┌────────────────────────────▼────────────────────────────────────────┐
│                  LAYER 1 — Molecular Properties                     │
│  DrugMolecule (RDKit) · ADMETPredictor (DeepChem MolNet / QSAR)   │
│  MolecularPropertyExtractor → DrugProfile (29-feature vector)      │
└─────────────────────────────────────────────────────────────────────┘
                             ▲
                       SMILES string
                   e.g. "CC(=O)Oc1ccccc1C(=O)O"
```

Data flows upward. The SMILES string enters at the bottom. By the time it reaches the RL agent, it has been transformed into a simulation of a trial cohort whose pharmacokinetics are grounded in the chemistry of that specific molecule.

---

## Layer 1 — Molecular Properties (RDKit + DeepChem)

**Location:** `Patient_Simulation/clinical_trial_gym/drug/`

Layer 1 converts a molecule's SMILES string into a structured drug profile. Nothing is hardcoded. Every property is computed or predicted from the molecular structure.

### DrugMolecule (`drug/molecule.py`)

Takes a SMILES string and computes 14 PK/PD-relevant molecular descriptors using RDKit:

| Descriptor | What it measures |
|---|---|
| MolWt | Molecular weight |
| MolLogP | Lipophilicity (fat vs water solubility) |
| NumHDonors / NumHAcceptors | Hydrogen bonding capacity |
| TPSA | Topological polar surface area (membrane permeability) |
| NumRotatableBonds | Molecular flexibility |
| RingCount | Aromatic / aliphatic ring count |
| NumAromaticRings | Aromatic ring count |
| FractionCSP3 | sp3 carbon fraction (drug-likeness) |
| BertzCT | Molecular complexity |
| HallKierAlpha | Molecular shape |
| Chi0v, Chi1v | Connectivity indices |
| LabuteASA | Accessible surface area |

Also applies Lipinski's Rule of 5 (drug-likeness filter) and PAINS detection (flags compounds that cause false positives in screens).

### ADMETPredictor (`drug/admet.py`)

Predicts ADMET properties — Absorption, Distribution, Metabolism, Excretion, Toxicity — using two backends:

**Primary: DeepChem MolNet**
Trained on real experimental datasets:
- BBBP (2039 compounds) — Blood-brain barrier penetration
- Tox21 (7831 compounds) — 12 toxicity endpoints
- ClinTox (1478 compounds) — Clinical toxicity vs FDA approval
- Lipo (4200 compounds) — Lipophilicity (logD)
- Clearance — Hepatic clearance
- HPPB — Human plasma protein binding

These datasets are downloaded once on first run and cached to disk. All subsequent runs work offline.

**Fallback: QSAR models**
When MolNet data cannot be downloaded, a set of Random Forest models trained on 35 reference drugs (Aspirin, Ibuprofen, Metformin, Warfarin, etc.) with literature ADMET values provides a working approximation. The QSAR backend is built on RDKit descriptors and StandardScaler normalization.

### MolecularPropertyExtractor (`drug/properties.py`)

Combines descriptors and ADMET predictions into a `DrugProfile`:

- **29-feature observation vector** — grouped: 14 RDKit descriptors, 10 ADMET predictions, 4 PK parameters, 1 QED drug-likeness score
- **Safety flags** — DILI risk, hERG cardiac risk, CYP inhibition profile, BBB penetration, PAINS flag
- **Composite risk score** — [0, 1] aggregated from all safety signals
- **PK/PD parameters** — ka (absorption), F (bioavailability), CL (clearance), Vc, Vp (volumes), Q (inter-compartmental flow)

The DrugProfile is the output of Layer 1 and the input to Layer 2.

---

## Layer 2 — PK/PD Simulation (Surrogate ODE + Allometric Scaling)

**Location:** `Patient_Simulation/clinical_trial_gym/pk_pd/`

Layer 2 takes the PK parameters from Layer 1 (which reflect rat/preclinical data) and turns them into a working pharmacokinetic simulation for a human population.

### AllometricScaler (`pk_pd/allometric_scaler.py`)

Drug behavior differs between species because body size affects how quickly drugs are metabolized and distributed. Allometric scaling is the validated method for translating animal PK to human PK.

The core formula (Boxenbaum 1982):

```
Parameter_human = Parameter_animal × (BW_human / BW_animal) ^ α
```

Exponents used (literature-validated):

| Parameter | Exponent | Rationale |
|---|---|---|
| CL (clearance) | 0.75 | ¾ power law — most validated in literature |
| Vc, Vp (volumes) | 1.00 | Linear with body mass |
| Q (inter-compartmental flow) | 1.00 | Scales with volumes it connects |
| ka (absorption rate) | -0.25 | Faster absorption in smaller animals |
| F, PPB, fu | 0.00 | Species-invariant |

For dose translation, the FDA Km-factor method (body surface area based) is used:

```
HED = Animal_dose × (Animal_Km / Human_Km)
```

Km factors: Mouse=3, Rat=6, Monkey=12, Dog=20, Human=37.

The scaler supports three methods: `simple` (pure power law), `correction` (Mahmood 1996 brain-weight correction for CNS drugs), and `pksim` (placeholder for PK-Sim API integration).

### SurrogateODE (`pk_pd/surrogate_ode.py`)

A two-compartment pharmacokinetic ODE model. This is the "fast surrogate" approach standard in computational pharmacology: BioGears (the full-body physiological simulator) serves as the ground truth reference; the ODE is calibrated to match BioGears across a parameter grid and runs in the RL training loop at microseconds per call.

**The ODE system:**

```
d(Depot)/dt = -ka × Depot                                         (oral absorption)
d(Cc)/dt    = ka × Depot/Vc  -  (CL/Vc) × Cc  -  (Q/Vc) × Cc  +  (Q/Vp) × Cp
d(Cp)/dt    = (Q/Vc) × Cc  -  (Q/Vp) × Cp                        (peripheral distribution)
d(AUC)/dt   = Cc                                                   (AUC integration)
```

Where:
- `Depot` = unabsorbed drug in gut (oral dosing)
- `Cc` = drug concentration in central compartment (blood + well-perfused tissues), mg/L
- `Cp` = drug concentration in peripheral compartment (muscle, fat), mg/L
- `AUC` = area under the plasma concentration curve

Uses `scipy.integrate.solve_ivp` with RK45 adaptive step-size (rtol=1e-4, atol=1e-6). Handles multiple dose events via event-driven integration: the ODE stops and restarts at each dose event.

**Pharmacodynamic model (Emax/Hill equation):**

```
Effect = Emax × Cc^n / (EC50^n + Cc^n)
```

Toxicity score combines Cmax overage (60% weight) and AUC overage (40% weight) via sigmoid transition at the minimum toxic concentration (MTC).

### PatientAgent (`pk_pd/patient_agent.py`)

Wraps the SurrogateODE with realistic patient variability:

**Covariate effects:**
- Age: hepatic clearance declines ~0.5%/year past age 40 (floor 0.5×)
- Sex: women have ~15% lower Vd for hydrophilic drugs
- Renal factor: scales renal excretion (1.0 = healthy, 0.5 = half function)
- Hepatic factor: scales hepatic clearance

**Inter-individual variability (IIV):**
Uses the NONMEM log-normal IIV convention — the industry standard for population PK modeling:

```
CL_individual = CL_population × exp(η_CL),  η_CL ~ N(0, ω_CL)
```

This means each virtual patient has their own PK parameters sampled around the population mean, giving realistic trial-to-trial variation in plasma concentration curves.

**PatientPopulation:**
Generates cohorts of N patients with randomized demographics (age 25–75, weight 50–100 kg, sex, organ health factors) for use in the RL environment.

---

## Layer 3 — Multi-Agent Biology Simulation

**Location:** `rl_agent/server/agents.py`

Six agents simulate what happens inside each patient's body during the trial. Five are rule-based (deterministic, fast, suitable for the RL training loop). One is LLM-powered (the doctor).

### Agent 1: PatientAgent

Runs the 2-compartment ODE in a simplified Euler integration for speed. Each patient has randomized demographics. Drug PK parameters flow in from Layer 1/2 via the `drug_params` dict — when supplied, the ODE uses the real molecule's properties; when omitted, conservative defaults are used.

The ODE is integrated with 20 sub-steps per hour using forward Euler. All three derivatives (`dDepot`, `dBlood`, `dTissue`) are computed from the current state before any updates are applied, ensuring numerical consistency:

```python
dDepot  = -ka × depot
dBlood  = ka × depot / Vc  -  ke × blood  -  k12 × blood  +  k21 × tissue
dTissue = k12 × blood  -  k21 × tissue

# All updates applied together
depot       += dDepot  × dt
blood_conc  += dBlood  × dt
tissue_conc += dTissue × dt
```

Organ stress accumulates via trapezoidal AUC integration. If `dili_risk` is flagged by Layer 1, liver stress accumulates at 2× rate.

### Agent 2: HepatocyteAgent

Monitors cumulative liver stress (CYP450 enzyme saturation). Returns a signal in [0, 1]:

- Below `SATURATION_THRESHOLD` (80 mg/L·h AUC): linear response
- Above threshold: sigmoid acceleration (Michaelis-Menten-like saturation kinetics)
- At `MAX_STRESS` (200 mg/L·h AUC): signal = 1.0

This maps biological reality: at low drug exposure the liver metabolizes normally, but CYP enzymes become saturated at high concentrations causing nonlinear accumulation.

### Agent 3: ImmuneAgent

Monitors inflammatory response via a cytokine proxy. Returns a signal in [0, 1] based on peak blood concentration:

- Below 5 mg/L: no reaction
- Above 20 mg/L: maximal reaction (severe inflammatory response)
- Linear interpolation between

### Agent 4: RenalAgent

Monitors kidney function (GFR fraction). Returns a signal in [1, 0] (1 = healthy, 0 = failure) based on cumulative kidney stress:

- Below `STRESS_FOR_IMPAIRMENT` (100): no impairment
- Above `STRESS_FOR_FAILURE` (400): kidney failure
- Linear decline between

### Agent 5: MeasurementAgent

Simulates clinical lab blood tests and grades side effect severity per NCI CTCAE v5.0 criteria.

**Lab tests simulated:**

| Lab value | Normal range | What it detects |
|---|---|---|
| ALT (alanine aminotransferase) | < 56 U/L | Liver damage |
| Creatinine | < 1.5 mg/dL | Kidney impairment |
| WBC (white blood cells) | ≥ 4.5 × 10⁹/L | Bone marrow suppression |

**CTCAE grading:**

| Grade | Meaning | DLT? |
|---|---|---|
| 0 | No abnormality | No |
| 1 | Mild | No |
| 2 | Moderate | No |
| 3 | Severe | **Yes** |
| 4 | Life-threatening | **Yes** |

A patient counts as a DLT if any single lab test reaches Grade 3 or higher.

Lab values are deterministic: the same physiological state always produces the same lab result (seeded from a hash of the state values). This ensures reproducible episodes.

### Agent 6: DoctorAgent (LLM)

The only LLM-powered agent. Called once per environment step. Reads all signals — Cmax, DLT count, liver saturation, kidney function, immune response, current dose — plus drug-specific context from Layer 1 (drug name, CYP inhibition profile, DILI flag) and writes one sentence: ESCALATE, HOLD, or DE-ESCALATE.

The prompt is structured to prevent injection from drug names. If the LLM call fails for any reason, a deterministic rule-based fallback kicks in:

```
dlt_count >= 2              → DE-ESCALATE
dili_risk AND cyp > 0.5     → HOLD
cyp > 0.8                   → HOLD (liver saturation critical)
gfr < 0.5                   → HOLD (kidney impairment)
otherwise                   → ESCALATE
```

---

## Layer 4 — RL Environment Orchestrator

**Location:** `rl_agent/server/rl_agent_environment.py`

The environment wires all layers together and exposes the OpenEnv interface.

### Episode flow

```
reset()
  └─ create 3 PatientAgents with drug_params from Layer 1/2
  └─ run 24-hour simulation at starting dose (1/10 of allometric HED)
  └─ return initial observation

step(action)
  ├─ apply action: clamp dose to [0.1, 50] mg/kg
  ├─ create new cohort of N patients
  ├─ run 24-hour simulation at new dose
  ├─ collect signals from all 6 agents
  ├─ capture PK time-series curves (48 timepoints per patient)
  ├─ apply FDA stopping rules
  ├─ append to episode history
  ├─ compute reward
  ├─ call DoctorAgent
  └─ return observation
```

### FDA stopping rules

The environment enforces the 3+3 dose escalation design:

- If `dlt_count / cohort_size > 0.33` → FDA hard stop, episode ends
- If `step_count >= 10` → maximum steps reached, episode ends
- If `action.escalate == False` → agent voluntarily stops

The `rp2d_dose` tracks the **highest dose tested without triggering FDA stopping**. This is the agent's running estimate of the safe maximum — the RP2D.

### Drug profile integration

When a drug profile from Layer 1/2 is provided at construction time, three things change:

1. **PK parameters**: every PatientAgent ODE uses the molecule's real ka, CL, Vc, Vp, Q, F values
2. **Starting dose**: set to 1/10 of the Layer 2 allometric HED (FDA oncology guidance)
3. **Doctor context**: DoctorAgent prompt includes drug name, CYP inhibitions, DILI flag

When no drug profile is provided, conservative defaults are used and the starting dose is 1.0 mg/kg. Nothing breaks.

### Episode data

After an episode ends, `get_episode_data()` returns everything collected:

```python
{
    "drug_name":    "Aspirin",
    "drug_params":  { ... },          # Layer 1/2 PK parameters
    "safety_flags": { ... },          # Layer 1/2 safety profile
    "start_dose":   0.162,            # mg/kg (1/10 HED)
    "history":      [ ... ],          # one dict per step
    "pk_traces":    [ ... ],          # time-series per patient per step
    "cohort_log":   [ ... ],          # demographics + outcomes per patient
    "final_score": {
        "phase_i_dosing":     0.876,
        "allometric_scaling": 0.920,
        "combo_ddi":          0.741,
    },
    "rp2d_dose":   11.5,              # mg/kg
    "steps_taken": 7,
}
```

---

## The Three Tasks

### Task 1: `phase_i_dosing` (Easy)

The core task. The agent runs a Phase I dose escalation trial and must find the RP2D. It escalates through cohorts, reads DLT signals, and stops at the right dose.

**Scored on:**
- Distance between `rp2d_dose` and `TRUE_RP2D` (12 mg/kg)
- Whether any FDA stopping rule was triggered (−0.3 penalty)
- Speed: bonus for finding RP2D in fewer steps

| Error | Score |
|---|---|
| ≤ 10% | 1.0 |
| ≤ 25% | 0.8 |
| ≤ 50% | 0.5 |
| ≤ 75% | 0.3 |
| > 75% | 0.1 |

### Task 2: `allometric_scaling` (Medium)

Given a rat study dose, the agent must propose the correct human equivalent dose. Tests whether the agent understands species scaling.

**Scored on:** Distance between agent's first proposed dose and the allometric HED from Layer 2.

### Task 3: `combo_ddi` (Hard)

The agent must schedule two drugs simultaneously while managing CYP450 enzyme competition (drug-drug interaction). One drug inhibiting CYP3A4 affects the metabolism of another drug cleared by the same enzyme.

**Scored on:** Composite of efficacy (dose achieved relative to RP2D), safety (DLT rate), and DDI management (total DLT burden).

---

## Reward Function

Four components computed at every step, combined into a single reward in [0, 1]:

```
reward = 0.40 × safety + 0.35 × progress + 0.15 × stopping + 0.10 × organ
```

### Safety (40%)

| DLT rate | Reward |
|---|---|
| FDA stop triggered | 0.0 |
| 0% DLTs | 1.0 |
| ≤ 16.7% (0–1 of 6) | 0.8 |
| ≤ 33% (1 of 3) | 0.4 |
| > 33% | 0.0 |

### Progress (35%)

Rewards escalating when safe, de-escalating when DLTs appear:

- `dlt_rate == 0` and organs healthy and dose increased: `min(1.0, dose / 50.0)`
- DLTs present and dose held or reduced: 0.7
- DLTs present but dose increased anyway: 0.1

### Stopping (15%)

Rewards stopping at the right moment:

- Agent stops (`escalate=False`) with DLTs present at dose > 3 mg/kg: 1.0 (correct stopping point)
- Agent stops with no DLTs but at dose > 8 mg/kg: 0.7 (reasonable but could go higher)
- FDA forced stop: 0.0 (agent failed to stop in time)

### Organ health (10%)

```
organ = renal_signal × 0.5 + (1 − hepatocyte_signal) × 0.5
```

### RP2D transition bonus

When the episode history shows the first DLT occurrence (previous step had zero DLTs, current step has DLTs > 0), a +0.15 bonus is added. This rewards the agent for reaching the true transition point rather than stopping too early.

---

## Project Structure

```
meta/
├── requirements.txt                         # Python dependencies
│
├── Patient_Simulation/                      # Layer 1 + Layer 2
│   ├── main.py                              # End-to-end demo pipeline
│   ├── setup_models.py                      # DeepChem MolNet model training
│   └── clinical_trial_gym/
│       ├── drug/
│       │   ├── molecule.py                  # RDKit descriptor extraction
│       │   ├── admet.py                     # ADMET prediction (DeepChem + QSAR)
│       │   └── properties.py               # Layer 1 → 2 interface (DrugProfile)
│       ├── pk_pd/
│       │   ├── allometric_scaler.py         # Species-bridging dose scaling
│       │   ├── patient_agent.py             # Patient population simulator
│       │   └── surrogate_ode.py            # 2-compartment PK/PD ODE
│       └── tests/
│           └── test_layer1_layer2.py        # 51 tests covering Layers 1 and 2
│
└── rl_agent/                                # Layer 3 + Layer 4
    ├── models.py                            # Action / Observation data contracts
    ├── client.py                            # Python SDK for the environment
    ├── inference.py                         # LLM agent (what judges run to score)
    ├── openenv.yaml                         # OpenEnv task manifest
    └── server/
        ├── app.py                           # FastAPI HTTP + WebSocket server
        ├── agents.py                        # 6-agent biology simulation engine
        └── rl_agent_environment.py          # Main environment: reset / step / grade
```

---

## File-by-File Reference

### `Patient_Simulation/clinical_trial_gym/drug/molecule.py`

Parses a SMILES string into a `DrugMolecule` object. Computes 14 RDKit descriptors, applies Lipinski's Rule of 5, detects PAINS patterns. Generates a SHA-256 `mol_id` from the canonical SMILES so the same molecule always gets the same ID.

### `Patient_Simulation/clinical_trial_gym/drug/admet.py`

`ADMETPredictor` class. Tries DeepChem MolNet models first, falls back to QSAR Random Forest. Contains NumPy 2.x compatibility patches applied at import time (fixes ragged array handling and AxisError in `remove_missing_entries` for 1-D arrays — necessary because DeepChem 2.5 was written before NumPy 2).

The `to_pkpd_params()` method converts predicted ADMET values into PK parameters using empirical formulas derived from the relationships between logD, bioavailability, plasma protein binding, and clearance.

### `Patient_Simulation/clinical_trial_gym/drug/properties.py`

`MolecularPropertyExtractor` combines `DrugMolecule` and `ADMETPredictor` into a `DrugProfile`. Builds the 29-feature observation vector, aggregates safety flags, and computes the composite risk score.

### `Patient_Simulation/clinical_trial_gym/pk_pd/allometric_scaler.py`

`AllometricScaler` with a species database (mouse, rat, monkey, dog, human) and allometric exponents from Boxenbaum 1982 and Mordenti 1986. `scale(params)` returns human-scaled PK parameters. `scale_dose(dose_mgkg)` returns the Human Equivalent Dose using the FDA Km-factor method.

### `Patient_Simulation/clinical_trial_gym/pk_pd/surrogate_ode.py`

`SurrogateODE` implementing the 2-compartment PK + Emax PD model. `administer_dose()` schedules dose events. `simulate(duration_h)` runs the ODE via `scipy.integrate.solve_ivp`. `get_summary_stats()` returns AUC, Cmax, Tmax, T>MTC fraction, mean effect, peak toxicity.

### `Patient_Simulation/clinical_trial_gym/pk_pd/patient_agent.py`

`PatientAgent` wraps `SurrogateODE` with covariate effects (age, sex, renal, hepatic) and log-normal IIV sampling. `PatientPopulation` generates a cohort of N patients with randomized demographics from a given drug profile.

### `Patient_Simulation/setup_models.py`

Trains the DeepChem MolNet models for ADMET prediction. Run this once to pre-download and train on BBBP, Tox21, ClinTox, Lipo, Clearance, and HPPB datasets. Includes all NumPy 2 compatibility patches and Keras 3 legacy shims. After this runs, models are cached to disk and ADMET prediction works offline.

### `Patient_Simulation/main.py`

End-to-end demonstration. Takes Aspirin's SMILES, runs Layer 1 (ADMET), Layer 2 (allometric scaling, rat → human), and Layer 3 (cohort simulation). Prints the full pipeline output including predicted PK parameters, HED, and simulated plasma concentrations.

### `rl_agent/models.py`

Pydantic data contracts for the OpenEnv interface.

```python
class RlAgentAction(Action):
    next_dose: float       # mg/kg body weight — what dose to give next cohort
    cohort_size: int       # 3 or 6 patients
    escalate: bool         # False = agent thinks RP2D is found, stop escalating

class RlAgentObservation(Observation):
    phase: str                    # "phase_i"
    cohort_size: int              # patients in current cohort
    dose_level: float             # current dose mg/kg
    plasma_conc: float            # mean peak blood concentration mg/L
    dlt_count: int                # patients with Grade 3+ toxicity
    dlt_grade: List[int]          # per-patient grade (0–4)
    hepatocyte_signal: float      # liver stress [0=fine, 1=overwhelmed]
    immune_signal: float          # immune reaction [0=calm, 1=severe]
    renal_signal: float           # kidney function [1=healthy, 0=failed]
    doctor_recommendation: str    # ESCALATE / HOLD / DE-ESCALATE + reason
```

### `rl_agent/server/agents.py`

All 6 biological agents. `PatientAgent.advance()` runs the forward Euler ODE. `HepatocyteAgent.observe()`, `ImmuneAgent.observe()`, `RenalAgent.observe()` are pure functions taking a state dict and returning a float. `MeasurementAgent.grade_dlt()` returns the CTCAE grade. `DoctorAgent.recommend()` calls the LLM.

### `rl_agent/server/rl_agent_environment.py`

`RlAgentEnvironment` — the main environment class. `reset()` and `step()` implement the OpenEnv interface. Three graders (`_grade_phase_i`, `_grade_allometric`, `_grade_combo_ddi`) score the episode. `_compute_reward()` computes the 4-component reward. `_get_pk_trace()` re-simulates each patient at fine time resolution (48 points over 24h) for the episode data record. `get_episode_data()` returns the full episode record for downstream visualization or analysis.

### `rl_agent/server/app.py`

FastAPI server. Wraps the environment via `openenv.core.env_server.create_app()`. Adds a custom `/episode_data` GET endpoint that returns the full episode record after the trial ends.

### `rl_agent/inference.py`

The entry point that judges run to score a submission. Connects to the environment server, runs the LLM agent through a full episode, outputs the mandatory log format:

```
[START] task=phase_i_dosing env=rxgym model=Qwen2.5-72B-Instruct
[STEP]  step=1 action={"next_dose":2.0,"cohort_size":3,"escalate":true} reward=0.67 done=false error=null
...
[END]   success=true steps=7 score=0.876 rewards=0.67,0.70,0.76,0.87,0.99,0.81,0.63
```

### `rl_agent/client.py`

Python SDK. Lets other scripts connect to the environment cleanly:

```python
from client import ClinicalTrialClient
client = ClinicalTrialClient("http://localhost:8000")
obs = client.reset()
obs = client.step(next_dose=2.0, cohort_size=3, escalate=True)
```

---

## Setup and Running

### Prerequisites

- Python 3.10+
- Internet access on first run (to download MolNet datasets — subsequent runs work offline)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train ADMET models (first time only)

This downloads MolNet datasets (~100MB total) and trains DeepChem models. Run once; models cache to disk.

```bash
cd Patient_Simulation
python setup_models.py
```

If you cannot run this (no internet, or GPU memory constraints), the system automatically falls back to the QSAR backend. No action needed.

### Run the end-to-end demo (Layer 1 + 2)

```bash
cd Patient_Simulation
python main.py
```

This takes Aspirin's SMILES, predicts ADMET properties, scales rat → human, simulates a 6-patient cohort, and prints the full pipeline output.

### Start the RL environment server (Layers 3 + 4)

```bash
cd rl_agent
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the LLM agent

In a separate terminal:

```bash
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:8000

cd rl_agent
python inference.py
```

### Run Layer 1+2 tests

```bash
cd Patient_Simulation
python -m pytest clinical_trial_gym/tests/test_layer1_layer2.py -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace access token (for LLM router) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | LLM model for DoctorAgent and inference.py |
| `API_KEY` | `dummy` | API key (if using a different provider) |
| `ENV_URL` | `http://localhost:8000` | Environment server URL for inference.py |

---

## API Reference

### `POST /reset`

Start a new trial episode. Returns the initial `RlAgentObservation`.

```json
{}
```

### `POST /step`

Send an action, receive an observation.

```json
{
  "next_dose": 4.0,
  "cohort_size": 3,
  "escalate": true
}
```

Returns `RlAgentObservation`:

```json
{
  "phase": "phase_i",
  "cohort_size": 3,
  "dose_level": 4.0,
  "plasma_conc": 3.24,
  "dlt_count": 0,
  "dlt_grade": [0, 0, 1],
  "hepatocyte_signal": 0.21,
  "immune_signal": 0.05,
  "renal_signal": 0.98,
  "doctor_recommendation": "ESCALATE: No DLTs observed, liver and kidney signals within safe range.",
  "done": false,
  "reward": 0.813
}
```

### `GET /state`

Returns current episode state (episode ID, step count).

### `GET /episode_data`

Returns the full episode record (history, PK traces, scores). Call after `done=true`.

### `WebSocket /ws`

Persistent connection. Send JSON action objects, receive JSON observations. Useful for high-frequency RL training loops.

---

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **SMILES** | Simplified Molecular Input Line Entry System — a text notation for molecular structure, e.g. `CC(=O)Oc1ccccc1C(=O)O` for Aspirin |
| **ADMET** | Absorption, Distribution, Metabolism, Excretion, Toxicity — the five properties that determine how a drug behaves in the body |
| **PK/PD** | Pharmacokinetics (what the body does to the drug) / Pharmacodynamics (what the drug does to the body) |
| **Phase I trial** | First-in-human clinical trial; goal is safety and dose finding, not efficacy |
| **RP2D** | Recommended Phase II Dose — the highest dose that is safe enough to use in efficacy trials |
| **MTD** | Maximum Tolerated Dose — the highest dose where ≤ 1/3 of patients have DLTs |
| **DLT** | Dose Limiting Toxicity — a serious side effect (Grade 3 or higher) that limits dose escalation |
| **CTCAE** | Common Terminology Criteria for Adverse Events (NCI) — the grading scale for clinical trial side effects |
| **3+3 design** | The standard Phase I escalation rule: enroll 3 patients; if ≤ 1 DLT, escalate; if ≥ 2 DLTs, stop |
| **Allometric scaling** | Method for translating drug doses and PK parameters between species based on body weight |
| **HED** | Human Equivalent Dose — animal dose converted to human terms via body surface area |
| **CYP450** | Cytochrome P450 — liver enzymes responsible for metabolizing most drugs |
| **DDI** | Drug-Drug Interaction — when one drug affects the metabolism of another |
| **AUC** | Area Under the Curve — total drug exposure over time; used as a proxy for toxicity |
| **Cmax** | Maximum plasma concentration — peak drug level in blood after a dose |
| **Tmax** | Time to maximum concentration |
| **IIV** | Inter-Individual Variability — the natural variation in PK between different patients |
| **NONMEM** | Nonlinear Mixed Effects Modeling — the industry standard for population PK analysis |
| **Two-compartment model** | PK model with central (blood) and peripheral (tissue) compartments connected by drug transfer |
| **Surrogate model** | A fast approximation (ODE) calibrated to match a slower, higher-fidelity simulator (BioGears) |
| **PAINS** | Pan-Assay Interference Compounds — molecules that give false positives in drug screens |
| **QED** | Quantitative Estimate of Drug-likeness — a [0,1] score of how drug-like a molecule is |
| **BBB** | Blood-brain barrier — if a drug penetrates this, it affects the CNS |
| **DILI** | Drug-Induced Liver Injury — liver damage caused by a drug |
| **hERG** | A cardiac ion channel; drugs that block hERG cause dangerous heart arrhythmias |
