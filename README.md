---
title: RxGym
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# RxGym — A Gymnasium-Compliant RL Environment for Drug Trial Lifecycle Optimization

> **The first OpenEnv environment that bridges cheminformatics, physiological simulation, and reinforcement learning into a single drug-trial optimization pipeline — starting from a SMILES string, ending at an FDA-compliant Phase I recommendation.**

RxGym doesn't simulate a toy grid-world. It simulates the actual decision-making chain that a clinical pharmacologist faces: take a molecule, predict its ADMET properties, translate an animal dose to humans using allometric scaling, escalate through a Phase I trial with real PK/PD dynamics, manage drug-drug interactions in combination regimens — all under hard FDA safety constraints that can override the agent's actions.

---

## Why This Exists

Nature Digital Medicine (2025) explicitly calls for *"validated, scalable frameworks combining RL-driven protocol optimization with adaptive trial designs"* — and notes that **such frameworks do not yet exist**. RxGym is a direct answer to that call.

Clinical trials are the slowest, most expensive, and most failure-prone step in drug development. A single Phase I→III pipeline costs **$1–2 billion** and takes **10–15 years**. The core decisions — what dose to start, when to escalate, when to stop — are still made by heuristic rules from the 1990s (3+3 designs). RL can do better, but there's no realistic training environment. RxGym fills that gap.

---

## Architecture: 6 Layers, One Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  SMILES String (e.g., CC(=O)NC1=CC=C(O)C=C1)           │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Cheminformatics (RDKit + DeepChem)             │
│  ├─ 14 molecular descriptors (MolWt, LogP, TPSA, ...)   │
│  ├─ QED score, Lipinski Ro5, PAINS filter                │
│  └─ ADMET: F_oral, PPB, BBB, CYP inhibition, DILI,      │
│     hERG, ClinTox, Tox21 (GraphConv + QSAR fallback)    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 2: Physiological PK/PD Engine                     │
│  ├─ Allometric scaling (Boxenbaum 1982, 5 species)       │
│  ├─ Mahmood brain-weight correction for CNS drugs        │
│  ├─ FDA Km-factor HED computation (FDA Guidance 2005)    │
│  ├─ 2-compartment ODE (RK45) + Emax/Hill PD model       │
│  ├─ BioGears-calibrated surrogate with IIV (log-normal)  │
│  └─ Mechanistic CYP450 DDI (FDA M12 static model)       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 3: Multi-Agent Biological Simulation               │
│  ├─ PatientAgent — per-patient PK with IIV, sex, age     │
│  ├─ HepatocyteAgent — CYP450 saturation / liver stress   │
│  ├─ ImmuneAgent — IL-6 cytokine storm signal             │
│  ├─ RenalAgent — GFR decline tracking                    │
│  ├─ MeasurementAgent — NCI CTCAE lab grading             │
│  └─ DoctorAgent — LLM-powered clinical reasoning         │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 4: RL Environment + FDA Safety Wrapper             │
│  ├─ Gymnasium-compliant reset()/step()/state()           │
│  ├─ Hard-constraint safety layer (DLT thresholds,        │
│  │   forced de-escalation — overrides agent actions)     │
│  ├─ Multi-objective reward: [safety, progress,           │
│  │   stopping, organ_health] with configurable weights   │
│  └─ 3 graded tasks: dose escalation, allometric          │
│     scaling, combination DDI scheduling                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 5: FastAPI Server + Docker + HF Spaces             │
│  ├─ REST: /reset, /step, /state, /schema, /drug, /tasks  │
│  ├─ WebSocket: /ws (persistent sessions)                 │
│  └─ /drug endpoint: SMILES → full Layer 1+2 pipeline     │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 6: LLM Inference Agent                             │
│  ├─ Qwen/Qwen2.5-72B-Instruct via HuggingFace router    │
│  ├─ Drug-aware fragility profiling from ADMET             │
│  ├─ Task-specific escalation policies                    │
│  └─ Rule-based fallback on API failure                   │
└──────────────────────────────────────────────────────────┘
```

---

## Three Novel Contributions

### 1. Species-Bridging PK/PD as the Transition Model

No existing RL environment models the **translation gap** between animal and human pharmacology. RxGym does:

- The `AllometricScaler` implements power-law scaling (Boxenbaum 1982) with per-parameter exponents across **5 species** (mouse, rat, monkey, dog, human)
- CNS drugs get the **Mahmood brain-weight correction** (1996)
- FDA Km-factor HED computation follows the **2005 FDA Guidance for Industry**
- The agent must learn that animal-derived parameters are informative but systematically biased — and correct for that bias

### 2. Hard-Constraint FDA Safety Layer (Not Soft Reward)

Most RL safety work folds constraints into the reward function. That doesn't model reality — an ethics board doesn't negotiate with your reward signal. RxGym implements **irremovable safety wrappers**:

- If DLT rate exceeds 33% → **forced de-escalation** (agent's action is silently overridden)
- If ≥2/3 DLTs in a 3-patient cohort → **forced de-escalation**
- If ≥3/6 DLTs in a 6-patient expansion → **trial termination**
- Maximum dose ceiling: 50 mg/kg absolute cap
- These constraints are **not in the reward** — they are a `Gymnasium.Wrapper` that intercepts `step()` before the environment sees it

### 3. Multi-Objective Reward with Organ-Level Signals

The reward isn't a single number — it's a weighted composition of clinically meaningful objectives:

| Component | Weight | Signal Source |
|-----------|--------|---------------|
| **Safety** | 40% | DLT count, FDA stopping rules |
| **Progress** | 35% | Distance to molecule-specific target dose |
| **Stopping Quality** | 15% | Precision of RP2D identification |
| **Organ Health** | 10% | HepatocyteAgent + RenalAgent signals |

Plus a **+0.15 RP2D bonus** when the agent correctly identifies the recommended Phase II dose.

---

## Graded Tasks

### `phase_i_dosing` — Easy

Run a 3+3 dose-escalation from a molecule-derived starting dose. Find the RP2D.

The agent receives: plasma concentration, DLT counts/grades, organ signals from hepatocyte/immune/renal sub-agents, and a natural-language recommendation from the LLM DoctorAgent. Scoring is based on proximity to the drug-specific HED target (±10% = 1.0, ±25% = 0.8, ±50% = 0.5), with penalties for triggering FDA safety stops and bonuses for trial speed.

### `allometric_scaling` — Medium

Start from a rat reference dose. Predict the human-equivalent dose using allometric scaling, then refine it from observed human PK response.

The agent must propose a first-in-human dose close to the allometric HED anchor, then iteratively correct using actual patient data. Scoring rewards first-dose accuracy and penalizes overshoot — because in real Phase I trials, overshooting kills patients.

### `combo_ddi` — Hard

Manage a two-drug regimen where Drug A inhibits Drug B's CYP3A4 metabolism, increasing Drug B's exposure.

The environment implements **mechanistic CYP450 DDI** following the FDA M12 guidance:
- `CL_B_eff = CL_B × [fm_3A4 / (1 + [A]_free / Ki_A) + (1 - fm_3A4)]`
- DDI severity tiers: Weak (≥1.25× AUC), Moderate (≥2×), Strong (≥5×)
- The agent must find a safe, efficacious combination while managing temporal dose separation

---

## Molecule-Derived Parameters (Not Hardcoded)

Every parameter in RxGym traces back to the input molecule's SMILES string:

```
SMILES: CC(=O)NC1=CC=C(O)C=C1  (Acetaminophen)
  ↓ RDKit
14 molecular descriptors (MolWt=151.16, LogP=0.46, TPSA=49.33, ...)
  ↓ DeepChem GraphConv + QSAR fallback
ADMET profile (F_oral=0.87, PPB=0.25, DILI_flag=True, CYP3A4_inhib=0.12, ...)
  ↓ Allometric Scaler
PK parameters (CL=18.2 L/h, Vc=42.1 L, ka=1.2 h⁻¹, ...)
  ↓ FDA Km-factor
HED = 10 mg/kg × (6/37) = 1.62 mg/kg → starting dose = HED/10
```

This means **every task run is molecule-specific**. The same agent policy faces different dynamics for aspirin vs. diazepam vs. a novel compound. The environment isn't a fixed MDP — it's a family of MDPs parameterized by molecular structure.

---

## Biological Multi-Agent Layer

Each patient in a cohort is not just a number — they're simulated by a `PatientAgent` with:

- **Inter-individual variability**: `CL_i = CL_pop × exp(η)` (log-normal IIV, standard NONMEM model)
- **Demographic effects**: weight-scaled volumes, sex-adjusted distribution (0.85× for hydrophilic drugs in women), age-dependent CL decline (0.5%/year past 40)
- **DILI sensitivity**: if the molecule is DILI-flagged, liver stress multiplier doubles (2×)

Inside each patient, three biological sub-agents observe the PK state and emit structured signals:

| Sub-Agent | Signal | Mechanism |
|-----------|--------|-----------|
| **HepatocyteAgent** | CYP450 saturation [0,1] | Michaelis-Menten kinetics above AUC threshold |
| **ImmuneAgent** | Cytokine storm risk [0,1] | IL-6 level from Cmax (linear 5–20 mg/L) |
| **RenalAgent** | GFR fraction [1→0] | Kidney stress AUC (linear 100–400 mg/L·h) |

A **DoctorAgent** (LLM-powered, Qwen 72B) synthesizes all signals into a natural-language recommendation that becomes part of the RL agent's observation vector.

---

## Repository Layout

```
.
├── Dockerfile                          # Production image (Python 3.12-slim)
├── docker-compose.yml                  # API + inference + report profiles
├── inference.py                        # Competition entrypoint
├── openenv.yaml                        # OpenEnv task registry (3 graded tasks)
├── pyproject.toml                      # Package metadata
├── requirements.txt                    # Dependencies
│
├── rl_agent/                           # Core RL agent package
│   ├── inference.py                    # LLM-driven policy + scoring
│   ├── models.py                       # Pydantic action/observation schemas
│   ├── drug_profile_builder.py         # SMILES → full drug profile pipeline
│   └── server/
│       ├── app.py                      # FastAPI REST + WebSocket server
│       ├── agents.py                   # 6 agent types (Patient, Hepatocyte, ...)
│       ├── rl_agent_environment.py     # Gymnasium env + FDA safety wrapper
│       └── graders.py                  # Deterministic task grading
│
├── Patient_Simulation/                 # Physiological simulation engine
│   └── clinical_trial_gym/
│       ├── drug/
│       │   ├── molecule.py             # RDKit molecular descriptors + QED + PAINS
│       │   ├── admet.py                # DeepChem GraphConv + QSAR ADMET prediction
│       │   └── properties.py           # 29D feature vector + risk scoring
│       ├── pk_pd/
│       │   ├── allometric_scaler.py    # 5-species scaling + Mahmood correction
│       │   ├── surrogate_ode.py        # 2-compartment PK + Emax PD + DDI
│       │   └── patient_agent.py        # IIV population sampling + CTCAE grading
│       ├── envs/
│       │   ├── phase_i_env.py          # Phase I Gymnasium env (39D obs)
│       │   ├── allometric_env.py       # Allometric scaling env
│       │   └── combo_ddi_env.py        # Combination DDI env (84D obs)
│       └── science/
│           └── trial_priors.py         # Drug-specific Bayesian priors
│
└── scripts/
    ├── docker-entrypoint.sh
    └── validate-submission.sh
```

---

## Quick Start

### Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./rl_agent
uvicorn rl_agent.server.app:app --host 0.0.0.0 --port 8000
```

### Configure a Molecule (from any SMILES)

```bash
curl -X POST http://localhost:8000/drug \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)NC1=CC=C(O)C=C1", "name": "Acetaminophen"}'
```

Returns: HED, starting dose, full ADMET profile, safety flags, PK parameters.

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
python inference.py
```

### Docker

```bash
docker build -t rxgym .
docker run --rm -p 8000:8000 rxgym
```

---

## Technical References

- **Boxenbaum (1982)** — Interspecies scaling, allometric parameters
- **Mahmood & Balian (1996)** — Brain-weight correction for CNS drug scaling
- **FDA Guidance for Industry (2005)** — Km-factor table for HED estimation
- **FDA M12 Guidance (2024)** — Static DDI model for competitive CYP inhibition
- **FDA Project Optimus (2024)** — Dose optimization framework for oncology
- **NCI CTCAE v5.0** — Common Terminology Criteria for Adverse Events (DLT grading)
- **NONMEM Population PK** — Inter-individual variability via exponential error model
- **Nature Digital Medicine (2025)** — Call for RL-driven adaptive trial frameworks

---

## OpenEnv Compliance

| Requirement | Status |
|------------|--------|
| `openenv.yaml` with tasks + graders | 3 tasks, 3 graders |
| FastAPI server with `/reset`, `/step`, `/state`, `/schema` | All implemented |
| Typed Pydantic action/observation models | `RlAgentAction`, `RlAgentObservation` |
| Root-level `inference.py` with `[START]`/`[STEP]`/`[END]` logging | Implemented |
| Root-level `Dockerfile` | Python 3.12-slim with healthcheck |
| HF Spaces metadata + `openenv` tag | In README frontmatter |
| Scores strictly in (0, 1) | Clamped with open-interval guard |

---

*RxGym turns drug development's most expensive guesswork into a tractable optimization problem — grounded in real pharmacology, constrained by real regulations, and parameterized by real molecules.*
