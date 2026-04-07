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

# RxGym

RxGym is an OpenEnv environment for real-world clinical pharmacology decision making. It simulates three tasks that matter in actual drug development: Phase I dose escalation, animal-to-human dose translation, and combination scheduling under drug-drug interaction risk.

## Competition Fit

- Real-world task: dose finding and translational pharmacology are genuine human workflows.
- OpenEnv-compliant server: typed models, FastAPI app, `reset()/step()/state()`, and `openenv.yaml`.
- Three graded tasks: `phase_i_dosing`, `allometric_scaling`, `combo_ddi`.
- Dense reward shaping: partial progress, safety, refinement, and stopping are all rewarded.
- Baseline script: root-level `inference.py` with required `[START]`, `[STEP]`, `[END]` logging.
- Docker + HF Spaces ready: root `Dockerfile`, README front matter, `openenv` tag.

## Tasks

### `phase_i_dosing` — Easy

The agent runs a Phase I dose-escalation process and must stop near a molecule-specific RP2D.

Success depends on:
- escalating efficiently
- avoiding unsafe DLT rates
- using cohort expansion meaningfully
- refining or stopping near the correct target band

### `allometric_scaling` — Medium

The agent starts from a rat reference dose and must identify the correct human-equivalent dose, then refine it using observed human response.

Success depends on:
- proposing a human-equivalent dose close to the HED anchor
- avoiding overshoot
- stabilizing after informative response

### `combo_ddi` — Hard

The agent manages a two-drug regimen where one compound can alter the other’s exposure through CYP-mediated interaction.

Success depends on:
- advancing toward an effective combination target
- controlling safety and organ stress
- avoiding interaction-driven overexposure

## Action Space

The environment uses a typed action model:

- `next_dose: float`
- `cohort_size: int`
- `escalate: bool`

Defined in [rl_agent/models.py](/Users/simransota/meta/rl_agent/models.py).

## Observation Space

The environment returns a typed observation containing:

- `phase`
- `cohort_size`
- `dose_level`
- `plasma_conc`
- `dlt_count`
- `dlt_grade`
- `hepatocyte_signal`
- `immune_signal`
- `renal_signal`
- `doctor_recommendation`

Defined in [rl_agent/models.py](/Users/simransota/meta/rl_agent/models.py).

## Reward Design

Rewards are shaped over the whole trajectory and normalized to `[0, 1]`.

Components include:
- safety under DLT thresholds
- progress toward a molecule-derived task target
- stopping or refining near the correct boundary
- organ-health terms from hepatocyte and renal signals

This avoids sparse binary-only feedback and gives agents useful intermediate signal.

## Grading

Each task has a deterministic programmatic scoring path:

- `phase_i_dosing`: closeness to the drug-specific phase-I target, safety, and efficiency
- `allometric_scaling`: closeness to HED and safe refinement
- `combo_ddi`: closeness to the combination target band plus safety consistency

Scores are reported in `[0, 1]`.

## Repository Layout

```text
.
├── Dockerfile
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── scripts/
│   └── validate-submission.sh
├── Patient_Simulation/
│   └── clinical_trial_gym/
└── rl_agent/
    ├── inference.py
    ├── models.py
    ├── drug_profile_builder.py
    └── server/
        ├── app.py
        ├── agents.py
        ├── rl_agent_environment.py
        └── Dockerfile
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./rl_agent
uvicorn rl_agent.server.app:app --host 0.0.0.0 --port 8000
```

## Baseline Inference

The competition-required entrypoint is [inference.py](/Users/simransota/meta/inference.py), which delegates to [rl_agent/inference.py](/Users/simransota/meta/rl_agent/inference.py).

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Typical run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
export MEDICINE_NAME="Acetaminophen"
export MEDICINE_SMILES="CC(=O)NC1=CC=C(O)C=C1"
export SOURCE_SPECIES="rat"
export ANIMAL_DOSE_MGKG="10.0"
export TASK_NAME="phase_i_dosing"
python inference.py
```

The script emits the required stdout format:
- `[START]`
- `[STEP]`
- `[END]`

## Docker

The root image packages the whole project:
- OpenEnv/FastAPI server
- baseline `inference.py`
- researcher report generation under `analysis/`

Build the image:

```bash
docker build -t rxgym .
```

Run the API server:

```bash
docker run --rm -p 8000:8000 rxgym
```

Run the API server with Docker Compose:

```bash
docker compose up --build
```

Run baseline inference against the containerized API:

```bash
export HF_TOKEN="..."
export MEDICINE_NAME="Acetaminophen"
export MEDICINE_SMILES="CC(=O)NC1=CC=C(O)C=C1"
export SOURCE_SPECIES="rat"
export ANIMAL_DOSE_MGKG="10.0"
export TASK_NAME="phase_i_dosing"
docker compose --profile inference run --rm rxgym-inference
```

Generate a drug report inside the same image:

```bash
export REPORT_DRUG_NAME="Diazepam"
export REPORT_DRUG_SMILES="CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"
export REPORT_ANIMAL_DOSE_MGKG="2.0"
docker compose --profile report run --rm rxgym-report
```

The report artifacts will be written to `analysis/reports/<drug>/` on your host machine.

## Hugging Face Spaces

This repository is configured for Docker-based HF Spaces deployment and includes the required `openenv` tag in the README metadata.

## Pre-Submission Validation

Use the included validation helper:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space.hf.space .
```

## Submission Notes

- The root of the repository now contains the required `inference.py`, `openenv.yaml`, and `Dockerfile`.
- The baseline policy defaults to deterministic logic and only uses the OpenAI client when explicitly enabled.
- `/drug` no longer hard-fails when DeepChem is unavailable; it can use the trained QSAR backend.
- Task scoring and controller targets are molecule-derived rather than globally hard-coded.
