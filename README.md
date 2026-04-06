
# Clinical Trial Gym

An AI training environment that simulates Phase I clinical drug trials. An AI agent learns to find the right dose for a new drug — high enough to be effective, low enough to be safe — by running simulated trials on virtual patients.

---

## Project structure

```
rl_agent/
├── models.py                    # Data contracts (Action / Observation)
├── client.py                    # Python SDK for the environment
├── inference.py                 # LLM agent — what judges run to score
├── openenv.yaml                 # OpenEnv config (3 tasks)
├── validate.sh                  # Pre-submission checker
├── .env                         # Secrets (never pushed to GitHub)
└── server/
    ├── app.py                   # FastAPI HTTP / WebSocket server
    ├── agents.py                # 6-agent biology simulation engine
    ├── rl_agent_environment.py  # Main environment: reset / step / state
    ├── requirements.txt         # Server dependencies
    └── Dockerfile               # Container config
```

---

## File by file

### `models.py` — data contracts

Defines exactly what information flows between the AI agent and the environment at every step.

- `RlAgentAction` — what the agent sends: next dose, cohort size, whether to keep escalating
- `RlAgentObservation` — what the environment sends back: plasma concentration, DLT count, organ signals, doctor recommendation
- Both inherit from OpenEnv base classes so the hackathon validator can read them

---

### `server/agents.py` — biology simulation engine (6 agents)

The scientific core. Six agents simulate what happens inside a patient's body:

- **PatientAgent** — runs a 2-compartment ODE that tracks drug concentration in blood and tissue over 24 hours. Each patient has random weight, age, and organ health
- **HepatocyteAgent** — watches liver stress (CYP450 saturation). Returns 0 = healthy, 1 = overwhelmed. Rule-based, no LLM
- **ImmuneAgent** — watches immune/inflammatory response via cytokine proxy. Rule-based
- **RenalAgent** — watches kidney function (GFR). Rule-based
- **MeasurementAgent** — simulates real blood test results (ALT, creatinine, WBC) and grades side effect severity using NCI CTCAE criteria (Grade 0–4). Grade 3+ counts as a DLT
- **DoctorAgent** — the only LLM agent. Reads all signals once per step and writes one sentence: ESCALATE / HOLD / DE-ESCALATE. Falls back to rules if LLM fails

---

### `server/rl_agent_environment.py` — the game engine

Orchestrates the whole trial. Exposes three methods required by OpenEnv:

- `reset()` — starts a new trial at 1 mg/kg, creates first cohort of 3 patients
- `step(action)` — takes an agent action, runs the body simulation, reads all 6 agents, applies FDA stopping rules, computes reward, returns observation
- `state` — returns current episode state

Contains three graders (one per task), each returning a score 0.0–1.0:

- `_grade_phase_i()` — how close did the agent get to the true RP2D (12 mg/kg)?
- `_grade_allometric()` — was the proposed human dose correct from rodent data?
- `_grade_combo_ddi()` — did the agent balance two-drug efficacy vs drug-drug interaction?

Reward function — 4 components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Safety | 40% | DLT rate — penalizes side effects, hard zero if FDA stops trial |
| Escalation progress | 35% | Rewards escalating when safe, de-escalating when DLTs appear |
| Stopping behavior | 15% | Rewards stopping at the right moment |
| Organ health | 10% | Kidney + liver signals |

Plus a +0.15 bonus when the agent finds the exact DLT transition point (RP2D).

---

### `server/app.py` — the HTTP server

Wraps the environment in a FastAPI web server. Endpoints:

```
POST /reset    Start a new trial episode
POST /step     Send an action, receive an observation
GET  /state    Get current episode info
WS   /ws       WebSocket for persistent connections
```

Loads `.env` credentials on startup via python-dotenv.

---

### `inference.py` — the LLM agent

What the hackathon judges run to score the submission. An LLM (Qwen via HuggingFace router) reads the observation at each step and decides the next dose. Outputs the mandatory log format:

```
[START] task=phase_i_dosing env=gym_env model=Qwen2.5-72B
[STEP]  step=1 action={"next_dose":2.0,...} reward=0.67 done=false error=null
[END]   success=true steps=7 score=0.876 rewards=0.67,0.70,0.76,0.87,0.99,0.81,0.63
```

Uses OpenAI client pointed at HuggingFace router — no OpenAI account needed.

---

### `client.py` — Python SDK

Lets other Python code connect to the environment cleanly via WebSocket or HTTP. Handles serialization of `RlAgentAction` and deserialization of `RlAgentObservation`.

---

### `openenv.yaml` — hackathon config

Declares the three tasks and difficulty levels. First thing the hackathon validator reads.

---

### `validate.sh` — pre-submission checker

Runs all checks automatically: required files exist, 3 tasks defined, Python imports work, server starts and responds correctly, Docker builds, inference.py log format is correct.

---

## The three tasks

| Task | Difficulty | What the agent must do |
|------|------------|------------------------|
| `phase_i_dosing` | Easy | Find the right dose (RP2D) by escalating through patient cohorts while respecting FDA 3+3 stopping rules |
| `allometric_scaling` | Medium | Given rat study results, calculate the correct human starting dose using body weight scaling |
| `combo_ddi` | Hard | Schedule two drugs simultaneously while minimizing CYP450 enzyme competition between them |

---

## Quick start

```bash
# Install dependencies
uv sync

# Start the environment server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run the LLM agent (in a new terminal)
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
python inference.py
```

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace access token |
| `API_BASE_URL` | LLM API endpoint (default: HuggingFace router) |
| `MODEL_NAME` | Model identifier (default: Qwen2.5-7B-Instruct) |
| `ENV_URL` | Environment server URL (default: http://localhost:8000) |


---

## Run validation before submitting

```bash
chmod +x validate.sh
./validate.sh
```