# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Rl Agent Environment.

This module creates an HTTP server that exposes the RlAgentEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from dotenv import load_dotenv
load_dotenv()
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import RlAgentAction, RlAgentObservation
    from .rl_agent_environment import RlAgentEnvironment
except ModuleNotFoundError:
    from models import RlAgentAction, RlAgentObservation
    from server.rl_agent_environment import RlAgentEnvironment


# Shared drug profile used by the environment factory.
# OpenEnv creates a fresh environment instance per HTTP request.
_CURRENT_DRUG_PROFILE = None


def _env_factory():
    return RlAgentEnvironment(drug_profile=_CURRENT_DRUG_PROFILE)


# Create the app with web interface and README integration
app = create_app(
    _env_factory,
    RlAgentAction,
    RlAgentObservation,
    env_name="rl_agent",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# app.py — add this after create_app(...)

class DrugRequest(BaseModel):
    smiles: str
    name: Optional[str] = "investigational compound"
    source_species: Optional[str] = "rat"
    animal_dose_mgkg: Optional[float] = 8.0


@app.post("/drug")
def configure_drug(req: DrugRequest):
    """
    Configure the environment with a real molecule from Layer 1+2.

    Runs DrugProfileBuilder (RDKit → DeepChem ADMET → AllometricScaler) on the
    provided SMILES string and applies the resulting drug profile to the
    environment. The next /reset call will use this drug's PK parameters and
    start at 1/10 of the computed HED.

    Body:
        smiles           : SMILES string, e.g. "CC(=O)Oc1ccccc1C(=O)O"
        name             : drug name for logging (default "investigational compound")
        source_species   : preclinical species (default "rat")
        animal_dose_mgkg : reference dose for HED computation (default 8.0)

    Returns:
        status, drug name, HED, starting dose, and ADMET summary.
    """
    import sys, os
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    try:
        from drug_profile_builder import DrugProfileBuilder
    except ImportError:
        sys.path.insert(0, os.path.join(_repo, "rl_agent"))
        from drug_profile_builder import DrugProfileBuilder

    try:
        builder = DrugProfileBuilder(
            smiles=req.smiles,
            name=req.name,
            source_species=req.source_species,
            animal_dose_mgkg=req.animal_dose_mgkg,
        )
        profile = builder.build()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "status": "configuration_failed",
                "reason": str(exc),
                "drug": req.name,
                "smiles": req.smiles,
            },
        ) from exc
    global _CURRENT_DRUG_PROFILE
    _CURRENT_DRUG_PROFILE = profile
    start_dose = round(max(0.1, profile["human_equivalent_dose"] / 10.0), 3)

    return {
        "status":       "configured",
        "drug":         profile["name"],
        "smiles":       profile["smiles"],
        "hed_mgkg":     profile["human_equivalent_dose"],
        "start_dose":   start_dose,
        "drug_params":  profile["drug_params"],
        "admet_summary": profile["admet_summary"],
    }


@app.get("/episode_data")
def get_episode_data():
    """Episode data is only available in persistent WebSocket sessions."""
    return {
        "error": "episode_data is unavailable in stateless HTTP mode. Use /ws for persistent sessions."
    }

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m rl_agent.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn rl_agent.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
