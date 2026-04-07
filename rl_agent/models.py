import os
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional


_DEFAULT_DRUG_NAME = os.getenv("DEFAULT_DRUG_NAME", "")
_DEFAULT_DRUG_SMILES = os.getenv("DEFAULT_DRUG_SMILES", "")
_DEFAULT_SOURCE_SPECIES = os.getenv("DEFAULT_SOURCE_SPECIES", "rat")
_DEFAULT_ANIMAL_DOSE_MGKG = float(os.getenv("DEFAULT_ANIMAL_DOSE_MGKG", "8.0"))


class RlAgentAction(Action):
    """
    Action for the Clinical Trial environment.
    The AI agent fills this in at every step.
    """
    next_dose: float = Field(
        ...,
        description="Next dose to give patients, in mg per kg of body weight"
    )
    cohort_size: int = Field(
        default=3,
        description="How many patients in the next group (3 or 6)"
    )
    escalate: bool = Field(
        default=True,
        description="True = increase dose next step, False = hold or stop"
    )
    drug_name: Optional[str] = Field(
        default=_DEFAULT_DRUG_NAME or None,
        description="Optional drug label for UI-driven configuration. Leave unchanged to keep the current drug."
    )
    drug_smiles: Optional[str] = Field(
        default=_DEFAULT_DRUG_SMILES or None,
        description="Optional SMILES string for configuring the molecule directly from the OpenEnv UI."
    )
    source_species: Optional[str] = Field(
        default=_DEFAULT_SOURCE_SPECIES,
        description="Optional preclinical source species used for allometric scaling."
    )
    animal_dose_mgkg: Optional[float] = Field(
        default=_DEFAULT_ANIMAL_DOSE_MGKG,
        description="Optional preclinical dose in mg/kg used to derive the human equivalent dose."
    )


class RlAgentObservation(Observation):
    """
    Observation from the Clinical Trial environment.
    The AI agent reads this at every step to decide what to do next.
    """
    phase: str = Field(
        default="phase_i",
        description="Which trial phase we are in"
    )
    cohort_size: int = Field(
        default=3,
        description="Number of patients in current group"
    )
    dose_level: float = Field(
        default=0.0,
        description="Current dose in mg/kg"
    )
    plasma_conc: float = Field(
        default=0.0,
        description="Peak drug concentration in blood (mg/L)"
    )
    dlt_count: int = Field(
        default=0,
        description="Number of patients with serious side effects (Grade 3+)"
    )
    dlt_grade: List[int] = Field(
        default_factory=list,
        description="Side effect severity grade per patient (0=none, 4=life-threatening)"
    )
    hepatocyte_signal: float = Field(
        default=0.0,
        description="Liver stress level from 0 (fine) to 1 (overwhelmed)"
    )
    immune_signal: float = Field(
        default=0.0,
        description="Immune/inflammatory reaction from 0 (calm) to 1 (severe)"
    )
    renal_signal: float = Field(
        default=1.0,
        description="Kidney function from 1 (healthy) to 0 (failed)"
    )
    doctor_recommendation: str = Field(
        default="",
        description="One-sentence clinical recommendation from doctor agent"
    )
