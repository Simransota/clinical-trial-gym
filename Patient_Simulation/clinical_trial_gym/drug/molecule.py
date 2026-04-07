"""
DrugMolecule: Core representation of a drug candidate.

The pipeline entrance is a SMILES string (not hardcoded numbers).
RDKit computes 2D/3D descriptors; these feed into ADMET prediction and
ultimately initialize PK/PD model parameters.

Example
-------
>>> mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
>>> mol.molecular_weight
180.16
>>> mol.descriptors["MolLogP"]
1.31
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


# ---------------------------------------------------------------------------
# Descriptor names we extract (subset of RDKit's ~200 descriptors).
# These are the ones with established PK/PD relevance.
# ---------------------------------------------------------------------------
PKPD_RELEVANT_DESCRIPTORS = [
    "MolWt",           # Molecular weight — drives allometric scaling
    "MolLogP",         # Lipophilicity — predicts oral absorption / BBB
    "NumHDonors",      # H-bond donors — Lipinski Rule of 5
    "NumHAcceptors",   # H-bond acceptors — Lipinski Rule of 5
    "TPSA",            # Topological polar surface area — membrane permeability
    "NumRotatableBonds",  # Flexibility — relates to oral bioavailability
    "NumAromaticRings",
    "NumAliphaticRings",
    "RingCount",
    "FractionCSP3",    # sp3 carbon fraction — solubility proxy
    "HeavyAtomCount",
    "NumHeteroatoms",
    "MaxPartialCharge",
    "MinPartialCharge",
]


@dataclass
class DrugMolecule:
    """
    Immutable representation of a drug candidate, initialized from SMILES.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string (e.g. 'CC(=O)Oc1ccccc1C(=O)O').
    name : str, optional
        Human-readable drug name (e.g. 'Aspirin').
    dose_unit : str
        Dose unit used throughout the trial pipeline. Default 'mg/kg'.

    Attributes
    ----------
    descriptors : Dict[str, float]
        RDKit-computed molecular descriptors with PK/PD relevance.
    qed_score : float
        Quantitative Estimate of Drug-likeness in [0, 1].
    lipinski_pass : bool
        True if molecule passes all four Lipinski Rules of Five.
    feature_vector : np.ndarray
        Flat numpy array of descriptor values for use as RL observation.
    mol_id : str
        Deterministic SHA-256 hash of the canonical SMILES.
    """

    smiles: str
    name: str = "unnamed_drug"
    dose_unit: str = "mg/kg"

    # Filled during __post_init__
    _rdkit_mol: object = field(default=None, init=False, repr=False)
    descriptors: Dict[str, float] = field(default_factory=dict, init=False)
    qed_score: float = field(default=0.0, init=False)
    lipinski_pass: bool = field(default=False, init=False)
    pains_flag: bool = field(default=False, init=False)
    mol_id: str = field(default="", init=False)

    def __post_init__(self):
        self._parse_and_validate()
        self._compute_descriptors()
        self._compute_qed()
        self._check_lipinski()
        self._check_pains()
        self._compute_mol_id()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _parse_and_validate(self):
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(
                f"Invalid SMILES string: '{self.smiles}'. "
                "RDKit could not parse this molecule."
            )
        # Canonicalize
        self.smiles = Chem.MolToSmiles(mol, canonical=True)
        self._rdkit_mol = mol

    def _compute_descriptors(self):
        mol = self._rdkit_mol
        desc = {}
        for name in PKPD_RELEVANT_DESCRIPTORS:
            fn = getattr(Descriptors, name, None)
            if fn is not None:
                try:
                    val = fn(mol)
                    desc[name] = float(val) if val is not None else np.nan
                except Exception:
                    desc[name] = np.nan
        self.descriptors = desc

    def _compute_qed(self):
        self.qed_score = float(QED.qed(self._rdkit_mol))

    def _check_lipinski(self):
        d = self.descriptors
        self.lipinski_pass = (
            d.get("MolWt", 9999) <= 500
            and d.get("MolLogP", 9999) <= 5
            and d.get("NumHDonors", 9999) <= 5
            and d.get("NumHAcceptors", 9999) <= 10
        )

    def _check_pains(self):
        """Flag PAINS (Pan-Assay Interference Compounds) — frequent hitters in HTS."""
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        self.pains_flag = catalog.HasMatch(self._rdkit_mol)

    def _compute_mol_id(self):
        self.mol_id = hashlib.sha256(self.smiles.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def molecular_weight(self) -> float:
        return self.descriptors.get("MolWt", np.nan)

    @property
    def logp(self) -> float:
        return self.descriptors.get("MolLogP", np.nan)

    @property
    def tpsa(self) -> float:
        return self.descriptors.get("TPSA", np.nan)

    @property
    def feature_vector(self) -> np.ndarray:
        """
        Flat numpy array of descriptor values in a fixed, deterministic order.
        Used as part of the RL observation vector.
        Shape: (len(PKPD_RELEVANT_DESCRIPTORS),)
        """
        return np.array(
            [self.descriptors.get(k, np.nan) for k in PKPD_RELEVANT_DESCRIPTORS],
            dtype=np.float32,
        )

    @property
    def feature_names(self):
        return list(PKPD_RELEVANT_DESCRIPTORS)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mol_id": self.mol_id,
            "name": self.name,
            "smiles": self.smiles,
            "dose_unit": self.dose_unit,
            "descriptors": self.descriptors,
            "qed_score": self.qed_score,
            "lipinski_pass": self.lipinski_pass,
            "pains_flag": self.pains_flag,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DrugMolecule":
        return cls(smiles=d["smiles"], name=d["name"], dose_unit=d.get("dose_unit", "mg/kg"))

    def __repr__(self) -> str:
        return (
            f"DrugMolecule(name='{self.name}', smiles='{self.smiles[:30]}...', "
            f"MW={self.molecular_weight:.1f}, LogP={self.logp:.2f}, "
            f"QED={self.qed_score:.3f}, Lipinski={'✓' if self.lipinski_pass else '✗'})"
        )