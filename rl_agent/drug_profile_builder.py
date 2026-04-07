"""
DrugProfileBuilder — bridges Layer 1+2 (Patient_Simulation) into Layer 3+4 (rl_agent).

This is the missing integration glue. It takes a SMILES string and runs the full
Layer 1+2 pipeline automatically, returning a drug_profile dict that
RlAgentEnvironment accepts directly.

Pipeline executed:
    SMILES
      → DrugMolecule (RDKit descriptors, Lipinski, PAINS)
      → ADMETPredictor (DeepChem MolNet / QSAR fallback)
      → MolecularPropertyExtractor (29-feature obs vector, safety flags)
      → AllometricScaler (rat → human, FDA Km-factor HED)
      → drug_profile dict

Usage
-----
    from drug_profile_builder import DrugProfileBuilder

    builder = DrugProfileBuilder("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    profile = builder.build()          # runs Layer 1+2
    print(builder.summary())          # human-readable summary

    env = RlAgentEnvironment(drug_profile=profile)
    obs = env.reset()

Or via the HTTP API after the server is running:
    POST /drug  {"smiles": "...", "name": "Aspirin", "source_species": "rat"}
"""

import os
import sys
import warnings
import math

# ── path: add Patient_Simulation so Layer 1+2 imports resolve ────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PS_PATH   = os.path.join(_REPO_ROOT, "Patient_Simulation")
if _PS_PATH not in sys.path:
    sys.path.insert(0, _PS_PATH)


class DrugProfileBuilder:
    """
    Runs the full Layer 1+2 pipeline for a given molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the drug molecule.
        Example: "CC(=O)Oc1ccccc1C(=O)O" (Aspirin)
    name : str
        Human-readable drug name used in DoctorAgent prompts and logging.
    source_species : str
        Species from which the preclinical PK data comes. Default "rat".
        Supported: "mouse", "rat", "monkey", "dog", "human"
    target_species : str
        Species for the clinical trial. Default "human".
    animal_dose_mgkg : float
        Preclinical reference dose used to compute the Human Equivalent Dose (HED).
        Default 8.0 mg/kg (typical Phase I starting range in rodents).
    """

    def __init__(
        self,
        smiles: str,
        name: str = "investigational compound",
        source_species: str = "rat",
        target_species: str = "human",
        animal_dose_mgkg: float = 8.0,
    ):
        self.smiles            = smiles
        self.name              = name
        self.source_species    = source_species
        self.target_species    = target_species
        self.animal_dose_mgkg  = animal_dose_mgkg
        self._profile: dict | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def build(self) -> dict:
        """
        Execute the Layer 1+2 pipeline and return the drug_profile dict.

        Returns
        -------
        dict with the following keys, ready for RlAgentEnvironment:

            name                  : str    drug name
            smiles                : str    canonical SMILES
            drug_params           : dict   human-scaled PK params
                                          {ka, F, CL, Vc, Vp, Q, PPB, fu}
            safety_flags          : dict   safety signals from Layer 1
                                          {dili_risk, herg_risk, cyp_inhibitions,
                                           bbb_penetrant, overall_risk_score, ...}
            human_equivalent_dose : float  HED in mg/kg (FDA Km-factor method)
            observation_vector    : list   29-feature molecular descriptor vector
            admet_summary         : dict   key ADMET values for logging/display
        """
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler

        # Layer 1: molecule → ADMET → drug profile
        mol       = DrugMolecule(self.smiles, name=self.name)
        predictor = ADMETPredictor(use_deepchem=True, cache=True)
        profile   = MolecularPropertyExtractor(predictor).extract(mol)
        if profile.admet.source != "deepchem":
            raise RuntimeError(
                "DeepChem MolNet backend is required, but prediction did not use DeepChem. "
                "Run setup_models.py with internet access and ensure DeepChem/TensorFlow are installed."
            )

        # Layer 2: allometric scaling (source species → human)
        scaler      = AllometricScaler(self.source_species, self.target_species)
        human_params = {
            k: v for k, v in scaler.scale(profile.pkpd_params).items()
            if not k.startswith("_")
        }
        hed = scaler.scale_dose(self.animal_dose_mgkg)

        admet = profile.admet
        debug_drug = os.getenv("DEBUG_DRUG", "0").lower() in ("1", "true", "yes")
        if debug_drug:
            print(f"[DEBUG] ADMET source={admet.source}", file=sys.stderr, flush=True)
            print(
                f"[DEBUG] Raw ADMET summary="
                f"{{'F_oral': {admet.F_oral}, 'PPB': {admet.PPB}, 'BBB_probability': {admet.BBB_probability}, "
                f"'predicted_logD': {admet.predicted_logD}, 'clintox_toxic_prob': {admet.clintox_toxic_prob}}}",
                file=sys.stderr,
                flush=True,
            )
            print("[DEBUG] PK generation path=ADMETProperties.to_pkpd_params -> AllometricScaler.scale", file=sys.stderr, flush=True)
            print(f"[DEBUG] Derived PK pre-validation={human_params}", file=sys.stderr, flush=True)

        self._validate_pk_params(human_params)
        if debug_drug:
            print("[DEBUG] PK validation=PASS", file=sys.stderr, flush=True)

        self._profile = {
            "name":   self.name,
            "smiles": self.smiles,

            # PK parameters — human-scaled, ready for PatientAgent ODE
            "drug_params": human_params,

            # Safety flags — used by DoctorAgent and HepatocyteAgent
            "safety_flags": profile.safety_flags,

            # FDA-computed HED: environment starts at HED / 10
            "human_equivalent_dose": float(hed),

            # 29-feature vector for downstream RL observation augmentation
            "observation_vector": profile.observation_vector.tolist(),

            # Human-readable ADMET summary for logging and the /drug endpoint
            "admet_summary": {
                "source":             admet.source,
                "F_oral":             round(float(admet.F_oral), 4),
                "PPB":                round(float(admet.PPB), 4),
                "BBB_probability":    round(float(admet.BBB_probability), 4),
                "predicted_logD":     round(float(admet.predicted_logD), 4),
                "clintox_toxic_prob": round(float(admet.clintox_toxic_prob), 4),
                "half_life_class":    admet.half_life_class,
                "dili_risk":          bool(profile.safety_flags.get("dili_risk", False)),
                "herg_risk":          bool(profile.safety_flags.get("herg_risk", False)),
                "cyp_inhibitions":    list(profile.safety_flags.get("cyp_inhibitions", [])),
                "overall_risk_score": round(float(profile.safety_flags.get("overall_risk_score", 0.0)), 4),
            },
        }
        return self._profile

    @staticmethod
    def _validate_pk_params(params: dict) -> None:
        required = {"ka", "F", "CL", "Vc", "Vp", "Q", "PPB", "fu"}
        missing = sorted(required.difference(params.keys()))
        if missing:
            raise ValueError(f"PK validation failed: missing keys {missing}")

        def _num(name: str) -> float:
            v = params[name]
            try:
                x = float(v)
            except Exception as exc:
                raise ValueError(f"PK validation failed: {name} is non-numeric ({v!r})") from exc
            if not math.isfinite(x):
                raise ValueError(f"PK validation failed: {name} is non-finite ({x})")
            return x

        ka = _num("ka")
        F = _num("F")
        CL = _num("CL")
        Vc = _num("Vc")
        Vp = _num("Vp")
        Q = _num("Q")
        PPB = _num("PPB")
        fu = _num("fu")

        if not (0.0 < F <= 1.0):
            raise ValueError(f"PK validation failed: F out of bounds ({F})")
        if not (0.0 < fu <= 1.0):
            raise ValueError(f"PK validation failed: fu out of bounds ({fu})")
        if not (0.0 <= PPB <= 1.0):
            raise ValueError(f"PK validation failed: PPB out of bounds ({PPB})")
        if ka <= 0.0 or CL <= 0.0 or Vc <= 0.0 or Vp <= 0.0 or Q <= 0.0:
            raise ValueError(
                f"PK validation failed: expected ka/CL/Vc/Vp/Q > 0, got "
                f"ka={ka}, CL={CL}, Vc={Vc}, Vp={Vp}, Q={Q}"
            )

    @property
    def profile(self) -> dict:
        """Return the built profile, building it first if necessary."""
        if self._profile is None:
            self.build()
        return self._profile

    def summary(self) -> str:
        """Return a human-readable one-screen summary of the drug profile."""
        p = self.profile
        a = p["admet_summary"]
        dp = p["drug_params"]
        hed = p["human_equivalent_dose"]
        lines = [
            f"{'─'*56}",
            f"  Drug: {p['name']}",
            f"  SMILES: {p['smiles']}",
            f"  ADMET source: {a['source']}",
            f"{'─'*56}",
            f"  ADMET properties:",
            f"    Oral bioavailability (F):    {a['F_oral']:.2f}",
            f"    Plasma protein binding:      {a['PPB']:.2f}",
            f"    LogD (lipophilicity):        {a['predicted_logD']:.2f}",
            f"    BBB penetration prob:        {a['BBB_probability']:.2f}",
            f"    ClinTox probability:         {a['clintox_toxic_prob']:.2f}",
            f"    Half-life class:             {a['half_life_class']}",
            f"{'─'*56}",
            f"  Safety flags:",
            f"    DILI risk:      {'⚠ YES' if a['dili_risk'] else 'No'}",
            f"    hERG cardiac:   {'⚠ YES' if a['herg_risk'] else 'No'}",
            f"    CYP inhibitions: {', '.join(a['cyp_inhibitions']) or 'none'}",
            f"    Overall risk:   {a['overall_risk_score']:.2f} / 1.00",
            f"{'─'*56}",
            f"  Human-scaled PK parameters ({self.source_species} → {self.target_species}):",
            f"    ka  = {dp.get('ka',  0):.4f}  1/h       (absorption rate)",
            f"    F   = {dp.get('F',   0):.4f}             (bioavailability)",
            f"    CL  = {dp.get('CL',  0):.4f}  L/h/kg    (clearance)",
            f"    Vc  = {dp.get('Vc',  0):.4f}  L/kg      (central volume)",
            f"    Vp  = {dp.get('Vp',  0):.4f}  L/kg      (peripheral volume)",
            f"    Q   = {dp.get('Q',   0):.4f}  L/h/kg    (inter-comp flow)",
            f"{'─'*56}",
            f"  Dosing:",
            f"    Animal ref dose:        {self.animal_dose_mgkg:.1f} mg/kg ({self.source_species})",
            f"    Human Equivalent Dose:  {hed:.3f} mg/kg",
            f"    Trial starting dose:    {hed/10:.3f} mg/kg  (1/10 HED, FDA guidance)",
            f"{'─'*56}",
        ]
        return "\n".join(lines)

    def to_env_kwargs(self) -> dict:
        """Return dict suitable for RlAgentEnvironment(drug_profile=...)."""
        return {"drug_profile": self.profile}
