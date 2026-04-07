#!/usr/bin/env python3
"""
ClinicalTrialGym Setup: Download MolNet datasets & train DeepChem models.

Run this ONCE on a machine with internet access:

    python setup_models.py

What it does:
  1. Downloads 6 MoleculeNet datasets (~10 MB total) from DeepChem's S3 bucket
  2. Trains GraphConvModel on each dataset (takes 5-15 min depending on hardware)
  3. Caches everything to data/ and models/ directories in the repo

After setup, the pipeline works fully offline with real DeepChem models.
No QSAR fallback needed, no internet needed.

Requirements:
    pip install deepchem tensorflow rdkit scipy numpy scikit-learn

Fixes applied vs original:
  - HPPB AxisError: DeepChem's remove_missing_entries calls X.any(axis=1) which
    crashes under NumPy 2 when ConvMolFeaturizer returns a 1D object array.
    We patch deepchem.utils.remove_missing_entries BEFORE any dataset loader
    imports it, so the patch is in effect for all loaders.
  - NumPy 2 ragged-array patch for MolecularFeaturizer (carried over from original).
  - HPPB featurized cache is always cleared before loading to prevent stale-shard
    corruption from previous failed runs.
  - GraphConvModel Keras 3 / BatchNormalization fused=True incompatibility is
    detected early with a clear error message.
"""

import os
import sys
import time
import logging
import shutil
import importlib.util

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ── Keras legacy shim (must come before any TF/DeepChem import) ──────────────
if importlib.util.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# ── Configuration ────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data", "molnet")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
N_EPOCHS = 30

DATASETS = {
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "BBBP.csv",
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv.gz",
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv.gz",
    },
    "lipo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "filename": "Lipophilicity.csv",
    },
    "clearance": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clearance.csv",
        "filename": "clearance.csv",
    },
    "hppb": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hppb.csv",
        "filename": "hppb.csv",
    },
}

MODEL_CONFIGS = [
    # (name, loader_function_name, n_tasks, mode)
    ("bbbp",      "load_bbbp",      1,  "classification"),
    ("clintox",   "load_clintox",   2,  "classification"),
    ("tox21",     "load_tox21",     12, "classification"),
    ("lipo",      "load_lipo",      1,  "regression"),
    ("clearance", "load_clearance", 1,  "regression"),
    ("hppb",      "load_hppb",      1,  "regression"),
]


# ── Patch 1: NumPy 2 ragged-array fix for MolecularFeaturizer ────────────────

def _apply_deepchem_numpy2_compat_patch() -> bool:
    """
    Patch DeepChem 2.x MolecularFeaturizer for NumPy 2 ragged array behavior.

    DeepChem 2.5 returns np.asarray(features). With NumPy 2, this raises
    ValueError when any molecule fails featurization (mixed shapes).
    The patch returns an object array so failed rows can be filtered instead
    of crashing.
    """
    import numpy as np
    from deepchem.feat.base_classes import MolecularFeaturizer

    if getattr(MolecularFeaturizer.featurize, "_ctg_numpy2_patch", False):
        return False  # already applied

    if int(np.__version__.split(".")[0]) < 2:
        return False  # not needed on NumPy 1.x

    _original_featurize = MolecularFeaturizer.featurize

    def _patched_featurize(self, molecules, log_every_n=1000):
        try:
            return _original_featurize(self, molecules, log_every_n=log_every_n)
        except ValueError as exc:
            if "inhomogeneous shape" not in str(exc):
                raise

            from rdkit import Chem
            from rdkit.Chem import rdmolfiles, rdmolops
            from rdkit.Chem.rdchem import Mol

            if isinstance(molecules, (str, Mol)):
                molecules = [molecules]
            else:
                molecules = list(molecules)

            features = []
            for i, mol in enumerate(molecules):
                if i % log_every_n == 0:
                    logger.info("Featurizing datapoint %i", i)
                try:
                    if isinstance(mol, str):
                        mol = Chem.MolFromSmiles(mol)
                        new_order = rdmolfiles.CanonicalRankAtoms(mol)
                        mol = rdmolops.RenumberAtoms(mol, new_order)
                    features.append(self._featurize(mol))
                except Exception as inner_exc:
                    smiles_repr = Chem.MolToSmiles(mol) if isinstance(mol, Mol) else mol
                    logger.warning(
                        "Failed to featurize datapoint %d (%s): %s. "
                        "Appending empty array.",
                        i, smiles_repr, inner_exc,
                    )
                    features.append(np.array([]))

            return np.array(features, dtype=object)

    _patched_featurize._ctg_numpy2_patch = True
    MolecularFeaturizer.featurize = _patched_featurize
    return True


# ── Patch 2: remove_missing_entries AxisError fix ─────────────────────────────

def _patch_remove_missing_entries() -> bool:
    """
    Patch deepchem.utils.remove_missing_entries to handle 1-D X arrays.

    Root cause
    ----------
    DeepChem's hppb_datasets.py (and a few others) call:

        available_rows = X.any(axis=1)

    When ConvMolFeaturizer returns a 1-D object array (every molecule in a
    shard failed, or the dataset has exactly one sample), `axis=1` does not
    exist and NumPy 2 raises:

        numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1

    The patched version:
      - If X is 1-D → treat the single value as one "available" row (keep it).
      - If X is 2-D → use the original axis=1 check (unchanged behaviour).
      - If X is 0-D or empty → skip the shard gracefully.

    This patch is installed on `deepchem.utils` AND on the module-level name
    in `deepchem.molnet.load_function.hppb_datasets` so it takes effect
    regardless of which import path was already cached.
    """
    import numpy as np
    
    # Track if we successfully patched anything
    patched_any = False

    def _safe_remove_missing_entries(dataset):
        """NumPy-2-safe version of remove_missing_entries."""
        for i, (X, y, w, ids) in enumerate(dataset.itershards()):
            X = np.asarray(X)

            if X.ndim == 0 or X.size == 0:
                # Empty shard — nothing to filter, leave it alone.
                logger.debug("Shard %d is empty, skipping missing-entry filter.", i)
                continue

            if X.ndim == 1:
                # Single sample or object array of failed featurizations.
                # Treat the whole row as available (non-zero object reference).
                available_rows = np.array([bool(X.any())], dtype=bool)
                if not available_rows[0]:
                    # The one sample is all-zero → remove it.
                    X   = X[np.array([], dtype=int)]
                    y   = y[np.array([], dtype=int)]
                    w   = w[np.array([], dtype=int)]
                    ids = ids[np.array([], dtype=int)]
                    dataset.set_shard(i, X, y, w, ids)
                # Otherwise leave as-is (nothing to remove).
                continue

            # Normal 2-D path — identical to the original implementation.
            available_rows = X.any(axis=1)
            missing = int(np.count_nonzero(~available_rows))
            if missing:
                logger.info("Shard %d has %d missing entries.", i, missing)
            X   = X[available_rows]
            y   = y[available_rows]
            w   = w[available_rows]
            ids = ids[available_rows]
            dataset.set_shard(i, X, y, w, ids)

    _safe_remove_missing_entries._ctg_rme_patch = True

    # In DeepChem 2.5.0+, remove_missing_entries is defined inside dataset modules
    # rather than deepchem.utils. We must patch them individually.
    modules_to_patch = [
        "deepchem.utils",
        "deepchem.molnet.load_function.hppb_datasets",
        "deepchem.molnet.load_function.uv_datasets",
        "deepchem.molnet.load_function.kaggle_datasets",
        "deepchem.molnet.load_function.factors_datasets",
        "deepchem.molnet.load_function.kinase_datasets",
    ]

    import importlib
    for mod_name in modules_to_patch:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "remove_missing_entries"):
                # Check if it's already our patched version
                if getattr(mod.remove_missing_entries, "_ctg_rme_patch", False):
                    patched_any = True
                    continue
                # Overwrite with our safe version
                mod.remove_missing_entries = _safe_remove_missing_entries
                patched_any = True
        except ImportError:
            pass

    return patched_any


# ── Step 1: Download datasets ────────────────────────────────────────────────

def download_datasets():
    """Download all MolNet CSV files to data/molnet/."""
    import urllib.request

    os.makedirs(DATA_DIR, exist_ok=True)

    for name, info in DATASETS.items():
        dest = os.path.join(DATA_DIR, info["filename"])
        if os.path.exists(dest):
            logger.info("  [%s] Already downloaded: %s", name, dest)
            continue

        logger.info("  [%s] Downloading %s ...", name, info["url"])
        try:
            urllib.request.urlretrieve(info["url"], dest)
            size_kb = os.path.getsize(dest) / 1024
            logger.info("  [%s] Done (%.1f KB)", name, size_kb)
        except Exception as exc:
            logger.error("  [%s] FAILED: %s", name, exc)
            sys.exit(1)


# ── Step 2: Train models ─────────────────────────────────────────────────────

def train_models():
    """Train GraphConvModel on each dataset and save to models/."""
    import deepchem as dc
    from deepchem.models import GraphConvModel

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Apply all patches BEFORE any loader is called ────────────────────────
    patched_np2 = _apply_deepchem_numpy2_compat_patch()
    if patched_np2:
        logger.info("Applied DeepChem NumPy 2 compatibility patch (featurizer).")

    patched_rme = _patch_remove_missing_entries()
    if patched_rme:
        logger.info("Applied remove_missing_entries AxisError patch (HPPB fix).")

    featurizer = dc.feat.ConvMolFeaturizer()

    # Point DeepChem's data directory at our local copy.
    os.environ["DEEPCHEM_DATA_DIR"] = DATA_DIR

    for name, loader_name, n_tasks, mode in MODEL_CONFIGS:
        model_dir = os.path.join(MODEL_DIR, name)
        save_dir  = os.path.join(DATA_DIR, f"{name}_featurized")

        # ── Skip if already trained ──────────────────────────────────────────
        if os.path.isdir(model_dir) and any(
            f.endswith((".index", ".npy", ".npz", ".h5"))
            or f.startswith("checkpoint")
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f))
        ):
            logger.info("  [%s] Model already trained at %s", name, model_dir)
            continue

        # ── HPPB: always clear featurized cache ──────────────────────────────
        # The HPPB dataset's featurized cache is particularly prone to
        # corruption from previous failed runs because remove_missing_entries
        # mutates shards in-place. A stale cache can have shards where X was
        # already trimmed to 1-D. Clearing it guarantees a clean slate.
        if name == "hppb":
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir, ignore_errors=True)
                logger.info(
                    "  [hppb] Cleared stale featurized cache at %s "
                    "(prevents AxisError from prior failed run).",
                    save_dir,
                )

        # ── Load dataset ─────────────────────────────────────────────────────
        logger.info("  [%s] Loading dataset...", name)
        loader_fn = getattr(dc.molnet, loader_name)

        loader_kwargs = dict(
            featurizer=featurizer,
            splitter="scaffold",
            data_dir=DATA_DIR,
            save_dir=save_dir,
        )

        try:
            tasks, datasets, transformers = loader_fn(**loader_kwargs)
        except Exception as first_exc:
            logger.warning(
                "  [%s] Dataset load failed (%s). "
                "Clearing featurized cache and retrying without reload.",
                name, first_exc,
            )
            shutil.rmtree(save_dir, ignore_errors=True)
            loader_kwargs["reload"] = False
            try:
                tasks, datasets, transformers = loader_fn(**loader_kwargs)
            except Exception as second_exc:
                logger.error(
                    "  [%s] Dataset load failed again (%s). "
                    "Skipping this dataset — ADMET predictor will use QSAR fallback.",
                    name, second_exc,
                )
                continue

        train, valid, test = datasets

        # ── Train model ───────────────────────────────────────────────────────
        logger.info(
            "  [%s] Training GraphConvModel "
            "(%d tasks, %s, %d compounds, %d epochs)...",
            name, n_tasks, mode, len(train), N_EPOCHS,
        )

        try:
            model = GraphConvModel(
                n_tasks=n_tasks,
                mode=mode,
                model_dir=model_dir,
                batch_size=64,
                learning_rate=0.001,
            )
        except Exception as exc:
            msg = str(exc)
            if "BatchNormalization" in msg and "fused" in msg:
                raise RuntimeError(
                    "GraphConvModel is incompatible with Keras 3 in this environment.\n"
                    "Install legacy Keras support and retry:\n"
                    "    pip install tf_keras\n"
                    "Then set the environment variable:\n"
                    "    export TF_USE_LEGACY_KERAS=1"
                ) from exc
            raise

        t0 = time.time()
        model.fit(train, nb_epoch=N_EPOCHS)
        elapsed = time.time() - t0

        # ── Evaluate ──────────────────────────────────────────────────────────
        metric_fn = dc.metrics.Metric(
            dc.metrics.roc_auc_score
            if mode == "classification"
            else dc.metrics.pearson_r2_score
        )
        try:
            score = model.evaluate(test, [metric_fn], transformers)
            logger.info("  [%s] Done in %.1fs. Test score: %s", name, elapsed, score)
        except Exception:
            logger.info("  [%s] Done in %.1fs. (evaluation skipped)", name, elapsed)

    logger.info("")
    logger.info("All models trained and saved to: %s", MODEL_DIR)


# ── Step 3: Verify ────────────────────────────────────────────────────────────

def verify():
    """Quick verification: predict Aspirin's ADMET profile."""
    logger.info("Verifying with Aspirin prediction...")

    os.environ["DEEPCHEM_DATA_DIR"] = DATA_DIR
    os.environ["CLINICAL_TRIAL_GYM_MODEL_DIR"] = MODEL_DIR

    from clinical_trial_gym.drug.molecule import DrugMolecule
    from clinical_trial_gym.drug.admet import ADMETPredictor
    from clinical_trial_gym.drug.properties import MolecularPropertyExtractor

    mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    predictor = ADMETPredictor(model_dir=MODEL_DIR)
    extractor = MolecularPropertyExtractor(predictor)
    profile = extractor.extract(mol)

    logger.info("  Source  : %s", profile.admet.source)
    logger.info("  F_oral  : %.3f", profile.admet.F_oral)
    logger.info("  PPB     : %.3f", profile.admet.PPB)
    logger.info("  BBB     : %s (p=%.3f)",
                profile.admet.BBB_penetrant, profile.admet.BBB_probability)
    logger.info("  LogD    : %.2f", profile.admet.predicted_logD)
    logger.info("  DILI    : %s", profile.admet.DILI_flag)
    logger.info("  Tox21   : %d predictions", len(profile.admet.tox21_predictions))

    params = profile.pkpd_params
    logger.info("  PK/PD   : ka=%.3f  F=%.3f  CL=%.3f  Vc=%.3f",
                params["ka"], params["F"], params["CL"], params["Vc"])

    if profile.admet.source == "deepchem":
        logger.info("")
        logger.info("SUCCESS: All predictions from trained DeepChem MolNet models.")
    else:
        logger.warning(
            "NOTE: Predictions from QSAR fallback. "
            "DeepChem models may not have loaded (check logs above)."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ClinicalTrialGym — Model Setup")
    print("=" * 70)
    print()
    print(f"  Data directory  : {DATA_DIR}")
    print(f"  Model directory : {MODEL_DIR}")
    print(f"  Training epochs : {N_EPOCHS}")
    print()

    # ── Dependency check ──────────────────────────────────────────────────────
    try:
        import deepchem
        print(f"  DeepChem version: {deepchem.__version__}")
    except ImportError:
        print("ERROR: DeepChem not installed. Run:")
        print("    pip install deepchem tensorflow")
        sys.exit(1)

    try:
        import numpy as np
        print(f"  NumPy version   : {np.__version__}")
        if int(np.__version__.split(".")[0]) >= 2:
            print("  NumPy 2 detected — AxisError patches will be applied.")
    except ImportError:
        print("ERROR: NumPy not installed.")
        sys.exit(1)

    try:
        import rdkit
        print("  RDKit           : available")
    except ImportError:
        print("ERROR: RDKit not installed. Run:")
        print("    pip install rdkit")
        sys.exit(1)

    print()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("Step 1/3: Downloading MoleculeNet datasets")
    print("─" * 70)
    download_datasets()
    print()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("Step 2/3: Training GraphConvModels")
    print("─" * 70)
    train_models()
    print()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("─" * 70)
    print("Step 3/3: Verification")
    print("─" * 70)
    verify()
    print()

    print("=" * 70)
    print("Setup complete. You can now run:  python main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()