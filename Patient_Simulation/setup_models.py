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
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

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
        except Exception as e:
            logger.error("  [%s] FAILED: %s", name, e)
            sys.exit(1)


# ── Step 2: Train models ─────────────────────────────────────────────────────

def train_models():
    """Train GraphConvModel on each dataset and save to models/."""
    import deepchem as dc
    from deepchem.models import GraphConvModel

    os.makedirs(MODEL_DIR, exist_ok=True)
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # Point DeepChem's data directory at our local copy
    os.environ["DEEPCHEM_DATA_DIR"] = DATA_DIR

    for name, loader_name, n_tasks, mode in MODEL_CONFIGS:
        model_dir = os.path.join(MODEL_DIR, name)

        # Check if already trained
        if os.path.isdir(model_dir) and any(
            f.endswith((".index", ".npy", ".npz", ".h5")) or f.startswith("checkpoint")
            for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
        ):
            logger.info("  [%s] Model already trained at %s", name, model_dir)
            continue

        logger.info("  [%s] Loading dataset...", name)
        loader_fn = getattr(dc.molnet, loader_name)
        tasks, datasets, transformers = loader_fn(
            featurizer=featurizer,
            splitter="scaffold",
            data_dir=DATA_DIR,
            save_dir=os.path.join(DATA_DIR, f"{name}_featurized"),
        )
        train, valid, test = datasets

        logger.info("  [%s] Training GraphConvModel (%d tasks, %s, %d compounds, %d epochs)...",
                     name, n_tasks, mode, len(train), N_EPOCHS)

        model = GraphConvModel(
            n_tasks=n_tasks,
            mode=mode,
            model_dir=model_dir,
            batch_size=64,
            learning_rate=0.001,
        )

        t0 = time.time()
        model.fit(train, nb_epoch=N_EPOCHS)
        elapsed = time.time() - t0

        # Evaluate on test set
        metric_fn = dc.metrics.Metric(
            dc.metrics.roc_auc_score if mode == "classification" else dc.metrics.pearson_r2_score
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

    # Set env vars so ADMETPredictor finds our local data/models
    os.environ["DEEPCHEM_DATA_DIR"] = DATA_DIR
    os.environ["CLINICAL_TRIAL_GYM_MODEL_DIR"] = MODEL_DIR

    from clinical_trial_gym.drug.molecule import DrugMolecule
    from clinical_trial_gym.drug.admet import ADMETPredictor
    from clinical_trial_gym.drug.properties import MolecularPropertyExtractor

    mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    predictor = ADMETPredictor(model_dir=MODEL_DIR)
    extractor = MolecularPropertyExtractor(predictor)
    profile = extractor.extract(mol)

    logger.info("  Source: %s", profile.admet.source)
    logger.info("  F_oral: %.3f", profile.admet.F_oral)
    logger.info("  PPB: %.3f", profile.admet.PPB)
    logger.info("  BBB: %s (p=%.3f)", profile.admet.BBB_penetrant, profile.admet.BBB_probability)
    logger.info("  LogD: %.2f", profile.admet.predicted_logD)
    logger.info("  DILI: %s", profile.admet.DILI_flag)
    logger.info("  Tox21 assays: %d predictions", len(profile.admet.tox21_predictions))

    params = profile.pkpd_params
    logger.info("  PK/PD params: ka=%.3f F=%.3f CL=%.3f Vc=%.3f",
                 params["ka"], params["F"], params["CL"], params["Vc"])

    if profile.admet.source == "deepchem":
        logger.info("")
        logger.info("SUCCESS: All predictions from trained DeepChem MolNet models.")
    else:
        logger.warning("WARNING: Predictions from QSAR fallback. DeepChem models may not have loaded.")


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

    # Check dependencies
    try:
        import deepchem
        print(f"  DeepChem version: {deepchem.__version__}")
    except ImportError:
        print("ERROR: DeepChem not installed. Run: pip install deepchem tensorflow")
        sys.exit(1)

    try:
        import rdkit
        print(f"  RDKit: available")
    except ImportError:
        print("ERROR: RDKit not installed. Run: pip install rdkit")
        sys.exit(1)

    print()

    # Step 1
    print("─" * 70)
    print("Step 1/3: Downloading MoleculeNet datasets")
    print("─" * 70)
    download_datasets()
    print()

    # Step 2
    print("─" * 70)
    print("Step 2/3: Training GraphConvModels")
    print("─" * 70)
    train_models()
    print()

    # Step 3
    print("─" * 70)
    print("Step 3/3: Verification")
    print("─" * 70)
    verify()
    print()

    print("=" * 70)
    print("Setup complete. You can now run: python main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()