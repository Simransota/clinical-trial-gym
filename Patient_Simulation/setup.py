"""Package installation for ClinicalTrialGym."""
from setuptools import setup, find_packages

setup(
    name="clinical_trial_gym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "rdkit>=2023.3",
        "gymnasium>=0.29",
    ],
    extras_require={
        "deepchem": ["deepchem>=2.7", "tensorflow>=2.12"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    python_requires=">=3.9",
    description="Gymnasium-compliant RL environments for clinical trial optimization",
)
