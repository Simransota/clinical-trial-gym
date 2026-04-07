"""
pytest configuration — sets up sys.path so that tests can be run from any directory.

Run:
    cd Patient_Simulation
    pytest clinical_trial_gym/tests/ -v
"""

import sys
import os

# Ensure the Patient_Simulation directory is on sys.path so that
# `from clinical_trial_gym.xxx import yyy` works without installing the package.
_here = os.path.dirname(os.path.abspath(__file__))
_patient_sim_root = os.path.dirname(os.path.dirname(_here))  # .../Patient_Simulation
if _patient_sim_root not in sys.path:
    sys.path.insert(0, _patient_sim_root)
