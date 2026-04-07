"""
Tests for ClinicalTrialGym Layers 1 and 2.

Run with:
    pytest tests/test_layer1_layer2.py -v

Requirements:
    pytest, rdkit, scipy, numpy, deepchem, tensorflow
    
ALL ADMET predictions use trained DeepChem models — no fallback heuristics.
"""

import numpy as np
import pytest


# -----------------------------------------------------------------------
# Layer 1 tests
# -----------------------------------------------------------------------

class TestDrugMolecule:
    """Test RDKit-based molecular property computation."""

    ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
    IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    CAFFEINE_SMILES = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"

    def test_valid_smiles_parses(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        assert mol.molecular_weight > 0
        assert mol.smiles != ""

    def test_aspirin_molecular_weight(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        # Aspirin MW = 180.16
        assert 178 < mol.molecular_weight < 182

    def test_aspirin_logp(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        # Aspirin LogP ≈ 1.2-1.5 (RDKit)
        assert 0.5 < mol.logp < 2.5

    def test_aspirin_lipinski_passes(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        assert mol.lipinski_pass is True

    def test_feature_vector_shape(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule, PKPD_RELEVANT_DESCRIPTORS
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        vec = mol.feature_vector
        assert vec.shape == (len(PKPD_RELEVANT_DESCRIPTORS),)
        assert vec.dtype == np.float32

    def test_feature_vector_no_nan(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        assert not np.any(np.isnan(mol.feature_vector))

    def test_invalid_smiles_raises(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        with pytest.raises(ValueError, match="Invalid SMILES"):
            DrugMolecule("NOTASMILES_XYZ", name="Bad")

    def test_mol_id_deterministic(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol1 = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        mol2 = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        assert mol1.mol_id == mol2.mol_id

    def test_qed_score_in_range(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        assert 0.0 <= mol.qed_score <= 1.0

    def test_serialization_roundtrip(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        d = mol.to_dict()
        mol2 = DrugMolecule.from_dict(d)
        assert mol.mol_id == mol2.mol_id
        assert abs(mol.molecular_weight - mol2.molecular_weight) < 0.01

    def test_ibuprofen_different_from_aspirin(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        mol1 = DrugMolecule(self.ASPIRIN_SMILES, name="Aspirin")
        mol2 = DrugMolecule(self.IBUPROFEN_SMILES, name="Ibuprofen")
        assert mol1.mol_id != mol2.mol_id
        assert mol1.molecular_weight != mol2.molecular_weight


class TestADMETPredictor:
    """Test ADMET prediction using trained DeepChem models."""

    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
    LIPOPHILIC_DRUG = "CCCCCCCC(=O)Oc1ccccc1"   # octyl benzoate (high LogP)

    def test_predict_returns_admet_properties(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert props is not None

    def test_source_is_trained_model(self):
        """Verify predictions come from a trained model, not raw heuristics."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        # Either DeepChem MolNet or trained QSAR RF — never "heuristic"
        assert props.source in ("deepchem", "qsar_trained")

    def test_f_oral_in_range(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert 0.0 <= props.F_oral <= 1.0

    def test_ppb_in_range(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert 0.0 <= props.PPB <= 1.0

    def test_bbb_probability_in_range(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert 0.0 <= props.BBB_probability <= 1.0

    def test_tox21_predictions_populated(self):
        """Tox21 model should produce predictions for all 12 assays."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert len(props.tox21_predictions) == 12
        for task, val in props.tox21_predictions.items():
            assert 0.0 <= val <= 1.0, f"Tox21 {task} = {val} out of range"

    def test_clintox_probabilities_populated(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert 0.0 <= props.clintox_approved_prob <= 1.0
        assert 0.0 <= props.clintox_toxic_prob <= 1.0

    def test_predicted_logD_is_finite(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert np.isfinite(props.predicted_logD)

    def test_caching(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor(cache=True)
        props1 = predictor.predict(mol)
        props2 = predictor.predict(mol)  # should be cached
        assert props1 is props2  # same object reference

    def test_pkpd_params_keys(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        params = props.to_pkpd_params()
        required_keys = {"ka", "F", "CL", "Vc", "Vp", "Q", "PPB", "fu"}
        assert required_keys.issubset(params.keys())

    def test_pkpd_params_positive(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        params = props.to_pkpd_params()
        for k, v in params.items():
            assert v >= 0, f"{k} = {v} should be non-negative"

    def test_pkpd_params_no_nan(self):
        """All PK/PD params must be finite — no NaN allowed."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        params = props.to_pkpd_params()
        for k, v in params.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_different_molecules_different_predictions(self):
        """Different molecules should produce different ADMET profiles."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol1 = DrugMolecule(self.ASPIRIN, name="Aspirin")
        mol2 = DrugMolecule(self.LIPOPHILIC_DRUG, name="Lipophilic")
        predictor = ADMETPredictor()
        p1 = predictor.predict(mol1)
        p2 = predictor.predict(mol2)
        # LogD should differ for molecules with very different lipophilicity
        assert abs(p1.predicted_logD - p2.predicted_logD) > 0.1

    def test_half_life_class_populated(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        props = predictor.predict(mol)
        assert props.half_life_class in ("short", "medium", "long")


class TestMolecularPropertyExtractor:
    """Test the Layer 1 → Layer 2 interface."""

    ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

    def test_extract_returns_drug_profile(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)
        assert profile is not None

    def test_observation_vector_shape(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)
        assert profile.observation_vector.shape == (29,)

    def test_observation_vector_no_nan(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)
        assert not np.any(np.isnan(profile.observation_vector))
        assert not np.any(np.isinf(profile.observation_vector))

    def test_safety_flags_present(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)
        assert "overall_risk_score" in profile.safety_flags
        assert 0.0 <= profile.safety_flags["overall_risk_score"] <= 1.0

    def test_tox21_active_count_in_safety_flags(self):
        """Safety flags should include Tox21 assay hit count."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule(self.ASPIRIN, name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)
        assert "tox21_active_count" in profile.safety_flags
        assert isinstance(profile.safety_flags["tox21_active_count"], int)


# -----------------------------------------------------------------------
# Layer 2 tests
# -----------------------------------------------------------------------

class TestSurrogateODE:
    """Test the two-compartment PK/PD ODE."""

    DEFAULT_PARAMS = {
        "ka": 1.5, "F": 0.8, "CL": 0.5,
        "Vc": 0.6, "Vp": 0.4, "Q": 0.3,
        "PPB": 0.85, "fu": 0.15,
    }

    def test_simulate_returns_states(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=24.0, dt_h=1.0)
        assert len(states) > 0

    def test_concentration_peaks_and_declines(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0, route="oral")
        states = ode.simulate(duration_h=48.0, dt_h=0.5)
        Cc_vals = [s.Cc for s in states]
        # Concentration should peak then decline
        peak_idx = np.argmax(Cc_vals)
        assert peak_idx > 0       # not instant
        assert peak_idx < len(Cc_vals) - 1   # not at end

    def test_no_dose_zero_concentration(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        states = ode.simulate(duration_h=24.0)
        assert all(s.Cc == 0.0 for s in states)

    def test_higher_dose_higher_cmax(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode1 = SurrogateODE(self.DEFAULT_PARAMS)
        ode2 = SurrogateODE(self.DEFAULT_PARAMS)
        ode1.administer_dose(5.0, time_h=0.0)
        ode2.administer_dose(20.0, time_h=0.0)
        s1 = ode1.simulate(duration_h=24.0)
        s2 = ode2.simulate(duration_h=24.0)
        cmax1 = max(s.Cc for s in s1)
        cmax2 = max(s.Cc for s in s2)
        assert cmax2 > cmax1

    def test_state_array_shape(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE, PKPDState
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0)
        states = ode.simulate(duration_h=24.0, dt_h=1.0)
        arr = states[0].to_array()
        assert arr.shape == (PKPDState.STATE_DIM,)

    def test_effect_in_range(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0)
        states = ode.simulate(duration_h=24.0)
        for s in states:
            assert 0.0 <= s.effect <= 1.0 + 1e-6

    def test_auc_monotonically_increasing(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0)
        states = ode.simulate(duration_h=48.0)
        aucs = [s.cumulative_AUC for s in states]
        assert all(aucs[i] <= aucs[i+1] + 1e-6 for i in range(len(aucs)-1))

    def test_summary_stats_keys(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0)
        ode.simulate(duration_h=24.0)
        stats = ode.get_summary_stats()
        assert "AUC" in stats
        assert "Cmax" in stats
        assert "mean_effect" in stats

    def test_reset_clears_state(self):
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(self.DEFAULT_PARAMS)
        ode.administer_dose(10.0, time_h=0.0)
        ode.simulate(duration_h=24.0)
        ode.reset()
        assert ode.current_state.Cc == 0.0
        assert ode.current_state.cumulative_AUC == 0.0


class TestAllometricScaler:
    """Test species-bridging allometric scaling."""

    PARAMS = {
        "ka": 2.0, "F": 0.75, "CL": 3.2,
        "Vc": 0.8, "Vp": 0.5, "Q": 0.6,
        "PPB": 0.90, "fu": 0.10,
    }

    def test_rat_to_human_cl_increases(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
        scaler = AllometricScaler("rat", "human")
        scaled = scaler.scale(self.PARAMS)
        # Rat BW=0.25 kg, Human BW=70 kg → ratio=280
        # CL scales by 280^0.75 ≈ 80x
        assert scaled["CL"] > self.PARAMS["CL"]

    def test_vd_scales_linearly(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler, SPECIES_DB
        scaler = AllometricScaler("rat", "human")
        scaled = scaler.scale(self.PARAMS)
        expected_ratio = (SPECIES_DB["human"].body_weight_kg /
                          SPECIES_DB["rat"].body_weight_kg) ** 1.0
        actual_ratio = scaled["Vc"] / self.PARAMS["Vc"]
        assert abs(actual_ratio - expected_ratio) < 0.1

    def test_ppb_unchanged(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
        scaler = AllometricScaler("rat", "human")
        scaled = scaler.scale(self.PARAMS)
        assert abs(scaled["PPB"] - self.PARAMS["PPB"]) < 1e-6

    def test_hed_dose_scaling(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
        scaler = AllometricScaler("rat", "human")
        rat_dose = 10.0  # mg/kg
        hed = scaler.scale_dose(rat_dose)
        # FDA Km method: rat_Km=6, human_Km=37 → HED = 10 × (6/37) ≈ 1.62 mg/kg
        assert abs(hed - 10.0 * (6 / 37)) < 0.01

    def test_metadata_in_output(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
        scaler = AllometricScaler("mouse", "human")
        scaled = scaler.scale(self.PARAMS)
        assert "_scaling_metadata" in scaled
        assert scaled["_scaling_metadata"]["source_species"] == "mouse"

    def test_unknown_species_raises(self):
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
        with pytest.raises(ValueError, match="Unknown species"):
            AllometricScaler("elephant", "human")


class TestPatientAgent:
    """Test PatientAgent simulation with DeepChem-derived drug profiles."""

    def _make_profile(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        return extractor.extract(mol)

    def test_patient_initializes(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        agent = PatientAgent(profile, rng=np.random.default_rng(42))
        assert agent.is_active is True

    def test_step_advances_time(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        agent = PatientAgent(profile, rng=np.random.default_rng(42))
        agent.administer(10.0, time_h=0.0)
        agent.step(duration_h=24.0)
        assert agent.elapsed_h == 24.0

    def test_observation_shape(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        agent = PatientAgent(profile, rng=np.random.default_rng(42))
        agent.administer(10.0, time_h=0.0)
        agent.step(duration_h=24.0)
        obs = agent.observation
        assert obs.shape == (PatientAgent.OBS_DIM,)

    def test_observation_no_nan(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        agent = PatientAgent(profile, rng=np.random.default_rng(42))
        agent.administer(10.0, time_h=0.0)
        agent.step(duration_h=24.0)
        obs = agent.observation
        assert not np.any(np.isnan(obs))

    def test_iiv_produces_variability(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        # Multiple patients from same population should differ
        agents = [PatientAgent(profile, rng=np.random.default_rng(i)) for i in range(5)]
        cls = [a._individual_pkpd["pk"]["CL"] for a in agents]
        assert len(set(round(c, 4) for c in cls)) > 1  # not all identical

    def test_reset_clears_history(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
        profile = self._make_profile()
        agent = PatientAgent(profile, rng=np.random.default_rng(42))
        agent.administer(10.0, time_h=0.0)
        agent.step(duration_h=72.0)
        agent.reset()
        assert agent.elapsed_h == 0.0
        assert len(agent.dose_history) == 0

    def test_population_sampler(self):
        from clinical_trial_gym.pk_pd.patient_agent import PatientPopulation
        profile = self._make_profile()
        pop = PatientPopulation(profile, n_patients=6, rng_seed=42)
        cohort = pop.sample()
        assert len(cohort) == 6
        ages = [a.covariates.age for a in cohort]
        assert len(set(round(a) for a in ages)) > 1  # varied ages


# -----------------------------------------------------------------------
# Integration test: Layer 1 → Layer 2 pipeline
# -----------------------------------------------------------------------

class TestEndToEndPipeline:
    """
    End-to-end test: SMILES → DeepChem ADMET → PK/PD → Patient simulation.
    This is the core ClinicalTrialGym data flow.
    """

    SMILES_LIST = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    ]

    def test_full_pipeline_three_drugs(self):
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        from clinical_trial_gym.pk_pd.patient_agent import PatientAgent

        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)

        for smiles, name in self.SMILES_LIST:
            mol = DrugMolecule(smiles, name=name)
            profile = extractor.extract(mol)

            # Verify all ADMET predictions are from trained models
            assert profile.admet.source in ("deepchem", "qsar_trained")
            assert len(profile.admet.tox21_predictions) == 12

            agent = PatientAgent(profile, rng=np.random.default_rng(0))
            agent.administer(5.0, time_h=0.0)
            state = agent.step(duration_h=24.0)
            obs = agent.observation
            assert obs.shape == (PatientAgent.OBS_DIM,)
            assert not np.any(np.isnan(obs))

    def test_allometric_transfer(self):
        """Rat preclinical → human Phase I parameter transfer."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
        from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler

        mol = DrugMolecule("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)
        profile = extractor.extract(mol)

        # Scale from rat preclinical to human
        scaler = AllometricScaler("rat", "human")
        human_params = scaler.scale(profile.pkpd_params)

        # Strip metadata before passing to ODE
        clean_params = {k: v for k, v in human_params.items()
                        if not k.startswith("_")}

        # Direct ODE test with scaled params
        from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
        ode = SurrogateODE(clean_params)
        ode.administer_dose(2.0, time_h=0.0)   # HED dose
        states = ode.simulate(duration_h=24.0)
        assert len(states) > 0
        assert states[-1].Cmax > 0

    def test_no_hardcoded_fallbacks_in_pkpd(self):
        """Verify that PK/PD params are all derived from model predictions,
        not from hardcoded defaults."""
        from clinical_trial_gym.drug.molecule import DrugMolecule
        from clinical_trial_gym.drug.admet import ADMETPredictor
        from clinical_trial_gym.drug.properties import MolecularPropertyExtractor

        predictor = ADMETPredictor()
        extractor = MolecularPropertyExtractor(predictor)

        # Test three different molecules
        for smiles, name in self.SMILES_LIST:
            mol = DrugMolecule(smiles, name=name)
            profile = extractor.extract(mol)
            params = profile.pkpd_params

            # All params must be finite and positive
            for k, v in params.items():
                assert np.isfinite(v), f"{name}: {k} = {v} not finite"
                assert v > 0, f"{name}: {k} = {v} not positive"

        # Different molecules should produce different PK params
        profiles = []
        for smiles, name in self.SMILES_LIST:
            mol = DrugMolecule(smiles, name=name)
            profiles.append(extractor.extract(mol))

        # CL should differ across drugs
        cls = [p.pkpd_params["CL"] for p in profiles]
        assert len(set(round(c, 6) for c in cls)) > 1, \
            "All drugs got same CL — likely hardcoded"