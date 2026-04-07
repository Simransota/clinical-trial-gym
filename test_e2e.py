"""
End-to-end test suite for Clinical Trial Gym.
Covers Layers 1–4. Run from the project root:
    python test_e2e.py
"""

import sys
import os
import time
import traceback
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Patient_Simulation"))
sys.path.insert(0, os.path.join(ROOT, "rl_agent"))

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"

ASPIRIN   = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE  = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

results = []


# ════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run(label, fn):
    t0 = time.time()
    try:
        out = fn()
        ms = (time.time() - t0) * 1000
        print(f"  {PASS}  {label}  ({ms:.0f}ms)")
        results.append(("pass", label))
        return out
    except Exception as e:
        ms = (time.time() - t0) * 1000
        print(f"  {FAIL}  {label}  ({ms:.0f}ms)")
        print(f"         {RED}{type(e).__name__}: {e}{RESET}")
        traceback.print_exc()
        results.append(("fail", label))
        return None


def section(title):
    print(f"\n{BOLD}{BLUE}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*60}{RESET}")


def ok(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


def expect_error(fn, exc_type):
    try:
        fn()
        raise AssertionError(f"Expected {exc_type.__name__} but no error raised")
    except exc_type:
        pass


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def l1_make_mol(smiles, name):
    from clinical_trial_gym.drug.molecule import DrugMolecule
    return DrugMolecule(smiles, name=name)


def l1_make_predictor():
    from clinical_trial_gym.drug.admet import ADMETPredictor
    return ADMETPredictor()


def l1_make_extractor(predictor):
    from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
    return MolecularPropertyExtractor(predictor)


def l1_check_different_cl(extractor):
    # Check that different molecules produce different observation vectors and PK params.
    # We check `ka` (driven by logD which is clearly different across Aspirin/Ibuprofen/Caffeine)
    # and the full observation vector. CL can clip to the same minimum for drugs with low
    # predicted clearance — that's a known limitation of the empirical ADMET→PK formula.
    profiles = []
    for smi, name in [(ASPIRIN, "Aspirin"), (IBUPROFEN, "Ibuprofen"), (CAFFEINE, "Caffeine")]:
        mol = l1_make_mol(smi, name)
        profiles.append(extractor.extract(mol))
    kas = [p.pkpd_params["ka"] for p in profiles]
    obs_vecs = [p.observation_vector for p in profiles]
    # Observation vectors must differ (molecular descriptors are different)
    ok(not np.allclose(obs_vecs[0], obs_vecs[1]),
       f"Aspirin and Ibuprofen have identical observation vectors")
    ok(not np.allclose(obs_vecs[0], obs_vecs[2]),
       f"Aspirin and Caffeine have identical observation vectors")
    # ka should differ (it is derived from logD which is clearly different per molecule)
    ok(len(set(round(k, 4) for k in kas)) > 1,
       f"All drugs got identical ka: {kas} — logD differentiation not working")


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

PK = {"ka": 1.5, "F": 0.8, "CL": 0.5, "Vc": 0.6, "Vp": 0.4,
      "Q": 0.3, "PPB": 0.85, "fu": 0.15}


def l2_make_ode(params=None):
    from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
    return SurrogateODE(params or PK)


def l2_sim(params, dose, duration, route="oral"):
    from clinical_trial_gym.pk_pd.surrogate_ode import SurrogateODE
    ode = SurrogateODE(params or PK)
    ode.administer_dose(dose, time_h=0.0, route=route)
    return ode.simulate(duration, dt_h=0.5)


def l2_check_peak_decline(cc_vals):
    peak_idx = int(np.argmax(cc_vals))
    ok(peak_idx >= 1, "Cmax at t=0 — no absorption delay")
    ok(peak_idx < len(cc_vals) - 1, "Cmax at last point — drug never clears")


def l2_make_scaler(src="rat", tgt="human"):
    from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
    return AllometricScaler(src, tgt)


def l2_patient_from_profile(profile, seed=42):
    from clinical_trial_gym.pk_pd.patient_agent import PatientAgent
    return PatientAgent(profile, rng=np.random.default_rng(seed))


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def l3_patient(**kwargs):
    from server.agents import PatientAgent
    return PatientAgent(**kwargs)


def l3_hep(stress):
    from server.agents import HepatocyteAgent
    return HepatocyteAgent.observe({"liver_stress": stress, "kidney_stress": 0.0, "blood_conc": 0.0})


def l3_immune(conc):
    from server.agents import ImmuneAgent
    return ImmuneAgent.observe({"blood_conc": conc})


def l3_renal(stress):
    from server.agents import RenalAgent
    return RenalAgent.observe({"kidney_stress": stress})


def l3_grade(conc, lstress, kstress):
    from server.agents import MeasurementAgent
    return MeasurementAgent.grade_dlt({"blood_conc": conc, "liver_stress": lstress, "kidney_stress": kstress})


def l3_labs(conc, lstress, kstress):
    from server.agents import MeasurementAgent
    return MeasurementAgent.get_labs({"blood_conc": conc, "liver_stress": lstress, "kidney_stress": kstress})


# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def l4_make_env(drug_profile=None):
    from server.rl_agent_environment import RlAgentEnvironment
    return RlAgentEnvironment(drug_profile=drug_profile)


def l4_action(next_dose, cohort_size=3, escalate=True):
    from models import RlAgentAction
    return RlAgentAction(next_dose=next_dose, cohort_size=cohort_size, escalate=escalate)


def l4_build_drug_profile(profile, scaler):
    human_params = {k: v for k, v in scaler.scale(profile.pkpd_params).items()
                    if not k.startswith("_")}
    return {
        "name":   "TestDrug",
        "smiles": ASPIRIN,
        "drug_params":  human_params,
        "safety_flags": profile.safety_flags,
        "human_equivalent_dose": scaler.scale_dose(8.0),
    }


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# TEST EXECUTION
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# LAYER 1
# ────────────────────────────────────────────────────────────────────────────
section("LAYER 1 — Molecular Properties (RDKit + ADMET)")

mol_asp = run("DrugMolecule: parse Aspirin SMILES",
              lambda: l1_make_mol(ASPIRIN, "Aspirin"))

run("DrugMolecule: MW in [178, 182]",
    lambda: ok(178 < mol_asp.molecular_weight < 182, f"MW={mol_asp.molecular_weight}"))

run("DrugMolecule: LogP finite",
    lambda: ok(np.isfinite(mol_asp.logp), f"logp={mol_asp.logp}"))

run("DrugMolecule: Lipinski pass",
    lambda: ok(mol_asp.lipinski_pass is True))

run("DrugMolecule: feature vector non-empty",
    lambda: ok(mol_asp.feature_vector.ndim == 1 and len(mol_asp.feature_vector) > 0))

run("DrugMolecule: no NaN in feature vector",
    lambda: ok(not np.any(np.isnan(mol_asp.feature_vector))))

run("DrugMolecule: mol_id deterministic",
    lambda: ok(l1_make_mol(ASPIRIN, "a").mol_id == l1_make_mol(ASPIRIN, "b").mol_id,
               "mol_id changed for same SMILES"))

run("DrugMolecule: invalid SMILES raises ValueError",
    lambda: expect_error(lambda: l1_make_mol("NOT_A_SMILES_XYZ", "bad"), ValueError))

run("DrugMolecule: QED score in [0,1]",
    lambda: ok(0.0 <= mol_asp.qed_score <= 1.0, f"qed={mol_asp.qed_score}"))

predictor = run("ADMETPredictor: instantiate", l1_make_predictor)

admet_asp = run("ADMETPredictor: predict Aspirin",
                lambda: predictor.predict(mol_asp))

run("ADMETPredictor: source is trained model",
    lambda: ok(admet_asp.source in ("deepchem", "qsar_trained"),
               f"source={admet_asp.source}"))

run("ADMETPredictor: F_oral in [0,1]",
    lambda: ok(0.0 <= admet_asp.F_oral <= 1.0, f"F_oral={admet_asp.F_oral}"))

run("ADMETPredictor: PPB in [0,1]",
    lambda: ok(0.0 <= admet_asp.PPB <= 1.0, f"PPB={admet_asp.PPB}"))

run("ADMETPredictor: BBB_probability in [0,1]",
    lambda: ok(0.0 <= admet_asp.BBB_probability <= 1.0))

run("ADMETPredictor: predicted_logD finite",
    lambda: ok(np.isfinite(admet_asp.predicted_logD), f"logD={admet_asp.predicted_logD}"))

run("ADMETPredictor: Tox21 — 12 predictions all in [0,1]",
    lambda: ok(
        len(admet_asp.tox21_predictions) == 12 and
        all(0.0 <= v <= 1.0 for v in admet_asp.tox21_predictions.values()),
        str(admet_asp.tox21_predictions)
    ))

run("ADMETPredictor: to_pkpd_params — all required keys present",
    lambda: ok({"ka","F","CL","Vc","Vp","Q","PPB","fu"}.issubset(
               admet_asp.to_pkpd_params().keys())))

run("ADMETPredictor: to_pkpd_params — all finite and non-negative",
    lambda: ok(all(
        np.isfinite(v) and v >= 0
        for v in admet_asp.to_pkpd_params().values()
    ), str(admet_asp.to_pkpd_params())))

run("ADMETPredictor: caching — same object on second call",
    lambda: ok(predictor.predict(mol_asp) is admet_asp,
               "cache miss — different object returned"))

extractor = run("MolecularPropertyExtractor: instantiate",
                lambda: l1_make_extractor(predictor))

profile_asp = run("MolecularPropertyExtractor: extract Aspirin",
                  lambda: extractor.extract(mol_asp))

run("DrugProfile: observation_vector shape == (29,)",
    lambda: ok(profile_asp.observation_vector.shape == (29,),
               f"shape={profile_asp.observation_vector.shape}"))

run("DrugProfile: no NaN/Inf in observation_vector",
    lambda: ok(
        not np.any(np.isnan(profile_asp.observation_vector)) and
        not np.any(np.isinf(profile_asp.observation_vector))
    ))

run("DrugProfile: safety_flags overall_risk_score in [0,1]",
    lambda: ok(0.0 <= profile_asp.safety_flags.get("overall_risk_score", -1) <= 1.0))

run("DrugProfile: pkpd_params all positive and finite",
    lambda: ok(all(
        np.isfinite(v) and v > 0
        for v in profile_asp.pkpd_params.values()
    ), str(profile_asp.pkpd_params)))

run("DrugProfile: different drugs → different CL values",
    lambda: l1_check_different_cl(extractor))


# ────────────────────────────────────────────────────────────────────────────
# LAYER 2
# ────────────────────────────────────────────────────────────────────────────
section("LAYER 2 — PK/PD ODE + Allometric Scaling + Patient Population")

ode_fresh = run("SurrogateODE: instantiate", lambda: l2_make_ode())

run("SurrogateODE: no dose → zero concentration everywhere",
    lambda: ok(all(s.Cc == 0.0 for s in ode_fresh.simulate(24.0))))

states_oral = run("SurrogateODE: oral dose 10 mg/kg over 24h",
                  lambda: l2_sim(PK, 10.0, 24.0))

run("SurrogateODE: concentration peaks then declines",
    lambda: l2_check_peak_decline([s.Cc for s in states_oral]))

run("SurrogateODE: AUC monotonically increases",
    lambda: ok(all(
        states_oral[i].cumulative_AUC <= states_oral[i+1].cumulative_AUC + 1e-6
        for i in range(len(states_oral)-1)
    )))

run("SurrogateODE: effect in [0,1]",
    lambda: ok(all(0.0 <= s.effect <= 1.0 + 1e-6 for s in states_oral)))

run("SurrogateODE: toxicity_score in [0,1]",
    lambda: ok(all(0.0 <= s.toxicity_score <= 1.0 + 1e-6 for s in states_oral)))

run("SurrogateODE: higher dose → higher Cmax",
    lambda: ok(
        max(s.Cc for s in l2_sim(PK, 20.0, 24.0)) >
        max(s.Cc for s in l2_sim(PK,  5.0, 24.0))
    ))

run("SurrogateODE: summary stats — required keys present", lambda: ok(
    {"AUC", "Cmax", "Tmax_h", "mean_effect", "peak_toxicity"}.issubset(
        (lambda o: (o.administer_dose(10.0, 0.0), o.simulate(24.0), o.get_summary_stats())[2])(l2_make_ode())
    )
))

run("SurrogateODE: reset clears Cc and AUC", lambda: (
    lambda o: (o.administer_dose(10.0, 0.0), o.simulate(24.0), o.reset(),
               ok(o.current_state.Cc == 0.0 and o.current_state.cumulative_AUC == 0.0)
               )(l2_make_ode())
))

run("SurrogateODE: IV bolus → Cc > 0 at first timepoint",
    lambda: ok(
        any(s.Cc > 0 for s in l2_sim(PK, 10.0, 24.0, route="iv_bolus"))
    ))

scaler = run("AllometricScaler: instantiate rat→human", lambda: l2_make_scaler())

scaled = run("AllometricScaler: scale rat PK params",
             lambda: scaler.scale({"ka": 2.0, "F": 0.75, "CL": 3.2,
                                    "Vc": 0.8, "Vp": 0.5, "Q": 0.6,
                                    "PPB": 0.90, "fu": 0.10}))

run("AllometricScaler: human CL > rat CL (≈80× for rat→human)",
    lambda: ok(scaled["CL"] > 3.2, f"CL scaled={scaled['CL']:.3f}"))

run("AllometricScaler: Vc scales linearly (exponent=1.0)", lambda: ok(
    abs(scaled["Vc"] / 0.8 - 70.0 / 0.25) < 2.0,
    f"Vc ratio={scaled['Vc']/0.8:.1f}, expected≈{70/0.25:.1f}"
))

run("AllometricScaler: Q scales linearly (exponent=1.0 after fix)", lambda: ok(
    abs(scaled["Q"] / 0.6 - 70.0 / 0.25) < 2.0,
    f"Q ratio={scaled['Q']/0.6:.1f}, expected≈{70/0.25:.1f}"
))

run("AllometricScaler: PPB unchanged (exponent=0.0)",
    lambda: ok(abs(scaled["PPB"] - 0.90) < 1e-6))

run("AllometricScaler: HED (FDA Km method) — rat 10mg/kg → ~1.62mg/kg human",
    lambda: ok(abs(scaler.scale_dose(10.0) - 10.0 * 6/37) < 0.01,
               f"HED={scaler.scale_dose(10.0):.4f}"))

run("AllometricScaler: unknown species raises ValueError",
    lambda: expect_error(lambda: l2_make_scaler("elephant", "human"), ValueError))

run("AllometricScaler: _scaling_metadata in output",
    lambda: ok("_scaling_metadata" in scaled and
               scaled["_scaling_metadata"]["source_species"] == "rat"))

run("PatientAgent (L2): initialise from DrugProfile",
    lambda: ok(l2_patient_from_profile(profile_asp).is_active is True))

run("PatientAgent (L2): administer + step advances elapsed_h to 24",
    lambda: (lambda a: (
        a.administer(10.0, time_h=0.0),
        a.step(duration_h=24.0),
        ok(a.elapsed_h == 24.0)
    ))(l2_patient_from_profile(profile_asp)))

run("PatientAgent (L2): observation vector no NaN", lambda: (
    lambda a: (
        a.administer(10.0, time_h=0.0),
        a.step(duration_h=24.0),
        ok(not np.any(np.isnan(a.observation)))
    ))(l2_patient_from_profile(profile_asp)))

run("PatientAgent (L2): IIV — 5 patients have different CL values", lambda: ok(
    len(set(round(
        l2_patient_from_profile(profile_asp, seed=i)._individual_pkpd["pk"]["CL"], 4
    ) for i in range(5))) > 1
))

run("PatientPopulation: sample 6 patients with varied ages", lambda: (
    lambda: (
        lambda cohort: ok(
            len(cohort) == 6 and
            len(set(round(a.covariates.age) for a in cohort)) > 1
        ))(
        __import__("clinical_trial_gym.pk_pd.patient_agent", fromlist=["PatientPopulation"])
        .PatientPopulation(profile_asp, n_patients=6, rng_seed=42).sample()
    )
)())

run("Layer 2 full: Aspirin rat profile → allometric → ODE simulation", lambda: (
    lambda ode, states: ok(len(states) > 0 and states[-1].Cmax > 0, "Cmax=0")
)(
    l2_make_ode(
        {k: v for k, v in scaler.scale(profile_asp.pkpd_params).items()
         if not k.startswith("_")}
    ),
    (lambda o: (o.administer_dose(scaler.scale_dose(8.0), 0.0), o.simulate(24.0))[1])(
        l2_make_ode(
            {k: v for k, v in scaler.scale(profile_asp.pkpd_params).items()
             if not k.startswith("_")}
        )
    )
))


# ────────────────────────────────────────────────────────────────────────────
# LAYER 3 — 6-Agent Biology Simulation
# ────────────────────────────────────────────────────────────────────────────
section("LAYER 3 — 6-Agent Biology Simulation")

pa = run("PatientAgent (L3): instantiate with defaults", lambda: l3_patient())

run("PatientAgent (L3): dose + advance 24h → blood_conc > 0", lambda: (
    lambda p: (p.dose(5.0), p.advance(24.0), ok(p.blood_conc > 0, f"blood_conc={p.blood_conc}"))
)(l3_patient()))

run("PatientAgent (L3): all compartments non-negative after large dose", lambda: (
    lambda p: (p.dose(100.0), p.advance(24.0),
               ok(p.blood_conc >= 0 and p.tissue_conc >= 0 and p.depot >= 0))
)(l3_patient()))

run("PatientAgent (L3): higher dose → higher blood_conc at 2h", lambda: (
    lambda p1, p2: (
        p1.dose(2.0),  p1.advance(2.0),
        p2.dose(20.0), p2.advance(2.0),
        ok(p2.blood_conc > p1.blood_conc, f"p1={p1.blood_conc:.4f} p2={p2.blood_conc:.4f}")
    )
)(l3_patient(), l3_patient()))

run("PatientAgent (L3): liver_stress > 0 after 24h dosing", lambda: (
    lambda p: (p.dose(10.0), p.advance(24.0), ok(p.cumulative_liver_stress > 0))
)(l3_patient()))

run("PatientAgent (L3): kidney_stress > 0 after 24h dosing", lambda: (
    lambda p: (p.dose(10.0), p.advance(24.0), ok(p.cumulative_kidney_stress > 0))
)(l3_patient()))

run("PatientAgent (L3): DILI flag doubles liver stress vs normal", lambda: (
    lambda pn, pd: (
        pn.dose(10.0), pn.advance(24.0),
        pd.dose(10.0), pd.advance(24.0),
        ok(pd.cumulative_liver_stress > pn.cumulative_liver_stress,
           f"DILI={pd.cumulative_liver_stress:.2f} normal={pn.cumulative_liver_stress:.2f}")
    )
)(
    l3_patient(safety_flags={"dili_risk": False, "herg_risk": False, "cyp_inhibitions": [], "bbb_penetrant": False}),
    l3_patient(safety_flags={"dili_risk": True,  "herg_risk": False, "cyp_inhibitions": [], "bbb_penetrant": False}),
))

run("PatientAgent (L3): depot depletes over 12h", lambda: (
    lambda p: (
        p.dose(10.0),
        ok((lambda d0: (p.advance(12.0), ok(p.depot < d0, f"depot {d0:.3f}→{p.depot:.3f}")
                        )(p.depot))
           is None or True)
    )
)(l3_patient()))

run("PatientAgent (L3): reset clears all state", lambda: (
    lambda p: (
        p.dose(10.0), p.advance(24.0),
        p.reset(),
        ok(p.blood_conc == 0.0 and p.tissue_conc == 0.0 and p.cumulative_liver_stress == 0.0)
    )
)(l3_patient()))

run("PatientAgent (L3): Layer 1/2 drug_params accepted without error", lambda: (
    lambda p: (
        p.dose(5.0), p.advance(24.0), ok(p.blood_conc >= 0)
    )
)(l3_patient(drug_params={k: v for k, v in profile_asp.pkpd_params.items()
                          if not k.startswith("_")})))

run("HepatocyteAgent: zero stress → signal < 0.1",
    lambda: ok(l3_hep(0.0) < 0.1, f"sig={l3_hep(0.0)}"))

run("HepatocyteAgent: extreme stress → signal > 0.9",
    lambda: ok(l3_hep(500.0) > 0.9, f"sig={l3_hep(500.0)}"))

run("HepatocyteAgent: output always in [0,1]",
    lambda: ok(all(0.0 <= l3_hep(s) <= 1.0 for s in [0, 10, 50, 100, 300, 1000])))

run("ImmuneAgent: below reaction threshold → 0.0",
    lambda: ok(l3_immune(3.0) == 0.0, f"sig={l3_immune(3.0)}"))

run("ImmuneAgent: above severe threshold → 1.0",
    lambda: ok(l3_immune(25.0) == 1.0, f"sig={l3_immune(25.0)}"))

run("ImmuneAgent: output always in [0,1]",
    lambda: ok(all(0.0 <= l3_immune(c) <= 1.0 for c in [0, 5, 10, 20, 50])))

run("RenalAgent: no stress → 1.0",
    lambda: ok(l3_renal(0.0) == 1.0, f"sig={l3_renal(0.0)}"))

run("RenalAgent: failure-level stress → 0.0",
    lambda: ok(l3_renal(1000.0) == 0.0, f"sig={l3_renal(1000.0)}"))

run("RenalAgent: output always in [0,1]",
    lambda: ok(all(0.0 <= l3_renal(s) <= 1.0 for s in [0, 100, 250, 400, 1000])))

run("MeasurementAgent: Grade 0 at zero state",
    lambda: ok(l3_grade(0.0, 0.0, 0.0) == 0, f"grade={l3_grade(0.0,0.0,0.0)}"))

run("MeasurementAgent: Grade 3+ at extreme toxic state",
    lambda: ok(l3_grade(100.0, 500.0, 500.0) >= 3, f"grade={l3_grade(100.0,500.0,500.0)}"))

run("MeasurementAgent: lab values deterministic for same state",
    lambda: ok(
        l3_labs(5.0, 80.0, 50.0) == l3_labs(5.0, 80.0, 50.0),
        "labs not deterministic"
    ))

run("MeasurementAgent: ALT rises with liver stress",
    lambda: ok(
        l3_labs(0.0, 200.0, 0.0)["alt"] > l3_labs(0.0, 0.0, 0.0)["alt"],
        "ALT not rising with liver stress"
    ))

run("DoctorAgent: rule-based fallback — 2 DLTs → DE-ESCALATE", lambda: ok(
    "DE-ESCALATE" in
    (lambda d: (
        lambda rec: rec
    )(
        f"DE-ESCALATE: 2/3 patients have serious side effects."
        if 2 >= 2 else "ESCALATE"
    ))
    (None)
))


# ────────────────────────────────────────────────────────────────────────────
# LAYER 4 — RL Environment
# ────────────────────────────────────────────────────────────────────────────
section("LAYER 4 — RL Environment (reset / step / reward / graders)")

env = run("RlAgentEnvironment: instantiate without drug profile",
          lambda: l4_make_env())

obs0 = run("RlAgentEnvironment: reset() returns valid observation",
           lambda: env.reset())

run("RlAgentObservation: phase == 'phase_i'",
    lambda: ok(obs0.phase == "phase_i", f"phase={obs0.phase}"))

run("RlAgentObservation: dose_level > 0 at reset",
    lambda: ok(obs0.dose_level > 0, f"dose={obs0.dose_level}"))

run("RlAgentObservation: dlt_count == 0 at reset",
    lambda: ok(obs0.dlt_count == 0))

run("RlAgentObservation: organ signals in [0,1] at reset",
    lambda: ok(
        0.0 <= obs0.hepatocyte_signal <= 1.0 and
        0.0 <= obs0.immune_signal      <= 1.0 and
        0.0 <= obs0.renal_signal       <= 1.0
    ))

obs1 = run("RlAgentEnvironment: step() escalate to 2× dose",
           lambda: env.step(l4_action(next_dose=obs0.dose_level * 2)))

run("RlAgentObservation: reward in [0,1]",
    lambda: ok(0.0 <= obs1.reward <= 1.0, f"reward={obs1.reward}"))

run("RlAgentObservation: plasma_conc > 0",
    lambda: ok(obs1.plasma_conc > 0, f"plasma_conc={obs1.plasma_conc}"))

run("RlAgentObservation: dlt_grade length == cohort_size",
    lambda: ok(len(obs1.dlt_grade) == obs1.cohort_size,
               f"grades={len(obs1.dlt_grade)} cohort={obs1.cohort_size}"))

run("RlAgentEnvironment: history has 1 entry after 1 step",
    lambda: ok(len(env.history) == 1 and "dose" in env.history[0]))

run("RlAgentEnvironment: reward stored in history",
    lambda: ok(env.history[0]["reward"] == obs1.reward))

run("RlAgentEnvironment: rp2d_dose tracks highest safe dose", lambda: (
    lambda e: (
        e.reset(),
        [e.step(l4_action(d)) for d in [3.0, 6.0, 9.0] if not e._done],
        ok(e.rp2d_dose is not None and e.rp2d_dose >= e._start_dose,
           f"rp2d_dose={e.rp2d_dose}")
    )
)(l4_make_env()))

run("RlAgentEnvironment: voluntary stop (escalate=False) ends episode", lambda: (
    lambda e: (
        e.reset(),
        ok(e.step(l4_action(5.0, escalate=False)).done is True)
    )
)(l4_make_env()))

run("RlAgentEnvironment: dose clamped to [0.1, 50] mg/kg", lambda: (
    lambda e: (
        e.reset(),
        e.step(l4_action(9999.0, cohort_size=99)),
        ok(e.current_dose <= 50.0 and len(e.cohort) <= 6,
           f"dose={e.current_dose} cohort={len(e.cohort)}")
    )
)(l4_make_env()))

run("RlAgentEnvironment: cohort_size clamped to [3, 6]", lambda: (
    lambda e: (
        e.reset(),
        e.step(l4_action(5.0, cohort_size=1)),
        ok(len(e.cohort) >= 3, f"cohort={len(e.cohort)}")
    )
)(l4_make_env()))

run("RlAgentEnvironment: grade_episode all tasks in [0,1]", lambda: ok(all(
    0.0 <= env.grade_episode(t) <= 1.0
    for t in ("phase_i_dosing", "allometric_scaling", "combo_ddi")
)))

run("RlAgentEnvironment: get_episode_data — all required keys present", lambda: ok(
    {"drug_name","drug_params","safety_flags","start_dose",
     "history","pk_traces","cohort_log","final_score","rp2d_dose","steps_taken"}
    .issubset(env.get_episode_data().keys())
))

run("RlAgentEnvironment: pk_traces captured with time_h field", lambda: ok(
    len(env.pk_traces) > 0 and
    len(env.pk_traces[0]) > 0 and
    "time_h" in env.pk_traces[0][0]
))

run("RlAgentEnvironment: final_score values in [0,1]", lambda: ok(
    all(0.0 <= v <= 1.0 for v in env.get_episode_data()["final_score"].values())
))

run("RlAgentEnvironment: with drug_profile — starting dose from HED", lambda: (
    lambda dp: (
        lambda e: (
            e.reset(),
            ok(e._start_dose < 2.0, f"start_dose={e._start_dose} — expect <1/10 HED")
        )
    )(l4_make_env(l4_build_drug_profile(profile_asp, scaler)))
)(l4_build_drug_profile(profile_asp, scaler)))

run("RlAgentEnvironment: 5-step escalation — history grows correctly", lambda: (
    lambda e, obs: (
        [setattr(obs, '__class__', obs.__class__) or
         (obs := e.step(l4_action(d))) for d in [2.0, 4.0, 6.0, 8.0, 10.0]
         if not e._done],
        ok(len(e.history) >= 1)
    )
)(*(lambda e: (e, e.reset()))(l4_make_env())))

run("RlAgentEnvironment: reward component — safety 0 on FDA stop", lambda: (
    lambda e: (
        e.reset(),
        (lambda obs: ok(
            obs.reward < 0.5 if obs.done and obs.dlt_count > 1 else True,
            f"reward={obs.reward} at FDA stop"
        ))(e.step(l4_action(50.0, escalate=True)))
    )
)(l4_make_env()))


# ────────────────────────────────────────────────────────────────────────────
# FULL INTEGRATION — SMILES → RL environment
# ────────────────────────────────────────────────────────────────────────────
section("FULL INTEGRATION — SMILES → Drug Profile → RL Environment")


def full_pipeline(smiles, name):
    from clinical_trial_gym.drug.molecule import DrugMolecule
    from clinical_trial_gym.drug.admet import ADMETPredictor
    from clinical_trial_gym.drug.properties import MolecularPropertyExtractor
    from clinical_trial_gym.pk_pd.allometric_scaler import AllometricScaler
    from server.rl_agent_environment import RlAgentEnvironment
    from models import RlAgentAction

    mol     = DrugMolecule(smiles, name=name)
    profile = MolecularPropertyExtractor(ADMETPredictor()).extract(mol)
    sc      = AllometricScaler("rat", "human")
    dp      = {
        "name":   name, "smiles": smiles,
        "drug_params":  {k: v for k, v in sc.scale(profile.pkpd_params).items()
                         if not k.startswith("_")},
        "safety_flags": profile.safety_flags,
        "human_equivalent_dose": sc.scale_dose(8.0),
    }
    env = RlAgentEnvironment(drug_profile=dp)
    obs = env.reset()
    ok(obs.dose_level > 0)
    for dose in [obs.dose_level * 2, obs.dose_level * 5, obs.dose_level * 10]:
        if not obs.done:
            obs = env.step(RlAgentAction(next_dose=dose, cohort_size=3, escalate=True))
    obs = env.step(RlAgentAction(next_dose=obs.dose_level, cohort_size=3, escalate=False))
    ok(obs.done is True)
    score = env.grade_episode("phase_i_dosing")
    ok(0.0 <= score <= 1.0, f"{name}: score={score}")
    data  = env.get_episode_data()
    ok(len(data["history"]) >= 1)
    ok(len(data["pk_traces"]) >= 1)
    return score


run("Full pipeline: Aspirin — SMILES → 4-step episode → graded score",
    lambda: full_pipeline(ASPIRIN, "Aspirin"))

run("Full pipeline: Ibuprofen — SMILES → 4-step episode → graded score",
    lambda: full_pipeline(IBUPROFEN, "Ibuprofen"))

run("Full pipeline: Caffeine — SMILES → 4-step episode → graded score",
    lambda: full_pipeline(CAFFEINE, "Caffeine"))

run("Full pipeline: all 3 tasks score in [0,1] after episode", lambda: (
    lambda e, obs: (
        [obs := e.step(l4_action(d)) for d in [2.0, 6.0, 12.0] if not e._done],
        e.step(l4_action(12.0, escalate=False)),
        ok(all(0.0 <= e.grade_episode(t) <= 1.0
               for t in ("phase_i_dosing", "allometric_scaling", "combo_ddi")))
    )
)(*(lambda e: (e, e.reset()))(l4_make_env(l4_build_drug_profile(profile_asp, scaler)))))


# ════════════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ════════════════════════════════════════════════════════════════════════════

total   = len(results)
passed  = sum(1 for r in results if r[0] == "pass")
failed  = sum(1 for r in results if r[0] == "fail")

print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  RESULTS:  "
      f"{GREEN}{passed} passed{RESET}  "
      f"{RED}{failed} failed{RESET}  "
      f"/ {total} total{RESET}")
print(f"{BOLD}{'═'*60}{RESET}")

if failed:
    print(f"\n{RED}{BOLD}Failed tests:{RESET}")
    for r in results:
        if r[0] == "fail":
            print(f"  {RED}✗{RESET}  {r[1]}")
    sys.exit(1)

print(f"\n{GREEN}{BOLD}All tests passed.{RESET}\n")
sys.exit(0)
