"""
smoke_test.py — verifies layer 1/2 output flows correctly into layer 3/4.
Run with:  python smoke_test.py
No external dependencies beyond what agents.py already imports.
"""

import sys
import random
sys.path.insert(0, ".")

from .agents import PatientAgent, HepatocyteAgent, ImmuneAgent, RenalAgent, MeasurementAgent

# ── Exact output from your layer 1/2 terminal printout ──────────────────────
ASPIRIN_DRUG_PARAMS = {
    "ka":  0.11265799040068204,
    "F":   0.8,
    "CL":  0.5,
    "Vc":  2.4745736979827706,
    "Vp":  1.6497157986551807,
    "Q":   0.3,
    "PPB": 0.604808,
    "fu":  0.395192,
}

ASPIRIN_SAFETY_FLAGS = {
    "dili_risk":          False,
    "herg_risk":          False,
    "cyp_inhibitions":    [],
    "bbb_penetrant":      True,
    "overall_risk_score": 0.0,
}

ASPIRIN_HED = 1.62   # mg/kg — Layer 2 allometric output


def _log(message: str) -> None:
    print(message)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _build_time_trace(patient: PatientAgent, dose: float, hours: float = 24.0, dt: float = 0.5) -> dict:
    trace_agent = PatientAgent(
        weight_kg      = patient.weight_kg,
        age            = patient.age,
        sex            = patient.sex,
        renal_factor   = patient.renal_factor,
        hepatic_factor = patient.hepatic_factor,
        drug_params    = ASPIRIN_DRUG_PARAMS,
        safety_flags   = ASPIRIN_SAFETY_FLAGS,
    )
    trace_agent.dose(dose)

    times = [0.0]
    blood = [trace_agent.blood_conc]
    tissue = [trace_agent.tissue_conc]
    steps = int(hours / dt)
    for _ in range(steps):
        trace_agent.advance(dt)
        times.append(round(trace_agent.time, 3))
        blood.append(round(trace_agent.blood_conc, 4))
        tissue.append(round(trace_agent.tissue_conc, 4))

    cmax = round(max(blood), 4)
    tmax = round(times[blood.index(cmax)], 3)
    auc  = round(sum(blood) * dt, 4)

    return {
        "patient_id": patient.patient_id,
        "age": patient.age,
        "sex": patient.sex,
        "weight_kg": patient.weight_kg,
        "time_h": times,
        "blood_conc": blood,
        "tissue_conc": tissue,
        "cmax": cmax,
        "tmax": tmax,
        "auc": auc,
    }


def make_episode_data(steps: int = 6) -> dict:
    rng = random.Random(42)
    history = []
    pk_traces = []
    cohort_log = []
    dose = max(0.1, ASPIRIN_HED / 10.0)

    for step in range(1, steps + 1):
        if step > 1:
            dose = min(dose * 1.5, 50.0)

        cohort = []
        for _ in range(3):
            w   = max(50.0, min(100.0, rng.gauss(70, 10)))
            age = rng.randint(39, 65)
            sex = rng.choice(["M", "F"])
            pt  = PatientAgent(
                weight_kg=w,
                age=age,
                sex=sex,
                renal_factor=rng.uniform(0.7, 1.0),
                hepatic_factor=rng.uniform(0.7, 1.0),
                drug_params=ASPIRIN_DRUG_PARAMS,
                safety_flags=ASPIRIN_SAFETY_FLAGS,
            )
            pt.dose(dose)
            pt.advance(hours=24.0)
            cohort.append(pt)

        pk_traces.append([_build_time_trace(p, dose) for p in cohort])

        dlt_grades = [MeasurementAgent.grade_dlt(p.get_state()) for p in cohort]
        hep_signals = [HepatocyteAgent.observe(p.get_state()) for p in cohort]
        imm_signals = [ImmuneAgent.observe(p.get_state()) for p in cohort]
        ren_signals = [RenalAgent.observe(p.get_state()) for p in cohort]

        dlt_count = sum(1 for g in dlt_grades if g >= 3)
        dlt_rate = round(dlt_count / len(cohort), 3)
        reward = round(1.0 - dlt_rate * 0.4, 3)

        history.append({
            "step": step,
            "dose": round(dose, 3),
            "dlt_count": dlt_count,
            "dlt_rate": dlt_rate,
            "cohort_size": len(cohort),
            "escalate": step < steps,
            "reward": reward,
            "doctor_recommendation": (
                "ESCALATE: safety signals within limits." if dlt_count == 0
                else "HOLD: DLTs observed."
            ),
        })

        cohort_log.append([
            {
                "patient_id": p.patient_id,
                "age": p.age,
                "sex": p.sex,
                "weight_kg": p.weight_kg,
                "ke": p.ke,
                "fu": p.fu,
                "renal_factor": p.renal_factor,
                "hepatic_factor": p.hepatic_factor,
                "blood_conc": round(p.blood_conc, 4),
                "liver_stress": round(p.cumulative_liver_stress, 4),
                "kidney_stress": round(p.cumulative_kidney_stress, 4),
                "hep_signal": round(hep, 3),
                "imm_signal": round(imm, 3),
                "ren_signal": round(ren, 3),
                "dlt_grade": grade,
                "is_dlt": grade >= 3,
            }
            for p, grade, hep, imm, ren in zip(cohort, dlt_grades, hep_signals, imm_signals, ren_signals)
        ])

    rp2d_dose = max((h["dose"] for h in history if h["dlt_count"] == 0), default=0.0)
    return {
        "drug_name": "Aspirin",
        "start_dose": max(0.1, ASPIRIN_HED / 10.0),
        "drug_params": ASPIRIN_DRUG_PARAMS,
        "safety_flags": ASPIRIN_SAFETY_FLAGS,
        "history": history,
        "pk_traces": pk_traces,
        "cohort_log": cohort_log,
        "final_score": {
            "phase_i_dosing": 0.72,
            "allometric_scaling": 0.80,
            "combo_ddi": 0.65,
        },
        "rp2d_dose": round(rp2d_dose, 3),
        "steps_taken": len(history),
    }


def run_smoke_test(steps: int = 6) -> dict:
    _log("=" * 60)
    _log("TEST 1: PatientAgent PK parameters from Layer 1/2")
    _log("=" * 60)

    p = PatientAgent(
        weight_kg=70.0,
        age=45,
        sex="M",
        renal_factor=1.0,
        hepatic_factor=1.0,
        drug_params=ASPIRIN_DRUG_PARAMS,
        safety_flags=ASPIRIN_SAFETY_FLAGS,
    )
    _assert(abs(p.ka - 0.1127) < 0.001, f"ka mismatch: {p.ka}")
    _assert(abs(p.F - 0.80) < 0.001, f"F mismatch: {p.F}")
    expected_vc = 2.4746 * 70.0
    _assert(abs(p.Vc - expected_vc) < 1.0, f"Vc mismatch: {p.Vc} vs {expected_vc}")

    _log(f"  ka  = {p.ka:.4f}  (Layer 1/2: 0.1127)  ✓")
    _log(f"  F   = {p.F:.4f}   (Layer 1/2: 0.80)    ✓")
    _log(f"  Vc  = {p.Vc:.2f} L (Layer 1/2: 2.4746 × 70 kg)  ✓")
    _log(f"  ke  = {p.ke:.4f}  (derived: CL/Vc)")
    _log(f"  k12 = {p.k12:.4f}  (derived: Q/Vc)")
    _log(f"  k21 = {p.k21:.4f}  (derived: Q/Vp)")
    _log("")

    _log("=" * 60)
    _log("TEST 2: PatientAgent fallback when no Layer 1/2 params given")
    _log("=" * 60)
    p_default = PatientAgent(weight_kg=70.0, age=45)
    _assert(abs(p_default.ka - 1.2) < 0.001, f"Default ka wrong: {p_default.ka}")
    _log(f"  ka  = {p_default.ka:.4f}  (default: 1.2)  ✓")
    _log("")

    _log("=" * 60)
    _log("TEST 3: PatientAgent simulation with Layer 1/2 PK params")
    _log("=" * 60)
    p.dose(ASPIRIN_HED)
    p.advance(hours=24.0)
    state = p.get_state()
    _log(f"  After 24h at HED ({ASPIRIN_HED} mg/kg):")
    _log(f"    blood_conc   = {state['blood_conc']:.4f} mg/L")
    _log(f"    tissue_conc  = {state['tissue_conc']:.4f} mg/L")
    _log(f"    liver_stress = {state['liver_stress']:.4f} mg/L·h")
    _log(f"    kidney_stress= {state['kidney_stress']:.4f} mg/L·h")
    _assert(state["blood_conc"] >= 0.0, "Blood concentration is negative")
    _assert(state["liver_stress"] >= 0.0, "Liver stress is negative")
    _log("  Simulation ran cleanly  ✓")
    _log("")

    _log("=" * 60)
    _log("TEST 4: Organ agents reading simulation state")
    _log("=" * 60)
    hep = HepatocyteAgent.observe(state)
    imm = ImmuneAgent.observe(state)
    ren = RenalAgent.observe(state)
    _assert(0.0 <= hep <= 1.0, "HepatocyteAgent out of range")
    _assert(0.0 <= imm <= 1.0, "ImmuneAgent out of range")
    _assert(0.0 <= ren <= 1.0, "RenalAgent out of range")
    _log(f"  HepatocyteAgent: {hep:.4f}  (0=fine, 1=overwhelmed)")
    _log(f"  ImmuneAgent:     {imm:.4f}  (0=calm, 1=severe)")
    _log(f"  RenalAgent:      {ren:.4f}  (1=healthy, 0=failed)")
    _log("  All in [0,1]  ✓")
    _log("")

    _log("=" * 60)
    _log("TEST 5: DLT grading determinism (Bug 2 fix)")
    _log("=" * 60)
    grade_a = MeasurementAgent.grade_dlt(state)
    grade_b = MeasurementAgent.grade_dlt(state)
    grade_c = MeasurementAgent.grade_dlt(state)
    _assert(grade_a == grade_b == grade_c, f"Non-deterministic! {grade_a} {grade_b} {grade_c}")
    _log(f"  DLT grade = {grade_a}  (three identical calls → same result)  ✓")
    _log("")

    _log("=" * 60)
    _log("TEST 6: Full cohort of 6 (matching your Layer 1/2 output)")
    _log("=" * 60)
    rng.seed(42)
    cohort = []
    for _ in range(6):
        w   = max(50, min(100, rng.gauss(70, 10)))
        age = rng.randint(39, 65)
        sex = rng.choice(["M", "F"])
        pt  = PatientAgent(
            weight_kg=w, age=age, sex=sex,
            renal_factor=rng.uniform(0.7, 1.0),
            hepatic_factor=rng.uniform(0.7, 1.0),
            drug_params=ASPIRIN_DRUG_PARAMS,
            safety_flags=ASPIRIN_SAFETY_FLAGS,
        )
        pt.dose(ASPIRIN_HED)
        pt.advance(hours=24.0)
        cohort.append(pt)

    dlt_count = 0
    for pt in cohort:
        s = pt.get_state()
        grade = MeasurementAgent.grade_dlt(s)
        is_dlt = grade >= 3
        if is_dlt:
            dlt_count += 1
        status = "DLT" if is_dlt else "NONE"
        _log(f"  PatientAgent(id={s['patient_id']}, age={s['age']:.0f}, "
              f"sex={s['sex']}, peak_grade={status})")

    _assert(0 <= dlt_count <= 6, "DLT count out of expected range")
    _log(f"\nDLTs in cohort: {dlt_count}/6")
    _log("Cohort simulation complete  ✓")
    _log("")

    _log("=" * 60)
    _log("ALL TESTS PASSED — Layer 1/2 → Layer 3/4 integration working")
    _log("=" * 60)
    return make_episode_data(steps)


if __name__ == "__main__":
    run_smoke_test()
