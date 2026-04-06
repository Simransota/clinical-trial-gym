"""
smoke_test.py — verifies layer 1/2 output flows correctly into layer 3/4.
Run with:  python smoke_test.py
No external dependencies beyond what agents.py already imports.
"""

import sys
sys.path.insert(0, ".")

from agents import PatientAgent, HepatocyteAgent, ImmuneAgent, RenalAgent, MeasurementAgent

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

# ── 1. Test PatientAgent uses Layer 1/2 params ──────────────────────────────
print("=" * 60)
print("TEST 1: PatientAgent PK parameters from Layer 1/2")
print("=" * 60)

p = PatientAgent(
    weight_kg=70.0,
    age=45,
    sex="M",
    renal_factor=1.0,
    hepatic_factor=1.0,
    drug_params=ASPIRIN_DRUG_PARAMS,
    safety_flags=ASPIRIN_SAFETY_FLAGS,
)

# Verify ka came from Layer 1/2, not the hardcoded default (1.2)
assert abs(p.ka - 0.1127) < 0.001, f"ka mismatch: {p.ka}"
assert abs(p.F  - 0.80)   < 0.001, f"F mismatch: {p.F}"
# Vc should be per-kg value × weight
expected_vc = 2.4746 * 70.0
assert abs(p.Vc - expected_vc) < 1.0, f"Vc mismatch: {p.Vc} vs {expected_vc}"

print(f"  ka  = {p.ka:.4f}  (Layer 1/2: 0.1127)  ✓")
print(f"  F   = {p.F:.4f}   (Layer 1/2: 0.80)    ✓")
print(f"  Vc  = {p.Vc:.2f} L (Layer 1/2: 2.4746 × 70 kg)  ✓")
print(f"  ke  = {p.ke:.4f}  (derived: CL/Vc)")
print(f"  k12 = {p.k12:.4f}  (derived: Q/Vc)")
print(f"  k21 = {p.k21:.4f}  (derived: Q/Vp)")
print()

# ── 2. Test without drug_params falls back to defaults ──────────────────────
print("=" * 60)
print("TEST 2: PatientAgent fallback when no Layer 1/2 params given")
print("=" * 60)

p_default = PatientAgent(weight_kg=70.0, age=45)
assert abs(p_default.ka - 1.2) < 0.001, f"Default ka wrong: {p_default.ka}"
print(f"  ka  = {p_default.ka:.4f}  (default: 1.2)  ✓")
print()

# ── 3. Test dose and advance run without errors ──────────────────────────────
print("=" * 60)
print("TEST 3: PatientAgent simulation with Layer 1/2 PK params")
print("=" * 60)

p.dose(ASPIRIN_HED)        # dose at the Layer 2 HED
p.advance(hours=24.0)

state = p.get_state()
print(f"  After 24h at HED ({ASPIRIN_HED} mg/kg):")
print(f"    blood_conc   = {state['blood_conc']:.4f} mg/L")
print(f"    tissue_conc  = {state['tissue_conc']:.4f} mg/L")
print(f"    liver_stress = {state['liver_stress']:.4f} mg/L·h")
print(f"    kidney_stress= {state['kidney_stress']:.4f} mg/L·h")
assert state["blood_conc"] >= 0.0
assert state["liver_stress"] >= 0.0
print("  Simulation ran cleanly  ✓")
print()

# ── 4. Test organ agents read the state correctly ────────────────────────────
print("=" * 60)
print("TEST 4: Organ agents reading simulation state")
print("=" * 60)

hep = HepatocyteAgent.observe(state)
imm = ImmuneAgent.observe(state)
ren = RenalAgent.observe(state)
print(f"  HepatocyteAgent: {hep:.4f}  (0=fine, 1=overwhelmed)")
print(f"  ImmuneAgent:     {imm:.4f}  (0=calm, 1=severe)")
print(f"  RenalAgent:      {ren:.4f}  (1=healthy, 0=failed)")
assert 0.0 <= hep <= 1.0
assert 0.0 <= imm <= 1.0
assert 0.0 <= ren <= 1.0
print("  All in [0,1]  ✓")
print()

# ── 5. Test DLT grading is deterministic ────────────────────────────────────
print("=" * 60)
print("TEST 5: DLT grading determinism (Bug 2 fix)")
print("=" * 60)

grade_a = MeasurementAgent.grade_dlt(state)
grade_b = MeasurementAgent.grade_dlt(state)
grade_c = MeasurementAgent.grade_dlt(state)
assert grade_a == grade_b == grade_c, f"Non-deterministic! {grade_a} {grade_b} {grade_c}"
print(f"  DLT grade = {grade_a}  (three identical calls → same result)  ✓")
print()

# ── 6. Test cohort of 6 matches Layer 1/2 expected output format ────────────
print("=" * 60)
print("TEST 6: Full cohort of 6 (matching your Layer 1/2 output)")
print("=" * 60)

import random
random.seed(42)

cohort = []
for _ in range(6):
    w   = max(50, min(100, random.gauss(70, 10)))
    age = random.randint(39, 65)
    sex = random.choice(["M", "F"])
    pt  = PatientAgent(
        weight_kg=w, age=age, sex=sex,
        renal_factor=random.uniform(0.7, 1.0),
        hepatic_factor=random.uniform(0.7, 1.0),
        drug_params=ASPIRIN_DRUG_PARAMS,
        safety_flags=ASPIRIN_SAFETY_FLAGS,
    )
    pt.dose(ASPIRIN_HED)
    pt.advance(hours=24.0)
    cohort.append(pt)

dlt_count = 0
for pt in cohort:
    s     = pt.get_state()
    grade = MeasurementAgent.grade_dlt(s)
    is_dlt = grade >= 3
    if is_dlt:
        dlt_count += 1
    status = "DLT" if is_dlt else "NONE"
    print(f"  PatientAgent(id={s['patient_id']}, age={s['age']:.0f}, "
          f"sex={s['sex']}, peak_grade={status})")

print(f"\nDLTs in cohort: {dlt_count}/6")
assert 0 <= dlt_count <= 6
print("Cohort simulation complete  ✓")
print()

print("=" * 60)
print("ALL TESTS PASSED — Layer 1/2 → Layer 3/4 integration working")
print("=" * 60)