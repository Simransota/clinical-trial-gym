"""
analysis.py — Layer 5: Analysis & Visualization
Goes in: /Users/simransota/meta/rl_agent/analysis.py (root folder)

What this file does:
  1. Runs a complete trial and collects all data at every step
  2. Plots PK concentration-time curves (how drug moves through body)
  3. Plots organ stress over time (liver, kidney, immune)
  4. Plots DLT grades per patient per step
  5. Plots reward curve (how well the agent did over time)
  6. Exports all data to CSV for further analysis
  7. Generates a full HTML dashboard using Plotly

Usage:
  python analysis.py
  python analysis.py --task phase_i_dosing
  python analysis.py --task allometric_scaling
  python analysis.py --task combo_ddi
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp, trapezoid
import requests
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
ENV_URL   = os.getenv("ENV_URL", "http://localhost:8000")
OUT_DIR   = "layer5_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dummy Layer 1 drug properties ────────────────────────────────────────────
DRUG = {
    "name":             "Aspirin-like compound",
    "smiles":           "CC(=O)Oc1ccccc1C(=O)O",
    "molecular_weight": 180.16,
    "half_life_hours":  4.6,
    "predicted_vd":     49.0,
    "cyp_inhibition":   0.15,
}

# ── 1. PK CURVE using scipy ODE solver ───────────────────────────────────────
# This is the real scipy integration (Layer 5 requirement)
# Models how drug concentration changes in blood over 24 hours

def pk_ode(t, y, ke, k12, k21):
    """
    2-compartment PK ODE system.
    y[0] = blood concentration (mg/L)
    y[1] = tissue concentration (mg/L)
    """
    blood, tissue = y
    d_blood  = -ke * blood - k12 * blood + k21 * tissue
    d_tissue =  k12 * blood - k21 * tissue
    return [d_blood, d_tissue]


def compute_pk_curve(dose_mg_per_kg: float, weight_kg: float = 70.0,
                     hepatic_factor: float = 1.0) -> dict:
    """
    Use scipy.integrate.solve_ivp to compute full PK curve.
    Returns time points and concentrations.
    """
    Vd  = 0.7 * weight_kg
    ke  = 0.15 * hepatic_factor
    k12 = 0.08
    k21 = 0.04

    initial_blood = (dose_mg_per_kg * weight_kg) / Vd
    y0 = [initial_blood, 0.0]
    t_span = (0, 48)  # 48 hours
    t_eval = np.linspace(0, 48, 500)

    sol = solve_ivp(pk_ode, t_span, y0, t_eval=t_eval,
                    args=(ke, k12, k21), method="RK45")

    cmax = max(sol.y[0])
    tmax = sol.t[np.argmax(sol.y[0])]
    auc  = trapezoid(sol.y[0], sol.t)
    t_half = np.log(2) / ke

    return {
        "time":       sol.t,
        "blood_conc": sol.y[0],
        "tissue_conc":sol.y[1],
        "cmax":       round(cmax, 4),
        "tmax":       round(tmax, 2),
        "auc":        round(auc, 2),
        "t_half":     round(t_half, 2),
        "dose":       dose_mg_per_kg,
    }


def plot_pk_curves(doses: list, output_path: str):
    """
    Plot PK concentration-time curves for multiple doses.
    Shows how drug concentration in blood changes over 48 hours.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(doses)))

    for i, dose in enumerate(doses):
        pk = compute_pk_curve(dose)
        axes[0].plot(pk["time"], pk["blood_conc"],
                     color=colors[i], label=f"{dose} mg/kg",
                     linewidth=2)
        axes[1].plot(pk["time"], pk["tissue_conc"],
                     color=colors[i], label=f"{dose} mg/kg",
                     linewidth=2)

    # Blood compartment
    axes[0].set_title("Blood Concentration vs Time (Layer 2 PK)", fontsize=13)
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Concentration (mg/L)")
    axes[0].legend(title="Dose", fontsize=8)
    axes[0].axhline(y=5.0,  color="orange", linestyle="--",
                    alpha=0.7, label="Immune threshold")
    axes[0].axhline(y=20.0, color="red",    linestyle="--",
                    alpha=0.7, label="Toxicity threshold")
    axes[0].grid(alpha=0.3)

    # Tissue compartment
    axes[1].set_title("Tissue Concentration vs Time", fontsize=13)
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Concentration (mg/L)")
    axes[1].legend(title="Dose", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Layer 1→2 PK Curves: {DRUG['name']} (MW={DRUG['molecular_weight']})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── 2. RUN TRIAL & COLLECT DATA ───────────────────────────────────────────────

def run_trial_collect_data(task: str = "phase_i_dosing") -> pd.DataFrame:
    """
    Run a complete trial via the environment HTTP API.
    Collect all observations and rewards at every step.
    Returns a pandas DataFrame with all trial data.
    """
    print(f"\n  Running trial for task: {task}")

    # Reset
    resp = requests.post(f"{ENV_URL}/reset", timeout=30)
    data = resp.json()
    obs  = data.get("observation", data)

    rows = []

    # Dose escalation schedule (same as test_dummy.py)
    dose_schedule = [
        (1.0,  3, True),
        (2.0,  3, True),
        (4.0,  3, True),
        (8.0,  3, True),
        (12.0, 3, True),
        (18.0, 3, True),
        (25.0, 3, False),
    ]

    for step, (dose, cohort, escalate) in enumerate(dose_schedule, 1):
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": {
                "next_dose":   dose,
                "cohort_size": cohort,
                "escalate":    escalate,
            }},
            timeout=30,
        )
        data = resp.json()
        obs  = data.get("observation", data)

        # Also compute PK curve for this dose
        pk = compute_pk_curve(dose)

        rows.append({
            "step":               step,
            "dose_mg_per_kg":     dose,
            "cohort_size":        obs.get("cohort_size", cohort),
            "plasma_conc_env":    obs.get("plasma_conc", 0),
            "plasma_conc_scipy":  pk["cmax"],
            "cmax":               pk["cmax"],
            "auc":                pk["auc"],
            "tmax":               pk["tmax"],
            "dlt_count":          obs.get("dlt_count", 0),
            "dlt_grades":         str(obs.get("dlt_grade", [])),
            "hepatocyte_signal":  obs.get("hepatocyte_signal", 0),
            "immune_signal":      obs.get("immune_signal", 0),
            "renal_signal":       obs.get("renal_signal", 1),
            "doctor_rec":         obs.get("doctor_recommendation", ""),
            "reward":             data.get("reward", 0),
            "done":               data.get("done", False),
            "escalate":           escalate,
        })

        print(f"  Step {step}: dose={dose}mg/kg, "
              f"DLTs={obs.get('dlt_count',0)}, "
              f"reward={data.get('reward',0):.3f}, "
              f"done={data.get('done',False)}")

        if data.get("done", False):
            break

    df = pd.DataFrame(rows)
    return df


# ── 3. MATPLOTLIB PLOTS ───────────────────────────────────────────────────────

def plot_trial_dashboard(df: pd.DataFrame, task: str, output_path: str):
    """
    4-panel matplotlib dashboard showing:
    1. Dose escalation + DLT count over steps
    2. Organ signals over steps
    3. Reward curve
    4. PK summary (Cmax vs dose)
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    steps = df["step"].values

    # ── Panel 1: Dose escalation + DLT ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()

    bars = ax1.bar(steps, df["dose_mg_per_kg"], color="#4472C4",
                   alpha=0.7, label="Dose (mg/kg)")
    ax1_twin.plot(steps, df["dlt_count"], "ro-",
                  linewidth=2, markersize=8, label="DLT count")
    ax1_twin.axhline(y=1, color="orange", linestyle="--",
                     alpha=0.7, linewidth=1.5, label="DLT=1 threshold")
    ax1_twin.axhline(y=2, color="red", linestyle="--",
                     alpha=0.7, linewidth=1.5, label="DLT=2 stop")

    ax1.axvline(x=5, color="green", linestyle=":",
                alpha=0.5, linewidth=2, label="True RP2D (12mg/kg)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Dose (mg/kg)", color="#4472C4")
    ax1_twin.set_ylabel("DLT Count", color="red")
    ax1.set_title("Dose Escalation + DLT Count", fontweight="bold")
    ax1.set_xticks(steps)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Organ signals ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, df["hepatocyte_signal"], "s-",
             color="#E74C3C", linewidth=2, markersize=7, label="Liver stress")
    ax2.plot(steps, df["immune_signal"],     "^-",
             color="#F39C12", linewidth=2, markersize=7, label="Immune reaction")
    ax2.plot(steps, df["renal_signal"],      "o-",
             color="#27AE60", linewidth=2, markersize=7, label="Kidney function")

    ax2.axhline(y=0.7, color="#E74C3C", linestyle="--", alpha=0.4)
    ax2.axhline(y=0.5, color="#27AE60", linestyle="--", alpha=0.4)
    ax2.fill_between(steps, 0.7, 1.0,
                     color="#E74C3C", alpha=0.05, label="Liver danger zone")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Signal (0=bad for liver, 1=good for kidney)")
    ax2.set_title("Organ Signals Over Time", fontweight="bold")
    ax2.set_xticks(steps)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Reward curve ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, df["reward"], "D-",
             color="#8E44AD", linewidth=2.5, markersize=9, label="Step reward")
    ax3.fill_between(steps, 0, df["reward"],
                     color="#8E44AD", alpha=0.15)

    # Mark the best step
    best_step = df.loc[df["reward"].idxmax(), "step"]
    best_rew  = df["reward"].max()
    ax3.annotate(f"Peak\n{best_rew:.3f}",
                 xy=(best_step, best_rew),
                 xytext=(best_step + 0.3, best_rew - 0.05),
                 fontsize=9, color="#8E44AD",
                 arrowprops=dict(arrowstyle="->", color="#8E44AD"))

    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward (0.0 – 1.0)")
    ax3.set_title("Reward Curve", fontweight="bold")
    ax3.set_xticks(steps)
    ax3.axhline(y=df["reward"].mean(), color="gray", linestyle="--",
                alpha=0.6, label=f"Average: {df['reward'].mean():.3f}")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # ── Panel 4: Cmax vs Dose (PK summary) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(df["dose_mg_per_kg"], df["plasma_conc_env"],
                color="#2980B9", s=100, zorder=5, label="Env Cmax (ODE)")
    ax4.scatter(df["dose_mg_per_kg"], df["cmax"],
                color="#E74C3C", s=100, marker="^", zorder=5,
                label="scipy Cmax (Layer 2)")

    # Fit a line through scipy values
    z = np.polyfit(df["dose_mg_per_kg"], df["cmax"], 1)
    p = np.poly1d(z)
    dose_range = np.linspace(df["dose_mg_per_kg"].min(),
                             df["dose_mg_per_kg"].max(), 100)
    ax4.plot(dose_range, p(dose_range), "--",
             color="#E74C3C", alpha=0.5, linewidth=1.5, label="Linear fit")

    ax4.axvline(x=12.0, color="green", linestyle=":",
                linewidth=2, label="True RP2D (12 mg/kg)")
    ax4.set_xlabel("Dose (mg/kg)")
    ax4.set_ylabel("Peak Concentration Cmax (mg/L)")
    ax4.set_title("PK Summary: Cmax vs Dose", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    fig.suptitle(
        f"rxgym Layer 5 Dashboard — Task: {task}\n"
        f"Drug: {DRUG['name']}  |  Avg reward: {df['reward'].mean():.3f}  |  "
        f"Steps: {len(df)}",
        fontsize=13, fontweight="bold"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ── 4. PLOTLY INTERACTIVE HTML DASHBOARD ─────────────────────────────────────

def plot_plotly_dashboard(df: pd.DataFrame, task: str, output_path: str):
    """
    Interactive Plotly dashboard saved as standalone HTML.
    Includes hover tooltips, zoom, and pan.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  Plotly not installed — skipping HTML dashboard")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Dose Escalation & DLT Count",
            "Organ Signals Over Steps",
            "Reward Curve",
            "PK Summary: Cmax vs Dose",
        ],
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
    )

    steps = df["step"].values

    # Panel 1: Dose + DLT
    fig.add_trace(go.Bar(
        x=steps, y=df["dose_mg_per_kg"],
        name="Dose (mg/kg)", marker_color="steelblue", opacity=0.7,
        hovertemplate="Step %{x}<br>Dose: %{y} mg/kg<extra></extra>",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps, y=df["dlt_count"],
        name="DLT count", mode="lines+markers",
        line=dict(color="red", width=2),
        marker=dict(size=8),
        hovertemplate="Step %{x}<br>DLTs: %{y}<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    # Panel 2: Organ signals
    fig.add_trace(go.Scatter(
        x=steps, y=df["hepatocyte_signal"],
        name="Liver stress", mode="lines+markers",
        line=dict(color="#E74C3C", width=2),
        hovertemplate="Step %{x}<br>Liver: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=steps, y=df["immune_signal"],
        name="Immune", mode="lines+markers",
        line=dict(color="#F39C12", width=2),
        hovertemplate="Step %{x}<br>Immune: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=steps, y=df["renal_signal"],
        name="Kidney", mode="lines+markers",
        line=dict(color="#27AE60", width=2),
        hovertemplate="Step %{x}<br>Kidney: %{y:.3f}<extra></extra>",
    ), row=1, col=2)

    # Panel 3: Reward
    fig.add_trace(go.Scatter(
        x=steps, y=df["reward"],
        name="Reward", mode="lines+markers",
        fill="tozeroy", fillcolor="rgba(142,68,173,0.1)",
        line=dict(color="#8E44AD", width=2.5),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate=(
            "Step %{x}<br>Reward: %{y:.3f}<br>"
            "Dose: " + df["dose_mg_per_kg"].astype(str) + " mg/kg<extra></extra>"
        ),
    ), row=2, col=1)

    # Panel 4: Cmax vs Dose
    fig.add_trace(go.Scatter(
        x=df["dose_mg_per_kg"], y=df["cmax"],
        name="scipy Cmax", mode="markers",
        marker=dict(size=12, color="#E74C3C", symbol="triangle-up"),
        hovertemplate="Dose: %{x} mg/kg<br>Cmax: %{y:.3f} mg/L<extra></extra>",
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=df["dose_mg_per_kg"], y=df["plasma_conc_env"],
        name="Env Cmax", mode="markers",
        marker=dict(size=12, color="#2980B9"),
        hovertemplate="Dose: %{x} mg/kg<br>Cmax: %{y:.3f} mg/L<extra></extra>",
    ), row=2, col=2)

    # Vertical line at RP2D
    fig.add_vline(x=12.0, line_dash="dot", line_color="green",
                  annotation_text="RP2D", row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"rxgym Layer 5 Interactive Dashboard — {task}<br>"
                 f"<sub>{DRUG['name']} | Avg reward: {df['reward'].mean():.3f}</sub>",
            font=dict(size=16),
        ),
        height=750,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
    )

    fig.write_html(output_path)
    print(f"  Saved: {output_path}")


# ── 5. CSV EXPORT ─────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, output_path: str):
    """Export full trial data to CSV for further analysis."""
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


# ── 6. PK SUMMARY TABLE ───────────────────────────────────────────────────────

def print_pk_summary(df: pd.DataFrame):
    """Print a clean PK summary table to terminal."""
    print("\n" + "="*70)
    print("  LAYER 5 — PK/PD Summary Table")
    print("="*70)
    print(f"  {'Step':>4}  {'Dose':>8}  {'Cmax(env)':>10}  "
          f"{'Cmax(scipy)':>12}  {'AUC':>8}  {'DLTs':>5}  {'Reward':>8}")
    print("  " + "-"*66)
    for _, row in df.iterrows():
        print(f"  {int(row['step']):>4}  "
              f"{row['dose_mg_per_kg']:>7.1f}  "
              f"{row['plasma_conc_env']:>10.3f}  "
              f"{row['cmax']:>12.3f}  "
              f"{row['auc']:>8.1f}  "
              f"{int(row['dlt_count']):>5}  "
              f"{row['reward']:>8.3f}")
    print("  " + "-"*66)
    print(f"  Average reward: {df['reward'].mean():.3f}")
    print(f"  Best step:      Step {df.loc[df['reward'].idxmax(),'step']} "
          f"at {df.loc[df['reward'].idxmax(),'dose_mg_per_kg']} mg/kg")
    print(f"  True RP2D:      12.0 mg/kg")
    print("="*70)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global ENV_URL
    parser = argparse.ArgumentParser(description="rxgym Layer 5 Analysis")
    parser.add_argument("--task", default="phase_i_dosing",
                        choices=["phase_i_dosing", "allometric_scaling", "combo_ddi"])
    parser.add_argument("--env-url", default=ENV_URL)
    args = parser.parse_args()

    ENV_URL = args.env_url
    task    = args.task
    ts      = datetime.now().strftime("%H%M%S")

    print("\n" + "="*55)
    print("  rxgym — Layer 5: Analysis & Visualization")
    print(f"  Task: {task}")
    print(f"  Drug: {DRUG['name']}")
    print("="*55)

    # ── Step 1: Plot PK curves for multiple doses ────────────────────────────
    print("\n[1/5] Generating PK curves (scipy ODE)...")
    pk_path = f"{OUT_DIR}/pk_curves_{ts}.png"
    plot_pk_curves(
        doses=[1.0, 4.0, 8.0, 12.0, 18.0, 25.0],
        output_path=pk_path,
    )

    # ── Step 2: Run trial and collect data ───────────────────────────────────
    print("\n[2/5] Running trial via environment API...")
    try:
        df = run_trial_collect_data(task)
    except Exception as e:
        print(f"  ERROR: Could not connect to environment at {ENV_URL}")
        print(f"  Make sure server is running: uv run uvicorn server.app:app --port 8000")
        print(f"  Error details: {e}")
        sys.exit(1)

    # ── Step 3: PK summary table ─────────────────────────────────────────────
    print("\n[3/5] PK/PD summary...")
    print_pk_summary(df)

    # ── Step 4: Matplotlib dashboard ─────────────────────────────────────────
    print("\n[4/5] Generating matplotlib dashboard...")
    mpl_path = f"{OUT_DIR}/trial_dashboard_{task}_{ts}.png"
    plot_trial_dashboard(df, task, mpl_path)

    # ── Step 5: Plotly HTML dashboard ────────────────────────────────────────
    print("\n[5/5] Generating Plotly interactive dashboard...")
    html_path = f"{OUT_DIR}/trial_dashboard_{task}_{ts}.html"
    plot_plotly_dashboard(df, task, html_path)

    # ── Step 6: Export CSV ───────────────────────────────────────────────────
    csv_path = f"{OUT_DIR}/trial_data_{task}_{ts}.csv"
    export_csv(df, csv_path)

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ALL DONE — outputs saved to ./{OUT_DIR}/")
    print(f"{'='*55}")
    print(f"  PK curves:           {pk_path}")
    print(f"  Trial dashboard:     {mpl_path}")
    print(f"  Interactive HTML:    {html_path}  ← open in browser")
    print(f"  Raw data CSV:        {csv_path}")
    print(f"{'='*55}")
    print(f"\n  Open the HTML file in your browser:")
    print(f"  open {html_path}")


if __name__ == "__main__":
    main()
