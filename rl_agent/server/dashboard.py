"""
Gradio dashboard for RxGym — visual trial simulation for evaluators.

Mounted on the FastAPI app so it shares the same port on HF Spaces.
Provides preset drug configs + custom SMILES input, runs a full episode,
and renders 6-panel clinical trial visualizations.
"""

from __future__ import annotations

import io
import base64
import traceback

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# ── Preset drugs for one-click evaluation ────────────────────────────────────

PRESET_DRUGS = {
    "Acetaminophen (Easy — Phase I Dosing)": {
        "name": "Acetaminophen",
        "smiles": "CC(=O)NC1=CC=C(O)C=C1",
        "source_species": "rat",
        "animal_dose_mgkg": 10.0,
    },
    "Naproxen (Medium — Allometric Scaling)": {
        "name": "Naproxen",
        "smiles": "COC1=CC=C2C(CC(C)C(=O)O)=CC=C2C1",
        "source_species": "rat",
        "animal_dose_mgkg": 8.0,
    },
    "Diazepam (Hard — Combo DDI)": {
        "name": "Diazepam",
        "smiles": "CN1C(=O)CN=C(C2=CC=CC=C2)C2=CC(Cl)=CC=C21",
        "source_species": "rat",
        "animal_dose_mgkg": 2.0,
    },
    "Aspirin": {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "source_species": "rat",
        "animal_dose_mgkg": 50.0,
    },
    "Ibuprofen": {
        "name": "Ibuprofen",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "source_species": "rat",
        "animal_dose_mgkg": 20.0,
    },
    "Custom (enter your own)": {
        "name": "",
        "smiles": "",
        "source_species": "rat",
        "animal_dose_mgkg": 8.0,
    },
}

# ── Colours ──────────────────────────────────────────────────────────────────

_GRADE_COLOURS = {0: "#4caf50", 1: "#8bc34a", 2: "#ffb300", 3: "#f44336", 4: "#b71c1c"}
_ORGAN_COLOURS = {"hepatocyte": "#ef6c00", "immune": "#8e24aa", "renal": "#1565c0"}


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


# ── Run episode ──────────────────────────────────────────────────────────────

def _run_episode(drug_name: str, smiles: str, species: str, animal_dose: float, num_steps: int):
    """Run a full trial episode and return episode_data dict."""
    try:
        from .rl_agent_environment import RlAgentEnvironment
        from ..drug_profile_builder import DrugProfileBuilder
        from ..models import RlAgentAction
    except ImportError:
        from rl_agent.server.rl_agent_environment import RlAgentEnvironment
        from rl_agent.drug_profile_builder import DrugProfileBuilder
        from rl_agent.models import RlAgentAction

    builder = DrugProfileBuilder(
        smiles=smiles,
        name=drug_name or "investigational compound",
        source_species=species or "rat",
        animal_dose_mgkg=float(animal_dose or 8.0),
    )
    profile = builder.build()

    env = RlAgentEnvironment(drug_profile=profile, use_llm=False)
    obs = env.reset()

    dose = env._start_dose
    for step_i in range(int(num_steps)):
        if obs.done if hasattr(obs, "done") else False:
            break

        escalate = True
        if obs.dlt_count > 0:
            dose = max(0.1, dose * 0.7)
            escalate = False
        elif obs.hepatocyte_signal > 0.6:
            escalate = False
        elif obs.renal_signal < 0.5:
            escalate = False
        else:
            dose = min(dose * 1.3, 50.0)

        action = RlAgentAction(
            next_dose=round(dose, 3),
            cohort_size=3,
            escalate=escalate,
        )
        obs = env.step(action)

    return env.get_episode_data()


# ── Plotting ─────────────────────────────────────────────────────────────────

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_dose_escalation(history, safety_limits, rp2d_dose):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa")
    if not history:
        ax.text(0.5, 0.5, "No history data", ha="center", va="center", fontsize=14, color="grey")
        return fig

    steps = [h["step"] for h in history]
    doses = [h["dose"] for h in history]
    dlt_rates = [h["dlt_rate"] for h in history]
    dlt_cnts = [h["dlt_count"] for h in history]
    dlt_limit = float(safety_limits.get("dlt_rate_limit", 0.33))
    seen = set()

    ax.plot(steps, doses, "o-", color="#1565c0", linewidth=2.5, markersize=8, zorder=3, label="Dose (mg/kg)")
    for s, d, r, cnt in zip(steps, doses, dlt_rates, dlt_cnts):
        if dlt_limit > 0 and r >= dlt_limit:
            c, lbl = "#f44336", "Boundary reached"
        elif r > 0:
            c, lbl = "#ffb300", "DLT observed"
        else:
            c, lbl = "#4caf50", "No DLT observed"
        show = lbl not in seen
        seen.add(lbl)
        ax.scatter(s, d, s=max(60, r * 600 + 80), c=c, edgecolors="#333",
                   linewidths=0.8, zorder=4, alpha=0.85, label=lbl if show else None)
        if cnt > 0:
            ax.annotate(f"{cnt}DLT", (s, d), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color="#b71c1c", fontweight="bold")

    if rp2d_dose:
        ax.axhline(rp2d_dose, color="#4caf50", linestyle="--", linewidth=1.8,
                   alpha=0.8, label=f"RP2D est. {rp2d_dose:.2f} mg/kg")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Dose (mg/kg)", fontsize=11)
    ax.set_title("Dose Escalation Trajectory", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_pk_curves(pk_traces, history):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa")
    if not pk_traces:
        ax.text(0.5, 0.5, "No PK trace data", ha="center", va="center", fontsize=14, color="grey")
        return fig

    cmap = matplotlib.colormaps["Blues"].resampled(len(pk_traces) + 2)
    all_cmaxes = []
    for step_idx, step_traces in enumerate(pk_traces):
        dose = history[step_idx]["dose"] if step_idx < len(history) else 0
        colour = cmap(step_idx + 2)
        first = True
        for trace in step_traces:
            times = trace.get("time_h", [])
            concs = trace.get("blood_conc", [])
            if not times or not concs:
                continue
            label = f"Step {step_idx + 1} ({dose:.1f} mg/kg)" if first else None
            ax.plot(times, concs, color=colour, linewidth=1.2, alpha=0.7, label=label)
            first = False
            all_cmaxes.append(trace.get("cmax", max(concs)))

    if all_cmaxes:
        peak = max(all_cmaxes)
        ax.axhline(peak, color="#f44336", linestyle=":", linewidth=1.2,
                   alpha=0.6, label=f"Peak Cmax {peak:.2f}")

    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel("Blood Concentration (mg/L)", fontsize=11)
    ax.set_title("PK Concentration-Time Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_dlt_heatmap(cohort_log):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa")
    if not cohort_log:
        ax.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=14, color="grey")
        return fig

    n_steps = len(cohort_log)
    max_pts = max(len(step) for step in cohort_log)
    matrix = np.full((n_steps, max_pts), -1, dtype=float)
    for s_idx, step in enumerate(cohort_log):
        for p_idx, pat in enumerate(step):
            matrix[s_idx, p_idx] = pat.get("dlt_grade", 0)

    colours = ["#cccccc", "#4caf50", "#8bc34a", "#ffb300", "#f44336", "#b71c1c"]
    cmap_custom = LinearSegmentedColormap.from_list("dlt", colours, N=6)
    ax.imshow(matrix, cmap=cmap_custom, vmin=-1, vmax=4, aspect="auto", interpolation="nearest")

    for s in range(n_steps):
        for p in range(max_pts):
            val = matrix[s, p]
            if val >= 0:
                text_col = "white" if val >= 3 else "black"
                ax.text(p, s, str(int(val)), ha="center", va="center",
                        fontsize=10, color=text_col, fontweight="bold")

    ax.set_xlabel("Patient", fontsize=11)
    ax.set_ylabel("Step", fontsize=11)
    ax.set_xticks(range(max_pts))
    ax.set_xticklabels([f"P{i+1}" for i in range(max_pts)], fontsize=9)
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([f"Step {i+1}" for i in range(n_steps)], fontsize=9)
    ax.set_title("DLT Grade Heatmap (Grade 3+ = DLT)", fontsize=13, fontweight="bold")

    legend_patches = [Patch(color=_GRADE_COLOURS[g],
                            label=f"Grade {g}" + (" (DLT)" if g >= 3 else ""))
                      for g in range(5)]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right",
              bbox_to_anchor=(1.0, -0.08), ncol=5)
    fig.tight_layout()
    return fig


def _plot_organ_signals(cohort_log):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa")
    if not cohort_log:
        ax.text(0.5, 0.5, "No cohort data", ha="center", va="center", fontsize=14, color="grey")
        return fig

    steps = list(range(1, len(cohort_log) + 1))
    hep = [np.mean([p["hep_signal"] for p in step]) for step in cohort_log]
    imm = [np.mean([p["imm_signal"] for p in step]) for step in cohort_log]
    ren = [np.mean([p["ren_signal"] for p in step]) for step in cohort_log]

    ax.plot(steps, hep, "o-", color=_ORGAN_COLOURS["hepatocyte"], linewidth=2.2, markersize=7, label="Liver Stress")
    ax.plot(steps, imm, "s-", color=_ORGAN_COLOURS["immune"], linewidth=2.2, markersize=7, label="Immune Response")
    ax.plot(steps, ren, "^-", color=_ORGAN_COLOURS["renal"], linewidth=2.2, markersize=7, label="Renal Function")

    ax.axhline(0.7, color="#f44336", linestyle=":", linewidth=1, alpha=0.5, label="Danger threshold")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Signal Intensity", fontsize=11)
    ax.set_title("Organ Safety Signals", fontsize=13, fontweight="bold")
    ax.set_xticks(steps)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_rewards(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafafa")
    if not history:
        ax.text(0.5, 0.5, "No history data", ha="center", va="center", fontsize=14, color="grey")
        return fig

    steps = [h["step"] for h in history]
    rewards = [h.get("reward", 0) for h in history]
    cumulative = np.cumsum(rewards).tolist()
    ax2 = ax.twinx()

    ax.bar(steps, rewards, color="#42a5f5", alpha=0.7, label="Step Reward", zorder=2)
    ax2.plot(steps, cumulative, "D-", color="#e53935", linewidth=2.2,
             markersize=7, label="Cumulative", zorder=3)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Step Reward", fontsize=11, color="#42a5f5")
    ax2.set_ylabel("Cumulative Reward", fontsize=11, color="#e53935")
    ax.set_title("Reward Trajectory", fontsize=13, fontweight="bold")
    ax.set_xticks(steps)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", labelcolor="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#e53935")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_summary_html(data: dict) -> str:
    drug = data.get("drug_name", "unknown")
    params = data.get("drug_params", {})
    flags = data.get("safety_flags", {})
    scores = data.get("final_score", {})
    rp2d = data.get("rp2d_dose")
    steps = data.get("steps_taken", 0)
    start = data.get("start_dose", 0)

    cyps = flags.get("cyp_inhibitions", [])
    risk = flags.get("overall_risk_score", 0)

    score_rows = "".join(
        f"<tr><td style='padding:4px 12px;'>{_display_name(k)}</td>"
        f"<td style='padding:4px 12px;font-weight:bold;color:#1565c0;'>{v:.3f}</td></tr>"
        for k, v in scores.items()
    )

    return f"""
    <div style="font-family:system-ui;max-width:700px;">
      <h2 style="color:#1a237e;margin-bottom:4px;">Drug: {drug}</h2>

      <h3 style="color:#333;margin-top:16px;">PK Parameters (Human-Scaled)</h3>
      <table style="border-collapse:collapse;">
        <tr><td style="padding:2px 12px;">ka</td><td><b>{params.get('ka', 'n/a')}</b> 1/h</td></tr>
        <tr><td style="padding:2px 12px;">CL</td><td><b>{params.get('CL', 'n/a')}</b> L/h/kg</td></tr>
        <tr><td style="padding:2px 12px;">Vc</td><td><b>{params.get('Vc', 'n/a')}</b> L/kg</td></tr>
        <tr><td style="padding:2px 12px;">Vp</td><td><b>{params.get('Vp', 'n/a')}</b> L/kg</td></tr>
        <tr><td style="padding:2px 12px;">PPB</td><td><b>{params.get('PPB', 'n/a')}</b></td></tr>
      </table>

      <h3 style="color:#333;margin-top:16px;">Safety Flags</h3>
      <table style="border-collapse:collapse;">
        <tr><td style="padding:2px 12px;">DILI Risk</td>
            <td style="color:{'#f44336' if flags.get('dili_risk') else '#4caf50'};font-weight:bold;">
            {'YES' if flags.get('dili_risk') else 'No'}</td></tr>
        <tr><td style="padding:2px 12px;">hERG Risk</td>
            <td style="color:{'#f44336' if flags.get('herg_risk') else '#4caf50'};font-weight:bold;">
            {'YES' if flags.get('herg_risk') else 'No'}</td></tr>
        <tr><td style="padding:2px 12px;">CYP Inhibitions</td>
            <td><b>{', '.join(cyps) if cyps else 'None'}</b></td></tr>
        <tr><td style="padding:2px 12px;">Overall Risk</td>
            <td><b>{risk:.2f}</b></td></tr>
      </table>

      <h3 style="color:#333;margin-top:16px;">Episode Summary</h3>
      <table style="border-collapse:collapse;">
        <tr><td style="padding:2px 12px;">Starting Dose</td><td><b>{start:.3f} mg/kg</b></td></tr>
        <tr><td style="padding:2px 12px;">RP2D Estimate</td>
            <td><b>{f'{rp2d:.3f} mg/kg' if rp2d else 'Not found'}</b></td></tr>
        <tr><td style="padding:2px 12px;">Steps Taken</td><td><b>{steps}</b></td></tr>
      </table>

      <h3 style="color:#1a237e;margin-top:16px;">Task Scores</h3>
      <table style="border-collapse:collapse;background:#f5f5f5;border-radius:8px;">
        <tr style="background:#e3f2fd;"><th style="padding:6px 12px;text-align:left;">Task</th>
            <th style="padding:6px 12px;text-align:left;">Score</th></tr>
        {score_rows}
      </table>
    </div>
    """


# ── Main Gradio handler ─────────────────────────────────────────────────────

def run_trial(preset, drug_name, smiles, species, animal_dose, num_steps):
    """Run a trial and return all visualization figures + summary."""
    try:
        # Use preset values if not custom
        if preset and preset != "Custom (enter your own)":
            p = PRESET_DRUGS[preset]
            drug_name = p["name"]
            smiles = p["smiles"]
            species = p["source_species"]
            animal_dose = p["animal_dose_mgkg"]

        if not smiles or not smiles.strip():
            raise ValueError("Please provide a valid SMILES string.")

        data = _run_episode(drug_name, smiles.strip(), species, float(animal_dose), int(num_steps))

        fig_dose = _plot_dose_escalation(data["history"], data["safety_limits"], data["rp2d_dose"])
        fig_pk = _plot_pk_curves(data["pk_traces"], data["history"])
        fig_dlt = _plot_dlt_heatmap(data["cohort_log"])
        fig_organ = _plot_organ_signals(data["cohort_log"])
        fig_reward = _plot_rewards(data["history"])
        summary = _build_summary_html(data)

        return fig_dose, fig_pk, fig_dlt, fig_organ, fig_reward, summary

    except Exception as e:
        empty = plt.figure(figsize=(8, 5))
        plt.close(empty)
        err_html = f"<div style='color:#f44336;padding:20px;'><h3>Error</h3><pre>{traceback.format_exc()}</pre></div>"
        return empty, empty, empty, empty, empty, err_html


def on_preset_change(preset):
    """Fill in fields when a preset is selected."""
    if preset and preset in PRESET_DRUGS:
        p = PRESET_DRUGS[preset]
        is_custom = preset == "Custom (enter your own)"
        return (
            gr.update(value=p["name"], interactive=is_custom),
            gr.update(value=p["smiles"], interactive=is_custom),
            gr.update(value=p["source_species"]),
            gr.update(value=p["animal_dose_mgkg"]),
        )
    return gr.update(), gr.update(), gr.update(), gr.update()


# ── Build Gradio app ─────────────────────────────────────────────────────────

def create_dashboard() -> gr.Blocks:
    with gr.Blocks(title="RxGym — Clinical Trial Simulator") as demo:
        gr.Markdown(
            "# RxGym — Clinical Trial Simulator\n"
            "### From SMILES to FDA-Compliant Phase I Recommendation",
            elem_classes=["main-title"]
        )
        gr.Markdown(
            "Select a preset drug or enter your own molecule. "
            "RxGym runs the full pipeline: RDKit descriptors, DeepChem ADMET prediction, "
            "allometric scaling, multi-agent PK/PD simulation, and FDA safety-constrained dose escalation.",
            elem_classes=["subtitle"]
        )

        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(
                    choices=list(PRESET_DRUGS.keys()),
                    value="Acetaminophen (Easy — Phase I Dosing)",
                    label="Preset Drug",
                    info="Pick a preset or choose 'Custom' to enter your own SMILES"
                )
                drug_name = gr.Textbox(
                    value="Acetaminophen", label="Drug Name", interactive=False
                )
                smiles = gr.Textbox(
                    value="CC(=O)NC1=CC=C(O)C=C1", label="SMILES String", interactive=False
                )
                species = gr.Dropdown(
                    choices=["mouse", "rat", "monkey", "dog"],
                    value="rat", label="Source Species"
                )
                animal_dose = gr.Number(
                    value=10.0, label="Animal Dose (mg/kg)", minimum=0.1, maximum=500
                )
                num_steps = gr.Slider(
                    minimum=2, maximum=10, value=6, step=1,
                    label="Max Trial Steps"
                )
                run_btn = gr.Button("Run Trial Simulation", variant="primary", size="lg")

            with gr.Column(scale=2):
                summary_html = gr.HTML(label="Episode Summary")

        gr.Markdown("---")
        gr.Markdown("## Trial Visualizations")

        with gr.Row():
            plot_dose = gr.Plot(label="Dose Escalation")
            plot_pk = gr.Plot(label="PK Curves")

        with gr.Row():
            plot_dlt = gr.Plot(label="DLT Heatmap")
            plot_organ = gr.Plot(label="Organ Signals")

        with gr.Row():
            plot_reward = gr.Plot(label="Rewards")

        # Wire events
        preset.change(
            fn=on_preset_change,
            inputs=[preset],
            outputs=[drug_name, smiles, species, animal_dose],
        )

        run_btn.click(
            fn=run_trial,
            inputs=[preset, drug_name, smiles, species, animal_dose, num_steps],
            outputs=[plot_dose, plot_pk, plot_dlt, plot_organ, plot_reward, summary_html],
        )

    return demo
