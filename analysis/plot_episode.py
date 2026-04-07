"""
Episode Visualization for Clinical Trial Gym.

Renders all data captured by RlAgentEnvironment.get_episode_data() into a
multi-panel figure. Works standalone (pass a saved JSON file) or inline
(pass the dict directly from a running environment).

Panels produced
───────────────
1. Dose escalation trajectory     — dose vs step, DLT rate as bubble size
2. PK concentration curves        — Cc(t) for all patients at each step
3. DLT grade heatmap              — patient × step, colour = CTCAE grade
4. Organ signal timeline          — hepatocyte, immune, renal vs step
5. Reward trajectory              — cumulative and per-step reward
6. Drug ADMET summary             — text panel (flags, HED, risk score)

Usage
─────
# From a running environment:
    from analysis.plot_episode import EpisodePlotter
    data = env.get_episode_data()
    plotter = EpisodePlotter(data)
    plotter.plot(save_path="episode.png")

# From a saved JSON file:
    python analysis/plot_episode.py episode_data.json

# Quick one-liner after running the environment in test_e2e.py:
    python analysis/plot_episode.py --demo
"""

import sys
import os
import json
import argparse
import textwrap

import numpy as np

# ── optional matplotlib — import lazily so the module can be imported without it
def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch
        return matplotlib, plt, gridspec, LinearSegmentedColormap, Patch
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization.\n"
            "Install it with:  pip install matplotlib"
        )


# ── colour palette ────────────────────────────────────────────────────────────
_GRADE_COLOURS = {
    0: "#4caf50",   # green  — no toxicity
    1: "#8bc34a",   # light green — mild
    2: "#ffb300",   # amber — moderate
    3: "#f44336",   # red — severe (DLT)
    4: "#b71c1c",   # dark red — life-threatening
}

_ORGAN_COLOURS = {
    "hepatocyte": "#ef6c00",   # orange
    "immune":     "#8e24aa",   # purple
    "renal":      "#1565c0",   # blue
}


class EpisodePlotter:
    """
    Renders a complete episode from get_episode_data() output.

    Parameters
    ----------
    data : dict
        Output of RlAgentEnvironment.get_episode_data().
    figsize : tuple
        Figure size in inches. Default (20, 24).
    style : str
        Matplotlib style. Default "seaborn-v0_8-whitegrid".
    """

    def __init__(self, data: dict, figsize=(20, 24), style="seaborn-v0_8-whitegrid"):
        self.data    = data
        self.figsize = figsize
        self.style   = style
        self._validate()

    def _validate(self):
        required = {"history", "pk_traces", "cohort_log", "final_score"}
        missing  = required - set(self.data.keys())
        if missing:
            raise ValueError(f"episode data missing keys: {missing}")

    # ── main entry point ─────────────────────────────────────────────────────

    def plot(self, save_path: str = None, show: bool = True) -> "plt.Figure":
        """
        Render all panels and optionally save/show.

        Parameters
        ----------
        save_path : str, optional
            File path to save the figure (PNG, PDF, SVG). If None, not saved.
        show : bool
            Call plt.show() after rendering. Default True.

        Returns
        -------
        matplotlib.figure.Figure
        """
        _mpl, plt, gridspec, _, _ = _require_matplotlib()

        try:
            plt.style.use(self.style)
        except Exception:
            pass  # style not available — use default

        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor("#fafafa")

        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            hspace=0.45,
            wspace=0.35,
            top=0.93,
            bottom=0.06,
        )

        ax_dose   = fig.add_subplot(gs[0, 0])   # dose escalation
        ax_pk     = fig.add_subplot(gs[0, 1])   # PK curves
        ax_dlt    = fig.add_subplot(gs[1, 0])   # DLT heatmap
        ax_organ  = fig.add_subplot(gs[1, 1])   # organ signals
        ax_reward = fig.add_subplot(gs[2, 0])   # reward trajectory
        ax_text   = fig.add_subplot(gs[2, 1])   # ADMET text summary

        self._plot_dose_escalation(ax_dose, plt)
        self._plot_pk_curves(ax_pk, plt)
        self._plot_dlt_heatmap(ax_dlt, plt, fig)
        self._plot_organ_signals(ax_organ, plt)
        self._plot_reward(ax_reward, plt)
        self._plot_admet_text(ax_text)

        drug  = self.data.get("drug_name", "investigational compound")
        score = self.data.get("final_score", {})
        title = (
            f"Clinical Trial Gym — Episode Report\n"
            f"Drug: {drug}   |   "
            f"Phase I: {score.get('phase_i_dosing', 0):.3f}   "
            f"Allometric: {score.get('allometric_scaling', 0):.3f}   "
            f"DDI: {score.get('combo_ddi', 0):.3f}"
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to: {save_path}")

        if show:
            plt.show()

        return fig

    # ── panel 1: dose escalation ─────────────────────────────────────────────

    def _plot_dose_escalation(self, ax, plt):
        history = self.data["history"]
        if not history:
            ax.set_visible(False)
            return

        steps    = [h["step"]      for h in history]
        doses    = [h["dose"]      for h in history]
        dlt_rates= [h["dlt_rate"]  for h in history]
        dlt_cnts = [h["dlt_count"] for h in history]
        rewards  = [h.get("reward", 0.0) for h in history]

        # Line: dose trajectory
        ax.plot(steps, doses, "o-", color="#1565c0", linewidth=2.5,
                markersize=8, zorder=3, label="Dose (mg/kg)")

        # Bubble: DLT count at each step (size ∝ DLT rate)
        for s, d, r, cnt in zip(steps, doses, dlt_rates, dlt_cnts):
            colour = "#f44336" if r > 0.33 else ("#ffb300" if r > 0 else "#4caf50")
            ax.scatter(s, d, s=max(60, r * 600 + 80), c=colour,
                       edgecolors="#333", linewidths=0.8, zorder=4, alpha=0.85)
            if cnt > 0:
                ax.annotate(f"{cnt}DLT", (s, d),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color="#b71c1c", fontweight="bold")

        # RP2D line
        rp2d = self.data.get("rp2d_dose")
        if rp2d:
            ax.axhline(rp2d, color="#4caf50", linestyle="--", linewidth=1.8,
                       alpha=0.8, label=f"RP2D est. {rp2d:.1f} mg/kg")

        # FDA threshold annotation
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Dose (mg/kg)", fontsize=11)
        ax.set_title("Dose Escalation Trajectory", fontsize=12, fontweight="bold")
        ax.set_xticks(steps)

        legend_elements = [
            plt.scatter([], [], s=80,  c="#4caf50", label="No DLTs"),
            plt.scatter([], [], s=150, c="#ffb300", label="DLTs present"),
            plt.scatter([], [], s=250, c="#f44336", label="FDA stop zone"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + legend_elements,
                  labels=labels + ["No DLTs", "DLTs present", "FDA stop zone"],
                  fontsize=8, loc="upper left")

    # ── panel 2: PK concentration curves ─────────────────────────────────────

    def _plot_pk_curves(self, ax, plt):
        pk_traces = self.data.get("pk_traces", [])
        history   = self.data.get("history", [])
        if not pk_traces:
            ax.text(0.5, 0.5, "No PK trace data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="grey")
            return

        import matplotlib
        cmap   = matplotlib.colormaps["Blues"].resampled(len(pk_traces) + 2)
        all_cmaxes = []

        for step_idx, step_traces in enumerate(pk_traces):
            dose = history[step_idx]["dose"] if step_idx < len(history) else "?"
            colour = cmap(step_idx + 2)
            first = True
            for trace in step_traces:
                times = trace.get("time_h", [])
                concs = trace.get("blood_conc", [])
                if not times or not concs:
                    continue
                label = f"Step {step_idx+1} ({dose:.1f} mg/kg)" if first else None
                ax.plot(times, concs, color=colour, linewidth=1.2,
                        alpha=0.7, label=label)
                first = False
                cmax = trace.get("cmax", max(concs))
                all_cmaxes.append(cmax)

        if all_cmaxes:
            ax.axhline(max(all_cmaxes), color="#f44336", linestyle=":",
                       linewidth=1.2, alpha=0.6, label=f"Peak Cmax {max(all_cmaxes):.2f}")

        ax.set_xlabel("Time (h)", fontsize=11)
        ax.set_ylabel("Blood Concentration (mg/L)", fontsize=11)
        ax.set_title("PK Concentration–Time Curves", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    # ── panel 3: DLT grade heatmap ────────────────────────────────────────────

    def _plot_dlt_heatmap(self, ax, plt, fig):
        _, _, _, LinearSegmentedColormap, Patch = _require_matplotlib()
        cohort_log = self.data.get("cohort_log", [])
        if not cohort_log:
            ax.set_visible(False)
            return

        n_steps = len(cohort_log)
        # Build grade matrix: rows=steps, cols=patients (max cohort across steps)
        max_pts = max(len(step) for step in cohort_log)
        matrix  = np.full((n_steps, max_pts), -1, dtype=float)

        for s_idx, step in enumerate(cohort_log):
            for p_idx, pat in enumerate(step):
                matrix[s_idx, p_idx] = pat.get("dlt_grade", 0)

        # Custom colourmap: grey for missing, then green→red for grades 0–4
        colours = ["#cccccc", "#4caf50", "#8bc34a", "#ffb300", "#f44336", "#b71c1c"]
        cmap    = LinearSegmentedColormap.from_list("dlt", colours, N=6)

        im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=4,
                       aspect="auto", interpolation="nearest")

        # Annotate grade values
        for s in range(n_steps):
            for p in range(max_pts):
                val = matrix[s, p]
                if val >= 0:
                    text_col = "white" if val >= 3 else "black"
                    ax.text(p, s, str(int(val)), ha="center", va="center",
                            fontsize=9, color=text_col, fontweight="bold")

        ax.set_xlabel("Patient", fontsize=11)
        ax.set_ylabel("Step", fontsize=11)
        ax.set_xticks(range(max_pts))
        ax.set_xticklabels([f"P{i+1}" for i in range(max_pts)], fontsize=9)
        ax.set_yticks(range(n_steps))
        ax.set_yticklabels([f"Step {i+1}" for i in range(n_steps)], fontsize=9)
        ax.set_title("DLT Grade Heatmap (CTCAE)", fontsize=12, fontweight="bold")

        legend_patches = [
            Patch(color=_GRADE_COLOURS[g], label=f"Grade {g}" + (" (DLT)" if g >= 3 else ""))
            for g in range(5)
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc="upper right",
                  bbox_to_anchor=(1.0, -0.12), ncol=5)

    # ── panel 4: organ signal timeline ───────────────────────────────────────

    def _plot_organ_signals(self, ax, plt):
        cohort_log = self.data.get("cohort_log", [])
        if not cohort_log:
            ax.set_visible(False)
            return

        steps = list(range(1, len(cohort_log) + 1))
        hep_means = [np.mean([p["hep_signal"] for p in step]) for step in cohort_log]
        imm_means = [np.mean([p["imm_signal"] for p in step]) for step in cohort_log]
        ren_means = [np.mean([p["ren_signal"] for p in step]) for step in cohort_log]

        ax.plot(steps, hep_means, "o-", color=_ORGAN_COLOURS["hepatocyte"],
                linewidth=2.2, markersize=7, label="Liver (CYP saturation)")
        ax.plot(steps, imm_means, "s-", color=_ORGAN_COLOURS["immune"],
                linewidth=2.2, markersize=7, label="Immune (cytokine)")
        ax.plot(steps, ren_means, "^-", color=_ORGAN_COLOURS["renal"],
                linewidth=2.2, markersize=7, label="Renal (GFR fraction)")

        # Safety thresholds
        ax.axhline(0.7, color=_ORGAN_COLOURS["hepatocyte"],
                   linestyle=":", linewidth=1.2, alpha=0.5)
        ax.axhline(0.5, color=_ORGAN_COLOURS["renal"],
                   linestyle=":", linewidth=1.2, alpha=0.5, label="Caution thresholds")

        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Signal [0 = bad → 1 = healthy]", fontsize=11)
        ax.set_title("Organ Safety Signals", fontsize=12, fontweight="bold")
        ax.set_xticks(steps)
        ax.legend(fontsize=9)

        # Fill danger zones
        ax.fill_between(steps, hep_means, 0.7,
                         where=[h > 0.7 for h in hep_means],
                         alpha=0.12, color=_ORGAN_COLOURS["hepatocyte"])

    # ── panel 5: reward trajectory ────────────────────────────────────────────

    def _plot_reward(self, ax, plt):
        history = self.data.get("history", [])
        if not history:
            ax.set_visible(False)
            return

        steps   = [h["step"]           for h in history]
        rewards = [h.get("reward", 0)  for h in history]
        cumulative = np.cumsum(rewards).tolist()

        ax2 = ax.twinx()

        ax.bar(steps, rewards, color="#42a5f5", alpha=0.7,
               label="Step reward", zorder=2)
        ax2.plot(steps, cumulative, "D-", color="#e53935", linewidth=2.2,
                 markersize=7, label="Cumulative reward", zorder=3)

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Step Reward", fontsize=11, color="#42a5f5")
        ax2.set_ylabel("Cumulative Reward", fontsize=11, color="#e53935")
        ax.set_title("Reward Trajectory", fontsize=12, fontweight="bold")
        ax.set_xticks(steps)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="y", labelcolor="#1565c0")
        ax2.tick_params(axis="y", labelcolor="#e53935")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

        # Annotate final episode score
        scores = self.data.get("final_score", {})
        if scores:
            score_text = "  ".join(
                f"{k.replace('_', ' ')}: {v:.3f}"
                for k, v in scores.items()
            )
            ax.text(0.5, 0.03, score_text,
                    ha="center", va="bottom", transform=ax.transAxes,
                    fontsize=8.5, color="#555",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", alpha=0.8))

    # ── panel 6: ADMET text summary ───────────────────────────────────────────

    def _plot_admet_text(self, ax):
        ax.axis("off")

        drug_params  = self.data.get("drug_params",  {})
        safety_flags = self.data.get("safety_flags", {})
        drug_name    = self.data.get("drug_name", "investigational compound")
        start_dose   = self.data.get("start_dose", "—")
        rp2d         = self.data.get("rp2d_dose")
        steps_taken  = self.data.get("steps_taken", "—")
        scores       = self.data.get("final_score", {})

        dili   = safety_flags.get("dili_risk", False)
        herg   = safety_flags.get("herg_risk", False)
        cyps   = safety_flags.get("cyp_inhibitions", [])
        risk   = safety_flags.get("overall_risk_score", None)

        lines = [
            ("DRUG PROFILE", True),
            (f"Name:  {drug_name}", False),
            ("", False),
            ("PK Parameters (human-scaled)", True),
            (f"  ka  = {drug_params.get('ka',  'n/a'):.4f}  1/h", False) if isinstance(drug_params.get('ka'), float) else (f"  ka  = n/a", False),
            (f"  CL  = {drug_params.get('CL',  'n/a'):.4f}  L/h/kg", False) if isinstance(drug_params.get('CL'), float) else (f"  CL  = n/a", False),
            (f"  Vc  = {drug_params.get('Vc',  'n/a'):.4f}  L/kg", False)  if isinstance(drug_params.get('Vc'), float) else (f"  Vc  = n/a", False),
            (f"  Vp  = {drug_params.get('Vp',  'n/a'):.4f}  L/kg", False)  if isinstance(drug_params.get('Vp'), float) else (f"  Vp  = n/a", False),
            ("", False),
            ("Safety Flags", True),
            (f"  DILI risk:    {'⚠ YES' if dili else 'No'}", False),
            (f"  hERG risk:    {'⚠ YES' if herg else 'No'}", False),
            (f"  CYP inhib.:   {', '.join(cyps) if cyps else 'none'}", False),
            (f"  Risk score:   {risk:.2f}" if risk is not None else "  Risk score: n/a", False),
            ("", False),
            ("Episode Summary", True),
            (f"  Starting dose:  {start_dose} mg/kg", False),
            (f"  RP2D estimate:  {rp2d:.2f} mg/kg" if rp2d else "  RP2D estimate:  not found", False),
            (f"  Steps taken:    {steps_taken}", False),
            ("", False),
            ("Final Scores", True),
        ] + [(f"  {k.replace('_', ' ').title()}:  {v:.3f}", False) for k, v in scores.items()]

        y = 0.97
        for line, bold in lines:
            ax.text(0.05, y, line,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold" if bold else "normal",
                    color="#1a237e" if bold else "#212121",
                    va="top", fontfamily="monospace" if not bold else None)
            y -= 0.055 if bold else 0.048
            if y < 0:
                break

        ax.set_title("Drug & Episode Summary", fontsize=12, fontweight="bold")


# ════════════════════════════════════════════════════════════════════════════
# Standalone helpers
# ════════════════════════════════════════════════════════════════════════════

def plot_from_json(path: str, save_path: str = None, show: bool = True):
    """Load episode data from a JSON file and plot it."""
    with open(path) as f:
        data = json.load(f)
    EpisodePlotter(data).plot(save_path=save_path, show=show)


def run_demo(save_path: str = "demo_episode.png"):
    """
    Run a complete demo episode (no server needed) and plot the result.
    Uses Aspirin as the test drug.
    """
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_root, "rl_agent"))
    sys.path.insert(0, os.path.join(_root, "Patient_Simulation"))

    from drug_profile_builder import DrugProfileBuilder
    from server.rl_agent_environment import RlAgentEnvironment
    from models import RlAgentAction

    print("Building Aspirin drug profile (Layer 1+2)...")
    builder = DrugProfileBuilder("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    profile = builder.build()
    print(builder.summary())

    print("\nRunning 6-step episode (Layer 3+4)...")
    env = RlAgentEnvironment(drug_profile=profile)
    obs = env.reset()

    doses  = [obs.dose_level * m for m in [2, 4, 7, 11, 15, 20]]
    for dose in doses:
        if obs.done:
            break
        obs = env.step(RlAgentAction(next_dose=dose, cohort_size=3, escalate=True))
        print(f"  Step {env._state.step_count}: dose={obs.dose_level:.2f}  "
              f"DLTs={obs.dlt_count}  reward={obs.reward:.3f}  "
              f"liver={obs.hepatocyte_signal:.2f}  "
              f"done={obs.done}")

    if not obs.done:
        obs = env.step(RlAgentAction(next_dose=obs.dose_level, cohort_size=3, escalate=False))

    data = env.get_episode_data()
    print(f"\nFinal scores: {data['final_score']}")
    print(f"RP2D estimate: {data['rp2d_dose']}")

    print(f"\nPlotting to {save_path}...")
    EpisodePlotter(data).plot(save_path=save_path, show=True)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a Clinical Trial Gym episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Plot from a saved JSON file:
          python analysis/plot_episode.py episode_data.json

          # Run a live demo (Aspirin, no server needed):
          python analysis/plot_episode.py --demo

          # Save without showing:
          python analysis/plot_episode.py episode.json --save out.png --no-show
        """),
    )
    parser.add_argument("json_path", nargs="?", help="Path to episode_data JSON file")
    parser.add_argument("--demo",    action="store_true", help="Run a demo episode and plot it")
    parser.add_argument("--save",    default=None, help="Save figure to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    args = parser.parse_args()

    if args.demo:
        run_demo(save_path=args.save or "demo_episode.png")
    elif args.json_path:
        plot_from_json(args.json_path, save_path=args.save, show=not args.no_show)
    else:
        parser.print_help()
