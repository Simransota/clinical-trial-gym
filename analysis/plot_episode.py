"""
Episode visualization for RxGym.

Renders the full episode data emitted by `RlAgentEnvironment.get_episode_data()`
into a multi-panel figure intended for research review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch

        return matplotlib, plt, gridspec, LinearSegmentedColormap, Patch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization.\n"
            "Install it with: pip install matplotlib"
        ) from exc


_GRADE_COLOURS = {
    0: "#4caf50",
    1: "#8bc34a",
    2: "#ffb300",
    3: "#f44336",
    4: "#b71c1c",
}

_ORGAN_COLOURS = {
    "hepatocyte": "#ef6c00",
    "immune": "#8e24aa",
    "renal": "#1565c0",
}


def _display_name(name: str) -> str:
    return name.replace("_", " ").title()


class EpisodePlotter:
    def __init__(self, data: dict, figsize=(20, 24), style="seaborn-v0_8-whitegrid"):
        self.data = data
        self.figsize = figsize
        self.style = style
        self._validate()

    def _validate(self):
        required = {"history", "pk_traces", "cohort_log", "final_score"}
        missing = required - set(self.data.keys())
        if missing:
            raise ValueError(f"episode data missing keys: {missing}")

    def plot(self, save_path: str | None = None, show: bool = True):
        _mpl, plt, gridspec, _, _ = _require_matplotlib()
        try:
            plt.style.use(self.style)
        except Exception:
            pass

        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor("#fafafa")
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35, top=0.93, bottom=0.06)

        ax_dose = fig.add_subplot(gs[0, 0])
        ax_pk = fig.add_subplot(gs[0, 1])
        ax_dlt = fig.add_subplot(gs[1, 0])
        ax_organ = fig.add_subplot(gs[1, 1])
        ax_reward = fig.add_subplot(gs[2, 0])
        ax_text = fig.add_subplot(gs[2, 1])

        self._plot_dose_escalation(ax_dose, plt)
        self._plot_pk_curves(ax_pk, plt)
        self._plot_dlt_heatmap(ax_dlt, plt)
        self._plot_organ_signals(ax_organ, plt)
        self._plot_reward(ax_reward, plt)
        self._plot_admet_text(ax_text)

        drug = self.data.get("drug_name", "investigational compound")
        score = self.data.get("final_score", {})
        score_text = "   ".join(f"{_display_name(name)}: {value:.3f}" for name, value in score.items())
        title = f"Clinical Trial Gym — Episode Report\nDrug: {drug}"
        if score_text:
            title = f"{title}   |   {score_text}"
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to: {save_path}")
        if show:
            plt.show()
        return fig

    def _plot_dose_escalation(self, ax, plt):
        history = self.data["history"]
        if not history:
            ax.set_visible(False)
            return

        steps = [h["step"] for h in history]
        doses = [h["dose"] for h in history]
        dlt_rates = [h["dlt_rate"] for h in history]
        dlt_cnts = [h["dlt_count"] for h in history]
        dlt_limit = float(self.data.get("safety_limits", {}).get("dlt_rate_limit", 0.0) or 0.0)
        seen_statuses = set()

        ax.plot(steps, doses, "o-", color="#1565c0", linewidth=2.5, markersize=8, zorder=3, label="Dose (mg/kg)")
        for s, d, r, cnt in zip(steps, doses, dlt_rates, dlt_cnts):
            if dlt_limit > 0 and r >= dlt_limit:
                colour = "#f44336"
                status_label = "Boundary reached"
            elif r > 0:
                colour = "#ffb300"
                status_label = "DLT observed"
            else:
                colour = "#4caf50"
                status_label = "No DLT observed"
            show_label = status_label not in seen_statuses
            seen_statuses.add(status_label)
            ax.scatter(
                s,
                d,
                s=max(60, r * 600 + 80),
                c=colour,
                edgecolors="#333",
                linewidths=0.8,
                zorder=4,
                alpha=0.85,
                label=status_label if show_label else None,
            )
            if cnt > 0:
                ax.annotate(
                    f"{cnt}DLT",
                    (s, d),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color="#b71c1c",
                    fontweight="bold",
                )

        rp2d = self.data.get("rp2d_dose")
        if rp2d:
            ax.axhline(rp2d, color="#4caf50", linestyle="--", linewidth=1.8, alpha=0.8, label=f"RP2D est. {rp2d:.1f} mg/kg")
        if dlt_limit > 0:
            ax.text(0.99, 0.03, f"DLT boundary = {dlt_limit:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#555")

        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Dose (mg/kg)", fontsize=11)
        ax.set_title("Dose Escalation Trajectory", fontsize=12, fontweight="bold")
        ax.set_xticks(steps)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="upper left")

    def _plot_pk_curves(self, ax, plt):
        pk_traces = self.data.get("pk_traces", [])
        history = self.data.get("history", [])
        if not pk_traces:
            ax.text(0.5, 0.5, "No PK trace data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="grey")
            return

        import matplotlib

        cmap = matplotlib.colormaps["Blues"].resampled(len(pk_traces) + 2)
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
                label = f"Step {step_idx + 1} ({dose:.1f} mg/kg)" if first else None
                ax.plot(times, concs, color=colour, linewidth=1.2, alpha=0.7, label=label)
                first = False
                all_cmaxes.append(trace.get("cmax", max(concs)))

        if all_cmaxes:
            peak = max(all_cmaxes)
            ax.axhline(peak, color="#f44336", linestyle=":", linewidth=1.2, alpha=0.6, label=f"Peak Cmax {peak:.2f}")

        ax.set_xlabel("Time (h)", fontsize=11)
        ax.set_ylabel("Blood Concentration (mg/L)", fontsize=11)
        ax.set_title("PK Concentration-Time Curves", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    def _plot_dlt_heatmap(self, ax, plt):
        _, _, _, LinearSegmentedColormap, Patch = _require_matplotlib()
        cohort_log = self.data.get("cohort_log", [])
        if not cohort_log:
            ax.set_visible(False)
            return

        n_steps = len(cohort_log)
        max_pts = max(len(step) for step in cohort_log)
        matrix = np.full((n_steps, max_pts), -1, dtype=float)
        for s_idx, step in enumerate(cohort_log):
            for p_idx, pat in enumerate(step):
                matrix[s_idx, p_idx] = pat.get("dlt_grade", 0)

        colours = ["#cccccc", "#4caf50", "#8bc34a", "#ffb300", "#f44336", "#b71c1c"]
        cmap = LinearSegmentedColormap.from_list("dlt", colours, N=6)
        ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=4, aspect="auto", interpolation="nearest")

        for s in range(n_steps):
            for p in range(max_pts):
                val = matrix[s, p]
                if val >= 0:
                    text_col = "white" if val >= 3 else "black"
                    ax.text(p, s, str(int(val)), ha="center", va="center", fontsize=9, color=text_col, fontweight="bold")

        ax.set_xlabel("Patient", fontsize=11)
        ax.set_ylabel("Step", fontsize=11)
        ax.set_xticks(range(max_pts))
        ax.set_xticklabels([f"P{i + 1}" for i in range(max_pts)], fontsize=9)
        ax.set_yticks(range(n_steps))
        ax.set_yticklabels([f"Step {i + 1}" for i in range(n_steps)], fontsize=9)
        ax.set_title("DLT Grade Heatmap", fontsize=12, fontweight="bold")
        legend_patches = [Patch(color=_GRADE_COLOURS[g], label=f"Grade {g}" + (" (DLT)" if g >= 3 else "")) for g in range(5)]
        ax.legend(handles=legend_patches, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, -0.12), ncol=5)

    def _plot_organ_signals(self, ax, plt):
        cohort_log = self.data.get("cohort_log", [])
        if not cohort_log:
            ax.set_visible(False)
            return

        steps = list(range(1, len(cohort_log) + 1))
        hep_means = [np.mean([p["hep_signal"] for p in step]) for step in cohort_log]
        imm_means = [np.mean([p["imm_signal"] for p in step]) for step in cohort_log]
        ren_means = [np.mean([p["ren_signal"] for p in step]) for step in cohort_log]

        ax.plot(steps, hep_means, "o-", color=_ORGAN_COLOURS["hepatocyte"], linewidth=2.2, markersize=7, label="Liver")
        ax.plot(steps, imm_means, "s-", color=_ORGAN_COLOURS["immune"], linewidth=2.2, markersize=7, label="Immune")
        ax.plot(steps, ren_means, "^-", color=_ORGAN_COLOURS["renal"], linewidth=2.2, markersize=7, label="Renal")

        if hep_means:
            hep_warn = float(np.quantile(hep_means, 0.75))
            ax.axhline(hep_warn, color=_ORGAN_COLOURS["hepatocyte"], linestyle=":", linewidth=1.2, alpha=0.5, label=f"Liver upper quartile ({hep_warn:.2f})")
            ax.fill_between(steps, hep_means, hep_warn, where=[h >= hep_warn for h in hep_means], alpha=0.12, color=_ORGAN_COLOURS["hepatocyte"])
        if ren_means:
            ren_warn = float(np.quantile(ren_means, 0.25))
            ax.axhline(ren_warn, color=_ORGAN_COLOURS["renal"], linestyle=":", linewidth=1.2, alpha=0.5, label=f"Renal lower quartile ({ren_warn:.2f})")

        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Signal", fontsize=11)
        ax.set_title("Organ Safety Signals", fontsize=12, fontweight="bold")
        ax.set_xticks(steps)
        ax.legend(fontsize=9)

    def _plot_reward(self, ax, plt):
        history = self.data.get("history", [])
        if not history:
            ax.set_visible(False)
            return

        steps = [h["step"] for h in history]
        rewards = [h.get("reward", 0) for h in history]
        cumulative = np.cumsum(rewards).tolist()
        ax2 = ax.twinx()

        ax.bar(steps, rewards, color="#42a5f5", alpha=0.7, label="Step reward", zorder=2)
        ax2.plot(steps, cumulative, "D-", color="#e53935", linewidth=2.2, markersize=7, label="Cumulative reward", zorder=3)

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

        scores = self.data.get("final_score", {})
        if scores:
            score_text = "  ".join(f"{_display_name(k)}: {v:.3f}" for k, v in scores.items())
            ax.text(
                0.5,
                0.03,
                score_text,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", alpha=0.8),
            )

    def _plot_admet_text(self, ax):
        ax.axis("off")
        drug_params = self.data.get("drug_params", {})
        safety_flags = self.data.get("safety_flags", {})
        drug_name = self.data.get("drug_name", "investigational compound")
        start_dose = self.data.get("start_dose", "—")
        rp2d = self.data.get("rp2d_dose")
        steps_taken = self.data.get("steps_taken", "—")
        scores = self.data.get("final_score", {})

        dili = safety_flags.get("dili_risk", False)
        herg = safety_flags.get("herg_risk", False)
        cyps = safety_flags.get("cyp_inhibitions", [])
        risk = safety_flags.get("overall_risk_score", None)

        lines = [
            ("DRUG PROFILE", True),
            (f"Name:  {drug_name}", False),
            ("", False),
            ("PK Parameters", True),
            (f"  ka  = {drug_params.get('ka', 'n/a'):.4f}  1/h", False) if isinstance(drug_params.get("ka"), float) else ("  ka  = n/a", False),
            (f"  CL  = {drug_params.get('CL', 'n/a'):.4f}  L/h/kg", False) if isinstance(drug_params.get("CL"), float) else ("  CL  = n/a", False),
            (f"  Vc  = {drug_params.get('Vc', 'n/a'):.4f}  L/kg", False) if isinstance(drug_params.get("Vc"), float) else ("  Vc  = n/a", False),
            (f"  Vp  = {drug_params.get('Vp', 'n/a'):.4f}  L/kg", False) if isinstance(drug_params.get("Vp"), float) else ("  Vp  = n/a", False),
            ("", False),
            ("Safety Flags", True),
            (f"  DILI risk:    {'YES' if dili else 'No'}", False),
            (f"  hERG risk:    {'YES' if herg else 'No'}", False),
            (f"  CYP inhib.:   {', '.join(cyps) if cyps else 'none'}", False),
            (f"  Risk score:   {risk:.2f}" if risk is not None else "  Risk score: n/a", False),
            ("", False),
            ("Episode Summary", True),
            (f"  Starting dose:  {start_dose} mg/kg", False),
            (f"  RP2D estimate:  {rp2d:.2f} mg/kg" if rp2d else "  RP2D estimate:  not found", False),
            (f"  Steps taken:    {steps_taken}", False),
            ("", False),
            ("Final Scores", True),
        ] + [(f"  {_display_name(k)}:  {v:.3f}", False) for k, v in scores.items()]

        y = 0.97
        for line, bold in lines:
            ax.text(
                0.05,
                y,
                line,
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold" if bold else "normal",
                color="#1a237e" if bold else "#212121",
                va="top",
                fontfamily="monospace" if not bold else None,
            )
            y -= 0.055 if bold else 0.048
            if y < 0:
                break

        ax.set_title("Drug & Episode Summary", fontsize=12, fontweight="bold")


def plot_from_json(path: str, save_path: str | None = None, show: bool = True):
    with open(path) as f:
        data = json.load(f)
    EpisodePlotter(data).plot(save_path=save_path, show=show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot an RxGym episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python analysis/plot_episode.py episode_data.json
              python analysis/plot_episode.py episode.json --save out.png --no-show
            """
        ),
    )
    parser.add_argument("json_path", nargs="?", help="Path to episode JSON")
    parser.add_argument("--save", help="Optional output image path")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    args = parser.parse_args()

    if not args.json_path:
        raise SystemExit("Provide a JSON episode path.")
    plot_from_json(args.json_path, save_path=args.save, show=not args.no_show)
