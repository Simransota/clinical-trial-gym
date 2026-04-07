"""
Research-facing visualization workflow for RxGym.

Generates a complete per-drug report:
- runs all 3 tasks with the baseline controller
- saves episode JSON for each task
- renders a detailed figure for each task
- renders a cross-task overview figure
- writes a markdown summary for researchers

Example
-------
python analysis/generate_drug_report.py \
  --name Acetaminophen \
  --smiles "CC(=O)NC1=CC=C(O)C=C1" \
  --animal-dose 10.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Patient_Simulation"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from rl_agent.drug_profile_builder import DrugProfileBuilder
from rl_agent.models import RlAgentAction
from rl_agent.server.rl_agent_environment import RlAgentEnvironment
from rl_agent.inference import (
    MAX_STEPS,
    choose_action,
    compute_terminal_score,
    derive_fragility_profile,
    resolve_task_targets,
)
try:
    from analysis.plot_episode import EpisodePlotter, _require_matplotlib
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_episode import EpisodePlotter, _require_matplotlib

def display_name(name: str) -> str:
    return name.replace("_", " ").title()


def discover_tasks(task_targets: dict) -> List[str]:
    return [str(task) for task in task_targets.keys()]


def run_task_episode(
    task_name: str,
    drug_profile: dict,
    hed_anchor: float,
    fragility_profile: dict,
    task_targets: dict,
) -> Tuple[dict, float]:
    env = RlAgentEnvironment(drug_profile=drug_profile)
    obs = env.reset()

    observations_seen: List[dict] = [obs.model_dump()]
    actions_taken: List[dict] = []
    rewards: List[float] = []
    current_dose = float(obs.dose_level)

    for _step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        action = choose_action(
            task_name=task_name,
            obs=obs.model_dump(),
            current_dose=current_dose,
            prev_actions=actions_taken,
            prev_observations=observations_seen,
            hed=hed_anchor,
            cyp_inhibitions=list(fragility_profile.get("cyp_inhibitions", []) or []),
            fragility_profile=fragility_profile,
            task_targets=task_targets,
        )

        obs = env.step(RlAgentAction(**action))
        current_dose = float(obs.dose_level)
        observations_seen.append(obs.model_dump())
        actions_taken.append(action)
        rewards.append(float(obs.reward))

        if obs.done:
            break

    data = env.get_episode_data()
    score = compute_terminal_score(
        task_name=task_name,
        rewards=rewards,
        actions=actions_taken,
        observations=observations_seen,
        hed=hed_anchor,
        task_targets=task_targets,
    )
    data["task_name"] = task_name
    data["baseline_score"] = score
    data["task_target"] = float(task_targets[task_name])
    data["hed_mgkg"] = float(hed_anchor)
    return data, score


def summarize_episode(data: dict) -> Dict[str, float | str]:
    history = data.get("history", [])
    cohort_log = data.get("cohort_log", [])
    safety_limits = data.get("safety_limits", {})
    dlt_limit = float(safety_limits.get("dlt_rate_limit", 0.0) or 0.0)

    max_dose = max((h["dose"] for h in history), default=0.0)
    total_patients = int(sum(h["cohort_size"] for h in history))
    max_dlt_rate = max((h["dlt_rate"] for h in history), default=0.0)
    first_dlt_step = next((h["step"] for h in history if h["dlt_count"] > 0), None)
    mean_hep = float(np.mean([np.mean([p["hep_signal"] for p in step]) for step in cohort_log])) if cohort_log else 0.0
    mean_ren = float(np.mean([np.mean([p["ren_signal"] for p in step]) for step in cohort_log])) if cohort_log else 1.0
    target = float(data.get("task_target", 0.0) or 0.0)
    target_ratio = (max_dose / target) if target > 0 else 0.0
    reward_trace = [float(h.get("reward", 0.0)) for h in history]
    score = float(data.get("baseline_score", 0.0) or 0.0)
    boundary_crossed = max_dlt_rate >= dlt_limit if dlt_limit > 0 else max_dlt_rate > 0

    interpretation_bits: List[str] = []
    if target > 0:
        if target_ratio < 0.9:
            interpretation_bits.append("dose exploration stayed below the nominal task target")
        elif target_ratio <= 1.15:
            interpretation_bits.append("dose exploration stayed close to the nominal task target")
        else:
            interpretation_bits.append("dose exploration overshot the nominal task target before stopping")
    if dlt_limit > 0:
        if boundary_crossed:
            interpretation_bits.append(f"toxicity signals touched or exceeded the configured DLT boundary ({dlt_limit:.2f})")
        else:
            interpretation_bits.append("toxicity signals stayed below the configured DLT boundary")
    if mean_hep > 0.5 or mean_ren < 0.7:
        interpretation_bits.append("organ safety signals showed meaningful stress")
    else:
        interpretation_bits.append("organ safety signals remained relatively stable")
    if reward_trace:
        interpretation_bits.append("reward trend improved across the episode" if reward_trace[-1] >= reward_trace[0] else "reward trend weakened by the end of the episode")
    interpretation = "; ".join(interpretation_bits).capitalize() + "."

    return {
        "max_dose_mgkg": round(max_dose, 4),
        "total_patients": total_patients,
        "max_dlt_rate": round(max_dlt_rate, 4),
        "first_dlt_step": first_dlt_step or "none",
        "mean_hep_signal": round(mean_hep, 4),
        "mean_renal_signal": round(mean_ren, 4),
        "target_ratio": round(target_ratio, 4),
        "interpretation": interpretation,
    }


def render_overview(profile: dict, episodes: List[dict], outpath: Path) -> None:
    _, plt, _, _, _ = _require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#fafafa")

    task_names = [e["task_name"] for e in episodes]
    task_labels = [display_name(name) for name in task_names]
    scores = [e["baseline_score"] for e in episodes]
    max_doses = [max((h["dose"] for h in e["history"]), default=0.0) for e in episodes]
    patient_counts = [sum(h["cohort_size"] for h in e["history"]) for e in episodes]
    max_dlt_rates = [max((h["dlt_rate"] for h in e["history"]), default=0.0) for e in episodes]
    task_targets = [e["task_target"] for e in episodes]
    dlt_limit = max(float(e.get("safety_limits", {}).get("dlt_rate_limit", 0.0) or 0.0) for e in episodes)
    cmap = plt.get_cmap("viridis")
    task_colors = [cmap(value) for value in np.linspace(0.15, 0.85, max(len(task_names), 1))]

    axes[0, 0].bar(task_labels, scores, color=task_colors)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Task Scores")
    axes[0, 0].set_ylabel("Score")

    x = np.arange(len(task_names))
    axes[0, 1].bar(x - 0.15, max_doses, width=0.3, label="Max dose", color="#ef6c00")
    axes[0, 1].bar(x + 0.15, task_targets, width=0.3, label="Task target", color="#90caf9")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(task_labels)
    axes[0, 1].set_title("Dose Reached vs Target")
    axes[0, 1].set_ylabel("mg/kg")
    axes[0, 1].legend()

    axes[1, 0].bar(task_labels, patient_counts, color=task_colors)
    axes[1, 0].set_title("Patients Enrolled")
    axes[1, 0].set_ylabel("Patients")

    axes[1, 1].bar(task_labels, max_dlt_rates, color=task_colors)
    if dlt_limit > 0:
        axes[1, 1].axhline(dlt_limit, linestyle="--", color="#424242", linewidth=1.2, label="Configured DLT boundary")
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].set_title("Peak DLT Rate")
    axes[1, 1].set_ylabel("DLT rate")
    if dlt_limit > 0:
        axes[1, 1].legend()

    admet = profile["admet_summary"]
    fig.suptitle(
        f"RxGym Overview Report — {profile['name']}\n"
        f"HED={profile['human_equivalent_dose']:.3f} mg/kg | "
        f"Risk={admet['overall_risk_score']:.3f} | "
        f"Half-life={admet['half_life_class']} | "
        f"Source={admet['source']}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(profile: dict, episodes: List[dict], outpath: Path) -> None:
    lines = [
        f"# RxGym Report — {profile['name']}",
        "",
        "## Drug Summary",
        f"- SMILES: `{profile['smiles']}`",
        f"- Human equivalent dose (HED): `{profile['human_equivalent_dose']:.4f} mg/kg`",
        f"- ADMET source: `{profile['admet_summary']['source']}`",
        f"- Overall risk score: `{profile['admet_summary']['overall_risk_score']:.4f}`",
        f"- CYP inhibitions: `{', '.join(profile['admet_summary']['cyp_inhibitions']) or 'none'}`",
        "",
        "## Task Findings",
    ]

    for episode in episodes:
        summary = summarize_episode(episode)
        lines.extend([
            f"### {episode['task_name']}",
            f"- Display name: `{display_name(episode['task_name'])}`",
            f"- Baseline score: `{episode['baseline_score']:.3f}`",
            f"- Task target: `{episode['task_target']:.4f} mg/kg`",
            f"- Max dose reached: `{summary['max_dose_mgkg']:.4f} mg/kg`",
            f"- Dose/target ratio: `{summary['target_ratio']:.3f}`",
            f"- Total patients enrolled: `{summary['total_patients']}`",
            f"- Peak DLT rate: `{summary['max_dlt_rate']:.3f}`",
            f"- First DLT step: `{summary['first_dlt_step']}`",
            f"- Mean liver signal: `{summary['mean_hep_signal']:.3f}`",
            f"- Mean renal signal: `{summary['mean_renal_signal']:.3f}`",
            f"- Clinical interpretation: {summary['interpretation']}",
            "",
        ])

    lines.extend([
        "## Generated Files",
        "- `overview.png`: cross-task comparison for the drug",
        "- `*_episode.png`: detailed task-level figure",
        "- `*_episode.json`: structured episode record for audit and re-analysis",
        "",
    ])
    outpath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a researcher-facing visualization report for one drug.")
    parser.add_argument("--name", required=True, help="Drug name")
    parser.add_argument("--smiles", required=True, help="Drug SMILES string")
    parser.add_argument("--animal-dose", type=float, required=True, help="Rat study dose in mg/kg")
    parser.add_argument("--source-species", default="rat", help="Preclinical species, default rat")
    parser.add_argument("--outdir", default=None, help="Output directory, default analysis/reports/<drug>")
    args = parser.parse_args()

    safe_name = args.name.lower().replace(" ", "_")
    outdir = Path(args.outdir) if args.outdir else ROOT / "analysis" / "reports" / safe_name
    outdir.mkdir(parents=True, exist_ok=True)

    builder = DrugProfileBuilder(
        smiles=args.smiles,
        name=args.name,
        source_species=args.source_species,
        animal_dose_mgkg=args.animal_dose,
    )
    profile = builder.build()
    hed_anchor = float(profile["human_equivalent_dose"])
    fragility_profile = derive_fragility_profile(
        {
            "hed_mgkg": hed_anchor,
            "admet_summary": profile["admet_summary"],
            "drug_params": profile["drug_params"],
            "task_targets": profile.get("task_targets", {}),
        }
    )
    task_targets = resolve_task_targets(profile, hed_anchor)
    tasks = discover_tasks(task_targets)

    episodes: List[dict] = []
    for task_name in tasks:
        data, _score = run_task_episode(
            task_name=task_name,
            drug_profile=profile,
            hed_anchor=hed_anchor,
            fragility_profile=fragility_profile,
            task_targets=task_targets,
        )
        json_path = outdir / f"{task_name}_episode.json"
        png_path = outdir / f"{task_name}_episode.png"
        json_path.write_text(json.dumps(data, indent=2))
        EpisodePlotter(data).plot(save_path=str(png_path), show=False)
        episodes.append(data)

    render_overview(profile, episodes, outdir / "overview.png")
    write_markdown_report(profile, episodes, outdir / "report.md")
    print(f"Report generated at {outdir}")


if __name__ == "__main__":
    main()
