"""Compare completed phi-sweep experiments across inventory-cap settings."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_COMPARISON_INPUTS: dict[str, Path] = {
    "no_cap": Path("experiments/phi_sweep_nocap"),
    "cap_50": Path("experiments/phi_sweep_cap50"),
    "cap_20": Path("experiments/phi_sweep_cap20"),
}
DEFAULT_COMPARISON_OUTPUT_DIR = Path("experiments/inventory_cap_comparison")
DEFAULT_EVALUATION_MODE = "greedy"

CAP_STYLES: dict[str, tuple[str, str]] = {
    "no_cap": ("tab:blue", "no_cap"),
    "cap_50": ("tab:orange", "cap_50"),
    "cap_20": ("tab:green", "cap_20"),
}

COMPARISON_METRICS: tuple[str, ...] = (
    "average_spread",
    "average_depth",
    "zero_return_fraction",
    "volatility",
    "tail_exposure",
    "max_drawdown",
    "one_sided_book_fraction",
    "undefined_midprice_fraction",
    "max_consecutive_one_sided_duration",
    "average_abs_ending_inventory",
    "max_abs_inventory",
    "pipeline_issue_seed_fraction",
    "midprice_gap_without_missing_side_fraction",
    "defined_midprice_despite_missing_side_fraction",
)

PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("volatility", "Volatility vs Phi by Inventory Cap", "Volatility", "volatility_vs_phi_by_inventory_cap.png"),
    ("average_spread", "Spread vs Phi by Inventory Cap", "Spread", "spread_vs_phi_by_inventory_cap.png"),
    ("average_depth", "Average Depth vs Phi by Inventory Cap", "Depth", "average_depth_vs_phi_by_inventory_cap.png"),
    ("tail_exposure", "Tail Exposure vs Phi by Inventory Cap", "Tail exposure", "tail_exposure_vs_phi_by_inventory_cap.png"),
    (
        "one_sided_book_fraction",
        "One-Sided Book Fraction vs Phi by Inventory Cap",
        "Fraction",
        "one_sided_book_fraction_vs_phi_by_inventory_cap.png",
    ),
    (
        "undefined_midprice_fraction",
        "Undefined Midprice Fraction vs Phi by Inventory Cap",
        "Fraction",
        "undefined_midprice_fraction_vs_phi_by_inventory_cap.png",
    ),
    (
        "max_consecutive_one_sided_duration",
        "Max Consecutive One-Sided Duration vs Phi by Inventory Cap",
        "Seconds",
        "max_consecutive_one_sided_duration_vs_phi_by_inventory_cap.png",
    ),
    (
        "average_abs_ending_inventory",
        "Average Absolute Inventory vs Phi by Inventory Cap",
        "Average abs ending inventory",
        "average_abs_inventory_vs_phi_by_inventory_cap.png",
    ),
)

SUMMARY_COLUMN_MAP = {
    "average_spread": "average_spread_mean",
    "average_depth": "average_depth_mean",
    "zero_return_fraction": "zero_return_fraction_mean",
    "volatility": "volatility_mean",
    "tail_exposure": "tail_exposure_mean",
    "max_drawdown": "max_drawdown_mean",
    "one_sided_book_fraction": "one_sided_book_fraction_mean",
    "undefined_midprice_fraction": "undefined_midprice_fraction_mean",
    "max_consecutive_one_sided_duration": "max_consecutive_one_sided_duration_mean",
    "average_abs_ending_inventory": "evaluation_average_abs_ending_inventory_mean",
    "max_abs_inventory": "evaluation_max_abs_inventory_reached_max",
    "pipeline_issue_seed_fraction": "pipeline_issue_seed_fraction",
    "midprice_gap_without_missing_side_fraction": "midprice_gap_without_missing_side_fraction_mean",
    "defined_midprice_despite_missing_side_fraction": "defined_midprice_despite_missing_side_fraction_mean",
    "consistency_status": "consistency_status",
}


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_inventory_cap_comparison_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _expected_inventory_cap(cap_label: str) -> int | None:
    if cap_label == "no_cap":
        return None
    if cap_label == "cap_50":
        return 50
    if cap_label == "cap_20":
        return 20
    raise ValueError(f"Unknown inventory-cap label: {cap_label}")


def _require_experiment_files(experiment_dir: Path) -> None:
    required_paths = (
        experiment_dir / "experiment_config.json",
        experiment_dir / "phi_sweep_summary.csv",
        experiment_dir / "phi_sweep_summary.json",
        experiment_dir / "phi_sweep_report.md",
        experiment_dir / "summaries" / "per_seed_market_metrics.csv",
        experiment_dir / "summaries" / "per_seed_rl_diagnostics.csv",
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Experiment folder {experiment_dir} is missing required outputs: {missing_list}")


def _safe_float(value: Any) -> float:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    return float(series.iloc[0]) if not series.empty else float("nan")


def _value_at_phi(frame: pd.DataFrame, column: str, phi: float) -> float:
    matches = frame.loc[np.isclose(frame["phi"], float(phi)), column]
    if matches.empty:
        return float("nan")
    return _safe_float(matches.iloc[0])


def _artifact_summary(cap_frame: pd.DataFrame) -> dict[str, float | bool]:
    midprice_gap = pd.to_numeric(cap_frame["midprice_gap_without_missing_side_fraction"], errors="coerce").fillna(0.0)
    defined_midprice = pd.to_numeric(
        cap_frame["defined_midprice_despite_missing_side_fraction"],
        errors="coerce",
    ).fillna(0.0)
    pipeline_issue = pd.to_numeric(cap_frame["pipeline_issue_seed_fraction"], errors="coerce").fillna(0.0)
    return {
        "max_midprice_gap_without_missing_side_fraction": float(midprice_gap.abs().max()) if len(midprice_gap) else 0.0,
        "max_defined_midprice_despite_missing_side_fraction": (
            float(defined_midprice.abs().max()) if len(defined_midprice) else 0.0
        ),
        "max_pipeline_issue_seed_fraction": float(pipeline_issue.max()) if len(pipeline_issue) else 0.0,
        "artifacts_near_zero": bool(
            float(midprice_gap.abs().max()) == 0.0
            and float(defined_midprice.abs().max()) == 0.0
            and float(pipeline_issue.max()) == 0.0
        )
        if len(cap_frame)
        else True,
    }


def _deterioration_assessment(cap_frame: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "volatility": (_value_at_phi(cap_frame, "volatility", 0.30) - _value_at_phi(cap_frame, "volatility", 0.20), "increase"),
        "average_spread": (_value_at_phi(cap_frame, "average_spread", 0.30) - _value_at_phi(cap_frame, "average_spread", 0.20), "increase"),
        "average_depth": (_value_at_phi(cap_frame, "average_depth", 0.30) - _value_at_phi(cap_frame, "average_depth", 0.20), "decrease"),
        "one_sided_book_fraction": (
            _value_at_phi(cap_frame, "one_sided_book_fraction", 0.30) - _value_at_phi(cap_frame, "one_sided_book_fraction", 0.20),
            "increase",
        ),
        "tail_exposure": (_value_at_phi(cap_frame, "tail_exposure", 0.30) - _value_at_phi(cap_frame, "tail_exposure", 0.20), "decrease"),
    }

    score = 0
    available = 0
    deltas: dict[str, float] = {}
    for metric, (delta, direction) in checks.items():
        deltas[metric] = float(delta)
        if np.isnan(delta):
            continue
        available += 1
        if direction == "increase" and delta > 0.0:
            score += 1
        elif direction == "decrease" and delta < 0.0:
            score += 1

    if available < 3:
        status = "inconclusive"
    elif score >= 4:
        status = "yes"
    elif score == 3:
        status = "mixed"
    else:
        status = "no"

    return {
        "status": status,
        "score": int(score),
        "available_checks": int(available),
        "deltas_0_20_to_0_30": deltas,
    }


def _one_sided_rise_assessment(cap_frame: pd.DataFrame) -> dict[str, Any]:
    start_value = _value_at_phi(cap_frame, "one_sided_book_fraction", float(cap_frame["phi"].min()))
    end_value = _value_at_phi(cap_frame, "one_sided_book_fraction", float(cap_frame["phi"].max()))
    threshold_delta = _value_at_phi(cap_frame, "one_sided_book_fraction", 0.30) - _value_at_phi(
        cap_frame,
        "one_sided_book_fraction",
        0.20,
    )
    if np.isnan(start_value) or np.isnan(end_value):
        status = "inconclusive"
    elif end_value > start_value and (np.isnan(threshold_delta) or threshold_delta >= 0.0):
        status = "yes"
    elif end_value > start_value or (not np.isnan(threshold_delta) and threshold_delta > 0.0):
        status = "mixed"
    else:
        status = "no"
    return {
        "status": status,
        "start_value": float(start_value),
        "end_value": float(end_value),
        "delta_0_20_to_0_30": float(threshold_delta),
    }


def _overall_robustness_label(
    *,
    no_cap_status: str,
    cap50_status: str,
    cap20_status: str,
    artifacts_ok: bool,
) -> str:
    if not artifacts_ok:
        return "inconclusive_due_to_artifacts"
    if no_cap_status != "yes":
        return "inconclusive"
    if cap50_status == "yes" and cap20_status == "yes":
        return "robust"
    if cap50_status in {"yes", "mixed"} and cap20_status in {"yes", "mixed"}:
        return "weakened_but_present"
    if cap50_status == "no" and cap20_status == "no":
        return "mostly_driven_by_unconstrained_inventory"
    return "weakened_materially"


def _format_float(value: float, *, suffix: str = "") -> str:
    return f"{value:.6f}{suffix}" if np.isfinite(value) else "n/a"


def _extract_phi_grid(config: Mapping[str, Any]) -> list[float]:
    raw_grid = config.get("phi_grid", [])
    return [float(phi) for phi in raw_grid]


def load_inventory_cap_experiment(
    experiment_dir: str | Path,
    *,
    cap_label: str,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> pd.DataFrame:
    """Load one completed phi-sweep experiment into comparison rows."""

    experiment_root = Path(experiment_dir)
    _require_experiment_files(experiment_root)

    config = json.loads((experiment_root / "experiment_config.json").read_text(encoding="utf-8"))
    summary_frame = pd.read_csv(experiment_root / "phi_sweep_summary.csv")

    expected_cap = _expected_inventory_cap(cap_label)
    config_cap = config.get("inventory_cap", None)
    if config_cap is not None:
        config_cap = int(config_cap)
    if config_cap != expected_cap:
        raise ValueError(
            f"Experiment {experiment_root} has inventory_cap={config_cap}, expected {expected_cap} for {cap_label}."
        )

    prefix = f"{evaluation_mode}_"
    required_columns = {
        "phi",
        *(prefix + source for source in SUMMARY_COLUMN_MAP.values()),
    }
    missing_columns = sorted(required_columns.difference(summary_frame.columns))
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(
            f"Experiment {experiment_root} is missing required {evaluation_mode} summary columns: {missing_list}"
        )

    phi_grid = _extract_phi_grid(config)
    if phi_grid and len(summary_frame) != len(phi_grid):
        raise ValueError(
            f"Experiment {experiment_root} has {len(summary_frame)} summary rows but phi_grid has {len(phi_grid)} values."
        )

    rows: list[dict[str, Any]] = []
    for row in summary_frame.sort_values("phi").itertuples(index=False):
        comparison_row: dict[str, Any] = {
            "phi": float(row.phi),
            "inventory_cap_label": cap_label,
            "inventory_cap_value": expected_cap if expected_cap is not None else float("nan"),
            "evaluation_mode": evaluation_mode,
            "experiment_dir": str(experiment_root),
            "experiment_config_path": str(experiment_root / "experiment_config.json"),
            "summary_csv_path": str(experiment_root / "phi_sweep_summary.csv"),
            "report_md_path": str(experiment_root / "phi_sweep_report.md"),
        }
        for metric, source_suffix in SUMMARY_COLUMN_MAP.items():
            value = getattr(row, prefix + source_suffix)
            comparison_row[metric] = value
        rows.append(comparison_row)
    return pd.DataFrame(rows)


def save_inventory_cap_comparison_plots(comparison_frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save comparison plots grouped by inventory-cap condition."""

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    ordered = comparison_frame.sort_values(["inventory_cap_label", "phi"])
    saved_paths: list[Path] = []

    for metric, title, ylabel, filename in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        finite_values: list[float] = []
        for cap_label in ("no_cap", "cap_50", "cap_20"):
            cap_frame = ordered.loc[ordered["inventory_cap_label"] == cap_label].sort_values("phi")
            if cap_frame.empty:
                continue
            color, legend_label = CAP_STYLES[cap_label]
            y_values = pd.to_numeric(cap_frame[metric], errors="coerce")
            finite_values.extend([float(value) for value in y_values.dropna().tolist()])
            ax.plot(
                cap_frame["phi"],
                y_values,
                marker="o",
                linewidth=1.6,
                label=legend_label,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel("phi")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        if finite_values:
            y_min = min(finite_values)
            y_max = max(finite_values)
            padding = 0.05 * (y_max - y_min) if y_max != y_min else max(0.05 * max(abs(y_max), 1.0), 0.05)
            ax.set_ylim(y_min - padding, y_max + padding)
        fig.tight_layout()
        path = plot_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths


def build_inventory_cap_comparison_report(
    comparison_frame: pd.DataFrame,
    *,
    comparison_inputs: Mapping[str, str | Path],
    evaluation_mode: str,
    plot_paths: list[Path],
) -> str:
    """Build a markdown report that directly answers the robustness questions."""

    assessments: dict[str, dict[str, Any]] = {}
    artifact_ok = True
    for cap_label in ("no_cap", "cap_50", "cap_20"):
        cap_frame = comparison_frame.loc[comparison_frame["inventory_cap_label"] == cap_label].sort_values("phi")
        deterioration = _deterioration_assessment(cap_frame)
        one_sided_rise = _one_sided_rise_assessment(cap_frame)
        artifacts = _artifact_summary(cap_frame)
        artifact_ok = artifact_ok and bool(artifacts["artifacts_near_zero"])
        assessments[cap_label] = {
            "deterioration": deterioration,
            "one_sided_rise": one_sided_rise,
            "artifacts": artifacts,
        }

    overall_label = _overall_robustness_label(
        no_cap_status=str(assessments["no_cap"]["deterioration"]["status"]),
        cap50_status=str(assessments["cap_50"]["deterioration"]["status"]),
        cap20_status=str(assessments["cap_20"]["deterioration"]["status"]),
        artifacts_ok=artifact_ok,
    )

    def answer_line(number: int, label: str, status: str, evidence: str) -> list[str]:
        return [f"{number}. {label}: {status}.", evidence]

    lines = [
        "# Inventory Cap Comparison",
        "",
        "## Inputs",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Evaluation mode used for comparison: {evaluation_mode}",
        f"- no_cap source: {comparison_inputs['no_cap']}",
        f"- cap_50 source: {comparison_inputs['cap_50']}",
        f"- cap_20 source: {comparison_inputs['cap_20']}",
        "",
        "## Direct Answers",
    ]

    for number, cap_label, prompt in (
        (1, "cap_50", "Does the deterioration around phi ≈ 0.30 still exist with cap 50"),
        (2, "cap_20", "Does the deterioration around phi ≈ 0.30 still exist with cap 20"),
    ):
        deterioration = assessments[cap_label]["deterioration"]
        deltas = deterioration["deltas_0_20_to_0_30"]
        evidence = (
            "Evidence from phi 0.20→0.30: "
            f"volatility {_format_float(deltas['volatility'])}, "
            f"spread {_format_float(deltas['average_spread'])}, "
            f"depth {_format_float(deltas['average_depth'])}, "
            f"one-sided {_format_float(deltas['one_sided_book_fraction'])}, "
            f"tail exposure {_format_float(deltas['tail_exposure'])}. "
            f"Deterioration score {deterioration['score']}/{deterioration['available_checks']}."
        )
        lines.extend(answer_line(number, prompt, str(deterioration["status"]), evidence))

    cap50_rise = assessments["cap_50"]["one_sided_rise"]
    cap20_rise = assessments["cap_20"]["one_sided_rise"]
    one_sided_status = "yes" if cap50_rise["status"] == "yes" and cap20_rise["status"] == "yes" else (
        "mixed" if "yes" in {cap50_rise["status"], cap20_rise["status"]} or "mixed" in {cap50_rise["status"], cap20_rise["status"]} else "no"
    )
    lines.extend(
        answer_line(
            3,
            "Does one-sided-book fraction still rise with phi under caps",
            one_sided_status,
            (
                f"cap_50 start/end {_format_float(cap50_rise['start_value'])}→{_format_float(cap50_rise['end_value'])}, "
                f"phi 0.20→0.30 {_format_float(cap50_rise['delta_0_20_to_0_30'])}; "
                f"cap_20 start/end {_format_float(cap20_rise['start_value'])}→{_format_float(cap20_rise['end_value'])}, "
                f"phi 0.20→0.30 {_format_float(cap20_rise['delta_0_20_to_0_30'])}."
            ),
        )
    )

    artifact_status = "yes" if artifact_ok else "no"
    artifact_evidence = []
    for cap_label in ("no_cap", "cap_50", "cap_20"):
        artifacts = assessments[cap_label]["artifacts"]
        artifact_evidence.append(
            f"{cap_label}: gap-without-missing-side {_format_float(artifacts['max_midprice_gap_without_missing_side_fraction'])}, "
            f"defined-midprice-despite-missing-side {_format_float(artifacts['max_defined_midprice_despite_missing_side_fraction'])}, "
            f"pipeline_issue_seed_fraction {_format_float(artifacts['max_pipeline_issue_seed_fraction'])}"
        )
    lines.extend(
        answer_line(
            4,
            "Are the artifact metrics near zero, meaning the one-sided-book results are real",
            artifact_status,
            "Artifact maxima by cap: " + "; ".join(artifact_evidence) + ".",
        )
    )

    overall_text = {
        "robust": "The original no-cap result looks robust to the tested inventory caps.",
        "weakened_but_present": "The original no-cap result weakens under inventory caps but still appears under both capped settings.",
        "weakened_materially": "The original no-cap result weakens materially once inventory is capped.",
        "mostly_driven_by_unconstrained_inventory": "The original no-cap result does not survive the tested caps and is likely driven mostly by unconstrained inventory behavior.",
        "inconclusive_due_to_artifacts": "The comparison is not trustworthy yet because artifact metrics are nonzero in at least one condition.",
        "inconclusive": "The capped comparison is inconclusive from the saved summaries alone.",
    }[overall_label]
    lines.extend(answer_line(5, "Is the original no-cap result likely robust, weakened, or mostly an artifact of unconstrained inventory behavior", overall_label, overall_text))

    lines.extend(
        [
            "",
            "## Saved Plots",
            *[f"- {path.name}" for path in plot_paths],
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_inventory_cap_comparison(
    experiment_dirs: Mapping[str, str | Path],
    *,
    output_dir: str | Path = DEFAULT_COMPARISON_OUTPUT_DIR,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> dict[str, Any]:
    """Read completed sweep folders and write a combined inventory-cap comparison."""

    comparison_inputs = {label: Path(path) for label, path in experiment_dirs.items()}
    missing_labels = {"no_cap", "cap_50", "cap_20"}.difference(comparison_inputs)
    if missing_labels:
        missing_text = ", ".join(sorted(missing_labels))
        raise ValueError(f"Missing required comparison inputs: {missing_text}")

    frames = [
        load_inventory_cap_experiment(comparison_inputs["no_cap"], cap_label="no_cap", evaluation_mode=evaluation_mode),
        load_inventory_cap_experiment(comparison_inputs["cap_50"], cap_label="cap_50", evaluation_mode=evaluation_mode),
        load_inventory_cap_experiment(comparison_inputs["cap_20"], cap_label="cap_20", evaluation_mode=evaluation_mode),
    ]
    comparison_frame = pd.concat(frames, ignore_index=True).sort_values(["inventory_cap_label", "phi"]).reset_index(drop=True)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plot_paths = save_inventory_cap_comparison_plots(comparison_frame, plots_dir)

    csv_path = output_root / "inventory_cap_comparison.csv"
    json_path = output_root / "inventory_cap_comparison.json"
    md_path = output_root / "inventory_cap_comparison.md"
    comparison_frame.to_csv(csv_path, index=False)

    report = build_inventory_cap_comparison_report(
        comparison_frame,
        comparison_inputs=comparison_inputs,
        evaluation_mode=evaluation_mode,
        plot_paths=plot_paths,
    )
    md_path.write_text(report, encoding="utf-8")

    assessments = {
        cap_label: {
            "deterioration": _deterioration_assessment(
                comparison_frame.loc[comparison_frame["inventory_cap_label"] == cap_label].sort_values("phi")
            ),
            "one_sided_rise": _one_sided_rise_assessment(
                comparison_frame.loc[comparison_frame["inventory_cap_label"] == cap_label].sort_values("phi")
            ),
            "artifacts": _artifact_summary(
                comparison_frame.loc[comparison_frame["inventory_cap_label"] == cap_label].sort_values("phi")
            ),
        }
        for cap_label in ("no_cap", "cap_50", "cap_20")
    }
    _write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(),
            "evaluation_mode": evaluation_mode,
            "comparison_inputs": comparison_inputs,
            "plots": [str(path) for path in plot_paths],
            "results": comparison_frame.to_dict(orient="records"),
            "assessments": assessments,
        },
    )

    return {
        "comparison_frame": comparison_frame,
        "csv_path": csv_path,
        "json_path": json_path,
        "md_path": md_path,
        "plot_paths": plot_paths,
        "output_dir": output_root,
    }
