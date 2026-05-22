"""Compare trained and non-learning baseline phi sweeps."""

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

DEFAULT_POLICY_COMPARISON_OUTPUT_DIR = Path("experiments/policy_baseline_comparison")
DEFAULT_EVALUATION_MODE = "greedy"

POLICY_STYLES: dict[str, tuple[str, str]] = {
    "trained": ("tab:blue", "trained"),
    "random_baseline": ("tab:orange", "random_baseline"),
}

SUMMARY_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "average_spread": ("average_spread_mean",),
    "average_depth": ("average_depth_mean",),
    "one_sided_book_fraction": ("one_sided_book_fraction_mean",),
    "undefined_midprice_fraction": ("undefined_midprice_fraction_mean",),
    "ask_side_failure_bias": ("ask_side_failure_bias_mean",),
    "mean_one_sided_episode_duration": ("mean_one_sided_episode_duration_mean",),
    "p90_one_sided_episode_duration": ("p90_one_sided_episode_duration_mean",),
    "quote_activity_fraction": ("quote_activity_fraction_mean",),
    "inactivity_fraction": ("inactivity_fraction_mean",),
    "market_breakdown_score": ("market_breakdown_score",),
    "average_abs_ending_inventory": ("average_abs_ending_inventory_mean", "evaluation_average_abs_ending_inventory_mean"),
    "transition_phi_one_sided": ("transition_phi_one_sided",),
    "transition_phi_undefined_midprice": ("transition_phi_undefined_midprice",),
    "transition_phi_depth_collapse": ("transition_phi_depth_collapse",),
    "transition_phi_spread_widening": ("transition_phi_spread_widening",),
}

PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("average_spread", "Average Spread vs Phi by Policy Type", "Spread", "average_spread_vs_phi_by_policy_type.png"),
    ("average_depth", "Average Depth vs Phi by Policy Type", "Depth", "average_depth_vs_phi_by_policy_type.png"),
    ("one_sided_book_fraction", "One-Sided Book Fraction vs Phi by Policy Type", "Fraction", "one_sided_book_fraction_vs_phi_by_policy_type.png"),
    ("ask_side_failure_bias", "Ask-Side Failure Bias vs Phi by Policy Type", "missing_ask_fraction - missing_bid_fraction", "ask_side_failure_bias_vs_phi_by_policy_type.png"),
    ("quote_activity_fraction", "Quote Activity Fraction vs Phi by Policy Type", "Buy fraction + sell fraction", "quote_activity_fraction_vs_phi_by_policy_type.png"),
    ("inactivity_fraction", "Inactivity Fraction vs Phi by Policy Type", "Hold fraction", "inactivity_fraction_vs_phi_by_policy_type.png"),
    ("market_breakdown_score", "Market Breakdown Score vs Phi by Policy Type", "Composite z-score index", "market_breakdown_score_vs_phi_by_policy_type.png"),
    ("p90_one_sided_episode_duration", "P90 One-Sided Episode Duration vs Phi by Policy Type", "Seconds", "p90_one_sided_episode_duration_vs_phi_by_policy_type.png"),
)


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_policy_baseline_comparison_matplotlib"
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


def _safe_float(value: Any) -> float:
    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    return float(series.iloc[0]) if not series.empty else float("nan")


def _format_float(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else "n/a"


def _value_at_phi(frame: pd.DataFrame, column: str, phi: float) -> float:
    matches = frame.loc[np.isclose(frame["phi"], float(phi)), column]
    if matches.empty:
        return float("nan")
    return _safe_float(matches.iloc[0])


def _resolve_prefixed_summary_column(
    summary_frame: pd.DataFrame,
    *,
    prefix: str,
    candidates: tuple[str, ...],
) -> str | None:
    for suffix in candidates:
        column = prefix + suffix
        if column in summary_frame.columns:
            return column
    return None


def load_policy_experiment(
    experiment_dir: str | Path,
    *,
    policy_label: str,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> pd.DataFrame:
    experiment_root = Path(experiment_dir)
    config = json.loads((experiment_root / "experiment_config.json").read_text(encoding="utf-8"))
    summary_frame = pd.read_csv(experiment_root / "phi_sweep_summary.csv")
    prefix = f"{evaluation_mode}_"

    if config.get("policy_type") != policy_label:
        raise ValueError(f"Experiment {experiment_root} has policy_type={config.get('policy_type')}, expected {policy_label}.")

    resolved_columns: dict[str, str] = {}
    missing_columns: list[str] = []
    for metric, candidates in SUMMARY_COLUMN_CANDIDATES.items():
        resolved = _resolve_prefixed_summary_column(summary_frame, prefix=prefix, candidates=candidates)
        if resolved is None:
            missing_columns.append(f"{metric} ({', '.join(prefix + candidate for candidate in candidates)})")
        else:
            resolved_columns[metric] = resolved
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Experiment {experiment_root} is missing required {evaluation_mode} summary columns: {missing_text}")

    rows: list[dict[str, Any]] = []
    for row in summary_frame.sort_values("phi").itertuples(index=False):
        comparison_row: dict[str, Any] = {
            "phi": float(row.phi),
            "policy_type": policy_label,
            "evaluation_mode": evaluation_mode,
            "experiment_dir": str(experiment_root),
            "experiment_config_path": str(experiment_root / "experiment_config.json"),
            "summary_csv_path": str(experiment_root / "phi_sweep_summary.csv"),
            "report_md_path": str(experiment_root / "phi_sweep_report.md"),
        }
        for metric, resolved_column in resolved_columns.items():
            comparison_row[metric] = getattr(row, resolved_column)
        rows.append(comparison_row)
    return pd.DataFrame(rows)


def save_policy_comparison_plots(comparison_frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    ordered = comparison_frame.sort_values(["policy_type", "phi"])
    saved_paths: list[Path] = []

    for metric, title, ylabel, filename in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        finite_values: list[float] = []
        for policy_label in ("trained", "random_baseline"):
            policy_frame = ordered.loc[ordered["policy_type"] == policy_label].sort_values("phi")
            if policy_frame.empty:
                continue
            color, legend_label = POLICY_STYLES[policy_label]
            y_values = pd.to_numeric(policy_frame[metric], errors="coerce")
            finite_values.extend([float(value) for value in y_values.dropna().tolist()])
            ax.plot(policy_frame["phi"], y_values, marker="o", linewidth=1.6, label=legend_label, color=color)
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


def _policy_delta_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "one_sided_delta_0_20_to_0_30": _value_at_phi(frame, "one_sided_book_fraction", 0.30)
        - _value_at_phi(frame, "one_sided_book_fraction", 0.20),
        "spread_delta_0_20_to_0_30": _value_at_phi(frame, "average_spread", 0.30)
        - _value_at_phi(frame, "average_spread", 0.20),
        "depth_delta_0_20_to_0_30": _value_at_phi(frame, "average_depth", 0.30)
        - _value_at_phi(frame, "average_depth", 0.20),
        "quote_activity_delta_0_20_to_0_30": _value_at_phi(frame, "quote_activity_fraction", 0.30)
        - _value_at_phi(frame, "quote_activity_fraction", 0.20),
        "inactivity_delta_0_20_to_0_30": _value_at_phi(frame, "inactivity_fraction", 0.30)
        - _value_at_phi(frame, "inactivity_fraction", 0.20),
        "breakdown_delta_0_20_to_0_30": _value_at_phi(frame, "market_breakdown_score", 0.30)
        - _value_at_phi(frame, "market_breakdown_score", 0.20),
        "transition_phi_one_sided": _value_at_phi(frame, "transition_phi_one_sided", float(frame["phi"].min())),
    }


def build_policy_baseline_comparison_report(
    comparison_frame: pd.DataFrame,
    *,
    comparison_inputs: Mapping[str, str | Path],
    evaluation_mode: str,
    plot_paths: list[Path],
) -> str:
    trained_frame = comparison_frame.loc[comparison_frame["policy_type"] == "trained"].sort_values("phi")
    baseline_frame = comparison_frame.loc[comparison_frame["policy_type"] == "random_baseline"].sort_values("phi")
    trained_deltas = _policy_delta_summary(trained_frame)
    baseline_deltas = _policy_delta_summary(baseline_frame)

    lines = [
        "# Policy Baseline Comparison",
        "",
        "## Inputs",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Evaluation mode used for comparison: {evaluation_mode}",
        f"- trained source: {comparison_inputs['trained']}",
        f"- random_baseline source: {comparison_inputs['random_baseline']}",
        "",
        "## Direct Answers",
        (
            "1. Does the learned policy deteriorate more sharply than the random baseline around phi ≈ 0.30: "
            f"trained one-sided Δ {_format_float(trained_deltas['one_sided_delta_0_20_to_0_30'])}, "
            f"spread Δ {_format_float(trained_deltas['spread_delta_0_20_to_0_30'])}, "
            f"depth Δ {_format_float(trained_deltas['depth_delta_0_20_to_0_30'])}, "
            f"breakdown Δ {_format_float(trained_deltas['breakdown_delta_0_20_to_0_30'])}; "
            f"baseline one-sided Δ {_format_float(baseline_deltas['one_sided_delta_0_20_to_0_30'])}, "
            f"spread Δ {_format_float(baseline_deltas['spread_delta_0_20_to_0_30'])}, "
            f"depth Δ {_format_float(baseline_deltas['depth_delta_0_20_to_0_30'])}, "
            f"breakdown Δ {_format_float(baseline_deltas['breakdown_delta_0_20_to_0_30'])}."
        ),
        (
            "2. Is the mechanism stronger under the learned policy: "
            f"trained quote-activity Δ {_format_float(trained_deltas['quote_activity_delta_0_20_to_0_30'])}, "
            f"inactivity Δ {_format_float(trained_deltas['inactivity_delta_0_20_to_0_30'])}; "
            f"baseline quote-activity Δ {_format_float(baseline_deltas['quote_activity_delta_0_20_to_0_30'])}, "
            f"inactivity Δ {_format_float(baseline_deltas['inactivity_delta_0_20_to_0_30'])}."
        ),
        (
            "3. When does one-sidedness first exceed the threshold: "
            f"trained transition phi {_format_float(trained_deltas['transition_phi_one_sided'])}; "
            f"baseline transition phi {_format_float(baseline_deltas['transition_phi_one_sided'])}."
        ),
        "",
        "## Saved Plots",
        *[f"- {path.name}" for path in plot_paths],
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_policy_baseline_comparison(
    experiment_dirs: Mapping[str, str | Path],
    *,
    output_dir: str | Path = DEFAULT_POLICY_COMPARISON_OUTPUT_DIR,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> dict[str, Any]:
    comparison_inputs = {label: Path(path) for label, path in experiment_dirs.items()}
    missing_labels = {"trained", "random_baseline"}.difference(comparison_inputs)
    if missing_labels:
        missing_text = ", ".join(sorted(missing_labels))
        raise ValueError(f"Missing required comparison inputs: {missing_text}")

    frames = [
        load_policy_experiment(comparison_inputs["trained"], policy_label="trained", evaluation_mode=evaluation_mode),
        load_policy_experiment(
            comparison_inputs["random_baseline"],
            policy_label="random_baseline",
            evaluation_mode=evaluation_mode,
        ),
    ]
    comparison_frame = pd.concat(frames, ignore_index=True).sort_values(["policy_type", "phi"]).reset_index(drop=True)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plot_paths = save_policy_comparison_plots(comparison_frame, plots_dir)

    csv_path = output_root / "policy_baseline_comparison.csv"
    json_path = output_root / "policy_baseline_comparison.json"
    md_path = output_root / "policy_baseline_comparison.md"
    comparison_frame.to_csv(csv_path, index=False)

    report = build_policy_baseline_comparison_report(
        comparison_frame,
        comparison_inputs=comparison_inputs,
        evaluation_mode=evaluation_mode,
        plot_paths=plot_paths,
    )
    md_path.write_text(report, encoding="utf-8")
    _write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(),
            "evaluation_mode": evaluation_mode,
            "comparison_inputs": comparison_inputs,
            "plots": [str(path) for path in plot_paths],
            "results": comparison_frame.to_dict(orient="records"),
            "trained_summary": _policy_delta_summary(
                comparison_frame.loc[comparison_frame["policy_type"] == "trained"].sort_values("phi")
            ),
            "random_baseline_summary": _policy_delta_summary(
                comparison_frame.loc[comparison_frame["policy_type"] == "random_baseline"].sort_values("phi")
            ),
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
