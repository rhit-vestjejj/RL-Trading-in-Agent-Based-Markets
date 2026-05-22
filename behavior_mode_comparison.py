"""Compare trained baseline behavior against trained anti-degeneracy behavior."""

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

BASELINE_BEHAVIOR_LABEL = "trained_baseline_behavior"
ANTI_DEGENERACY_BEHAVIOR_LABEL = "trained_anti_degeneracy_behavior"
DEFAULT_BEHAVIOR_MODE_COMPARISON_OUTPUT_DIR = Path("experiments/behavior_mode_comparison")
DEFAULT_EVALUATION_MODE = "stochastic"

BEHAVIOR_STYLES: dict[str, tuple[str, str]] = {
    BASELINE_BEHAVIOR_LABEL: ("tab:blue", "trained_baseline_behavior"),
    ANTI_DEGENERACY_BEHAVIOR_LABEL: ("tab:orange", "trained_anti_degeneracy_behavior"),
}

SUMMARY_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "quote_activity_fraction": ("quote_activity_fraction_mean",),
    "inactivity_fraction": ("inactivity_fraction_mean",),
    "one_sided_book_fraction": ("one_sided_book_fraction_mean",),
    "ask_side_failure_bias": ("ask_side_failure_bias_mean",),
    "p90_one_sided_episode_duration": ("p90_one_sided_episode_duration_mean",),
    "mean_hold_streak": ("mean_hold_streak_mean",),
    "max_hold_streak": ("max_hold_streak_mean",),
    "fraction_of_agents_with_near_total_inactivity": ("fraction_of_agents_with_near_total_inactivity_mean",),
    "market_breakdown_score": ("market_breakdown_score",),
    "anti_degeneracy_penalty_mean": ("anti_degeneracy_penalty_mean_mean",),
    "average_abs_ending_inventory": ("average_abs_ending_inventory_mean", "evaluation_average_abs_ending_inventory_mean"),
    "evaluation_inventory_at_cap_fraction": ("evaluation_inventory_at_cap_fraction_mean",),
    "transition_phi_one_sided": ("transition_phi_one_sided",),
    "fraction_of_decisions_in_hold_streak_ge_3": ("fraction_of_decisions_in_hold_streak_ge_3_mean",),
    "fraction_of_decisions_in_hold_streak_ge_5": ("fraction_of_decisions_in_hold_streak_ge_5_mean",),
}

PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("inactivity_fraction", "Inactivity Fraction vs Phi by Behavior Mode", "Hold fraction", "inactivity_fraction_vs_phi_by_behavior_mode.png"),
    ("quote_activity_fraction", "Quote Activity Fraction vs Phi by Behavior Mode", "Buy fraction + sell fraction", "quote_activity_fraction_vs_phi_by_behavior_mode.png"),
    ("one_sided_book_fraction", "One-Sided Book Fraction vs Phi by Behavior Mode", "Fraction", "one_sided_book_fraction_vs_phi_by_behavior_mode.png"),
    ("ask_side_failure_bias", "Ask-Side Failure Bias vs Phi by Behavior Mode", "missing_ask_fraction - missing_bid_fraction", "ask_side_failure_bias_vs_phi_by_behavior_mode.png"),
    ("p90_one_sided_episode_duration", "P90 One-Sided Episode Duration vs Phi by Behavior Mode", "Seconds", "p90_one_sided_episode_duration_vs_phi_by_behavior_mode.png"),
    ("mean_hold_streak", "Mean Hold Streak vs Phi by Behavior Mode", "Streak length", "mean_hold_streak_vs_phi_by_behavior_mode.png"),
    ("max_hold_streak", "Max Hold Streak vs Phi by Behavior Mode", "Streak length", "max_hold_streak_vs_phi_by_behavior_mode.png"),
    ("fraction_of_agents_with_near_total_inactivity", "Fraction Of Agents With Near-Total Inactivity vs Phi by Behavior Mode", "Fraction", "fraction_agents_near_total_inactivity_vs_phi_by_behavior_mode.png"),
    ("market_breakdown_score", "Market Breakdown Score vs Phi by Behavior Mode", "Composite z-score index", "market_breakdown_score_vs_phi_by_behavior_mode.png"),
    ("anti_degeneracy_penalty_mean", "Anti-Degeneracy Penalty Mean vs Phi", "Penalty", "anti_degeneracy_penalty_mean_vs_phi.png"),
)


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_behavior_mode_comparison_matplotlib"
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


def load_behavior_experiment(
    experiment_dir: str | Path,
    *,
    behavior_label: str,
    expected_anti_degeneracy_mode: str,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> pd.DataFrame:
    experiment_root = Path(experiment_dir)
    config = json.loads((experiment_root / "experiment_config.json").read_text(encoding="utf-8"))
    summary_frame = pd.read_csv(experiment_root / "phi_sweep_summary.csv")
    prefix = f"{evaluation_mode}_"

    if str(config.get("policy_type", "")).strip().lower() != "trained":
        raise ValueError(f"Experiment {experiment_root} must have policy_type=trained.")
    experiment_mode = str(config.get("anti_degeneracy_mode", "off")).strip().lower()
    if experiment_mode != expected_anti_degeneracy_mode:
        raise ValueError(
            f"Experiment {experiment_root} has anti_degeneracy_mode={experiment_mode}, expected {expected_anti_degeneracy_mode}."
        )

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
            "behavior_mode": behavior_label,
            "evaluation_mode": evaluation_mode,
            "experiment_dir": str(experiment_root),
            "experiment_config_path": str(experiment_root / "experiment_config.json"),
            "summary_csv_path": str(experiment_root / "phi_sweep_summary.csv"),
            "report_md_path": str(experiment_root / "phi_sweep_report.md"),
            "anti_degeneracy_mode": experiment_mode,
            "hold_streak_penalty": float(config.get("hold_streak_penalty", 0.0)),
            "hold_streak_grace": int(config.get("hold_streak_grace", 0)),
        }
        for metric, resolved_column in resolved_columns.items():
            comparison_row[metric] = getattr(row, resolved_column)
        rows.append(comparison_row)
    return pd.DataFrame(rows)


def save_behavior_mode_comparison_plots(comparison_frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    ordered = comparison_frame.sort_values(["behavior_mode", "phi"])
    saved_paths: list[Path] = []

    for metric, title, ylabel, filename in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        finite_values: list[float] = []
        for behavior_label in (BASELINE_BEHAVIOR_LABEL, ANTI_DEGENERACY_BEHAVIOR_LABEL):
            behavior_frame = ordered.loc[ordered["behavior_mode"] == behavior_label].sort_values("phi")
            if behavior_frame.empty:
                continue
            color, legend_label = BEHAVIOR_STYLES[behavior_label]
            y_values = pd.to_numeric(behavior_frame[metric], errors="coerce")
            finite_values.extend([float(value) for value in y_values.dropna().tolist()])
            ax.plot(behavior_frame["phi"], y_values, marker="o", linewidth=1.6, label=legend_label, color=color)
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


def _high_phi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame.loc[pd.to_numeric(frame["phi"], errors="coerce") >= 0.20].sort_values("phi")
    return subset if not subset.empty else frame.sort_values("phi")


def _mean_metric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _first_metric(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


def _behavior_profile(frame: pd.DataFrame) -> dict[str, float]:
    high_phi = _high_phi_frame(frame)
    return {
        "high_phi_mean_inactivity_fraction": _mean_metric(high_phi, "inactivity_fraction"),
        "high_phi_mean_quote_activity_fraction": _mean_metric(high_phi, "quote_activity_fraction"),
        "high_phi_mean_one_sided_book_fraction": _mean_metric(high_phi, "one_sided_book_fraction"),
        "high_phi_mean_ask_side_failure_bias": _mean_metric(high_phi, "ask_side_failure_bias"),
        "high_phi_mean_p90_one_sided_episode_duration": _mean_metric(high_phi, "p90_one_sided_episode_duration"),
        "high_phi_mean_mean_hold_streak": _mean_metric(high_phi, "mean_hold_streak"),
        "high_phi_mean_max_hold_streak": _mean_metric(high_phi, "max_hold_streak"),
        "high_phi_mean_fraction_of_agents_with_near_total_inactivity": _mean_metric(
            high_phi,
            "fraction_of_agents_with_near_total_inactivity",
        ),
        "high_phi_mean_market_breakdown_score": _mean_metric(high_phi, "market_breakdown_score"),
        "high_phi_mean_anti_degeneracy_penalty_mean": _mean_metric(high_phi, "anti_degeneracy_penalty_mean"),
        "high_phi_mean_average_abs_ending_inventory": _mean_metric(high_phi, "average_abs_ending_inventory"),
        "high_phi_mean_evaluation_inventory_at_cap_fraction": _mean_metric(
            high_phi,
            "evaluation_inventory_at_cap_fraction",
        ),
        "transition_phi_one_sided": _first_metric(frame.sort_values("phi"), "transition_phi_one_sided"),
    }


def _interpret_inactivity(baseline: dict[str, float], anti: dict[str, float]) -> str:
    reduced_signals = 0
    for anti_key, baseline_key in (
        ("high_phi_mean_inactivity_fraction", "high_phi_mean_inactivity_fraction"),
        ("high_phi_mean_mean_hold_streak", "high_phi_mean_mean_hold_streak"),
        (
            "high_phi_mean_fraction_of_agents_with_near_total_inactivity",
            "high_phi_mean_fraction_of_agents_with_near_total_inactivity",
        ),
    ):
        anti_value = anti.get(anti_key, float("nan"))
        baseline_value = baseline.get(baseline_key, float("nan"))
        if np.isfinite(anti_value) and np.isfinite(baseline_value) and anti_value <= 0.9 * baseline_value:
            reduced_signals += 1
    if reduced_signals >= 2:
        return "Inactivity collapsed substantially less under the anti-degeneracy intervention."
    if reduced_signals == 1:
        return "Inactivity weakens modestly under the anti-degeneracy intervention."
    return "Inactivity changes little under the anti-degeneracy intervention."


def _interpret_one_sided(baseline: dict[str, float], anti: dict[str, float]) -> str:
    baseline_value = baseline.get("high_phi_mean_one_sided_book_fraction", float("nan"))
    anti_value = anti.get("high_phi_mean_one_sided_book_fraction", float("nan"))
    if np.isfinite(anti_value) and anti_value >= 0.05 and (
        not np.isfinite(baseline_value) or anti_value >= 0.5 * baseline_value
    ):
        return "The one-sided-book phenomenon survives anti-degeneracy intervention."
    if np.isfinite(anti_value) and anti_value >= 0.05:
        return "The one-sided-book phenomenon weakens materially but remains present after the anti-degeneracy intervention."
    if np.isfinite(baseline_value) and baseline_value >= 0.10:
        return "The one-sided-book phenomenon appears largely attributable to inactivity collapse."
    return "The one-sided-book evidence is mixed after the anti-degeneracy intervention."


def _interpret_persistence(baseline: dict[str, float], anti: dict[str, float]) -> str:
    baseline_value = baseline.get("high_phi_mean_p90_one_sided_episode_duration", float("nan"))
    anti_value = anti.get("high_phi_mean_p90_one_sided_episode_duration", float("nan"))
    if np.isfinite(anti_value) and anti_value >= 5.0 and (
        not np.isfinite(baseline_value) or anti_value >= 0.5 * baseline_value
    ):
        return "Persistence of one-sided episodes survives anti-degeneracy intervention."
    if np.isfinite(anti_value) and anti_value > 0.0:
        return "Persistence weakens materially but does not disappear under the anti-degeneracy intervention."
    return "Persistent one-sided episodes appear largely attributable to inactivity collapse."


def _interpret_asymmetry(baseline: dict[str, float], anti: dict[str, float]) -> str:
    baseline_value = baseline.get("high_phi_mean_ask_side_failure_bias", float("nan"))
    anti_value = anti.get("high_phi_mean_ask_side_failure_bias", float("nan"))
    if np.isfinite(baseline_value) and np.isfinite(anti_value) and baseline_value > 0.0 and anti_value > 0.0:
        if anti_value >= 0.5 * baseline_value:
            return "The ask-side failure bias survives anti-degeneracy intervention."
        return "The ask-side failure bias remains positive but weakens materially."
    if np.isfinite(anti_value) and anti_value > 0.0:
        return "Ask-side asymmetry remains present after the anti-degeneracy intervention."
    return "Ask-side asymmetry weakens materially after the anti-degeneracy intervention."


def _interpret_inventory_signature(baseline: dict[str, float], anti: dict[str, float]) -> str:
    baseline_cap = baseline.get("high_phi_mean_evaluation_inventory_at_cap_fraction", float("nan"))
    anti_cap = anti.get("high_phi_mean_evaluation_inventory_at_cap_fraction", float("nan"))
    baseline_inv = baseline.get("high_phi_mean_average_abs_ending_inventory", float("nan"))
    anti_inv = anti.get("high_phi_mean_average_abs_ending_inventory", float("nan"))
    if (
        np.isfinite(anti_cap)
        and np.isfinite(baseline_cap)
        and anti_cap >= 0.5 * baseline_cap
        and np.isfinite(anti_inv)
        and np.isfinite(baseline_inv)
        and anti_inv >= 0.5 * baseline_inv
    ):
        return "The inventory-cap contact signature remains visible after the anti-degeneracy intervention."
    if np.isfinite(anti_cap) and anti_cap > 0.0:
        return "The inventory-cap contact signature weakens materially but remains present."
    return "The saved cap-contact signature weakens sharply under the anti-degeneracy intervention."


def _interpret_main_story(one_sided_text: str, persistence_text: str, asymmetry_text: str) -> str:
    positive_signals = sum(
        "survives anti-degeneracy intervention" in text
        for text in (one_sided_text, persistence_text, asymmetry_text)
    )
    if positive_signals >= 2:
        return "The main qualitative story survives anti-degeneracy intervention."
    if positive_signals == 1:
        return "The main qualitative story weakens materially under the anti-degeneracy intervention."
    return "The main qualitative story appears largely attributable to inactivity collapse."


def build_behavior_mode_comparison_report(
    comparison_frame: pd.DataFrame,
    *,
    comparison_inputs: Mapping[str, str | Path],
    evaluation_mode: str,
    plot_paths: list[Path],
) -> str:
    baseline_frame = comparison_frame.loc[comparison_frame["behavior_mode"] == BASELINE_BEHAVIOR_LABEL].sort_values("phi")
    anti_frame = comparison_frame.loc[comparison_frame["behavior_mode"] == ANTI_DEGENERACY_BEHAVIOR_LABEL].sort_values("phi")
    baseline_profile = _behavior_profile(baseline_frame)
    anti_profile = _behavior_profile(anti_frame)

    inactivity_text = _interpret_inactivity(baseline_profile, anti_profile)
    one_sided_text = _interpret_one_sided(baseline_profile, anti_profile)
    persistence_text = _interpret_persistence(baseline_profile, anti_profile)
    asymmetry_text = _interpret_asymmetry(baseline_profile, anti_profile)
    inventory_text = _interpret_inventory_signature(baseline_profile, anti_profile)
    main_story_text = _interpret_main_story(one_sided_text, persistence_text, asymmetry_text)

    lines = [
        "# Behavior Mode Comparison",
        "",
        "## Inputs",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Evaluation mode used for comparison: {evaluation_mode}",
        f"- trained_baseline_behavior source: {comparison_inputs['baseline']}",
        f"- trained_anti_degeneracy_behavior source: {comparison_inputs['anti_degeneracy']}",
        "",
        "## Direct Answers",
        (
            "1. Did inactivity collapse decrease: "
            f"baseline high-phi inactivity {_format_float(baseline_profile['high_phi_mean_inactivity_fraction'])}, "
            f"mean hold streak {_format_float(baseline_profile['high_phi_mean_mean_hold_streak'])}, "
            f"near-total inactivity {_format_float(baseline_profile['high_phi_mean_fraction_of_agents_with_near_total_inactivity'])}; "
            f"anti-degeneracy high-phi inactivity {_format_float(anti_profile['high_phi_mean_inactivity_fraction'])}, "
            f"mean hold streak {_format_float(anti_profile['high_phi_mean_mean_hold_streak'])}, "
            f"near-total inactivity {_format_float(anti_profile['high_phi_mean_fraction_of_agents_with_near_total_inactivity'])}. "
            f"{inactivity_text}"
        ),
        (
            "2. Did one-sided-book failures remain: "
            f"baseline high-phi one-sided {_format_float(baseline_profile['high_phi_mean_one_sided_book_fraction'])}, "
            f"transition phi {_format_float(baseline_profile['transition_phi_one_sided'])}; "
            f"anti-degeneracy high-phi one-sided {_format_float(anti_profile['high_phi_mean_one_sided_book_fraction'])}, "
            f"transition phi {_format_float(anti_profile['transition_phi_one_sided'])}. "
            f"{one_sided_text}"
        ),
        (
            "3. Did the ask-side failure bias remain: "
            f"baseline high-phi ask-bias {_format_float(baseline_profile['high_phi_mean_ask_side_failure_bias'])}; "
            f"anti-degeneracy high-phi ask-bias {_format_float(anti_profile['high_phi_mean_ask_side_failure_bias'])}. "
            f"{asymmetry_text}"
        ),
        (
            "4. Did persistence of one-sided episodes remain: "
            f"baseline high-phi p90 duration {_format_float(baseline_profile['high_phi_mean_p90_one_sided_episode_duration'])}; "
            f"anti-degeneracy high-phi p90 duration {_format_float(anti_profile['high_phi_mean_p90_one_sided_episode_duration'])}. "
            f"{persistence_text}"
        ),
        (
            "5. Did the inventory-cap signature remain: "
            f"baseline high-phi avg abs ending inventory {_format_float(baseline_profile['high_phi_mean_average_abs_ending_inventory'])}, "
            f"cap-contact {_format_float(baseline_profile['high_phi_mean_evaluation_inventory_at_cap_fraction'])}; "
            f"anti-degeneracy avg abs ending inventory {_format_float(anti_profile['high_phi_mean_average_abs_ending_inventory'])}, "
            f"cap-contact {_format_float(anti_profile['high_phi_mean_evaluation_inventory_at_cap_fraction'])}. "
            f"{inventory_text}"
        ),
        "",
        "## Interpretation",
        f"- {main_story_text}",
        f"- {inactivity_text}",
        f"- {one_sided_text}",
        f"- {persistence_text}",
        f"- {asymmetry_text}",
        "",
        "## Saved Plots",
        *[f"- {path.name}" for path in plot_paths],
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_behavior_mode_comparison(
    experiment_dirs: Mapping[str, str | Path],
    *,
    output_dir: str | Path = DEFAULT_BEHAVIOR_MODE_COMPARISON_OUTPUT_DIR,
    evaluation_mode: str = DEFAULT_EVALUATION_MODE,
) -> dict[str, Any]:
    comparison_inputs = {label: Path(path) for label, path in experiment_dirs.items()}
    missing_labels = {"baseline", "anti_degeneracy"}.difference(comparison_inputs)
    if missing_labels:
        missing_text = ", ".join(sorted(missing_labels))
        raise ValueError(f"Missing required comparison inputs: {missing_text}")

    frames = [
        load_behavior_experiment(
            comparison_inputs["baseline"],
            behavior_label=BASELINE_BEHAVIOR_LABEL,
            expected_anti_degeneracy_mode="off",
            evaluation_mode=evaluation_mode,
        ),
        load_behavior_experiment(
            comparison_inputs["anti_degeneracy"],
            behavior_label=ANTI_DEGENERACY_BEHAVIOR_LABEL,
            expected_anti_degeneracy_mode="hold_streak_penalty",
            evaluation_mode=evaluation_mode,
        ),
    ]
    comparison_frame = pd.concat(frames, ignore_index=True).sort_values(["behavior_mode", "phi"]).reset_index(drop=True)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / "plots"
    plot_paths = save_behavior_mode_comparison_plots(comparison_frame, plots_dir)

    csv_path = output_root / "behavior_mode_comparison.csv"
    json_path = output_root / "behavior_mode_comparison.json"
    md_path = output_root / "behavior_mode_comparison.md"
    comparison_frame.to_csv(csv_path, index=False)

    baseline_profile = _behavior_profile(
        comparison_frame.loc[comparison_frame["behavior_mode"] == BASELINE_BEHAVIOR_LABEL].sort_values("phi")
    )
    anti_profile = _behavior_profile(
        comparison_frame.loc[comparison_frame["behavior_mode"] == ANTI_DEGENERACY_BEHAVIOR_LABEL].sort_values("phi")
    )

    report = build_behavior_mode_comparison_report(
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
            "trained_baseline_behavior_summary": baseline_profile,
            "trained_anti_degeneracy_behavior_summary": anti_profile,
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
