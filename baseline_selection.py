"""Select and report a recommended phi=0 baseline from sweep outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from calibration import (
    CALIBRATION_PARAMETER_COLUMNS,
    CLASSIFICATION_ORDER,
    DiagnosticThresholds,
    summarize_parameter_sweep,
)

KEY_SUMMARY_COLUMNS = [
    "average_spread_mean",
    "average_spread_std",
    "average_relative_spread_mean",
    "average_relative_spread_std",
    "average_top_of_book_depth_mean",
    "average_top_of_book_depth_std",
    "return_variance_mean",
    "return_variance_std",
    "crash_rate_mean",
    "crash_rate_std",
    "tail_exposure_mean",
    "tail_exposure_std",
    "max_drawdown_mean",
    "max_drawdown_std",
    "trade_count_mean",
    "trade_count_std",
    "traded_volume_mean",
    "traded_volume_std",
    "mm_mean_abs_inventory_mean",
    "mm_mean_abs_inventory_std",
    "num_runs",
]


@dataclass(frozen=True)
class SelectionWeights:
    """Transparent point weights for candidate ranking."""

    spread: float = 20.0
    relative_spread: float = 5.0
    depth: float = 15.0
    trade_count: float = 15.0
    return_activity: float = 10.0
    crash_rate: float = 15.0
    tail_exposure: float = 8.0
    stable_share: float = 8.0
    stability: float = 10.0
    market_maker_inventory: float = 4.0
    thin_share: float = 4.0
    frozen_share: float = 8.0
    chaotic_share: float = 6.0
    crash_prone_share: float = 12.0


@dataclass(frozen=True)
class RobustnessConfig:
    """Simple thresholds for the nearby-setting robustness check."""

    score_gap: float = 8.0
    min_stable_share: float = 0.5
    min_similar_fraction: float = 0.5


def _parse_override(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise ValueError(f"Invalid override '{spec}'. Expected name=value.")
    name, raw_value = spec.split("=", 1)
    return name.strip(), float(raw_value.strip())


def apply_weight_overrides(
    weights: SelectionWeights,
    overrides: Sequence[str],
) -> SelectionWeights:
    """Apply repeated ``name=value`` weight overrides."""

    if not overrides:
        return weights

    valid_names = {field.name for field in fields(SelectionWeights)}
    updates: dict[str, float] = {}
    for spec in overrides:
        name, value = _parse_override(spec)
        if name not in valid_names:
            supported = ", ".join(sorted(valid_names))
            raise ValueError(f"Unsupported weight '{name}'. Supported: {supported}.")
        updates[name] = float(value)
    return replace(weights, **updates)


def apply_threshold_overrides(
    thresholds: DiagnosticThresholds,
    overrides: Sequence[str],
) -> DiagnosticThresholds:
    """Apply repeated ``name=value`` threshold overrides."""

    if not overrides:
        return thresholds

    valid_values = {
        field.name: getattr(thresholds, field.name)
        for field in fields(DiagnosticThresholds)
    }
    updates: dict[str, Any] = {}
    for spec in overrides:
        name, value = _parse_override(spec)
        if name not in valid_values:
            supported = ", ".join(sorted(valid_values))
            raise ValueError(f"Unsupported threshold '{name}'. Supported: {supported}.")
        updates[name] = int(value) if isinstance(valid_values[name], int) else float(value)
    return replace(thresholds, **updates)


def infer_parameter_columns(frame: pd.DataFrame) -> list[str]:
    """Return calibration parameter columns present in the given frame."""

    return [column for column in CALIBRATION_PARAMETER_COLUMNS if column in frame.columns]


def load_selection_inputs(
    *,
    runs_input: str | Path | None = None,
    summary_input: str | Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, list[str]]:
    """Load run-level and/or summary sweep results.

    Run-level input takes precedence because the grouped summary can be
    reconstructed deterministically from it using the current code.
    """

    if not runs_input and not summary_input:
        raise ValueError("At least one of runs_input or summary_input must be provided.")

    runs: pd.DataFrame | None = None
    if runs_input:
        runs = pd.read_csv(runs_input)
        parameter_columns = infer_parameter_columns(runs)
        summary = summarize_parameter_sweep(runs, parameter_columns=parameter_columns)
        return runs, prepare_summary(summary), parameter_columns

    summary = pd.read_csv(summary_input)
    parameter_columns = infer_parameter_columns(summary)
    return None, prepare_summary(summary), parameter_columns


def prepare_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Fill optional summary fields so ranking is stable on sparse inputs."""

    working = summary.copy()
    for column in KEY_SUMMARY_COLUMNS:
        if column not in working.columns:
            working[column] = 0.0

    share_columns = [f"share_{label}" for label in CLASSIFICATION_ORDER]
    for column in share_columns:
        if column not in working.columns:
            working[column] = 0.0

    for column in working.columns:
        if column.endswith("_std"):
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    if "num_runs" in working.columns:
        working["num_runs"] = pd.to_numeric(working["num_runs"], errors="coerce").fillna(1).astype(int)
    else:
        working["num_runs"] = 1

    return working


def _normalize_series(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().nunique() <= 1:
        return pd.Series(1.0, index=values.index, dtype=float)

    lower = float(numeric.min())
    upper = float(numeric.max())
    scale = (numeric - lower) / (upper - lower)
    normalized = scale if higher_is_better else 1.0 - scale
    return normalized.fillna(0.0).clip(lower=0.0, upper=1.0)


def _band_score(values: pd.Series, *, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if lower <= 0 or upper <= 0 or lower >= upper:
        return pd.Series(1.0, index=values.index, dtype=float)

    scores = pd.Series(1.0, index=values.index, dtype=float)
    below = numeric < lower
    above = numeric > upper
    scores.loc[below] = (numeric.loc[below] / lower).clip(lower=0.0, upper=1.0)
    scores.loc[above] = (upper / numeric.loc[above]).clip(lower=0.0, upper=1.0)
    return scores.fillna(0.0)


def _negative_tail_magnitude(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return numeric.clip(upper=0.0).abs()


def build_candidate_label(row: Mapping[str, Any], parameter_columns: Sequence[str]) -> str:
    """Return a compact human-readable parameter label."""

    if not parameter_columns:
        return "baseline"
    return ", ".join(f"{column}={row[column]}" for column in parameter_columns)


def dominant_class(row: Mapping[str, Any]) -> str:
    """Return the class label with the largest share."""

    shares = {
        label: float(row.get(f"share_{label}", 0.0))
        for label in CLASSIFICATION_ORDER
    }
    return max(shares, key=shares.get)


def compute_stability_penalty(summary: pd.DataFrame) -> pd.Series:
    """Return a cross-seed instability penalty for each parameter setting."""

    epsilon = 1e-12
    spread_cv = summary["average_spread_std"] / summary["average_spread_mean"].abs().clip(lower=epsilon)
    depth_cv = summary["average_top_of_book_depth_std"] / summary["average_top_of_book_depth_mean"].abs().clip(lower=epsilon)
    trade_cv = summary["trade_count_std"] / summary["trade_count_mean"].abs().clip(lower=epsilon)
    variance_cv = summary["return_variance_std"] / summary["return_variance_mean"].abs().clip(lower=epsilon)
    class_instability = 1.0 - summary.get("share_stable", 0.0).fillna(0.0)

    penalty = (
        0.30 * spread_cv.clip(lower=0.0, upper=5.0)
        + 0.20 * depth_cv.clip(lower=0.0, upper=5.0)
        + 0.20 * trade_cv.clip(lower=0.0, upper=5.0)
        + 0.10 * variance_cv.clip(lower=0.0, upper=5.0)
        + 0.20 * class_instability.clip(lower=0.0, upper=1.0)
    )
    return penalty.fillna(0.0)


def rank_candidate_settings(
    summary: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    weights: SelectionWeights | None = None,
    thresholds: DiagnosticThresholds | None = None,
) -> pd.DataFrame:
    """Rank candidate baseline settings from a grouped summary table."""

    if summary.empty:
        raise ValueError("Cannot rank candidates from an empty summary.")

    weights = weights or SelectionWeights()
    thresholds = thresholds or DiagnosticThresholds()
    ranked = prepare_summary(summary)

    ranked["candidate_label"] = ranked.apply(
        lambda row: build_candidate_label(row, parameter_columns),
        axis=1,
    )
    ranked["dominant_class"] = ranked.apply(dominant_class, axis=1)
    ranked["stability_penalty"] = compute_stability_penalty(ranked)

    spread_score = _normalize_series(ranked["average_spread_mean"], higher_is_better=False)
    relative_spread_score = _normalize_series(ranked["average_relative_spread_mean"], higher_is_better=False)
    depth_score = _normalize_series(ranked["average_top_of_book_depth_mean"], higher_is_better=True)
    trade_score = _normalize_series(ranked["trade_count_mean"], higher_is_better=True)
    return_activity_score = _band_score(
        ranked["return_variance_mean"],
        lower=thresholds.min_return_variance,
        upper=thresholds.max_return_variance,
    )
    crash_score = _normalize_series(ranked["crash_rate_mean"], higher_is_better=False)
    tail_score = _normalize_series(
        _negative_tail_magnitude(ranked["tail_exposure_mean"]),
        higher_is_better=False,
    )
    stability_score = _normalize_series(ranked["stability_penalty"], higher_is_better=False)
    inventory_score = _normalize_series(
        ranked["mm_mean_abs_inventory_mean"],
        higher_is_better=False,
    )

    ranked["spread_score"] = spread_score
    ranked["depth_score"] = depth_score
    ranked["trade_count_score"] = trade_score
    ranked["return_activity_score"] = return_activity_score
    ranked["crash_rate_score"] = crash_score
    ranked["tail_exposure_score"] = tail_score
    ranked["stability_score"] = stability_score
    ranked["market_maker_inventory_score"] = inventory_score

    ranked["selection_score"] = (
        weights.spread * spread_score
        + weights.relative_spread * relative_spread_score
        + weights.depth * depth_score
        + weights.trade_count * trade_score
        + weights.return_activity * return_activity_score
        + weights.crash_rate * crash_score
        + weights.tail_exposure * tail_score
        + weights.stable_share * ranked.get("share_stable", 0.0).fillna(0.0)
        + weights.stability * stability_score
        + weights.market_maker_inventory * inventory_score
        - weights.thin_share * ranked.get("share_thin", 0.0).fillna(0.0)
        - weights.frozen_share * ranked.get("share_frozen", 0.0).fillna(0.0)
        - weights.chaotic_share * ranked.get("share_chaotic", 0.0).fillna(0.0)
        - weights.crash_prone_share * ranked.get("share_crash-prone", 0.0).fillna(0.0)
    )

    ranked = ranked.sort_values(
        [
            "selection_score",
            "share_stable",
            "trade_count_mean",
            "average_spread_mean",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    ranked["candidate_rank"] = np.arange(1, len(ranked) + 1)
    best_score = float(ranked.loc[0, "selection_score"])
    ranked["score_gap_to_best"] = best_score - ranked["selection_score"]
    ranked["is_recommended"] = ranked["candidate_rank"] == 1
    return ranked


def recommend_baseline(
    ranked_candidates: pd.DataFrame,
) -> pd.Series:
    """Return the top-ranked candidate row."""

    if ranked_candidates.empty:
        raise ValueError("Cannot recommend a baseline from an empty ranking.")
    return ranked_candidates.sort_values("candidate_rank").iloc[0]


def assess_nearby_robustness(
    ranked_candidates: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    recommended: pd.Series | None = None,
    config: RobustnessConfig | None = None,
) -> dict[str, Any]:
    """Check whether the recommended point sits in a stable local region."""

    config = config or RobustnessConfig()
    recommended = recommended if recommended is not None else recommend_baseline(ranked_candidates)

    if not parameter_columns:
        return {
            "label": "insufficient nearby points",
            "neighbor_count": 0,
            "similar_neighbor_count": 0,
            "mean_neighbor_score_gap": float("nan"),
            "neighbors": pd.DataFrame(),
        }

    neighbor_indices: set[int] = set()
    for column in parameter_columns:
        values = list(pd.unique(ranked_candidates[column]))
        if len(values) <= 1 or recommended[column] not in values:
            continue
        base_position = values.index(recommended[column])
        adjacent_positions = [base_position - 1, base_position + 1]
        for position in adjacent_positions:
            if position < 0 or position >= len(values):
                continue
            neighbor_value = values[position]
            mask = ranked_candidates[column] == neighbor_value
            for other_column in parameter_columns:
                if other_column == column:
                    continue
                mask &= ranked_candidates[other_column] == recommended[other_column]
            neighbor_indices.update(ranked_candidates.index[mask].tolist())

    if not neighbor_indices:
        return {
            "label": "insufficient nearby points",
            "neighbor_count": 0,
            "similar_neighbor_count": 0,
            "mean_neighbor_score_gap": float("nan"),
            "neighbors": pd.DataFrame(),
        }

    neighbors = ranked_candidates.loc[sorted(neighbor_indices)].copy().reset_index(drop=True)
    score_gap = float(recommended["selection_score"]) - neighbors["selection_score"]
    similar_mask = (
        (score_gap <= config.score_gap)
        & (neighbors["share_stable"] >= config.min_stable_share)
    )
    similar_count = int(similar_mask.sum())
    neighbor_count = int(len(neighbors))
    similar_fraction = similar_count / neighbor_count
    label = "part of a stable region" if similar_fraction >= config.min_similar_fraction else "isolated / fragile"

    neighbors["score_gap_to_recommended"] = score_gap
    return {
        "label": label,
        "neighbor_count": neighbor_count,
        "similar_neighbor_count": similar_count,
        "mean_neighbor_score_gap": float(score_gap.mean()),
        "neighbors": neighbors.sort_values("selection_score", ascending=False).reset_index(drop=True),
    }


def format_parameter_block(row: Mapping[str, Any], parameter_columns: Sequence[str]) -> list[str]:
    """Return formatted parameter lines for a report."""

    if not parameter_columns:
        return ["- baseline: single evaluated configuration"]
    return [f"- {column}: {row[column]}" for column in parameter_columns]


def _format_metric(mean: float, std: float | None = None, precision: int = 6) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def explain_tradeoff(candidate: Mapping[str, Any], selected: Mapping[str, Any]) -> str:
    """Generate a short comparison against the selected baseline."""

    weaknesses: list[str] = []
    if candidate["share_stable"] < selected["share_stable"]:
        weaknesses.append("lower stable share across seeds")
    if candidate["stability_penalty"] > selected["stability_penalty"]:
        weaknesses.append("larger cross-seed variance")
    if candidate["average_spread_mean"] > selected["average_spread_mean"]:
        weaknesses.append("wider spreads")
    if candidate["average_top_of_book_depth_mean"] < selected["average_top_of_book_depth_mean"]:
        weaknesses.append("shallower top-of-book depth")
    if candidate["trade_count_mean"] < selected["trade_count_mean"]:
        weaknesses.append("less regular trading")
    if candidate["crash_rate_mean"] > selected["crash_rate_mean"]:
        weaknesses.append("higher crash rate")
    if candidate["tail_exposure_mean"] < selected["tail_exposure_mean"]:
        weaknesses.append("heavier left-tail exposure")

    if not weaknesses:
        weaknesses.append("a lower composite score without a single dominant failure mode")

    strengths: list[str] = []
    if candidate["average_spread_mean"] < selected["average_spread_mean"]:
        strengths.append("tighter spreads")
    if candidate["average_top_of_book_depth_mean"] > selected["average_top_of_book_depth_mean"]:
        strengths.append("deeper book")
    if candidate["trade_count_mean"] > selected["trade_count_mean"]:
        strengths.append("higher trade count")
    if candidate["crash_rate_mean"] < selected["crash_rate_mean"]:
        strengths.append("lower crash rate")

    prefix = f"{candidate['candidate_label']} scored lower"
    if strengths:
        prefix += f" despite {', '.join(strengths[:2])}"
    return f"{prefix} because of {', '.join(weaknesses[:2])}."


def build_recommendation_report(
    ranked_candidates: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    recommended: pd.Series | None = None,
    robustness: Mapping[str, Any] | None = None,
    max_tradeoff_candidates: int = 3,
) -> str:
    """Build a short markdown recommendation report."""

    recommended = recommended if recommended is not None else recommend_baseline(ranked_candidates)
    robustness = robustness or assess_nearby_robustness(
        ranked_candidates,
        parameter_columns=parameter_columns,
        recommended=recommended,
    )

    neighbor_rows = robustness.get("neighbors", pd.DataFrame())
    if isinstance(neighbor_rows, pd.DataFrame) and not neighbor_rows.empty:
        tradeoff_rows = neighbor_rows.head(max_tradeoff_candidates)
    else:
        tradeoff_rows = ranked_candidates[ranked_candidates["candidate_rank"] > 1].head(max_tradeoff_candidates)
    lines = [
        "# Recommended Phi=0 Baseline",
        "",
        "## Selected configuration",
        *format_parameter_block(recommended, parameter_columns),
        "",
        "## Why this setting was selected",
        f"- Selection score: {recommended['selection_score']:.2f}",
        f"- Dominant class: {recommended['dominant_class']}",
        f"- Stable share across seeds: {recommended['share_stable']:.2%}",
        f"- Stability penalty: {recommended['stability_penalty']:.3f}",
        f"- Nearby robustness: {robustness['label']}",
        "",
        "## Key diagnostics",
        f"- Average spread: {_format_metric(recommended['average_spread_mean'], recommended['average_spread_std'])}",
        f"- Average relative spread: {_format_metric(recommended['average_relative_spread_mean'], recommended['average_relative_spread_std'])}",
        f"- Average top-of-book depth: {_format_metric(recommended['average_top_of_book_depth_mean'], recommended['average_top_of_book_depth_std'], precision=3)}",
        f"- Trade count: {_format_metric(recommended['trade_count_mean'], recommended['trade_count_std'], precision=2)}",
        f"- Traded volume: {_format_metric(recommended['traded_volume_mean'], recommended['traded_volume_std'], precision=2)}",
        f"- Return variance: {_format_metric(recommended['return_variance_mean'], recommended['return_variance_std'])}",
        f"- Crash rate: {_format_metric(recommended['crash_rate_mean'], recommended['crash_rate_std'])}",
        f"- Tail exposure: {_format_metric(recommended['tail_exposure_mean'], recommended['tail_exposure_std'])}",
        f"- Max drawdown: {_format_metric(recommended['max_drawdown_mean'], recommended['max_drawdown_std'])}",
        f"- Market-maker mean absolute inventory: {_format_metric(recommended['mm_mean_abs_inventory_mean'], recommended['mm_mean_abs_inventory_std'], precision=3)}",
        "",
        "## Decision rationale",
        "- This candidate balances tighter spreads, usable depth, and regular trading without falling into frozen or crash-prone classifications.",
        "- The score favors low spreads, good depth, sufficient trading activity, contained crash and tail risk, and low cross-seed instability.",
    ]

    if robustness["neighbor_count"] > 0:
        lines.extend(
            [
                f"- Nearby settings checked: {robustness['neighbor_count']}",
                f"- Similar nearby settings: {robustness['similar_neighbor_count']}",
                f"- Mean score gap to nearby settings: {robustness['mean_neighbor_score_gap']:.2f}",
            ]
        )

    if not tradeoff_rows.empty:
        lines.extend(["", "## Rejected nearby candidates"])
        for _, candidate in tradeoff_rows.iterrows():
            lines.append(f"- {explain_tradeoff(candidate, recommended)}")

    return "\n".join(lines) + "\n"


def save_recommendation_report(report: str, output_path: str | Path) -> Path:
    """Persist the recommendation report to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def save_selection_plots(
    ranked_candidates: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> list[Path]:
    """Save a few minimal plots for the ranked candidate table."""

    if ranked_candidates.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save selection plots.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recommended = recommend_baseline(ranked_candidates)

    saved_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranked_candidates["candidate_rank"], ranked_candidates["selection_score"], marker="o")
    ax.scatter(
        [recommended["candidate_rank"]],
        [recommended["selection_score"]],
        color="red",
        label="recommended",
        zorder=3,
    )
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Selection score")
    ax.set_title("Selection Score by Candidate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    score_path = output_path / "selection_score_by_candidate.png"
    fig.tight_layout()
    fig.savefig(score_path, dpi=150)
    plt.close(fig)
    saved_paths.append(score_path)

    scatter_specs = [
        (
            "average_spread_mean",
            "crash_rate_mean",
            "Average Spread",
            "Crash Rate",
            "spread_vs_crash_rate.png",
        ),
        (
            "average_top_of_book_depth_mean",
            "trade_count_mean",
            "Average Top-of-Book Depth",
            "Trade Count",
            "depth_vs_trade_count.png",
        ),
    ]

    for x_column, y_column, xlabel, ylabel, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ranked_candidates[x_column], ranked_candidates[y_column], alpha=0.8)
        ax.scatter(
            [recommended[x_column]],
            [recommended[y_column]],
            color="red",
            label="recommended",
            zorder=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = output_path / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def selection_assumptions(weights: SelectionWeights, thresholds: DiagnosticThresholds) -> list[str]:
    """Return a compact list of assumptions behind the ranking rule."""

    return [
        "Scores are relative to the scanned candidate set, not absolute market realism scores.",
        "Lower spread, lower crash rate, and milder left-tail exposure are treated as preferable.",
        "Greater top-of-book depth and trade count are rewarded when they do not collapse into frozen or chaotic behavior.",
        f"Return variance is preferred inside [{thresholds.min_return_variance}, {thresholds.max_return_variance}] rather than maximized without bound.",
        f"Cross-seed instability and bad class shares are penalized using the configured weights {asdict(weights)}.",
    ]
