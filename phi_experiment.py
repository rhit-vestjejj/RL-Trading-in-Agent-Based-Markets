"""Automation helpers for the full phi-sweep experiment."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from analysis import log_returns, one_sided_book_metrics, summarize_market_frame
from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from config import MAX_PHI
from ppo_training import (
    PPOHyperparameters,
    SharedLinearPPOPolicy,
    save_policy_artifact,
    run_policy_episode,
    summarize_episode,
    train_shared_policy,
)
from rl_diagnostics import build_policy_evaluation_report, compute_policy_evaluation_diagnostics
from training_reporting import (
    build_combined_progress_frame,
    build_training_progress_report,
    save_training_progress_plots,
)
from visualization import save_market_plot

DEFAULT_PHI_GRID: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_EVALUATION_SEEDS: tuple[int, ...] = (7, 8, 9)
DEFAULT_EVALUATION_MODES: tuple[str, ...] = ("greedy", "stochastic")


def parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated list of floats."""

    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of integers."""

    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def format_phi(phi: float) -> str:
    """Return a stable folder/file label for one phi value."""

    return f"{float(phi):.2f}"


def default_experiment_output_dir(root: str | Path = "experiments") -> Path:
    """Return a timestamped experiment directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / f"phi_sweep_{timestamp}"


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_phi_experiment_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_ready(value: Any) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python types."""

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


def _has_midprice_pipeline_issue(metrics: dict[str, Any] | pd.Series) -> bool:
    """Return whether saved midprice diagnostics indicate a pipeline inconsistency."""

    defined_despite_missing = float(
        pd.to_numeric(
            pd.Series([metrics.get("defined_midprice_despite_missing_side_fraction", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    gap_without_missing_side = float(
        pd.to_numeric(
            pd.Series([metrics.get("midprice_gap_without_missing_side_fraction", 0.0)]),
            errors="coerce",
        ).fillna(0.0).iloc[0]
    )
    return defined_despite_missing > 0.0 or gap_without_missing_side > 0.0


def compute_extended_market_metrics(frame: pd.DataFrame, *, tick_size: float = 0.01) -> dict[str, float]:
    """Compute the market metrics used in the phi-sweep summary."""

    summary = summarize_market_frame(frame)
    summary.update(one_sided_book_metrics(frame))
    returns = log_returns(frame["midprice"])
    spread_series = pd.to_numeric(frame["spread"], errors="coerce").dropna()
    spread_ticks = np.rint(spread_series / float(tick_size)) if not spread_series.empty else np.array([], dtype=float)

    summary.update(
        {
            "zero_return_fraction": float((returns == 0.0).mean()) if len(returns) else float("nan"),
            "nonzero_return_count": float((returns != 0.0).sum()) if len(returns) else 0.0,
            "return_count": float(len(returns)),
            "inactivity_fraction": (
                float((pd.to_numeric(frame.get("traded_volume", 0.0), errors="coerce").fillna(0.0) <= 0.0).mean())
                if "traded_volume" in frame.columns
                else float("nan")
            ),
            "volatility": float(np.sqrt(max(summary["return_variance"], 0.0)))
            if not np.isnan(summary["return_variance"])
            else float("nan"),
            "average_depth": float(summary["average_top_of_book_depth"]),
            "spread_share_1_tick": float((spread_ticks == 1).mean()) if spread_ticks.size else float("nan"),
            "spread_share_2_ticks": float((spread_ticks == 2).mean()) if spread_ticks.size else float("nan"),
            "spread_share_3_plus_ticks": float((spread_ticks >= 3).mean()) if spread_ticks.size else float("nan"),
        }
    )
    return {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in summary.items()}


def _summarize_numeric_series(series: pd.Series) -> dict[str, float]:
    """Return an explicit across-run summary without coercing missing values to zero."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    sample_size = int(len(numeric))
    if sample_size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "stderr": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "n": 0.0,
        }

    mean_value = float(numeric.mean())
    if sample_size == 1:
        return {
            "mean": mean_value,
            "std": float("nan"),
            "stderr": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "n": 1.0,
        }

    std_value = float(numeric.std(ddof=1))
    stderr_value = float(std_value / np.sqrt(sample_size))
    ci_half_width = float(1.96 * stderr_value)
    return {
        "mean": mean_value,
        "std": std_value,
        "stderr": stderr_value,
        "ci95_lower": float(mean_value - ci_half_width),
        "ci95_upper": float(mean_value + ci_half_width),
        "n": float(sample_size),
    }


def _add_numeric_summary(aggregate: dict[str, float | str], prefix: str, series: pd.Series) -> None:
    """Attach mean/variance/sample-size statistics for one numeric series."""

    for suffix, value in _summarize_numeric_series(series).items():
        aggregate[f"{prefix}_{suffix}"] = value


def aggregate_mode_results(
    *,
    phi: float,
    evaluation_mode: str,
    episode_frame: pd.DataFrame,
    diagnostics_frame: pd.DataFrame,
    market_metrics_frame: pd.DataFrame,
    representative_seed: int,
) -> dict[str, float | str]:
    """Aggregate one evaluation mode across seeds."""

    aggregate: dict[str, float | str] = {
        "phi": float(phi),
        "evaluation_mode": str(evaluation_mode),
        "representative_seed": int(representative_seed),
        "num_evaluation_seeds": float(len(episode_frame)),
        "consistency_status": "not_evaluated",
        "pipeline_issue_seed_count": 0.0,
        "pipeline_issue_seed_fraction": 0.0,
    }
    for column_name, prefix in (
        ("total_training_reward", "evaluation_total_reward"),
        ("average_reward_per_rl_agent", "evaluation_average_reward_per_rl_agent"),
        ("average_abs_ending_inventory", "evaluation_average_abs_ending_inventory"),
        ("buy_fraction", "evaluation_buy_fraction"),
        ("hold_fraction", "evaluation_hold_fraction"),
        ("sell_fraction", "evaluation_sell_fraction"),
    ):
        if column_name in episode_frame.columns:
            _add_numeric_summary(aggregate, prefix, episode_frame[column_name])
    if not diagnostics_frame.empty:
        max_abs_inventory_reached = (
            float(diagnostics_frame["max_abs_inventory_reached"].max())
            if "max_abs_inventory_reached" in diagnostics_frame.columns
            else max(
                abs(float(diagnostics_frame["inventory_overall_min"].min()))
                if "inventory_overall_min" in diagnostics_frame.columns
                else 0.0,
                abs(float(diagnostics_frame["inventory_overall_max"].max()))
                if "inventory_overall_max" in diagnostics_frame.columns
                else 0.0,
            )
        )
        aggregate.update(
            {
                "inventory_cap_value": float(diagnostics_frame["inventory_cap_value"].dropna().iloc[0])
                if "inventory_cap_value" in diagnostics_frame.columns
                and not diagnostics_frame["inventory_cap_value"].dropna().empty
                else float("nan"),
                "evaluation_inventory_min_global": float(diagnostics_frame["inventory_overall_min"].min()),
                "evaluation_inventory_max_global": float(diagnostics_frame["inventory_overall_max"].max()),
                "evaluation_max_abs_inventory_reached_max": max_abs_inventory_reached,
                "evaluation_chosen_sell_count_total": float(diagnostics_frame["rl_sell_count"].sum()),
                "evaluation_chosen_hold_count_total": float(diagnostics_frame["rl_hold_count"].sum()),
                "evaluation_chosen_buy_count_total": float(diagnostics_frame["rl_buy_count"].sum()),
                "evaluation_executed_sell_count_total": float(diagnostics_frame["executed_sell_action_count"].sum()),
                "evaluation_executed_buy_count_total": float(diagnostics_frame["executed_buy_action_count"].sum()),
                "evaluation_executed_sell_volume_total": float(diagnostics_frame["rl_executed_sell_volume"].sum()),
                "evaluation_executed_buy_volume_total": float(diagnostics_frame["rl_executed_buy_volume"].sum()),
                "evaluation_submitted_sell_count_total": float(diagnostics_frame["submitted_sell_action_count"].sum()),
                "evaluation_submitted_buy_count_total": float(diagnostics_frame["submitted_buy_action_count"].sum()),
                "evaluation_blocked_sell_action_count_total": float(diagnostics_frame["blocked_sell_action_count"].sum())
                if "blocked_sell_action_count" in diagnostics_frame.columns
                else 0.0,
                "evaluation_blocked_buy_action_count_total": float(diagnostics_frame["blocked_buy_action_count"].sum())
                if "blocked_buy_action_count" in diagnostics_frame.columns
                else 0.0,
                "evaluation_inventory_at_cap_fraction_mean": float(diagnostics_frame["inventory_at_cap_fraction"].mean())
                if "inventory_at_cap_fraction" in diagnostics_frame.columns
                else 0.0,
                "evaluation_inventory_near_cap_fraction_mean": float(diagnostics_frame["inventory_near_cap_fraction"].mean())
                if "inventory_near_cap_fraction" in diagnostics_frame.columns
                else 0.0,
            }
        )
        pipeline_issue_mask = diagnostics_frame.apply(_has_midprice_pipeline_issue, axis=1)
        aggregate["pipeline_issue_seed_count"] = float(pipeline_issue_mask.sum())
        aggregate["pipeline_issue_seed_fraction"] = (
            float(pipeline_issue_mask.mean()) if len(diagnostics_frame) else 0.0
        )
        aggregate["consistency_status"] = "pipeline_issue" if bool(pipeline_issue_mask.any()) else "consistent"
        diagnostic_numeric_columns = [
            column
            for column in diagnostics_frame.columns
            if column not in {"evaluation_mode", "spread_gap_cause", "inventory_cap_present", "action_ordering"}
            and pd.api.types.is_numeric_dtype(diagnostics_frame[column])
        ]
        for column in diagnostic_numeric_columns:
            if column in aggregate:
                continue
            _add_numeric_summary(aggregate, column, diagnostics_frame[column])
    if not market_metrics_frame.empty:
        numeric_columns = [
            column
            for column in market_metrics_frame.columns
            if column not in {"phi", "seed", "evaluation_mode"} and pd.api.types.is_numeric_dtype(market_metrics_frame[column])
        ]
        for column in numeric_columns:
            _add_numeric_summary(aggregate, column, market_metrics_frame[column])
    return aggregate


def build_mode_report(
    *,
    phi: float,
    evaluation_mode: str,
    aggregate: dict[str, float | str],
    representative_diagnostics: dict[str, float | str],
    representative_agent_frame: pd.DataFrame,
) -> str:
    """Return a compact markdown report for one phi/mode evaluation."""

    lines = [
        f"# Phi {format_phi(phi)} {evaluation_mode.capitalize()} Evaluation",
        "",
        "## Aggregate",
        f"- Evaluation seeds: {int(aggregate['num_evaluation_seeds'])}",
        f"- Mean total reward: {float(aggregate['evaluation_total_reward_mean']):.6f}",
        f"- Mean reward per RL agent: {float(aggregate['evaluation_average_reward_per_rl_agent_mean']):.6f}",
        f"- Mean abs ending inventory: {float(aggregate['evaluation_average_abs_ending_inventory_mean']):.6f}",
        f"- Max abs inventory reached: {float(aggregate.get('evaluation_max_abs_inventory_reached_max', float('nan'))):.6f}",
        f"- Buy / hold / sell fractions: {float(aggregate['evaluation_buy_fraction_mean']):.6f} / {float(aggregate['evaluation_hold_fraction_mean']):.6f} / {float(aggregate['evaluation_sell_fraction_mean']):.6f}",
        f"- Blocked sell / buy actions: {float(aggregate.get('evaluation_blocked_sell_action_count_total', 0.0)):.0f} / {float(aggregate.get('evaluation_blocked_buy_action_count_total', 0.0)):.0f}",
        f"- Consistency status / pipeline-issue seeds: {aggregate.get('consistency_status', 'unknown')} / {float(aggregate.get('pipeline_issue_seed_count', 0.0)):.0f}",
    ]
    if "average_spread_mean" in aggregate:
        lines.extend(
            [
                f"- Average spread: {float(aggregate['average_spread_mean']):.6f}",
                f"- Average depth: {float(aggregate['average_depth_mean']):.6f}",
                f"- Inactivity fraction: {float(aggregate.get('inactivity_fraction_mean', float('nan'))):.6f}",
                f"- Zero-return fraction: {float(aggregate['zero_return_fraction_mean']):.6f}",
                f"- One-sided-book fraction (conditional on positive visible liquidity): {float(aggregate.get('one_sided_book_fraction_mean', float('nan'))):.6f}",
                f"- One-sided metric valid timestep count: {float(aggregate.get('one_sided_metric_valid_timestep_count_mean', float('nan'))):.2f}",
                f"- Empty-book fraction: {float(aggregate.get('empty_book_fraction_mean', float('nan'))):.6f}",
                f"- Average bid / ask volume: {float(aggregate.get('average_bid_volume_mean', float('nan'))):.6f} / {float(aggregate.get('average_ask_volume_mean', float('nan'))):.6f}",
                f"- Undefined-midprice fraction: {float(aggregate.get('undefined_midprice_fraction_mean', float('nan'))):.6f}",
                f"- Missing best bid / ask fraction: {float(aggregate.get('missing_bid_fraction_mean', float('nan'))):.6f} / {float(aggregate.get('missing_ask_fraction_mean', float('nan'))):.6f}",
                f"- Max consecutive one-sided duration: {float(aggregate.get('max_consecutive_one_sided_duration_mean', float('nan'))):.6f} seconds",
                f"- One-sided episodes / mean duration: {float(aggregate.get('num_one_sided_episodes_mean', float('nan'))):.6f} / {float(aggregate.get('mean_one_sided_episode_duration_mean', float('nan'))):.6f} seconds",
                f"- Missing-quote one-sided / both-sides-missing fraction: {float(aggregate.get('true_one_sided_book_fraction_mean', float('nan'))):.6f} / {float(aggregate.get('both_sides_missing_fraction_mean', float('nan'))):.6f}",
                f"- Midprice artifact checks (gap without missing side / defined despite missing side): {float(aggregate.get('midprice_gap_without_missing_side_fraction_mean', float('nan'))):.6f} / {float(aggregate.get('defined_midprice_despite_missing_side_fraction_mean', float('nan'))):.6f}",
                f"- Excess kurtosis: {float(aggregate['excess_kurtosis_mean']):.6f}",
                f"- Tail exposure: {float(aggregate['tail_exposure_mean']):.6f}",
                f"- Crash rate: {float(aggregate['crash_rate_mean']):.6f}",
                f"- Max drawdown: {float(aggregate['max_drawdown_mean']):.6f}",
                f"- Passive bid / ask / both submission rate: {float(aggregate.get('passive_bid_submission_rate_mean', float('nan'))):.6f} / {float(aggregate.get('passive_ask_submission_rate_mean', float('nan'))):.6f} / {float(aggregate.get('passive_both_quote_rate_mean', float('nan'))):.6f}",
                f"- Resting quote presence / quote fill rate: {float(aggregate.get('resting_order_presence_fraction_mean', float('nan'))):.6f} / {float(aggregate.get('quote_fill_rate_mean', float('nan'))):.6f}",
            ]
        )
    lines.extend(
        [
            "",
            f"## Representative Seed {int(aggregate['representative_seed'])}",
            build_policy_evaluation_report(representative_diagnostics, representative_agent_frame),
        ]
    )
    return "\n".join(lines)


def save_cross_phi_plots(summary_frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save top-level cross-phi comparison plots."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    saved_paths: list[Path] = []
    ordered = summary_frame.sort_values("phi")

    def resolve_errorbars(mode: str, column_suffix: str) -> tuple[np.ndarray, np.ndarray] | None:
        mean_column = f"{mode}_{column_suffix}"
        lower_column = mean_column.replace("_mean", "_ci95_lower")
        upper_column = mean_column.replace("_mean", "_ci95_upper")
        stderr_column = mean_column.replace("_mean", "_stderr")
        std_column = mean_column.replace("_mean", "_std")
        n_column = mean_column.replace("_mean", "_n")

        if lower_column in ordered.columns and upper_column in ordered.columns:
            lower = pd.to_numeric(ordered[lower_column], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(ordered[upper_column], errors="coerce").to_numpy(dtype=float)
            mean_values = pd.to_numeric(ordered[mean_column], errors="coerce").to_numpy(dtype=float)
            return mean_values - lower, upper - mean_values
        if stderr_column in ordered.columns:
            error = pd.to_numeric(ordered[stderr_column], errors="coerce").to_numpy(dtype=float)
            return error, error
        if std_column in ordered.columns and n_column in ordered.columns:
            std_values = pd.to_numeric(ordered[std_column], errors="coerce").to_numpy(dtype=float)
            n_values = pd.to_numeric(ordered[n_column], errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                stderr_values = std_values / np.sqrt(n_values)
            return stderr_values, stderr_values
        return None

    def plot_two_mode_metric(
        column_suffix: str,
        *,
        title: str,
        ylabel: str,
        filename: str,
        with_error_bars: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for mode, color in (("greedy", "tab:blue"), ("stochastic", "tab:orange")):
            column = f"{mode}_{column_suffix}"
            if column not in ordered.columns:
                continue
            y_values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
            if with_error_bars:
                errorbars = resolve_errorbars(mode, column_suffix)
                if errorbars is not None:
                    lower_err, upper_err = errorbars
                    ax.errorbar(
                        ordered["phi"],
                        y_values,
                        yerr=np.vstack([lower_err, upper_err]),
                        marker="o",
                        linewidth=1.6,
                        capsize=3.0,
                        label=mode,
                        color=color,
                    )
                    continue
            ax.plot(ordered["phi"], y_values, marker="o", linewidth=1.6, label=mode, color=color)
        ax.set_title(title)
        ax.set_xlabel("phi")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
        fig.tight_layout()
        path = output_root / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    plot_two_mode_metric("volatility_mean", title="Volatility vs Phi", ylabel="Volatility", filename="volatility_vs_phi.png")
    plot_two_mode_metric("average_spread_mean", title="Average Spread vs Phi", ylabel="Spread", filename="spread_vs_phi.png")
    plot_two_mode_metric("average_depth_mean", title="Average Depth vs Phi", ylabel="Depth", filename="average_depth_vs_phi.png")
    plot_two_mode_metric(
        "one_sided_book_fraction_mean",
        title="One-Sided Book Fraction vs Phi",
        ylabel="Fraction",
        filename="one_sided_book_fraction_vs_phi.png",
        with_error_bars=True,
    )
    plot_two_mode_metric(
        "undefined_midprice_fraction_mean",
        title="Undefined Midprice Fraction vs Phi",
        ylabel="Fraction",
        filename="undefined_midprice_fraction_vs_phi.png",
    )
    plot_two_mode_metric(
        "max_consecutive_one_sided_duration_mean",
        title="Max Consecutive One-Sided Duration vs Phi",
        ylabel="Seconds",
        filename="max_consecutive_one_sided_duration_vs_phi.png",
    )
    plot_two_mode_metric(
        "missing_bid_fraction_mean",
        title="Missing Best Bid Fraction vs Phi",
        ylabel="Fraction",
        filename="missing_bid_fraction_vs_phi.png",
    )
    plot_two_mode_metric(
        "missing_ask_fraction_mean",
        title="Missing Best Ask Fraction vs Phi",
        ylabel="Fraction",
        filename="missing_ask_fraction_vs_phi.png",
    )
    plot_two_mode_metric(
        "zero_return_fraction_mean",
        title="Zero-Return Fraction vs Phi",
        ylabel="Zero-return fraction",
        filename="zero_return_fraction_vs_phi.png",
    )
    plot_two_mode_metric(
        "inactivity_fraction_mean",
        title="Inactivity Fraction vs Phi",
        ylabel="Fraction",
        filename="inactivity_fraction_vs_phi.png",
    )
    plot_two_mode_metric("tail_exposure_mean", title="Tail Exposure vs Phi", ylabel="Tail exposure", filename="tail_exposure_vs_phi.png")
    plot_two_mode_metric("crash_rate_mean", title="Crash Rate vs Phi", ylabel="Crash rate", filename="crash_rate_vs_phi.png")
    plot_two_mode_metric("max_drawdown_mean", title="Max Drawdown vs Phi", ylabel="Max drawdown", filename="drawdown_vs_phi.png")
    plot_two_mode_metric(
        "evaluation_average_abs_ending_inventory_mean",
        title="Average Abs Inventory vs Phi",
        ylabel="Average abs ending inventory",
        filename="average_abs_inventory_vs_phi.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for axis, mode in zip(axes, ("greedy", "stochastic")):
        for action_name, color in (
            ("evaluation_buy_fraction_mean", "tab:green"),
            ("evaluation_hold_fraction_mean", "tab:gray"),
            ("evaluation_sell_fraction_mean", "tab:red"),
        ):
            column = f"{mode}_{action_name}"
            if column not in ordered.columns:
                continue
            axis.plot(
                ordered["phi"],
                ordered[column],
                marker="o",
                linewidth=1.5,
                label=action_name.replace("evaluation_", "").replace("_mean", ""),
                color=color,
            )
        axis.set_title(f"{mode.capitalize()} Action Fractions")
        axis.set_ylabel("Fraction")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("phi")
    fig.tight_layout()
    action_path = output_root / "rl_action_fractions_vs_phi.png"
    fig.savefig(action_path, dpi=150)
    plt.close(fig)
    saved_paths.append(action_path)
    return saved_paths


def build_phi_sweep_report(summary_frame: pd.DataFrame, experiment_config: dict[str, Any]) -> str:
    """Return a compact experiment-level markdown report."""

    def artifact_summary(mode: str) -> str | None:
        artifact_columns = [
            f"{mode}_midprice_gap_without_missing_side_fraction_mean",
            f"{mode}_spread_gap_without_missing_side_fraction_mean",
            f"{mode}_defined_midprice_despite_missing_side_fraction_mean",
            f"{mode}_defined_spread_despite_missing_side_fraction_mean",
        ]
        present_columns = [column for column in artifact_columns if column in summary_frame.columns]
        if not present_columns:
            return None

        max_artifact = max(
            float(summary_frame[column].fillna(0.0).abs().max()) for column in present_columns
        )
        if max_artifact == 0.0:
            return (
                f"- {mode.capitalize()}: fixed-grid logs carry forward the last exchange snapshot without "
                "interpolating missing sides, and the saved undefined midprice/spread values align entirely with missing best bid/ask state."
            )
        return (
            f"- {mode.capitalize()}: pipeline_issue detected; nonzero artifact checks remain in the saved summaries "
            f"(max inconsistent fraction {max_artifact:.6f}), so inspect the per-seed diagnostics before treating all gaps as true one-sided states."
        )

    lines = [
        "# Phi Sweep Report",
        "",
        "## Configuration",
        f"- Market profile: {experiment_config['market_profile']}",
        f"- Phi grid: {', '.join(f'{phi:.2f}' for phi in experiment_config['phi_grid'])}",
        f"- Training episodes: {experiment_config['episodes']}",
        f"- Evaluation seeds: {', '.join(str(seed) for seed in experiment_config['evaluation_seeds'])}",
        f"- End time: {experiment_config['end_time']}",
        f"- Log frequency: {experiment_config['log_frequency']}",
        f"- lambda_q: {experiment_config['lambda_q']}",
        f"- flat_hold_penalty: {experiment_config['flat_hold_penalty']}",
        f"- inventory_cap: {experiment_config['inventory_cap'] if experiment_config['inventory_cap'] is not None else 'disabled'}",
        "",
        "## Snapshot Integrity",
        "- The fixed-grid market logs use backward carry-forward of the most recent exchange snapshot and do not interpolate missing best bid, best ask, midprice, or spread values.",
    ]
    for mode in ("greedy", "stochastic"):
        summary_line = artifact_summary(mode)
        if summary_line is not None:
            lines.append(summary_line)
    lines.extend(
        [
            "",
            "## Per-Phi Summary",
        ]
    )
    for row in summary_frame.sort_values("phi").itertuples(index=False):
        lines.extend(
            [
                f"### phi = {row.phi:.2f}",
                f"- Greedy reward / hold fraction: {getattr(row, 'greedy_evaluation_total_reward_mean', float('nan')):.6f} / {getattr(row, 'greedy_evaluation_hold_fraction_mean', float('nan')):.6f}",
                f"- Stochastic reward / hold fraction: {getattr(row, 'stochastic_evaluation_total_reward_mean', float('nan')):.6f} / {getattr(row, 'stochastic_evaluation_hold_fraction_mean', float('nan')):.6f}",
                f"- Greedy spread / depth: {getattr(row, 'greedy_average_spread_mean', float('nan')):.6f} / {getattr(row, 'greedy_average_depth_mean', float('nan')):.6f}",
                f"- Stochastic spread / depth: {getattr(row, 'stochastic_average_spread_mean', float('nan')):.6f} / {getattr(row, 'stochastic_average_depth_mean', float('nan')):.6f}",
                f"- Greedy one-sided (valid-liquidity only) / undefined-mid / empty-book: {getattr(row, 'greedy_one_sided_book_fraction_mean', float('nan')):.6f} / {getattr(row, 'greedy_undefined_midprice_fraction_mean', float('nan')):.6f} / {getattr(row, 'greedy_empty_book_fraction_mean', float('nan')):.6f}",
                f"- Stochastic one-sided (valid-liquidity only) / undefined-mid / empty-book: {getattr(row, 'stochastic_one_sided_book_fraction_mean', float('nan')):.6f} / {getattr(row, 'stochastic_undefined_midprice_fraction_mean', float('nan')):.6f} / {getattr(row, 'stochastic_empty_book_fraction_mean', float('nan')):.6f}",
                f"- Greedy / stochastic valid timestep count: {getattr(row, 'greedy_one_sided_metric_valid_timestep_count_mean', float('nan')):.2f} / {getattr(row, 'stochastic_one_sided_metric_valid_timestep_count_mean', float('nan')):.2f}",
                f"- Greedy / stochastic max run / episodes: {getattr(row, 'greedy_max_consecutive_one_sided_duration_mean', float('nan')):.6f}s / {getattr(row, 'greedy_num_one_sided_episodes_mean', float('nan')):.6f} ; {getattr(row, 'stochastic_max_consecutive_one_sided_duration_mean', float('nan')):.6f}s / {getattr(row, 'stochastic_num_one_sided_episodes_mean', float('nan')):.6f}",
                f"- Greedy / stochastic consistency status: {getattr(row, 'greedy_consistency_status', 'unknown')} / {getattr(row, 'stochastic_consistency_status', 'unknown')}",
                f"- Greedy avg abs ending inventory: {getattr(row, 'greedy_evaluation_average_abs_ending_inventory_mean', float('nan')):.6f}",
                f"- Stochastic avg abs ending inventory: {getattr(row, 'stochastic_evaluation_average_abs_ending_inventory_mean', float('nan')):.6f}",
                f"- Greedy max abs inventory / blocked buy / blocked sell: {getattr(row, 'greedy_evaluation_max_abs_inventory_reached_max', float('nan')):.6f} / {getattr(row, 'greedy_evaluation_blocked_buy_action_count_total', float('nan')):.0f} / {getattr(row, 'greedy_evaluation_blocked_sell_action_count_total', float('nan')):.0f}",
                f"- Stochastic max abs inventory / blocked buy / blocked sell: {getattr(row, 'stochastic_evaluation_max_abs_inventory_reached_max', float('nan')):.6f} / {getattr(row, 'stochastic_evaluation_blocked_buy_action_count_total', float('nan')):.0f} / {getattr(row, 'stochastic_evaluation_blocked_sell_action_count_total', float('nan')):.0f}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_phi_experiment(
    *,
    phi_grid: Sequence[float],
    episodes: int,
    start_seed: int,
    evaluation_seeds: Sequence[int],
    output_dir: str | Path,
    end_time: str = "09:35:00",
    log_frequency: str = "1s",
    num_agents: int = 102,
    return_window: int = 10,
    lambda_q: float = 0.01,
    flat_hold_penalty: float = 0.02,
    inventory_cap: int | None = None,
    rl_liquidity_mode: str = "taker_only",
    rl_quoter_split: float = 0.5,
    rl_enable_passive_quotes: bool = True,
    rl_quote_mode: str = "at_best",
    rl_quote_offset_ticks: int = 0,
    rl_quote_size: int = 1,
    evaluation_interval: int = 5,
    checkpoint_interval: int = 5,
    training_seeds: Sequence[int] | None = None,
    hyperparameters: PPOHyperparameters | None = None,
    evaluation_modes: Sequence[str] = DEFAULT_EVALUATION_MODES,
) -> dict[str, Any]:
    """Run the full phi-sweep experiment and save all outputs."""

    if training_seeds and len(training_seeds) != int(episodes):
        raise ValueError("When --training-seeds is provided, its length must match --episodes.")
    if not phi_grid:
        raise ValueError("phi_grid must not be empty.")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty.")
    if not evaluation_modes:
        raise ValueError("evaluation_modes must not be empty.")
    for phi in phi_grid:
        if float(phi) < 0.0 or float(phi) > MAX_PHI:
            raise ValueError(f"phi={phi} is outside the allowed range [0.0, {MAX_PHI:.4f}].")

    root = Path(output_dir)
    config_dir = root / "config"
    summaries_dir = root / "summaries"
    plots_dir = root / "plots"
    root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    effective_hyperparameters = hyperparameters or PPOHyperparameters()
    experiment_config = {
        "timestamp": datetime.now().isoformat(),
        "market_profile": OFFICIAL_BASELINE_NAME,
        "phi_grid": [float(phi) for phi in phi_grid],
        "episodes": int(episodes),
        "start_seed": int(start_seed),
        "training_seeds": [int(seed) for seed in training_seeds] if training_seeds else None,
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "evaluation_modes": [str(mode) for mode in evaluation_modes],
        "evaluation_interval": int(evaluation_interval),
        "checkpoint_interval": int(checkpoint_interval),
        "num_agents": int(num_agents),
        "end_time": end_time,
        "log_frequency": log_frequency,
        "return_window": int(return_window),
        "lambda_q": float(lambda_q),
        "flat_hold_penalty": float(flat_hold_penalty),
        "inventory_cap": int(inventory_cap) if inventory_cap is not None else None,
        "rl_liquidity_mode": str(rl_liquidity_mode),
        "rl_quoter_split": float(rl_quoter_split),
        "rl_enable_passive_quotes": bool(rl_enable_passive_quotes),
        "rl_quote_mode": str(rl_quote_mode),
        "rl_quote_offset_ticks": int(rl_quote_offset_ticks),
        "rl_quote_size": int(rl_quote_size),
        "ppo_hyperparameters": asdict(effective_hyperparameters),
    }
    _write_json(root / "experiment_config.json", experiment_config)
    _write_json(config_dir / "experiment_config.json", experiment_config)

    summary_rows: list[dict[str, Any]] = []
    detailed_market_rows: list[pd.DataFrame] = []
    detailed_diagnostic_rows: list[pd.DataFrame] = []
    detailed_training_rows: list[pd.DataFrame] = []

    def save_interval_checkpoint(phi_checkpoint_dir: Path):
        def _callback(episode_index: int, seed: int, policy: object) -> None:
            snapshot_path = phi_checkpoint_dir / f"shared_ppo_policy_episode_{episode_index + 1:03d}_seed_{seed}.npz"
            save_policy_artifact(policy, snapshot_path)

        return _callback

    for phi in phi_grid:
        phi_value = float(phi)
        phi_label = format_phi(phi_value)
        phi_dir = root / f"phi_{phi_label}"
        checkpoint_dir = phi_dir / "checkpoints"
        training_dir = phi_dir / "training"
        evaluation_dirs = {
            "greedy": phi_dir / "evaluation_greedy",
            "stochastic": phi_dir / "evaluation_stochastic",
        }
        phi_plot_dir = phi_dir / "plots"
        for directory in (checkpoint_dir, training_dir, phi_plot_dir, *evaluation_dirs.values()):
            directory.mkdir(parents=True, exist_ok=True)

        phi_config_builder = build_abides_rmsc04_small_v1_config(
            phi=phi_value,
            num_agents=num_agents,
            end_time=end_time,
            log_frequency=log_frequency,
            return_window=return_window,
            lambda_q=lambda_q,
            flat_hold_penalty=flat_hold_penalty,
            inventory_cap=inventory_cap,
            rl_liquidity_mode=rl_liquidity_mode,
            rl_quoter_split=rl_quoter_split,
            rl_enable_passive_quotes=rl_enable_passive_quotes,
            rl_quote_mode=rl_quote_mode,
            rl_quote_offset_ticks=rl_quote_offset_ticks,
            rl_quote_size=rl_quote_size,
        )
        phi_config = {
            **experiment_config,
            "phi": phi_value,
            "agent_counts": phi_config_builder.agent_counts(),
            "rl_role_counts": (
                phi_config_builder.rl_role_counts()
                if hasattr(phi_config_builder, "rl_role_counts")
                else {}
            ),
        }
        _write_json(config_dir / f"phi_{phi_label}_config.json", phi_config)

        training_frame = pd.DataFrame()
        evaluation_during_training = pd.DataFrame()
        policy: object | None = None
        final_checkpoint_path: Path | None = None
        if phi_value > 0.0:
            policy, training_frame, evaluation_during_training = train_shared_policy(
                phi=phi_value,
                episodes=episodes,
                start_seed=start_seed,
                training_seeds=training_seeds,
                config_factory=build_abides_rmsc04_small_v1_config,
                config_overrides={
                    "num_agents": num_agents,
                    "end_time": end_time,
                    "log_frequency": log_frequency,
                    "return_window": return_window,
                    "lambda_q": lambda_q,
                    "flat_hold_penalty": flat_hold_penalty,
                    "inventory_cap": inventory_cap,
                    "rl_liquidity_mode": rl_liquidity_mode,
                    "rl_quoter_split": rl_quoter_split,
                    "rl_enable_passive_quotes": rl_enable_passive_quotes,
                    "rl_quote_mode": rl_quote_mode,
                    "rl_quote_offset_ticks": rl_quote_offset_ticks,
                    "rl_quote_size": rl_quote_size,
                },
                hyperparameters=effective_hyperparameters,
                evaluation_seeds=evaluation_seeds,
                evaluation_modes=evaluation_modes,
                evaluation_interval=evaluation_interval,
                checkpoint_interval=checkpoint_interval,
                checkpoint_callback=save_interval_checkpoint(checkpoint_dir) if checkpoint_interval > 0 else None,
            )
            training_frame.to_csv(training_dir / "training_log.csv", index=False)
            evaluation_during_training.to_csv(training_dir / "evaluation_during_training.csv", index=False)
            combined_progress = build_combined_progress_frame(training_frame, evaluation_during_training)
            combined_progress.to_csv(training_dir / "combined_progress.csv", index=False)
            (training_dir / "training_report.md").write_text(
                build_training_progress_report(training_frame, evaluation_during_training),
                encoding="utf-8",
            )
            final_checkpoint_path = save_policy_artifact(policy, checkpoint_dir / "shared_ppo_policy_final.npz")
            saved_phi_plots = save_training_progress_plots(training_frame, evaluation_during_training, phi_plot_dir)
            detailed_training_rows.append(training_frame.assign(phi=phi_value))
        else:
            saved_phi_plots = []

        phi_summary_row: dict[str, Any] = {
            "phi": phi_value,
            "inventory_cap": int(inventory_cap) if inventory_cap is not None else np.nan,
            "agent_counts_json": json.dumps(phi_config["agent_counts"], sort_keys=True),
            "checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else "",
            "training_completed": bool(phi_value > 0.0),
        }
        if not training_frame.empty:
            latest_training = training_frame.iloc[-1]
            phi_summary_row.update(
                {
                    "training_total_reward_last": float(latest_training["total_training_reward"]),
                    "training_average_abs_ending_inventory_last": float(latest_training["average_abs_ending_inventory"]),
                    "training_max_abs_inventory_last": float(latest_training["max_abs_inventory"]),
                    "training_blocked_buy_action_count_last": float(latest_training["blocked_buy_action_count"]),
                    "training_blocked_sell_action_count_last": float(latest_training["blocked_sell_action_count"]),
                    "training_entropy_last": float(latest_training["entropy"]),
                    "training_policy_loss_last": float(latest_training["policy_loss"]),
                    "training_value_loss_last": float(latest_training["value_loss"]),
                }
            )

        for evaluation_mode in evaluation_modes:
            mode_dir = evaluation_dirs[evaluation_mode]
            representative_seed = int(evaluation_seeds[0])
            original_deterministic: bool | None = None
            if hasattr(policy, "set_deterministic"):
                original_deterministic = bool(getattr(policy, "deterministic", False))
                getattr(policy, "set_deterministic")(evaluation_mode == "greedy")
            try:
                per_seed_rows: list[dict[str, Any]] = []
                diagnostics_rows: list[dict[str, Any]] = []
                market_metric_rows: list[dict[str, Any]] = []
                agent_frames: list[pd.DataFrame] = []
                representative_diagnostics: dict[str, Any] | None = None
                representative_agent_frame = pd.DataFrame()
                for episode_index, seed in enumerate(evaluation_seeds):
                    config = build_abides_rmsc04_small_v1_config(
                        phi=phi_value,
                        seed=int(seed),
                        num_agents=num_agents,
                        end_time=end_time,
                        log_frequency=log_frequency,
                        return_window=return_window,
                        lambda_q=lambda_q,
                        flat_hold_penalty=flat_hold_penalty,
                        inventory_cap=inventory_cap,
                        rl_liquidity_mode=rl_liquidity_mode,
                        rl_quoter_split=rl_quoter_split,
                        rl_enable_passive_quotes=rl_enable_passive_quotes,
                        rl_quote_mode=rl_quote_mode,
                        rl_quote_offset_ticks=rl_quote_offset_ticks,
                        rl_quote_size=rl_quote_size,
                    )
                    market_frame, rl_frame, transition_frame = run_policy_episode(config, shared_policy=policy)
                    summary_row = summarize_episode(
                        episode_index=episode_index,
                        phi=phi_value,
                        seed=int(seed),
                        rl_frame=rl_frame,
                        transition_frame=transition_frame,
                    )
                    summary_row["evaluation_mode"] = evaluation_mode
                    per_seed_rows.append(summary_row)

                    diagnostics, agent_frame = compute_policy_evaluation_diagnostics(
                        market_frame,
                        rl_frame,
                        transition_frame,
                        policy=policy,
                        inventory_cap=inventory_cap,
                    )
                    diagnostics = dict(diagnostics)
                    diagnostics["phi"] = phi_value
                    diagnostics["seed"] = int(seed)
                    diagnostics["evaluation_mode"] = evaluation_mode
                    diagnostics_rows.append(diagnostics)
                    if _has_midprice_pipeline_issue(diagnostics):
                        print(
                            "WARNING: pipeline_issue detected for "
                            f"phi={phi_label}, mode={evaluation_mode}, seed={int(seed)} "
                            f"(defined_midprice_despite_missing_side_fraction="
                            f"{float(diagnostics.get('defined_midprice_despite_missing_side_fraction', 0.0)):.6f}, "
                            f"midprice_gap_without_missing_side_fraction="
                            f"{float(diagnostics.get('midprice_gap_without_missing_side_fraction', 0.0)):.6f})"
                        )
                    if not agent_frame.empty:
                        agent_frames.append(
                            agent_frame.assign(phi=phi_value, seed=int(seed), evaluation_mode=evaluation_mode)
                        )
                    market_metrics = compute_extended_market_metrics(
                        market_frame,
                        tick_size=float(config.tick_size),
                    )
                    market_metrics.update({"phi": phi_value, "seed": int(seed), "evaluation_mode": evaluation_mode})
                    market_metric_rows.append(market_metrics)

                    if int(seed) == representative_seed:
                        representative_market_csv = mode_dir / f"representative_market_seed_{seed}.csv"
                        representative_market_plot = mode_dir / f"representative_market_seed_{seed}.png"
                        market_frame.to_csv(representative_market_csv, index=False)
                        save_market_plot(
                            market_frame,
                            representative_market_plot,
                            title=f"Phi {phi_label} {evaluation_mode.capitalize()} Seed {seed}",
                        )
                        save_market_plot(
                            market_frame,
                            phi_plot_dir / f"{evaluation_mode}_representative_market_seed_{seed}.png",
                            title=f"Phi {phi_label} {evaluation_mode.capitalize()} Seed {seed}",
                        )
                        rl_frame.to_csv(mode_dir / f"representative_rl_decisions_seed_{seed}.csv", index=False)
                        transition_frame.to_csv(
                            mode_dir / f"representative_rl_transitions_seed_{seed}.csv",
                            index=False,
                        )
                        representative_diagnostics = diagnostics
                        representative_agent_frame = (
                            agent_frame.assign(phi=phi_value, seed=int(seed), evaluation_mode=evaluation_mode)
                            if not agent_frame.empty
                            else pd.DataFrame()
                        )

                per_seed_frame = pd.DataFrame(per_seed_rows)
                diagnostics_frame = pd.DataFrame(diagnostics_rows)
                market_metrics_frame = pd.DataFrame(market_metric_rows)
                agent_diagnostics_frame = (
                    pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame()
                )

                per_seed_frame.to_csv(mode_dir / "evaluation_runs.csv", index=False)
                diagnostics_frame.to_csv(mode_dir / "diagnostics_by_seed.csv", index=False)
                market_metrics_frame.to_csv(mode_dir / "market_metrics_by_seed.csv", index=False)
                agent_diagnostics_frame.to_csv(mode_dir / "agent_diagnostics.csv", index=False)

                aggregate = aggregate_mode_results(
                    phi=phi_value,
                    evaluation_mode=evaluation_mode,
                    episode_frame=per_seed_frame,
                    diagnostics_frame=diagnostics_frame,
                    market_metrics_frame=market_metrics_frame,
                    representative_seed=representative_seed,
                )
                pd.DataFrame([aggregate]).to_csv(mode_dir / "evaluation_summary.csv", index=False)
                if representative_diagnostics is None:
                    representative_diagnostics = {}
                (mode_dir / "evaluation_report.md").write_text(
                    build_mode_report(
                        phi=phi_value,
                        evaluation_mode=evaluation_mode,
                        aggregate=aggregate,
                        representative_diagnostics=representative_diagnostics,
                        representative_agent_frame=representative_agent_frame,
                    ),
                    encoding="utf-8",
                )

                phi_summary_row.update(
                    {
                        f"{evaluation_mode}_{key}": value
                        for key, value in aggregate.items()
                        if key not in {"phi", "evaluation_mode"}
                    }
                )
                detailed_market_rows.append(market_metrics_frame)
                detailed_diagnostic_rows.append(diagnostics_frame)
            finally:
                if hasattr(policy, "set_deterministic"):
                    getattr(policy, "set_deterministic")(
                        bool(original_deterministic) if original_deterministic is not None else False
                    )

        summary_rows.append(phi_summary_row)

    summary_frame = pd.DataFrame(summary_rows).sort_values("phi").reset_index(drop=True)
    nested_results = []
    for row in summary_rows:
        nested = {
            "phi": float(row["phi"]),
        }
        nested["agent_counts"] = json.loads(str(row["agent_counts_json"]))
        nested["training"] = {
            key.replace("training_", ""): _json_ready(value)
            for key, value in row.items()
            if key.startswith("training_")
        }
        nested["paths"] = {
            "checkpoint_path": row.get("checkpoint_path", ""),
        }
        for mode in evaluation_modes:
            nested[mode] = {
                key[len(mode) + 1 :]: _json_ready(value)
                for key, value in row.items()
                if key.startswith(f"{mode}_")
            }
        nested_results.append(nested)
    _write_json(
        root / "phi_sweep_summary.json",
        {
            "experiment_config": experiment_config,
            "results": nested_results,
        },
    )
    summary_frame.to_csv(root / "phi_sweep_summary.csv", index=False)
    (root / "phi_sweep_report.md").write_text(
        build_phi_sweep_report(summary_frame, experiment_config),
        encoding="utf-8",
    )

    if detailed_market_rows:
        pd.concat(detailed_market_rows, ignore_index=True).to_csv(
            summaries_dir / "per_seed_market_metrics.csv",
            index=False,
        )
    if detailed_diagnostic_rows:
        pd.concat(detailed_diagnostic_rows, ignore_index=True).to_csv(
            summaries_dir / "per_seed_rl_diagnostics.csv",
            index=False,
        )
    if detailed_training_rows:
        pd.concat(detailed_training_rows, ignore_index=True).to_csv(
            summaries_dir / "training_episode_metrics.csv",
            index=False,
        )

    saved_top_level_plots = save_cross_phi_plots(summary_frame, plots_dir)
    return {
        "output_dir": root,
        "summary_frame": summary_frame,
        "saved_top_level_plots": saved_top_level_plots,
        "experiment_config": experiment_config,
    }
