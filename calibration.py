"""Baseline calibration, diagnostics, and sweep helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from analysis import summarize_market_frame
from config import SimulationConfig
from market import MarketSimulator

SUPPORTED_SWEEP_ALIASES: dict[str, str] = {
    "sigma_v": "sigma_v",
    "s_max": "zic_surplus_max_ticks",
    "zic_surplus_max_ticks": "zic_surplus_max_ticks",
    "delta": "value_delta_ticks",
    "value_delta_ticks": "value_delta_ticks",
    "mm_spread_ticks": "mm_spread_ticks",
    "alpha": "mm_alpha",
    "mm_alpha": "mm_alpha",
    "mm_count_scale": "market_maker_count_scale",
    "market_maker_count_scale": "market_maker_count_scale",
    "mm_quote_size": "mm_quote_size",
    "trend_arrival_rate": "trend_wake_up_frequency",
    "trend_wake_up_frequency": "trend_wake_up_frequency",
    "trend_passive_size": "trend_passive_order_size",
    "trend_passive_order_size": "trend_passive_order_size",
    "value_passive_size": "value_passive_order_size",
    "value_passive_order_size": "value_passive_order_size",
    "trend_order_size": "trend_order_size",
    "value_order_size": "value_order_size",
    "trend_aggressive_probability": "trend_aggressive_probability",
    "value_aggressive_probability": "value_aggressive_probability",
    "noise_limit_probability": "noise_limit_probability",
}

CALIBRATION_PARAMETER_COLUMNS = [
    "sigma_v",
    "zic_surplus_max_ticks",
    "value_delta_ticks",
    "mm_spread_ticks",
    "mm_alpha",
    "trend_wake_up_frequency",
]

RUN_DIAGNOSTIC_COLUMNS = [
    "average_spread",
    "average_relative_spread",
    "average_top_of_book_depth",
    "return_variance",
    "excess_kurtosis",
    "volatility_clustering",
    "max_drawdown",
    "crash_rate",
    "tail_exposure",
    "trade_count",
    "traded_volume",
    "baseline_quality_score",
    "mm_mean_inventory",
    "mm_mean_abs_inventory",
    "mm_max_abs_inventory",
]

CLASSIFICATION_ORDER = ["stable", "thin", "frozen", "chaotic", "crash-prone"]
_DEFAULT_CONFIG = SimulationConfig()


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Thresholds for coarse baseline run diagnostics."""

    min_trade_count: int = 10
    max_average_spread: float = 0.10
    max_average_relative_spread: float = 0.001
    min_average_top_depth: float = 1.0
    min_return_variance: float = 1e-8
    max_return_variance: float = 5e-4
    max_crash_rate: float = 0.02
    max_abs_drawdown: float = 0.15
    max_excess_kurtosis: float = 20.0


def normalize_sweep_parameter(name: str) -> str:
    """Map a user-facing sweep name to the config field name."""

    if name not in SUPPORTED_SWEEP_ALIASES:
        supported = ", ".join(sorted(SUPPORTED_SWEEP_ALIASES))
        raise ValueError(f"Unsupported sweep parameter '{name}'. Supported: {supported}.")
    return SUPPORTED_SWEEP_ALIASES[name]


def parse_sweep_spec(spec: str) -> tuple[str, list[Any]]:
    """Parse a single sweep spec of the form ``name=v1,v2,...``."""

    if "=" not in spec:
        raise ValueError(f"Invalid sweep specification '{spec}'. Expected name=v1,v2,...")
    raw_name, raw_values = spec.split("=", 1)
    parameter_name = normalize_sweep_parameter(raw_name.strip())
    values = [value.strip() for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError(f"Sweep parameter '{parameter_name}' requires at least one value.")
    return parameter_name, [parse_parameter_value(parameter_name, value) for value in values]


def parse_parameter_value(parameter_name: str, value: str) -> Any:
    """Parse a sweep value using the corresponding config field type."""

    current_value = getattr(_DEFAULT_CONFIG, parameter_name)
    if isinstance(current_value, bool):
        normalized = value.lower()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise ValueError(f"Invalid boolean value '{value}' for {parameter_name}.")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(value)
    if isinstance(current_value, float):
        return float(value)
    return value


def build_parameter_grid(
    sweep_specs: Sequence[tuple[str, Sequence[Any]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Expand parsed sweep specs into a Cartesian parameter grid."""

    parameter_columns = [name for name, _ in sweep_specs]
    if not parameter_columns:
        return [], [{}]

    value_lists = [list(values) for _, values in sweep_specs]
    grid = [
        dict(zip(parameter_columns, combination))
        for combination in product(*value_lists)
    ]
    return parameter_columns, grid


def consecutive_seeds(start_seed: int, num_runs: int) -> list[int]:
    """Return a simple deterministic seed schedule for calibration."""

    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")
    return [start_seed + offset for offset in range(num_runs)]


def classify_run(
    diagnostics: Mapping[str, float],
    thresholds: DiagnosticThresholds,
) -> str:
    """Classify a baseline run using simple threshold logic."""

    trade_count = float(diagnostics.get("trade_count", 0.0))
    average_depth = float(diagnostics.get("average_top_of_book_depth", 0.0))
    average_spread = float(diagnostics.get("average_spread", 0.0))
    average_relative_spread = float(diagnostics.get("average_relative_spread", 0.0))
    return_variance = float(diagnostics.get("return_variance", float("nan")))
    crash_rate = float(diagnostics.get("crash_rate", 0.0))
    max_drawdown = float(diagnostics.get("max_drawdown", 0.0))
    excess_kurtosis = float(diagnostics.get("excess_kurtosis", 0.0))

    if crash_rate > thresholds.max_crash_rate or max_drawdown < -thresholds.max_abs_drawdown:
        return "crash-prone"
    if pd.isna(return_variance) or trade_count < thresholds.min_trade_count or return_variance < thresholds.min_return_variance:
        return "frozen"
    if average_depth < thresholds.min_average_top_depth or average_relative_spread > thresholds.max_average_relative_spread:
        return "thin"
    if average_spread > thresholds.max_average_spread or return_variance > thresholds.max_return_variance or excess_kurtosis > thresholds.max_excess_kurtosis:
        return "chaotic"
    return "stable"


def compute_baseline_quality_score(
    diagnostics: Mapping[str, float],
    thresholds: DiagnosticThresholds,
) -> float:
    """Compute a coarse score that penalizes clearly bad baseline runs."""

    score = 100.0

    trade_count = max(float(diagnostics.get("trade_count", 0.0)), 0.0)
    if trade_count < thresholds.min_trade_count:
        score -= 35.0 * (1.0 - (trade_count / thresholds.min_trade_count))

    average_spread = max(float(diagnostics.get("average_spread", 0.0)), 0.0)
    if average_spread > thresholds.max_average_spread:
        score -= min(20.0, 20.0 * ((average_spread / thresholds.max_average_spread) - 1.0))

    average_depth = max(float(diagnostics.get("average_top_of_book_depth", 0.0)), 0.0)
    if average_depth < thresholds.min_average_top_depth:
        score -= 20.0 * (1.0 - (average_depth / thresholds.min_average_top_depth))

    crash_rate = max(float(diagnostics.get("crash_rate", 0.0)), 0.0)
    if crash_rate > thresholds.max_crash_rate:
        score -= min(25.0, 25.0 * (crash_rate / thresholds.max_crash_rate))

    return_variance = float(diagnostics.get("return_variance", float("nan")))
    if pd.isna(return_variance) or return_variance < thresholds.min_return_variance:
        score -= 20.0

    return max(0.0, min(100.0, score))


def compute_run_diagnostics(
    simulator: MarketSimulator,
    *,
    tail_quantile: float = 0.05,
    crash_threshold: float = -0.05,
    squared_return_lags: int = 5,
    thresholds: DiagnosticThresholds | None = None,
) -> dict[str, Any]:
    """Compute run-level diagnostics from a completed simulation."""

    if simulator.frame is None or simulator.end_state is None:
        raise RuntimeError("Simulation must be run before diagnostics are computed.")

    thresholds = thresholds or DiagnosticThresholds()
    frame = simulator.frame
    end_state = simulator.end_state

    diagnostics = summarize_market_frame(
        frame,
        squared_return_lags=squared_return_lags,
        tail_quantile=tail_quantile,
        crash_threshold=crash_threshold,
    )

    exchange = next(agent for agent in end_state["agents"] if getattr(agent, "type", "") == "ExchangeAgent")
    order_book = exchange.order_books[simulator.config.ticker]
    trade_count = len(order_book.buy_transactions) + len(order_book.sell_transactions)
    traded_volume = sum(quantity for _, quantity in order_book.buy_transactions)
    traded_volume += sum(quantity for _, quantity in order_book.sell_transactions)
    diagnostics["trade_count"] = int(trade_count)
    diagnostics["traded_volume"] = float(traded_volume)
    diagnostics["num_rows"] = int(len(frame))

    mm_agents = [
        agent for agent in end_state["agents"] if getattr(agent, "type", "") == "AdaptiveMarketMaker"
    ]
    inventories = [int(agent.get_holdings(simulator.config.ticker)) for agent in mm_agents]
    if inventories:
        diagnostics["mm_mean_inventory"] = float(pd.Series(inventories, dtype=float).mean())
        diagnostics["mm_mean_abs_inventory"] = float(pd.Series(inventories, dtype=float).abs().mean())
        diagnostics["mm_max_abs_inventory"] = float(pd.Series(inventories, dtype=float).abs().max())
    else:
        diagnostics["mm_mean_inventory"] = 0.0
        diagnostics["mm_mean_abs_inventory"] = 0.0
        diagnostics["mm_max_abs_inventory"] = 0.0

    diagnostics["baseline_quality_score"] = compute_baseline_quality_score(
        diagnostics,
        thresholds=thresholds,
    )
    diagnostics["classification"] = classify_run(
        diagnostics,
        thresholds=thresholds,
    )
    return diagnostics


def run_parameter_sweep(
    base_config: SimulationConfig,
    *,
    seeds: Sequence[int],
    parameter_grid: Sequence[Mapping[str, Any]] | None = None,
    parameter_columns: Sequence[str] | None = None,
    thresholds: DiagnosticThresholds | None = None,
    tail_quantile: float = 0.05,
    crash_threshold: float = -0.05,
    squared_return_lags: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a phi=0 baseline sweep across seeds and parameter settings."""

    if not seeds:
        raise ValueError("At least one seed is required for a parameter sweep.")

    thresholds = thresholds or DiagnosticThresholds()
    parameter_grid = list(parameter_grid or [{}])
    parameter_columns = list(parameter_columns or [])

    run_records = []
    baseline_config = replace(base_config, phi=0.0)

    for parameter_setting in parameter_grid:
        config_template = replace(baseline_config, **parameter_setting)
        for seed in seeds:
            config = replace(config_template, seed=int(seed))
            simulator = MarketSimulator(config)
            simulator.run()
            diagnostics = compute_run_diagnostics(
                simulator,
                tail_quantile=tail_quantile,
                crash_threshold=crash_threshold,
                squared_return_lags=squared_return_lags,
                thresholds=thresholds,
            )
            run_record = {
                "seed": int(seed),
                **{column: getattr(config, column) for column in CALIBRATION_PARAMETER_COLUMNS},
                **diagnostics,
            }
            run_records.append(run_record)

    runs = pd.DataFrame(run_records).sort_values(parameter_columns + ["seed"] if parameter_columns else ["seed"])
    runs = runs.reset_index(drop=True)
    summary = summarize_parameter_sweep(
        runs=runs,
        parameter_columns=parameter_columns,
    )
    return runs, summary


def run_baseline_calibration(
    base_config: SimulationConfig,
    *,
    seeds: Sequence[int],
    tail_quantile: float = 0.05,
    crash_threshold: float = -0.05,
    squared_return_lags: int = 5,
    thresholds: DiagnosticThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated baseline simulations and summarize their diagnostics."""

    return run_parameter_sweep(
        base_config,
        seeds=seeds,
        parameter_grid=[{}],
        parameter_columns=[],
        thresholds=thresholds,
        tail_quantile=tail_quantile,
        crash_threshold=crash_threshold,
        squared_return_lags=squared_return_lags,
    )


def summarize_parameter_sweep(
    runs: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
) -> pd.DataFrame:
    """Group run-level diagnostics by parameter setting and summarize them."""

    working = runs.copy()
    group_columns = list(parameter_columns)
    dummy_group = "__baseline_group"
    if not group_columns:
        working[dummy_group] = "baseline"
        group_columns = [dummy_group]

    grouped = working.groupby(group_columns, dropna=False)
    metric_summary = grouped[RUN_DIAGNOSTIC_COLUMNS].agg(["mean", "std", "min", "max"]).reset_index()
    metric_summary.columns = list(group_columns) + [
        f"{metric}_{stat}"
        for metric, stat in metric_summary.columns[len(group_columns):]
    ]

    class_shares = (
        grouped["classification"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for label in CLASSIFICATION_ORDER:
        if label not in class_shares.columns:
            class_shares[label] = 0.0
    class_shares = class_shares[group_columns + CLASSIFICATION_ORDER]
    class_shares = class_shares.rename(columns={label: f"share_{label}" for label in CLASSIFICATION_ORDER})

    counts = grouped.size().reset_index(name="num_runs")
    summary = metric_summary.merge(class_shares, on=group_columns, how="left").merge(
        counts,
        on=group_columns,
        how="left",
    )

    summary["ranking_score"] = (
        summary["baseline_quality_score_mean"]
        + 20.0 * summary.get("share_stable", 0.0)
        - 20.0 * summary.get("share_crash-prone", 0.0)
        - 10.0 * summary.get("share_frozen", 0.0)
    )

    if dummy_group in summary.columns:
        summary = summary.drop(columns=[dummy_group])

    return summary.sort_values("ranking_score", ascending=False).reset_index(drop=True)


def recommend_baseline_setting(summary: pd.DataFrame) -> pd.Series:
    """Return the top-ranked summary row."""

    if summary.empty:
        raise ValueError("Cannot recommend a baseline from an empty summary.")
    return summary.sort_values("ranking_score", ascending=False).iloc[0]


def save_sweep_plots(
    summary: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    output_dir: str | Path,
) -> list[Path]:
    """Save a few simple sweep plots when a single sweep parameter is used."""

    if len(parameter_columns) != 1 or summary.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save sweep plots.") from exc

    parameter = parameter_columns[0]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("average_spread_mean", "Average Spread", f"average_spread_vs_{parameter}.png"),
        (
            "average_top_of_book_depth_mean",
            "Average Top-of-Book Depth",
            f"average_depth_vs_{parameter}.png",
        ),
        ("crash_rate_mean", "Crash Rate", f"crash_rate_vs_{parameter}.png"),
        ("tail_exposure_mean", "Tail Exposure", f"tail_exposure_vs_{parameter}.png"),
    ]

    saved_paths: list[Path] = []
    x = summary[parameter]
    is_numeric = pd.api.types.is_numeric_dtype(x)

    for metric_column, ylabel, filename in plot_specs:
        if metric_column not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        if is_numeric:
            ax.plot(x, summary[metric_column], marker="o")
        else:
            positions = range(len(summary))
            ax.plot(list(positions), summary[metric_column], marker="o")
            ax.set_xticks(list(positions))
            ax.set_xticklabels([str(value) for value in x], rotation=45, ha="right")
        ax.set_xlabel(parameter)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {parameter}")
        ax.grid(True, alpha=0.3)
        path = output_path / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
