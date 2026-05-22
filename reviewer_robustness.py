"""Reviewer-facing robustness controls and diagnostic plots.

This module intentionally builds on the existing phi-sweep implementation rather
than replacing it.  The key control here is composition matching: for each
``phi`` and taker/quoter split, compare trained RL behavior against nonlearned
policies under the same RL liquidity-mode composition.  That does not eliminate
the passive/aggressive confound, but it separates "learned behavior" from the
mechanical change in order-type composition more cleanly than the original
single mixed setting.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from env import InventoryAwarePolicy, InventoryAwareQuoterPolicy, RandomPolicy, RandomQuoterPolicy
from phi_experiment import (
    compute_extended_market_metrics,
    format_phi,
    _add_numeric_summary,
    _json_ready,
)
from ppo_training import load_policy_artifact, run_policy_episode, summarize_episode
from rl_diagnostics import compute_policy_evaluation_diagnostics


DEFAULT_ROBUSTNESS_METRICS: tuple[str, ...] = (
    "buy_fraction",
    "sell_fraction",
    "signed_action_imbalance",
    "quote_action_fraction",
    "market_order_submission_rate",
    "quote_order_submission_rate",
    "one_sided_book_fraction",
    "missing_bid_fraction",
    "missing_ask_fraction",
    "average_spread",
    "average_depth",
    "volatility",
    "zero_return_fraction",
    "trade_timestep_count",
    "traded_volume_total",
)


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_reviewer_robustness_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _policy_for_control(control_policy: str) -> object | None:
    normalized = control_policy.strip().lower()
    if normalized == "random":
        return None
    if normalized == "inventory_aware":
        from ppo_training import SharedPolicyBundle

        return SharedPolicyBundle(
            taker_policy=InventoryAwarePolicy(),
            quoter_policy=InventoryAwareQuoterPolicy(),
        )
    raise ValueError(f"unknown control_policy={control_policy!r}")


def _policy_factory_names(control_policy: str) -> dict[str, str]:
    normalized = control_policy.strip().lower()
    if normalized == "random":
        return {"rl_policy_name": "random", "rl_quoter_policy_name": "random_quoter"}
    if normalized == "inventory_aware":
        return {
            "rl_policy_name": "inventory_aware",
            "rl_quoter_policy_name": "inventory_aware_quoter",
        }
    raise ValueError(f"unknown control_policy={control_policy!r}")


def _checkpoint_for_phi(experiment_dir: str | Path, phi: float) -> Path | None:
    path = Path(experiment_dir) / f"phi_{format_phi(phi)}" / "checkpoints" / "shared_ppo_policy_final.npz"
    return path if path.exists() else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _trade_activity_metrics(market_frame: pd.DataFrame) -> dict[str, float]:
    if "traded_volume" not in market_frame.columns:
        return {
            "trade_timestep_count": float("nan"),
            "trade_timestep_fraction": float("nan"),
            "traded_volume_total": float("nan"),
        }
    volume = pd.to_numeric(market_frame["traded_volume"], errors="coerce").fillna(0.0)
    return {
        "trade_timestep_count": float((volume > 0.0).sum()),
        "trade_timestep_fraction": float((volume > 0.0).mean()) if len(volume) else float("nan"),
        "traded_volume_total": float(volume.sum()),
    }


def _derived_mechanism_metrics(row: dict[str, Any]) -> dict[str, float]:
    buy = float(row.get("buy_fraction", 0.0))
    sell = float(row.get("sell_fraction", 0.0))
    quote_bid = float(row.get("quote_bid_fraction", 0.0))
    quote_ask = float(row.get("quote_ask_fraction", 0.0))
    quote_both = float(row.get("quote_both_fraction", 0.0))
    decisions = float(row.get("num_rl_decisions", 0.0))
    taker_decisions = float(row.get("num_rl_taker_decisions", 0.0))
    passive_orders = float(row.get("total_rl_passive_order_count", 0.0))
    aggressive_orders = float(row.get("total_rl_aggressive_order_count", 0.0))
    submitted_buy = float(row.get("submitted_buy_action_count", 0.0))
    submitted_sell = float(row.get("submitted_sell_action_count", 0.0))
    return {
        "signed_action_imbalance": buy - sell,
        "abs_signed_action_imbalance": abs(buy - sell),
        "quote_action_fraction": quote_bid + quote_ask + quote_both,
        "quote_side_imbalance": quote_bid - quote_ask,
        "quote_order_submission_rate": _safe_ratio(passive_orders, decisions),
        "market_order_submission_rate": _safe_ratio(aggressive_orders, decisions),
        "submitted_taker_order_rate": _safe_ratio(submitted_buy + submitted_sell, taker_decisions),
        "submitted_buy_minus_sell_count": submitted_buy - submitted_sell,
    }


def _one_sided_onset_event_study(
    market_frame: pd.DataFrame,
    *,
    window_steps: int = 5,
) -> pd.DataFrame:
    """Return market-state windows around visible one-sided-book onsets."""

    if market_frame.empty or not {"bid_depth", "ask_depth"}.issubset(market_frame.columns):
        return pd.DataFrame()
    bid_depth = pd.to_numeric(market_frame["bid_depth"], errors="coerce").fillna(0.0)
    ask_depth = pd.to_numeric(market_frame["ask_depth"], errors="coerce").fillna(0.0)
    visible = (bid_depth + ask_depth) > 0.0
    one_sided = visible & ((bid_depth > 0.0) ^ (ask_depth > 0.0))
    onset = one_sided & ~one_sided.shift(fill_value=False)
    onset_indices = list(np.flatnonzero(onset.to_numpy(dtype=bool)))
    if not onset_indices:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    traded_volume = pd.to_numeric(
        market_frame.get("traded_volume", pd.Series(np.nan, index=market_frame.index)),
        errors="coerce",
    )
    spread = pd.to_numeric(
        market_frame.get("spread", pd.Series(np.nan, index=market_frame.index)),
        errors="coerce",
    )
    midprice = pd.to_numeric(
        market_frame.get("midprice", pd.Series(np.nan, index=market_frame.index)),
        errors="coerce",
    )
    for event_index, center in enumerate(onset_indices):
        start = max(0, center - int(window_steps))
        stop = min(len(market_frame), center + int(window_steps) + 1)
        base_midprice = float(midprice.iloc[center]) if not pd.isna(midprice.iloc[center]) else float("nan")
        for position in range(start, stop):
            relative_step = int(position - center)
            rows.append(
                {
                    "event_index": float(event_index),
                    "relative_step": float(relative_step),
                    "bid_depth": float(bid_depth.iloc[position]),
                    "ask_depth": float(ask_depth.iloc[position]),
                    "depth_imbalance": _safe_ratio(
                        float(bid_depth.iloc[position] - ask_depth.iloc[position]),
                        float(bid_depth.iloc[position] + ask_depth.iloc[position]),
                    ),
                    "spread": float(spread.iloc[position]) if not pd.isna(spread.iloc[position]) else float("nan"),
                    "traded_volume": float(traded_volume.iloc[position])
                    if not pd.isna(traded_volume.iloc[position])
                    else float("nan"),
                    "midprice_change_from_onset": (
                        float(midprice.iloc[position] - base_midprice)
                        if not pd.isna(midprice.iloc[position]) and not np.isnan(base_midprice)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_seed_results(seed_frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate per-seed rows without mixing seeds and episodes."""

    rows: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in seed_frame.columns
        if column not in set(group_columns) | {"checkpoint_path"}
        and pd.api.types.is_numeric_dtype(seed_frame[column])
    ]
    for group_key, group in seed_frame.groupby(list(group_columns), dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        aggregate: dict[str, Any] = {
            column: value for column, value in zip(group_columns, key_values)
        }
        aggregate["num_seeds"] = float(group["seed"].nunique()) if "seed" in group.columns else float(len(group))
        for column in numeric_columns:
            _add_numeric_summary(aggregate, column, group[column])
        rows.append(aggregate)
    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)


def run_composition_control(
    *,
    phi_grid: Sequence[float],
    quoter_splits: Sequence[float],
    evaluation_seeds: Sequence[int],
    output_dir: str | Path,
    end_time: str,
    log_frequency: str,
    num_agents: int,
    return_window: int,
    lambda_q: float,
    flat_hold_penalty: float,
    inventory_cap: int | None,
    evaluation_mode: str,
    trained_experiment_dir: str | Path | None = None,
    include_trained: bool = True,
    control_policy: str = "random",
    quote_mode: str = "at_best",
    quote_offset_ticks: int = 0,
    quote_size: int = 1,
    event_window_steps: int = 5,
) -> dict[str, Any]:
    """Run matched-composition trained-vs-nonlearned robustness evaluations."""

    if not phi_grid:
        raise ValueError("phi_grid must not be empty.")
    if not quoter_splits:
        raise ValueError("quoter_splits must not be empty.")
    if not evaluation_seeds:
        raise ValueError("evaluation_seeds must not be empty.")
    normalized_mode = evaluation_mode.strip().lower()
    if normalized_mode not in {"greedy", "stochastic"}:
        raise ValueError("evaluation_mode must be greedy or stochastic.")

    root = Path(output_dir)
    summaries_dir = root / "summaries"
    plots_dir = root / "plots"
    root.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "phi_grid": [float(phi) for phi in phi_grid],
        "quoter_splits": [float(split) for split in quoter_splits],
        "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
        "evaluation_mode": normalized_mode,
        "trained_experiment_dir": str(trained_experiment_dir) if trained_experiment_dir else None,
        "include_trained": bool(include_trained),
        "control_policy": control_policy,
        "end_time": end_time,
        "log_frequency": log_frequency,
        "num_agents": int(num_agents),
        "return_window": int(return_window),
        "lambda_q": float(lambda_q),
        "flat_hold_penalty": float(flat_hold_penalty),
        "inventory_cap": int(inventory_cap) if inventory_cap is not None else None,
        "quote_mode": quote_mode,
        "quote_offset_ticks": int(quote_offset_ticks),
        "quote_size": int(quote_size),
    }
    _write_json(root / "experiment_config.json", config_payload)

    per_seed_rows: list[dict[str, Any]] = []
    event_study_frames: list[pd.DataFrame] = []
    control_factory_names = _policy_factory_names(control_policy)

    for phi in phi_grid:
        phi_value = float(phi)
        for quoter_split in quoter_splits:
            split_value = float(quoter_split)
            liquidity_mode = "taker_only" if split_value <= 0.0 else ("quoter_only" if split_value >= 1.0 else "mixed")
            conditions: list[tuple[str, object | None, str]] = [
                (f"matched_{control_policy}", _policy_for_control(control_policy), "")
            ]
            if include_trained and phi_value > 0.0 and trained_experiment_dir is not None:
                checkpoint_path = _checkpoint_for_phi(trained_experiment_dir, phi_value)
                if checkpoint_path is not None:
                    conditions.append(
                        (
                            "trained",
                            load_policy_artifact(checkpoint_path, deterministic=(normalized_mode == "greedy")),
                            str(checkpoint_path),
                        )
                    )

            for condition, policy, checkpoint_path in conditions:
                for seed in evaluation_seeds:
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
                        rl_liquidity_mode=liquidity_mode,
                        rl_quoter_split=split_value,
                        rl_enable_passive_quotes=True,
                        rl_quote_mode=quote_mode,
                        rl_quote_offset_ticks=quote_offset_ticks,
                        rl_quote_size=quote_size,
                        **control_factory_names,
                    )
                    market_frame, rl_frame, transition_frame = run_policy_episode(config, shared_policy=policy)
                    episode_row = summarize_episode(
                        episode_index=0,
                        phi=phi_value,
                        seed=int(seed),
                        rl_frame=rl_frame,
                        transition_frame=transition_frame,
                    )
                    diagnostics, _agent_frame = compute_policy_evaluation_diagnostics(
                        market_frame,
                        rl_frame,
                        transition_frame,
                        policy=policy,
                        inventory_cap=inventory_cap,
                    )
                    market_metrics = compute_extended_market_metrics(market_frame, tick_size=float(config.tick_size))
                    row: dict[str, Any] = {
                        "condition": condition,
                        "phi": phi_value,
                        "seed": int(seed),
                        "evaluation_mode": normalized_mode,
                        "rl_liquidity_mode": liquidity_mode,
                        "rl_quoter_split": split_value,
                        "checkpoint_path": checkpoint_path,
                        "agent_counts_json": json.dumps(config.agent_counts(), sort_keys=True),
                        "rl_role_counts_json": json.dumps(config.rl_role_counts(), sort_keys=True),
                    }
                    row.update(episode_row)
                    row.update(diagnostics)
                    row.update(market_metrics)
                    row.update(_trade_activity_metrics(market_frame))
                    row.update(_derived_mechanism_metrics(row))
                    per_seed_rows.append(row)

                    events = _one_sided_onset_event_study(
                        market_frame,
                        window_steps=event_window_steps,
                    )
                    if not events.empty:
                        events.insert(0, "condition", condition)
                        events.insert(1, "phi", phi_value)
                        events.insert(2, "seed", int(seed))
                        events.insert(3, "rl_quoter_split", split_value)
                        events.insert(4, "evaluation_mode", normalized_mode)
                        event_study_frames.append(events)

    per_seed_frame = pd.DataFrame(per_seed_rows)
    per_seed_path = summaries_dir / "composition_control_by_seed.csv"
    per_seed_frame.to_csv(per_seed_path, index=False)
    aggregate_frame = aggregate_seed_results(
        per_seed_frame,
        ["condition", "evaluation_mode", "rl_quoter_split", "phi"],
    )
    aggregate_path = summaries_dir / "composition_control_summary.csv"
    aggregate_frame.to_csv(aggregate_path, index=False)

    event_study_frame = (
        pd.concat(event_study_frames, ignore_index=True) if event_study_frames else pd.DataFrame()
    )
    event_study_path = summaries_dir / "one_sided_onset_event_study.csv"
    event_study_frame.to_csv(event_study_path, index=False)

    saved_plots = save_reviewer_plots(
        aggregate_frame,
        event_study_frame,
        plots_dir,
    )
    report_path = root / "reviewer_robustness_report.md"
    report_path.write_text(
        build_reviewer_report(
            aggregate_frame,
            config_payload,
            saved_plots=saved_plots,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": root,
        "per_seed_path": per_seed_path,
        "aggregate_path": aggregate_path,
        "event_study_path": event_study_path,
        "saved_plots": saved_plots,
        "report_path": report_path,
    }


def _metric_error_columns(metric: str) -> tuple[str, str, str]:
    return f"{metric}_mean", f"{metric}_ci95_lower", f"{metric}_ci95_upper"


def _plot_metric_grid(
    aggregate_frame: pd.DataFrame,
    output_dir: Path,
    *,
    metric: str,
    ylabel: str,
    filename: str,
    title: str,
) -> Path | None:
    mean_column, lower_column, upper_column = _metric_error_columns(metric)
    required = {"condition", "rl_quoter_split", "phi", mean_column}
    if aggregate_frame.empty or not required.issubset(aggregate_frame.columns):
        return None

    plt = _load_matplotlib()
    splits = sorted(float(value) for value in aggregate_frame["rl_quoter_split"].dropna().unique())
    if not splits:
        return None
    fig, axes = plt.subplots(1, len(splits), figsize=(6.0 * len(splits), 4.3), sharey=True)
    if len(splits) == 1:
        axes = [axes]
    for axis, split in zip(axes, splits):
        split_frame = aggregate_frame[np.isclose(aggregate_frame["rl_quoter_split"].astype(float), split)]
        for condition, condition_frame in split_frame.groupby("condition"):
            ordered = condition_frame.sort_values("phi")
            x_values = pd.to_numeric(ordered["phi"], errors="coerce").to_numpy(dtype=float)
            y_values = pd.to_numeric(ordered[mean_column], errors="coerce").to_numpy(dtype=float)
            if lower_column in ordered.columns and upper_column in ordered.columns:
                lower = pd.to_numeric(ordered[lower_column], errors="coerce").to_numpy(dtype=float)
                upper = pd.to_numeric(ordered[upper_column], errors="coerce").to_numpy(dtype=float)
                yerr = np.vstack([y_values - lower, upper - y_values])
                axis.errorbar(x_values, y_values, yerr=yerr, marker="o", capsize=3, label=condition)
            else:
                axis.plot(x_values, y_values, marker="o", label=condition)
        axis.set_title(f"quoter split = {split:.2f}")
        axis.set_xlabel("phi")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_mechanism_panel(aggregate_frame: pd.DataFrame, output_dir: Path) -> Path | None:
    metrics = [
        ("buy_fraction", "Buy fraction"),
        ("sell_fraction", "Sell fraction"),
        ("signed_action_imbalance", "Buy - sell"),
        ("quote_action_fraction", "Quote action fraction"),
        ("market_order_submission_rate", "Market order submissions / RL decision"),
        ("quote_order_submission_rate", "Passive order submissions / RL decision"),
    ]
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        mean_column = f"{metric}_mean"
        if mean_column not in aggregate_frame.columns:
            axis.set_visible(False)
            continue
        for (condition, split), group in aggregate_frame.groupby(["condition", "rl_quoter_split"]):
            ordered = group.sort_values("phi")
            axis.plot(
                ordered["phi"],
                ordered[mean_column],
                marker="o",
                linewidth=1.4,
                label=f"{condition}, q={float(split):.2f}",
            )
        axis.set_title(label)
        axis.set_xlabel("phi")
        axis.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("fraction")
    axes[1, 0].set_ylabel("rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle("Mechanism diagnostics: directional and quote participation")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_dir / "mechanism_action_quote_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_shutdown_panel(aggregate_frame: pd.DataFrame, output_dir: Path) -> Path | None:
    metrics = [
        ("quote_order_submission_rate", "Quote submission rate"),
        ("zero_return_fraction", "Zero-return fraction"),
        ("trade_timestep_count", "Trade timesteps"),
        ("traded_volume_total", "Total traded volume"),
    ]
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        mean_column = f"{metric}_mean"
        if mean_column not in aggregate_frame.columns:
            axis.set_visible(False)
            continue
        for (condition, split), group in aggregate_frame.groupby(["condition", "rl_quoter_split"]):
            ordered = group.sort_values("phi")
            axis.plot(
                ordered["phi"],
                ordered[mean_column],
                marker="o",
                linewidth=1.4,
                label=f"{condition}, q={float(split):.2f}",
            )
        axis.axvline(0.50, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
        axis.set_title(label)
        axis.set_xlabel("phi")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle("Phi = 0.5 shutdown diagnostics")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_dir / "phi_0_5_shutdown_diagnostics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_market_quality_panel(aggregate_frame: pd.DataFrame, output_dir: Path) -> Path | None:
    metrics = [
        ("average_spread", "Average spread"),
        ("average_depth", "Average depth"),
        ("volatility", "Return volatility"),
        ("zero_return_fraction", "Zero-return fraction"),
        ("trade_timestep_count", "Trade timesteps"),
        ("traded_volume_total", "Total traded volume"),
    ]
    plt = _load_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        mean_column = f"{metric}_mean"
        if mean_column not in aggregate_frame.columns:
            axis.set_visible(False)
            continue
        for (condition, split), group in aggregate_frame.groupby(["condition", "rl_quoter_split"]):
            ordered = group.sort_values("phi")
            axis.plot(
                ordered["phi"],
                ordered[mean_column],
                marker="o",
                linewidth=1.4,
                label=f"{condition}, q={float(split):.2f}",
            )
        axis.set_title(label)
        axis.set_xlabel("phi")
        axis.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle("Market-quality diagnostics")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = output_dir / "market_quality_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_event_study(event_study_frame: pd.DataFrame, output_dir: Path) -> Path | None:
    if event_study_frame.empty:
        return None
    plt = _load_matplotlib()
    grouped = (
        event_study_frame.groupby(["condition", "rl_quoter_split", "relative_step"], as_index=False)
        .agg(
            bid_depth=("bid_depth", "mean"),
            ask_depth=("ask_depth", "mean"),
            depth_imbalance=("depth_imbalance", "mean"),
            traded_volume=("traded_volume", "mean"),
        )
        .sort_values("relative_step")
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for (condition, split), group in grouped.groupby(["condition", "rl_quoter_split"]):
        label = f"{condition}, q={float(split):.2f}"
        axes[0].plot(group["relative_step"], group["bid_depth"], marker="o", label=f"bid {label}")
        axes[0].plot(group["relative_step"], group["ask_depth"], marker="x", label=f"ask {label}")
        axes[1].plot(group["relative_step"], group["depth_imbalance"], marker="o", label=label)
        axes[2].plot(group["relative_step"], group["traded_volume"], marker="o", label=label)
    axes[0].set_title("Depth around onset")
    axes[1].set_title("Depth imbalance around onset")
    axes[2].set_title("Trading around onset")
    for axis in axes:
        axis.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        axis.set_xlabel("steps from one-sided onset")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("depth")
    axes[1].set_ylabel("(bid - ask) / total")
    axes[2].set_ylabel("traded volume")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)))
    fig.suptitle("Event study around one-sided-book onset")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    path = output_dir / "one_sided_onset_event_study.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_reviewer_plots(
    aggregate_frame: pd.DataFrame,
    event_study_frame: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    panel_paths = [
        _plot_mechanism_panel(aggregate_frame, output_root),
        _plot_shutdown_panel(aggregate_frame, output_root),
        _plot_market_quality_panel(aggregate_frame, output_root),
        _plot_event_study(event_study_frame, output_root),
    ]
    for path in panel_paths:
        if path is not None:
            saved.append(path)

    metric_specs = [
        ("one_sided_book_fraction", "One-sided book fraction", "one_sided_book_fraction_with_ci.png", "One-sided book fraction with 95% CI"),
        ("average_spread", "Average spread", "average_spread_with_ci.png", "Average spread with 95% CI"),
        ("average_depth", "Average depth", "average_depth_with_ci.png", "Average depth with 95% CI"),
        ("volatility", "Volatility", "volatility_with_ci.png", "Return volatility with 95% CI"),
        ("zero_return_fraction", "Zero-return fraction", "zero_return_fraction_with_ci.png", "Zero-return fraction with 95% CI"),
        ("trade_timestep_count", "Trade timesteps", "trade_timestep_count_with_ci.png", "Trade timesteps with 95% CI"),
    ]
    for metric, ylabel, filename, title in metric_specs:
        path = _plot_metric_grid(
            aggregate_frame,
            output_root,
            metric=metric,
            ylabel=ylabel,
            filename=filename,
            title=title,
        )
        if path is not None:
            saved.append(path)
    return saved


def build_reviewer_report(
    aggregate_frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    saved_plots: Sequence[Path],
) -> str:
    lines = [
        "# Reviewer Robustness Diagnostics",
        "",
        "## What this addresses",
        "- Composition-control rows compare learned policies against nonlearned controls under the same `phi`, `rl_liquidity_mode`, and taker/quoter split.",
        "- This is a control for the passive/aggressive order-flow composition confound; it does not claim aggregate passive quoting is perfectly held constant.",
        "- All summary statistics are aggregated across evaluation seeds only. Training episodes are not mixed into the seed-level confidence intervals.",
        "",
        "## Configuration",
        f"- Phi grid: {', '.join(f'{float(phi):.2f}' for phi in config['phi_grid'])}",
        f"- Quoter splits: {', '.join(f'{float(split):.2f}' for split in config['quoter_splits'])}",
        f"- Evaluation seeds: {', '.join(str(seed) for seed in config['evaluation_seeds'])}",
        f"- Evaluation mode: {config['evaluation_mode']}",
        f"- Control policy: {config['control_policy']}",
        f"- Trained experiment dir: {config.get('trained_experiment_dir') or 'not used'}",
        "",
        "## Plot Files",
    ]
    lines.extend(f"- `{path.name}`" for path in saved_plots)
    if aggregate_frame.empty:
        lines.append("")
        lines.append("No aggregate rows were produced.")
        return "\n".join(lines) + "\n"

    lines.extend(["", "## Quick Read"])
    for (condition, split), group in aggregate_frame.groupby(["condition", "rl_quoter_split"]):
        ordered = group.sort_values("phi")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        lines.append(
            "- "
            f"{condition}, quoter split {float(split):.2f}: "
            f"one-sided {float(first.get('one_sided_book_fraction_mean', np.nan)):.4f} -> {float(last.get('one_sided_book_fraction_mean', np.nan)):.4f}; "
            f"zero-return {float(first.get('zero_return_fraction_mean', np.nan)):.4f} -> {float(last.get('zero_return_fraction_mean', np.nan)):.4f}; "
            f"quote-order rate {float(first.get('quote_order_submission_rate_mean', np.nan)):.4f} -> {float(last.get('quote_order_submission_rate_mean', np.nan)):.4f}."
        )
    return "\n".join(lines) + "\n"
