"""Narrow price-discovery ablation helpers for the phi=0 baseline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from calibration import build_parameter_grid, parse_sweep_spec
from config import SimulationConfig
from logging_utils import (
    extract_limit_order_lifecycle_dataframe,
    extract_trade_history_dataframe,
)
from market import MarketSimulator
from realism_diagnostics import compute_realism_diagnostics
from abides_support import bootstrap_abides_paths

bootstrap_abides_paths()

from abides_markets.agents import ExchangeAgent

AGENT_TYPE_SLUGS = {
    "AdaptiveMarketMaker": "adaptivemarketmaker",
    "NoiseTrader": "noisetrader",
    "TrendFollowerTrader": "trendfollowertrader",
    "ValueTrader": "valuetrader",
    "ZICTrader": "zictrader",
}

CORE_ABLATION_COLUMNS = [
    "midprice_range",
    "fraction_zero_midprice_change",
    "average_nonzero_midprice_change",
    "midprice_fundamental_change_correlation",
    "average_spread",
    "average_depth",
    "trade_count",
    "best_bid_mm_share",
    "best_ask_mm_share",
    "fraction_trade_count_against_mm_quotes",
    "fraction_traded_volume_against_mm_quotes",
    "fraction_non_mm_passive_join_inside",
    "fraction_non_mm_passive_improve_inside",
    "fraction_spread_one_tick",
    "fraction_spread_two_ticks",
    "fraction_spread_three_plus_ticks",
]


def _slugify(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


def compute_passive_accumulation_metrics(
    frame: pd.DataFrame,
    *,
    passive_orders: pd.DataFrame,
    trade_history: pd.DataFrame,
    market_open_ns: int,
    market_close_ns: int,
) -> dict[str, float]:
    """Measure passive-book accumulation and depth contribution by agent type."""

    def _is_present(value: Any) -> bool:
        return value is not None and not (isinstance(value, float) and np.isnan(value))

    passive_only = passive_orders[passive_orders["is_passive"] == True].copy()
    if passive_only.empty:
        return {}

    passive_only["accepted_or_submitted_ns"] = pd.to_numeric(
        passive_only["accepted_time_ns"].fillna(passive_only["time_submitted_ns"]),
        errors="coerce",
    )
    passive_only["quantity"] = pd.to_numeric(passive_only["quantity"], errors="coerce").fillna(0.0)
    passive_only["executed_quantity"] = pd.to_numeric(
        passive_only.get("executed_quantity"),
        errors="coerce",
    ).fillna(0.0)

    trade_execs = trade_history.groupby(["passive_order_id", "time_ns"], as_index=False)["quantity"].sum()
    trade_execs["passive_order_id"] = pd.to_numeric(trade_execs["passive_order_id"], errors="coerce")
    trade_execs["time_ns"] = pd.to_numeric(trade_execs["time_ns"], errors="coerce")
    trade_execs["quantity"] = pd.to_numeric(trade_execs["quantity"], errors="coerce").fillna(0.0)

    event_rows: list[dict[str, object]] = []
    type_rows: list[dict[str, float]] = []

    for record in passive_only.itertuples(index=False):
        accepted_time_ns = getattr(record, "accepted_or_submitted_ns", float("nan"))
        if pd.isna(accepted_time_ns):
            continue

        order_id = int(record.order_id)
        agent_type = str(record.agent_type)
        side = str(record.side)
        quantity = float(record.quantity)
        executed_quantity = float(record.executed_quantity)
        cancelled_time_ns = getattr(record, "cancelled_time_ns", None)
        cancelled_time_ns = (
            None
            if not _is_present(cancelled_time_ns)
            else int(cancelled_time_ns)
        )

        type_rows.append(
            {
                "agent_type": agent_type,
                "arrival_count": 1.0,
                "cancel_count": 1.0 if cancelled_time_ns is not None else 0.0,
                "execution_count": 1.0 if bool(getattr(record, "was_executed", False)) else 0.0,
                "never_materially_affects_price": 1.0
                if (str(record.placement_bucket) == "behind" and not bool(getattr(record, "was_executed", False)))
                else 0.0,
            }
        )

        event_rows.append(
            {
                "time_ns": int(accepted_time_ns),
                "agent_type": agent_type,
                "side": side,
                "depth_delta": quantity,
                "order_delta": 1.0,
            }
        )

        order_execs = trade_execs[trade_execs["passive_order_id"] == order_id]
        last_exec_ns = None
        cumulative_exec = 0.0
        for exec_record in order_execs.itertuples(index=False):
            exec_quantity = float(exec_record.quantity)
            cumulative_exec += exec_quantity
            last_exec_ns = int(exec_record.time_ns)
            event_rows.append(
                {
                    "time_ns": last_exec_ns,
                    "agent_type": agent_type,
                    "side": side,
                    "depth_delta": -exec_quantity,
                    "order_delta": 0.0,
                }
            )

        remaining_quantity = max(quantity - cumulative_exec, 0.0)
        if cancelled_time_ns is not None:
            event_rows.append(
                {
                    "time_ns": cancelled_time_ns,
                    "agent_type": agent_type,
                    "side": side,
                    "depth_delta": -remaining_quantity,
                    "order_delta": -1.0,
                }
            )
        elif cumulative_exec >= quantity and last_exec_ns is not None:
            event_rows.append(
                {
                    "time_ns": int(last_exec_ns),
                    "agent_type": agent_type,
                    "side": side,
                    "depth_delta": 0.0,
                    "order_delta": -1.0,
                }
            )

    metrics: dict[str, float] = {}
    type_frame = pd.DataFrame(type_rows)
    if not type_frame.empty:
        grouped_counts = type_frame.groupby("agent_type", as_index=False).sum(numeric_only=True)
        for row in grouped_counts.itertuples(index=False):
            slug = _slugify(str(row.agent_type))
            metrics[f"passive_arrival_count_{slug}"] = float(row.arrival_count)
            metrics[f"passive_cancel_count_{slug}"] = float(row.cancel_count)
            metrics[f"passive_execution_count_{slug}"] = float(row.execution_count)
            metrics[f"fraction_passive_never_material_{slug}"] = (
                float(row.never_materially_affects_price) / float(row.arrival_count)
                if row.arrival_count > 0
                else float("nan")
            )

    event_frame = pd.DataFrame(event_rows)
    if event_frame.empty:
        return metrics

    event_frame = event_frame.sort_values("time_ns")
    grid = pd.DataFrame(
        {
            "time_ns": market_open_ns + pd.to_numeric(frame["time"], errors="coerce").fillna(0).astype("int64")
        }
    )

    def cumulative_state(filter_mask: pd.Series) -> pd.DataFrame:
        filtered = event_frame.loc[filter_mask, ["time_ns", "depth_delta", "order_delta"]]
        if filtered.empty:
            return pd.DataFrame({"time_ns": grid["time_ns"], "depth": 0.0, "orders": 0.0})
        aggregated = filtered.groupby("time_ns", as_index=False).sum(numeric_only=True).sort_values("time_ns")
        aggregated["depth"] = aggregated["depth_delta"].cumsum()
        aggregated["orders"] = aggregated["order_delta"].cumsum()
        merged = pd.merge_asof(
            grid.sort_values("time_ns"),
            aggregated[["time_ns", "depth", "orders"]].sort_values("time_ns"),
            on="time_ns",
            direction="backward",
        )
        merged["depth"] = merged["depth"].fillna(0.0)
        merged["orders"] = merged["orders"].fillna(0.0)
        return merged

    bid_state = cumulative_state(event_frame["side"] == "bid")
    ask_state = cumulative_state(event_frame["side"] == "ask")
    metrics["mean_total_resting_bid_depth"] = float(bid_state["depth"].mean())
    metrics["mean_total_resting_ask_depth"] = float(ask_state["depth"].mean())
    metrics["max_total_resting_bid_depth"] = float(bid_state["depth"].max())
    metrics["max_total_resting_ask_depth"] = float(ask_state["depth"].max())
    metrics["mean_inside_queue_size"] = float(
        (pd.to_numeric(frame["bid_depth"], errors="coerce") + pd.to_numeric(frame["ask_depth"], errors="coerce")).mean()
    )

    for agent_type in sorted(passive_only["agent_type"].dropna().astype(str).unique()):
        slug = _slugify(agent_type)
        type_state = cumulative_state(event_frame["agent_type"] == agent_type)
        metrics[f"mean_resting_orders_{slug}"] = float(type_state["orders"].mean())
        metrics[f"mean_resting_depth_{slug}"] = float(type_state["depth"].mean())
        metrics[f"max_resting_depth_{slug}"] = float(type_state["depth"].max())

    return metrics


def _safe_change_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    left_values = aligned.iloc[:, 0].to_numpy(dtype=float)
    right_values = aligned.iloc[:, 1].to_numpy(dtype=float)
    if np.std(left_values) == 0.0 or np.std(right_values) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def compute_ablation_metrics(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    passive_orders: pd.DataFrame,
    trade_history: pd.DataFrame,
    market_open_ns: int,
    market_close_ns: int,
) -> dict[str, float]:
    """Compute the narrow price-discovery metrics for one completed run."""

    diagnostics = compute_realism_diagnostics(
        frame,
        tick_size=tick_size,
        passive_orders=passive_orders,
        trade_history=trade_history,
    )

    midprice = pd.to_numeric(frame["midprice"], errors="coerce")
    fundamental = pd.to_numeric(frame["fundamental_value"], errors="coerce")
    midprice_changes = midprice.diff()
    fundamental_changes = fundamental.diff()

    metrics: dict[str, float] = {
        "midprice_range": float(midprice.max() - midprice.min()),
        "fraction_zero_midprice_change": float(diagnostics["fraction_zero_midprice_change"]),
        "average_nonzero_midprice_change": float(diagnostics["average_nonzero_midprice_change"]),
        "midprice_fundamental_change_correlation": _safe_change_correlation(
            midprice_changes,
            fundamental_changes,
        ),
        "average_spread": float(diagnostics["average_spread"]),
        "average_depth": float(diagnostics["average_depth"]),
        "trade_count": float(diagnostics.get("trade_count", 0.0)),
        "best_bid_mm_share": float(diagnostics["best_bid_mm_share"]),
        "best_ask_mm_share": float(diagnostics["best_ask_mm_share"]),
        "fraction_trade_count_against_mm_quotes": float(
            diagnostics.get("fraction_trade_count_against_mm_quotes", float("nan"))
        ),
        "fraction_traded_volume_against_mm_quotes": float(
            diagnostics.get("fraction_traded_volume_against_mm_quotes", float("nan"))
        ),
        "fraction_non_mm_passive_join_inside": float(
            diagnostics.get("fraction_non_mm_passive_join_inside", float("nan"))
        ),
        "fraction_non_mm_passive_improve_inside": float(
            diagnostics.get("fraction_non_mm_passive_improve_inside", float("nan"))
        ),
        "fraction_spread_one_tick": float(diagnostics["fraction_spread_one_tick"]),
        "fraction_spread_two_ticks": float(diagnostics["fraction_spread_two_ticks"]),
        "fraction_spread_three_plus_ticks": float(diagnostics["fraction_spread_three_plus_ticks"]),
    }
    metrics.update(
        compute_passive_accumulation_metrics(
            frame,
            passive_orders=passive_orders,
            trade_history=trade_history,
            market_open_ns=market_open_ns,
            market_close_ns=market_close_ns,
        )
    )

    for agent_type, slug in AGENT_TYPE_SLUGS.items():
        metrics[f"passive_fill_rate_{slug}"] = float(
            diagnostics.get(f"passive_fill_rate_{slug}", float("nan"))
        )
        metrics[f"passive_mean_lifetime_seconds_{slug}"] = float(
            diagnostics.get(f"passive_mean_lifetime_seconds_{slug}", float("nan"))
        )
        metrics[f"passive_join_share_{slug}"] = float(
            diagnostics.get(f"passive_join_share_{slug}", float("nan"))
        )
        metrics[f"passive_improve_share_{slug}"] = float(
            diagnostics.get(f"passive_improve_share_{slug}", float("nan"))
        )

    for column_name, source_column in [
        ("passive_execution_share", "passive_agent_type"),
        ("aggressive_execution_share", "aggressor_agent_type"),
    ]:
        if trade_history.empty or source_column not in trade_history.columns:
            continue
        shares = (
            trade_history[source_column]
            .fillna("UNKNOWN")
            .value_counts(normalize=True)
            .to_dict()
        )
        for agent_type, slug in AGENT_TYPE_SLUGS.items():
            metrics[f"{column_name}_{slug}"] = float(shares.get(agent_type, 0.0))

    return metrics


def run_price_discovery_ablation(
    base_config: SimulationConfig,
    *,
    seeds: Sequence[int],
    parameter_grid: Sequence[Mapping[str, Any]] | None = None,
    parameter_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run a small phi=0 ablation grid and return run-level diagnostics."""

    parameter_grid = list(parameter_grid or [{}])
    parameter_columns = list(parameter_columns or [])
    rows: list[dict[str, Any]] = []

    for parameter_setting in parameter_grid:
        for seed in seeds:
            config = replace(
                base_config,
                phi=0.0,
                seed=seed,
                **parameter_setting,
            )
            simulator = MarketSimulator(config)
            frame = simulator.run()
            passive_orders = extract_limit_order_lifecycle_dataframe(simulator.end_state)
            trade_history = extract_trade_history_dataframe(simulator.end_state, config.ticker)
            exchange = next(
                agent for agent in simulator.end_state["agents"] if isinstance(agent, ExchangeAgent)
            )
            metrics = compute_ablation_metrics(
                frame,
                tick_size=config.tick_size,
                passive_orders=passive_orders,
                trade_history=trade_history,
                market_open_ns=int(exchange.mkt_open),
                market_close_ns=int(exchange.mkt_close),
            )

            row = {
                "seed": seed,
                "phi": config.phi,
                **{name: getattr(config, name) for name in parameter_columns},
                **metrics,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_ablation_runs(
    runs: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate ablation runs by parameter setting."""

    if runs.empty:
        return pd.DataFrame()

    group_columns = list(parameter_columns)
    if not group_columns:
        working = runs.copy()
        working["scenario"] = "baseline"
        group_columns = ["scenario"]
    else:
        working = runs

    summary = (
        working.groupby(group_columns, dropna=False)[CORE_ABLATION_COLUMNS]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    return summary


def parse_ablation_specs(
    raw_specs: Sequence[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse CLI ablation specs using the shared sweep parser."""

    parsed = [parse_sweep_spec(spec) for spec in raw_specs]
    return build_parameter_grid(parsed)
