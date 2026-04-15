"""Simulation logging helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from abides_support import bootstrap_abides_paths
from agents.base import MarketObservation

bootstrap_abides_paths()

from abides_markets.agents import ExchangeAgent
from abides_core.utils import str_to_ns

SNAPSHOT_COLUMNS = [
    "time",
    "best_bid",
    "best_ask",
    "bid_depth",
    "ask_depth",
    "best_bid_agent_id",
    "best_ask_agent_id",
    "best_bid_agent_type",
    "best_ask_agent_type",
]


def _is_rl_agent(agent: object) -> bool:
    return getattr(agent, "type", "") in {"RLTrader", "RLQuotingTrader"}


def _cents_to_dollars(series: pd.Series) -> pd.Series:
    return series.astype(float) / 100.0


def _snapshot_best_level(levels) -> tuple[Optional[int], int]:
    if levels is None or len(levels) == 0:
        return None, 0
    best_price = int(levels[0][0])
    best_depth = int(levels[0][1])
    return best_price, best_depth


def _safe_midprice(best_bid: Optional[int], best_ask: Optional[int], last_trade: int) -> float:
    del last_trade
    if best_bid is not None and best_ask is not None:
        return float(best_bid + best_ask) / 2.0
    return float("nan")


def _safe_spread(best_bid: Optional[int], best_ask: Optional[int]) -> float:
    if best_bid is None or best_ask is None:
        return float("nan")
    return float(best_ask - best_bid)


def _safe_imbalance(bid_depth: int, ask_depth: int) -> float:
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return 0.0
    return (bid_depth - ask_depth) / total_depth


def _aggregate_transactions(
    times: pd.Series,
    transactions: Iterable[tuple[int, int]],
    column_name: str,
) -> pd.Series:
    frame = pd.DataFrame(list(transactions), columns=["time", column_name])
    if frame.empty:
        return pd.Series(0.0, index=times.index)

    frame = frame.groupby("time", as_index=False)[column_name].sum()
    frame[f"cum_{column_name}"] = frame[column_name].cumsum()
    merged = pd.merge_asof(
        pd.DataFrame({"time": times}).sort_values("time"),
        frame[["time", f"cum_{column_name}"]].sort_values("time"),
        on="time",
        direction="backward",
    )
    cumulative = merged[f"cum_{column_name}"].fillna(0.0)
    return cumulative.diff().fillna(cumulative)


def _build_fixed_time_grid(
    market_open: int,
    market_close: int,
    frequency: str,
) -> pd.DataFrame:
    frequency_ns = int(str_to_ns(frequency))
    if frequency_ns <= 0:
        raise ValueError("log_frequency must convert to a positive nanosecond interval.")

    times = list(range(market_open, market_close + frequency_ns, frequency_ns))
    if times[-1] > market_close:
        times[-1] = market_close
    return pd.DataFrame({"time_ns": pd.Series(times, dtype="int64")})


def _resample_to_fixed_grid(
    grid: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge_asof(
        grid.sort_values("time_ns"),
        observations.sort_values("time_ns"),
        on="time_ns",
        direction="backward",
    )

    first_snapshot_index = merged["time"].first_valid_index() if "time" in merged.columns else None
    if first_snapshot_index is not None and first_snapshot_index > 0:
        for column in SNAPSHOT_COLUMNS:
            if column in merged.columns:
                merged.loc[: first_snapshot_index - 1, column] = merged.loc[first_snapshot_index, column]

    for column in ("best_bid", "best_ask", "bid_depth", "ask_depth"):
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    for column in ("best_bid_agent_id", "best_ask_agent_id"):
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("Int64")
    for column in ("best_bid_agent_type", "best_ask_agent_type"):
        if column in merged.columns:
            merged[column] = pd.Series(merged[column], dtype="string")

    best_bid = pd.to_numeric(merged.get("best_bid"), errors="coerce")
    best_ask = pd.to_numeric(merged.get("best_ask"), errors="coerce")
    bid_depth = pd.to_numeric(merged.get("bid_depth"), errors="coerce").fillna(0.0)
    ask_depth = pd.to_numeric(merged.get("ask_depth"), errors="coerce").fillna(0.0)

    both_sides = best_bid.notna() & best_ask.notna()
    merged["midprice"] = pd.Series(np.nan, index=merged.index, dtype=float)
    merged.loc[both_sides, "midprice"] = (best_bid[both_sides] + best_ask[both_sides]) / 2.0

    merged["spread"] = pd.Series(np.nan, index=merged.index, dtype=float)
    merged.loc[both_sides, "spread"] = best_ask[both_sides] - best_bid[both_sides]

    imbalance = pd.Series(0.0, index=merged.index, dtype=float)
    total_depth = bid_depth + ask_depth
    valid_depth = total_depth > 0
    imbalance.loc[valid_depth] = (bid_depth[valid_depth] - ask_depth[valid_depth]) / total_depth[valid_depth]
    merged["imbalance"] = imbalance
    return merged


def extract_abides_dataframe(
    end_state: dict,
    ticker: str,
    log_frequency: str = "1s",
) -> pd.DataFrame:
    """Convert ABIDES end-state data into the experiment log DataFrame."""

    agents = end_state["agents"]
    exchange = next(agent for agent in agents if isinstance(agent, ExchangeAgent))
    order_book = exchange.order_books[ticker]
    agent_type_by_id = {int(agent.id): str(agent.type) for agent in agents}
    snapshots = order_book.book_log2
    if not snapshots:
        return pd.DataFrame()

    market_open = int(exchange.mkt_open)
    rows: List[Dict[str, float]] = []
    for snapshot in snapshots:
        best_bid, bid_depth = _snapshot_best_level(snapshot["bids"])
        best_ask, ask_depth = _snapshot_best_level(snapshot["asks"])
        midprice = _safe_midprice(best_bid=best_bid, best_ask=best_ask, last_trade=order_book.last_trade)
        spread = _safe_spread(best_bid=best_bid, best_ask=best_ask)
        rows.append(
            {
                "time": int(snapshot["QuoteTime"]) - market_open,
                "time_ns": int(snapshot["QuoteTime"]),
                "best_bid": float(best_bid) if best_bid is not None else float("nan"),
                "best_ask": float(best_ask) if best_ask is not None else float("nan"),
                "best_bid_agent_id": (
                    int(snapshot["best_bid_agent_id"])
                    if snapshot.get("best_bid_agent_id") is not None
                    else pd.NA
                ),
                "best_ask_agent_id": (
                    int(snapshot["best_ask_agent_id"])
                    if snapshot.get("best_ask_agent_id") is not None
                    else pd.NA
                ),
                "best_bid_agent_type": (
                    agent_type_by_id.get(int(snapshot["best_bid_agent_id"]), "UNKNOWN")
                    if snapshot.get("best_bid_agent_id") is not None
                    else pd.NA
                ),
                "best_ask_agent_type": (
                    agent_type_by_id.get(int(snapshot["best_ask_agent_id"]), "UNKNOWN")
                    if snapshot.get("best_ask_agent_id") is not None
                    else pd.NA
                ),
                "midprice": midprice,
                "spread": spread,
                "bid_depth": float(bid_depth),
                "ask_depth": float(ask_depth),
                "imbalance": _safe_imbalance(bid_depth=bid_depth, ask_depth=ask_depth),
            }
        )

    observations = pd.DataFrame(rows).sort_values("time_ns").reset_index(drop=True)
    grid = _build_fixed_time_grid(
        market_open=market_open,
        market_close=int(exchange.mkt_close),
        frequency=log_frequency,
    )
    frame = _resample_to_fixed_grid(grid=grid, observations=observations)
    frame["time"] = frame["time_ns"] - market_open

    buy_volume = _aggregate_transactions(frame["time_ns"], order_book.buy_transactions, "buy_volume")
    sell_volume = _aggregate_transactions(frame["time_ns"], order_book.sell_transactions, "sell_volume")
    frame["traded_volume"] = buy_volume + sell_volume
    frame["signed_order_flow"] = buy_volume - sell_volume

    oracle = exchange.kernel.oracle
    fundamental = pd.DataFrame(oracle.f_log[ticker]).rename(
        columns={"FundamentalTime": "time_ns", "FundamentalValue": "fundamental_value"}
    )
    fundamental["time_ns"] = fundamental["time_ns"].astype("int64")
    frame = pd.merge_asof(
        frame.sort_values("time_ns"),
        fundamental.sort_values("time_ns"),
        on="time_ns",
        direction="backward",
    )

    rl_agents = [agent for agent in agents if _is_rl_agent(agent)]
    for agent in rl_agents:
        metrics = pd.DataFrame(agent.metrics_log)
        prefix = agent.name.lower().replace(" ", "_")
        if metrics.empty:
            continue
        renamed = {}
        for column in metrics.columns:
            if column == "time":
                renamed[column] = "time_ns"
            elif column == "agent_id":
                renamed[column] = f"{prefix}_agent_id"
            elif column == "agent_name":
                renamed[column] = f"{prefix}_agent_name"
            else:
                renamed[column] = f"{prefix}_{column}"
        metrics = metrics.rename(columns=renamed)
        metrics["time_ns"] = metrics["time_ns"].astype("int64")
        frame = pd.merge_asof(
            frame.sort_values("time_ns"),
            metrics.sort_values("time_ns"),
            on="time_ns",
            direction="backward",
        )

    dollar_columns = [
        "best_bid",
        "best_ask",
        "midprice",
        "spread",
        "fundamental_value",
    ]
    dollar_columns.extend(
        column
        for column in frame.columns
        if column.endswith("_cash")
        or column.endswith("_wealth")
        or column.endswith("_reward")
        or column.endswith("_spread")
        or column.endswith("_midprice")
        or column.endswith("_best_bid")
        or column.endswith("_best_ask")
    )
    for column in dollar_columns:
        if column in frame.columns:
            frame[column] = _cents_to_dollars(frame[column])

    inventory_columns = [column for column in frame.columns if column.endswith("_inventory")]
    for column in inventory_columns:
        frame[column] = frame[column].fillna(0.0)

    metric_columns = [
        column
        for column in frame.columns
        if column.endswith("_cash") or column.endswith("_wealth") or column.endswith("_reward")
    ]
    for column in metric_columns:
        frame[column] = frame[column].ffill().fillna(0.0)

    frame = frame.drop(columns=["time_ns"])
    return frame


def extract_rl_decision_dataframe(end_state: dict) -> pd.DataFrame:
    """Return one long-form row per RL-agent decision."""

    agents = end_state["agents"]
    rl_agents = [agent for agent in agents if _is_rl_agent(agent)]
    if not rl_agents:
        return pd.DataFrame()

    exchange = next(agent for agent in agents if isinstance(agent, ExchangeAgent))
    market_open = int(exchange.mkt_open)
    frames: List[pd.DataFrame] = []
    for agent in rl_agents:
        metrics = pd.DataFrame(agent.metrics_log)
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["time_ns"] = pd.to_numeric(metrics["time"], errors="coerce").astype("int64")
        metrics["time"] = metrics["time_ns"] - market_open
        frames.append(metrics)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True).sort_values(["time_ns", "agent_id"]).reset_index(drop=True)

    dollar_columns = [
        column
        for column in (
            "cash",
            "wealth",
            "reward",
            "spread",
            "midprice",
            "best_bid",
            "best_ask",
            "cash_delta_since_last_decision",
            "wealth_delta_since_last_decision",
            "inventory_penalty_since_last_decision",
            "flat_hold_penalty_since_last_decision",
            "passive_fill_bonus_since_last_decision",
            "two_sided_quote_reward_since_last_decision",
            "missing_quote_penalty_since_last_decision",
        )
        if column in frame.columns
    ]
    for column in dollar_columns:
        frame[column] = _cents_to_dollars(frame[column])

    return frame


def extract_rl_transition_dataframe(end_state: dict) -> pd.DataFrame:
    """Return one long-form row per realized RL transition."""

    agents = end_state["agents"]
    rl_agents = [agent for agent in agents if _is_rl_agent(agent)]
    if not rl_agents:
        return pd.DataFrame()

    exchange = next(agent for agent in agents if isinstance(agent, ExchangeAgent))
    market_open = int(exchange.mkt_open)
    frames: List[pd.DataFrame] = []
    for agent in rl_agents:
        transitions = pd.DataFrame(getattr(agent, "transition_log", []))
        if transitions.empty:
            continue
        transitions = transitions.copy()
        transitions["time_ns"] = pd.to_numeric(transitions["time"], errors="coerce").astype("int64")
        transitions["time_next_ns"] = pd.to_numeric(transitions["time_next"], errors="coerce").astype("int64")
        transitions["time"] = transitions["time_ns"] - market_open
        transitions["time_next"] = transitions["time_next_ns"] - market_open

        state_length = int(max(len(state) for state in transitions["state"]))
        for index in range(state_length):
            transitions[f"state_{index:02d}"] = transitions["state"].apply(
                lambda state: float(state[index]) if index < len(state) else float("nan")
            )
            transitions[f"next_state_{index:02d}"] = transitions["next_state"].apply(
                lambda state: float(state[index]) if index < len(state) else float("nan")
            )
        transitions = transitions.drop(columns=["state", "next_state"])
        frames.append(transitions)

    if not frames:
        return pd.DataFrame()

    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["time_ns", "agent_id", "decision_index"])
        .reset_index(drop=True)
    )

    dollar_columns = [
        column
        for column in (
            "reward",
            "wealth_delta",
            "inventory_penalty",
            "flat_hold_penalty",
            "passive_fill_bonus",
            "two_sided_quote_reward",
            "missing_quote_penalty",
            "cash_before",
            "cash_after",
            "midprice_before",
            "midprice_after",
        )
        if column in frame.columns
    ]
    for column in dollar_columns:
        frame[column] = _cents_to_dollars(frame[column])

    return frame


def extract_trade_history_dataframe(
    end_state: dict,
    ticker: str,
) -> pd.DataFrame:
    """Return trade history with passive/aggressor agent types for one symbol."""

    agents = end_state["agents"]
    exchange = next(agent for agent in agents if isinstance(agent, ExchangeAgent))
    order_book = exchange.order_books[ticker]
    market_open = int(exchange.mkt_open)
    agent_type_by_id = {int(agent.id): str(agent.type) for agent in agents}

    rows: List[Dict[str, object]] = []
    for event in order_book.history:
        if event.get("type") != "EXEC":
            continue
        passive_agent_id = int(event["agent_id"])
        aggressor_agent_id = int(event["oppos_agent_id"])
        rows.append(
            {
                "time": int(event["time"]) - market_open,
                "time_ns": int(event["time"]),
                "quantity": int(event["quantity"]),
                "passive_order_id": int(event["order_id"]),
                "passive_agent_id": passive_agent_id,
                "passive_agent_type": agent_type_by_id.get(passive_agent_id, "UNKNOWN"),
                "aggressor_order_id": int(event["oppos_order_id"]),
                "aggressor_agent_id": aggressor_agent_id,
                "aggressor_agent_type": agent_type_by_id.get(
                    aggressor_agent_id,
                    "UNKNOWN",
                ),
                "passive_side": str(event["side"]).lower(),
            }
        )

    return pd.DataFrame(rows)


def extract_limit_order_lifecycle_dataframe(end_state: dict) -> pd.DataFrame:
    """Return per-limit-order lifecycle data recorded by the custom agents."""

    quick_execution_ns = int(str_to_ns("1s"))
    rows: List[Dict[str, object]] = []
    for agent in end_state["agents"]:
        lifecycle = getattr(agent, "limit_order_lifecycle", None)
        if not lifecycle:
            continue
        for record in lifecycle.values():
            row = dict(record)
            accepted_time = row.get("accepted_time_ns")
            first_execution_time = row.get("first_execution_time_ns")
            submitted_time = row.get("time_submitted_ns")
            row["was_accepted"] = accepted_time is not None
            row["was_executed"] = first_execution_time is not None
            row["rested_before_execution"] = (
                accepted_time is not None and first_execution_time is not None
            )
            row["time_to_accept_seconds"] = (
                (int(accepted_time) - int(submitted_time)) / 1_000_000_000.0
                if accepted_time is not None and submitted_time is not None
                else float("nan")
            )
            row["time_to_first_execution_seconds"] = (
                (int(first_execution_time) - int(submitted_time)) / 1_000_000_000.0
                if first_execution_time is not None and submitted_time is not None
                else float("nan")
            )
            row["time_to_cancel_seconds"] = (
                (int(row["cancelled_time_ns"]) - int(submitted_time)) / 1_000_000_000.0
                if row.get("cancelled_time_ns") is not None and submitted_time is not None
                else float("nan")
            )
            terminal_time = first_execution_time
            cancelled_time = row.get("cancelled_time_ns")
            if terminal_time is None:
                terminal_time = cancelled_time
            elif cancelled_time is not None:
                terminal_time = min(int(terminal_time), int(cancelled_time))
            row["time_to_terminal_event_seconds"] = (
                (int(terminal_time) - int(submitted_time)) / 1_000_000_000.0
                if terminal_time is not None and submitted_time is not None
                else float("nan")
            )
            row["rested_before_execution"] = bool(
                first_execution_time is None
                or (
                    submitted_time is not None
                    and (int(first_execution_time) - int(submitted_time)) > quick_execution_ns
                )
            )
            row["executed_quickly"] = bool(
                first_execution_time is not None
                and submitted_time is not None
                and (int(first_execution_time) - int(submitted_time)) <= quick_execution_ns
            )
            row["executed_quickly_after_rest"] = row["executed_quickly"]
            rows.append(row)

    return pd.DataFrame(rows)


@dataclass
class SimulationLogger:
    """Collect simulation records and export them as a DataFrame or CSV."""

    records: List[Dict[str, float]] = field(default_factory=list)

    def log_step(
        self,
        time: int,
        observation: MarketObservation,
        traded_volume: int,
        signed_order_flow: int,
        fundamental_value: float,
        rl_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """Append a single step to the in-memory log."""

        record: Dict[str, float] = {
            "time": float(time),
            "best_bid": observation.best_bid,
            "best_ask": observation.best_ask,
            "midprice": observation.midprice,
            "spread": observation.spread,
            "bid_depth": float(observation.bid_depth),
            "ask_depth": float(observation.ask_depth),
            "imbalance": observation.imbalance,
            "traded_volume": float(traded_volume),
            "signed_order_flow": float(signed_order_flow),
            "fundamental_value": fundamental_value,
        }

        for agent_id, metrics in sorted(rl_metrics.items()):
            prefix = agent_id.replace("-", "_")
            record[f"{prefix}_inventory"] = metrics["inventory"]
            record[f"{prefix}_cash"] = metrics["cash"]
            record[f"{prefix}_wealth"] = metrics["wealth"]
            record[f"{prefix}_reward"] = metrics["reward"]

        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the logged records as a pandas DataFrame."""

        return pd.DataFrame.from_records(self.records)

    def to_csv(self, path: str | Path, index: bool = False) -> None:
        """Write the logs to a CSV file."""

        self.to_dataframe().to_csv(path, index=index)
