"""Diagnostics for post-integration RL runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analysis import (
    average_spread,
    average_top_of_book_depth,
    crash_rate,
    excess_kurtosis,
    log_returns,
    max_drawdown,
    one_sided_book_metrics,
    tail_exposure,
)

ACTION_LABEL_TO_INDEX = {
    "sell": 0,
    "hold": 1,
    "buy": 2,
}

QUOTER_ACTION_LABEL_TO_INDEX = {
    "hold": 0,
    "quote_bid": 1,
    "quote_ask": 2,
    "quote_both": 3,
}


def _safe_fraction(mask: Iterable[bool]) -> float:
    series = pd.Series(mask, dtype=bool)
    if series.empty:
        return float("nan")
    return float(series.mean())


def _role_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame.get("agent_role", pd.Series("taker", index=frame.index)),
        index=frame.index,
        dtype="string",
    ).fillna("taker")


def _resolve_inventory_cap(
    rl_frame: pd.DataFrame,
    transition_frame: pd.DataFrame | None = None,
    inventory_cap: float | None = None,
) -> float | None:
    if inventory_cap is not None:
        return float(inventory_cap)
    for frame in (transition_frame, rl_frame):
        if frame is None or frame.empty or "inventory_cap" not in frame.columns:
            continue
        series = pd.to_numeric(frame["inventory_cap"], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[0])
    return None


def diagnose_spread_gaps(frame: pd.DataFrame) -> dict[str, float | str]:
    """Classify spread gaps as one-sided-book events or pipeline issues."""

    missing_bid = frame["best_bid"].isna() if "best_bid" in frame.columns else pd.Series(False, index=frame.index)
    missing_ask = frame["best_ask"].isna() if "best_ask" in frame.columns else pd.Series(False, index=frame.index)
    spread_nan = frame["spread"].isna() if "spread" in frame.columns else pd.Series(False, index=frame.index)
    midprice_nan = frame["midprice"].isna() if "midprice" in frame.columns else pd.Series(False, index=frame.index)
    one_sided = missing_bid | missing_ask
    gap_without_missing_side = spread_nan & ~one_sided
    midprice_gap_without_missing_side = midprice_nan & ~one_sided
    both_sides_missing = missing_bid & missing_ask
    defined_midprice_despite_missing_side = (~midprice_nan) & one_sided
    defined_spread_despite_missing_side = (~spread_nan) & one_sided

    if not bool(spread_nan.any() or midprice_nan.any() or one_sided.any()):
        cause = "none"
    elif bool(
        gap_without_missing_side.any()
        or midprice_gap_without_missing_side.any()
        or defined_midprice_despite_missing_side.any()
        or defined_spread_despite_missing_side.any()
    ):
        cause = "pipeline_issue"
    elif bool(both_sides_missing.any()):
        cause = "missing_snapshot_or_empty_book"
    else:
        cause = "true_one_sided_book"

    diagnostics = one_sided_book_metrics(frame)
    diagnostics.update(
        {
            "spread_nan_fraction": _safe_fraction(spread_nan),
            "midprice_nan_fraction": _safe_fraction(midprice_nan),
            "missing_bid_fraction": _safe_fraction(missing_bid),
            "missing_ask_fraction": _safe_fraction(missing_ask),
            "spread_gap_without_missing_side_fraction": _safe_fraction(gap_without_missing_side),
            "midprice_gap_without_missing_side_fraction": _safe_fraction(midprice_gap_without_missing_side),
            "both_sides_missing_fraction": _safe_fraction(both_sides_missing),
            "defined_midprice_despite_missing_side_fraction": _safe_fraction(defined_midprice_despite_missing_side),
            "defined_spread_despite_missing_side_fraction": _safe_fraction(defined_spread_despite_missing_side),
            "spread_gap_cause": cause,
        }
    )
    return diagnostics


def diagnose_return_series(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize zero inflation and nonzero-return behavior."""

    returns = log_returns(frame["midprice"])
    nonzero_returns = returns[returns != 0]

    return {
        "return_count": float(len(returns)),
        "zero_return_fraction": _safe_fraction(returns == 0),
        "nonzero_return_count": float(len(nonzero_returns)),
        "positive_nonzero_return_fraction": _safe_fraction(nonzero_returns > 0),
        "negative_nonzero_return_fraction": _safe_fraction(nonzero_returns < 0),
        "mean_abs_nonzero_return": float(nonzero_returns.abs().mean()) if len(nonzero_returns) else float("nan"),
        "max_abs_nonzero_return": float(nonzero_returns.abs().max()) if len(nonzero_returns) else float("nan"),
        "excess_kurtosis_full": float(excess_kurtosis(returns)),
        "excess_kurtosis_nonzero_only": float(excess_kurtosis(nonzero_returns)),
    }


def diagnose_rl_behavior(rl_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize RL action balance, executions, inventory, and reward."""

    if rl_frame.empty:
        return {
            "rl_buy_count": 0.0,
            "rl_sell_count": 0.0,
            "rl_hold_count": 0.0,
            "rl_quote_hold_count": 0.0,
            "rl_quote_bid_count": 0.0,
            "rl_quote_ask_count": 0.0,
            "rl_quote_both_count": 0.0,
            "rl_executed_buy_volume": 0.0,
            "rl_executed_sell_volume": 0.0,
            "rl_net_signed_order_flow": 0.0,
            "rl_average_ending_inventory": 0.0,
            "rl_inventory_std": 0.0,
            "rl_inventory_min": 0.0,
            "rl_inventory_median": 0.0,
            "rl_inventory_max": 0.0,
            "rl_max_abs_inventory_reached": 0.0,
            "rl_average_reward_per_agent": 0.0,
            "rl_taker_agent_count": 0.0,
            "rl_quoter_agent_count": 0.0,
        }

    role_series = _role_series(rl_frame)
    taker_frame = rl_frame.loc[role_series == "taker"].copy()
    quoter_frame = rl_frame.loc[role_series == "quoter"].copy()
    executed_buy_volume = float(
        taker_frame.loc[
            taker_frame["previous_action"] == 2,
            "filled_quantity_since_last_decision",
        ].sum()
    )
    executed_sell_volume = float(
        taker_frame.loc[
            taker_frame["previous_action"] == 0,
            "filled_quantity_since_last_decision",
        ].sum()
    )
    ending_inventories = rl_frame.sort_values("time_ns").groupby("agent_id").tail(1)["inventory"]
    average_reward_by_agent = rl_frame.groupby("agent_id")["reward"].mean()

    return {
        "rl_buy_count": float((taker_frame["action"] == 2).sum()) if not taker_frame.empty else 0.0,
        "rl_sell_count": float((taker_frame["action"] == 0).sum()) if not taker_frame.empty else 0.0,
        "rl_hold_count": float((taker_frame["action"] == 1).sum()) if not taker_frame.empty else 0.0,
        "rl_quote_hold_count": float((quoter_frame["action"] == 0).sum()) if not quoter_frame.empty else 0.0,
        "rl_quote_bid_count": float((quoter_frame["action"] == 1).sum()) if not quoter_frame.empty else 0.0,
        "rl_quote_ask_count": float((quoter_frame["action"] == 2).sum()) if not quoter_frame.empty else 0.0,
        "rl_quote_both_count": float((quoter_frame["action"] == 3).sum()) if not quoter_frame.empty else 0.0,
        "rl_executed_buy_volume": executed_buy_volume,
        "rl_executed_sell_volume": executed_sell_volume,
        "rl_net_signed_order_flow": executed_buy_volume - executed_sell_volume,
        "rl_average_ending_inventory": float(ending_inventories.mean()),
        "rl_inventory_std": float(ending_inventories.std(ddof=0)),
        "rl_inventory_min": float(ending_inventories.min()),
        "rl_inventory_median": float(ending_inventories.median()),
        "rl_inventory_max": float(ending_inventories.max()),
        "rl_max_abs_inventory_reached": float(
            pd.to_numeric(rl_frame["inventory"], errors="coerce").fillna(0.0).abs().max()
        ),
        "rl_average_reward_per_agent": float(average_reward_by_agent.mean()),
        "rl_taker_agent_count": float(taker_frame["agent_id"].nunique()) if not taker_frame.empty else 0.0,
        "rl_quoter_agent_count": float(quoter_frame["agent_id"].nunique()) if not quoter_frame.empty else 0.0,
    }


def diagnose_passive_provision(rl_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize whether RL participation is actually providing passive liquidity."""

    diagnostics = {
        "passive_bid_submission_rate": 0.0,
        "passive_ask_submission_rate": 0.0,
        "passive_both_quote_rate": 0.0,
        "quote_fill_rate": 0.0,
        "resting_order_presence_fraction": 0.0,
        "fraction_of_rl_agents_with_resting_bids": 0.0,
        "fraction_of_rl_agents_with_resting_asks": 0.0,
        "fraction_of_rl_agents_with_both_quotes": 0.0,
        "total_rl_passive_order_count": 0.0,
        "total_rl_aggressive_order_count": 0.0,
    }
    if rl_frame.empty:
        return diagnostics

    role_series = _role_series(rl_frame)
    quote_frame = rl_frame.loc[role_series == "quoter"].copy()
    if quote_frame.empty:
        quote_frame = rl_frame.copy()

    passive_bid = pd.to_numeric(
        quote_frame.get("submitted_passive_bid_order_count", pd.Series(0.0, index=quote_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    passive_ask = pd.to_numeric(
        quote_frame.get("submitted_passive_ask_order_count", pd.Series(0.0, index=quote_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    passive_total = passive_bid + passive_ask
    passive_fill = pd.to_numeric(
        quote_frame.get("passive_filled_quantity_since_last_decision", pd.Series(0.0, index=quote_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    resting_bid = pd.to_numeric(
        quote_frame.get("resting_bid", pd.Series(False, index=quote_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    resting_ask = pd.to_numeric(
        quote_frame.get("resting_ask", pd.Series(False, index=quote_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    aggressive_total = pd.to_numeric(
        rl_frame.get("submitted_aggressive_order_count", pd.Series(0.0, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)

    total_passive_order_count = float(passive_total.sum())
    diagnostics.update(
        {
            "passive_bid_submission_rate": float((passive_bid > 0).mean()) if not passive_bid.empty else 0.0,
            "passive_ask_submission_rate": float((passive_ask > 0).mean()) if not passive_ask.empty else 0.0,
            "passive_both_quote_rate": float(((passive_bid > 0) & (passive_ask > 0)).mean())
            if not passive_total.empty
            else 0.0,
            "quote_fill_rate": (
                float(min(1.0, passive_fill.sum() / total_passive_order_count))
                if total_passive_order_count > 0.0
                else 0.0
            ),
            "resting_order_presence_fraction": float(((resting_bid > 0) | (resting_ask > 0)).mean())
            if not quote_frame.empty
            else 0.0,
            "fraction_of_rl_agents_with_resting_bids": float((resting_bid > 0).mean()) if not quote_frame.empty else 0.0,
            "fraction_of_rl_agents_with_resting_asks": float((resting_ask > 0).mean()) if not quote_frame.empty else 0.0,
            "fraction_of_rl_agents_with_both_quotes": float(((resting_bid > 0) & (resting_ask > 0)).mean())
            if not quote_frame.empty
            else 0.0,
            "total_rl_passive_order_count": total_passive_order_count,
            "total_rl_aggressive_order_count": float(aggressive_total.sum()),
        }
    )
    return diagnostics


def summarize_rl_agents(
    rl_frame: pd.DataFrame,
    *,
    inventory_alert_level: float = 6.0,
    inventory_cap: float | None = None,
) -> pd.DataFrame:
    """Return one diagnostic row per RL agent."""

    if rl_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    ordered = rl_frame.sort_values(["agent_id", "time_ns"])
    resolved_inventory_cap = _resolve_inventory_cap(rl_frame, inventory_cap=inventory_cap)
    near_cap_threshold = None
    if resolved_inventory_cap is not None and resolved_inventory_cap > 0:
        near_cap_threshold = max(
            float(resolved_inventory_cap) - 1.0,
            float(np.ceil(0.9 * float(resolved_inventory_cap))),
        )
    for agent_id, agent_frame in ordered.groupby("agent_id"):
        role = str(agent_frame.get("agent_role", pd.Series(["taker"])).iloc[0])
        inventory = pd.to_numeric(agent_frame["inventory"], errors="coerce").fillna(0.0)
        filled = pd.to_numeric(
            agent_frame["filled_quantity_since_last_decision"],
            errors="coerce",
        ).fillna(0.0)
        passive_filled = pd.to_numeric(
            agent_frame.get("passive_filled_quantity_since_last_decision", pd.Series(0.0, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        previous_action = pd.to_numeric(agent_frame["previous_action"], errors="coerce").fillna(-1).astype(int)
        resting_bid = pd.to_numeric(
            agent_frame.get("resting_bid", pd.Series(False, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        resting_ask = pd.to_numeric(
            agent_frame.get("resting_ask", pd.Series(False, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        passive_order_count = pd.to_numeric(
            agent_frame.get("submitted_passive_order_count", pd.Series(0.0, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        aggressive_order_count = pd.to_numeric(
            agent_frame.get("submitted_aggressive_order_count", pd.Series(0.0, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        blocked_buy = pd.to_numeric(
            agent_frame.get("blocked_buy_due_to_cap", pd.Series(False, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        blocked_sell = pd.to_numeric(
            agent_frame.get("blocked_sell_due_to_cap", pd.Series(False, index=agent_frame.index)),
            errors="coerce",
        ).fillna(0.0)
        ending_inventory = float(inventory.iloc[-1]) if len(inventory) else 0.0
        at_cap_fraction = 0.0
        near_cap_fraction = 0.0
        if resolved_inventory_cap is not None and resolved_inventory_cap > 0:
            at_cap_fraction = float((inventory.abs() >= float(resolved_inventory_cap)).mean())
            near_cap_fraction = float((inventory.abs() >= float(near_cap_threshold)).mean())
        rows.append(
            {
                "agent_id": float(agent_id),
                "agent_role": role,
                "inventory_min": float(inventory.min()),
                "inventory_max": float(inventory.max()),
                "max_abs_inventory": float(inventory.abs().max()),
                "ending_inventory": ending_inventory,
                "ending_abs_inventory": abs(ending_inventory),
                "inventory_alert_hits": float((inventory.abs() >= inventory_alert_level).sum()),
                "blocked_buy_actions": float(blocked_buy.sum()),
                "blocked_sell_actions": float(blocked_sell.sum()),
                "inventory_at_cap_fraction": at_cap_fraction,
                "inventory_near_cap_fraction": near_cap_fraction,
                "executed_buy_actions": float(((previous_action == 2) & (filled > 0)).sum()),
                "executed_sell_actions": float(((previous_action == 0) & (filled > 0)).sum()),
                "unfilled_buy_attempts": float(((previous_action == 2) & (filled <= 0)).sum()),
                "unfilled_sell_attempts": float(((previous_action == 0) & (filled <= 0)).sum()),
                "executed_buy_volume": float(filled[previous_action == 2].sum()),
                "executed_sell_volume": float(filled[previous_action == 0].sum()),
                "passive_filled_quantity": float(passive_filled.sum()),
                "passive_order_count": float(passive_order_count.sum()),
                "aggressive_order_count": float(aggressive_order_count.sum()),
                "resting_bid_fraction": float((resting_bid > 0).mean()),
                "resting_ask_fraction": float((resting_ask > 0).mean()),
                "resting_both_quote_fraction": float(((resting_bid > 0) & (resting_ask > 0)).mean()),
                "average_reward": float(pd.to_numeric(agent_frame["reward"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def diagnose_inventory_cap_behavior(
    rl_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    *,
    inventory_cap: float | None = None,
) -> dict[str, float]:
    """Summarize RL inventory-cap blocking and cap-contact behavior."""

    diagnostics = {
        "inventory_cap_value": float("nan"),
        "blocked_buy_action_count": 0.0,
        "blocked_sell_action_count": 0.0,
        "blocked_action_count": 0.0,
        "inventory_at_cap_fraction": 0.0,
        "inventory_near_cap_fraction": 0.0,
        "max_abs_inventory_reached": 0.0,
    }
    resolved_inventory_cap = _resolve_inventory_cap(
        rl_frame,
        transition_frame=transition_frame,
        inventory_cap=inventory_cap,
    )
    if resolved_inventory_cap is not None:
        diagnostics["inventory_cap_value"] = float(resolved_inventory_cap)
    if rl_frame.empty:
        return diagnostics
    blocked_buy = pd.to_numeric(
        rl_frame.get("blocked_buy_due_to_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    blocked_sell = pd.to_numeric(
        rl_frame.get("blocked_sell_due_to_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    inventory = pd.to_numeric(rl_frame.get("inventory", pd.Series(0.0, index=rl_frame.index)), errors="coerce").fillna(0.0)

    diagnostics["blocked_buy_action_count"] = float(blocked_buy.sum())
    diagnostics["blocked_sell_action_count"] = float(blocked_sell.sum())
    diagnostics["blocked_action_count"] = float(blocked_buy.sum() + blocked_sell.sum())
    diagnostics["max_abs_inventory_reached"] = float(inventory.abs().max()) if not inventory.empty else 0.0

    if resolved_inventory_cap is not None and resolved_inventory_cap > 0:
        near_cap_threshold = max(
            float(resolved_inventory_cap) - 1.0,
            float(np.ceil(0.9 * float(resolved_inventory_cap))),
        )
        diagnostics["inventory_cap_value"] = float(resolved_inventory_cap)
        diagnostics["inventory_at_cap_fraction"] = float((inventory.abs() >= float(resolved_inventory_cap)).mean())
        diagnostics["inventory_near_cap_fraction"] = float((inventory.abs() >= float(near_cap_threshold)).mean())
    return diagnostics


def diagnose_transition_dynamics(transition_frame: pd.DataFrame) -> dict[str, float | str]:
    """Summarize action execution and inventory-state transitions."""

    diagnostics: dict[str, float | str] = {
        "submitted_buy_action_count": 0.0,
        "submitted_sell_action_count": 0.0,
        "buy_fill_rate": 0.0,
        "sell_fill_rate": 0.0,
        "average_buy_fill_delay_seconds": float("nan"),
        "average_sell_fill_delay_seconds": float("nan"),
        "inventory_transition_0_to_pos1": 0.0,
        "inventory_transition_0_to_neg1": 0.0,
        "inventory_transition_neg1_to_0": 0.0,
        "inventory_transition_pos1_to_0": 0.0,
        "inventory_transition_counts": "{}",
    }
    if transition_frame.empty:
        return diagnostics
    required_columns = {"action", "inventory_before", "inventory_after"}
    if not required_columns.issubset(transition_frame.columns):
        return diagnostics

    role_series = _role_series(transition_frame)
    working = transition_frame.loc[role_series == "taker"].copy()
    if working.empty:
        return diagnostics

    inventory_before = pd.to_numeric(working["inventory_before"], errors="coerce").fillna(0.0)
    inventory_after = pd.to_numeric(working["inventory_after"], errors="coerce").fillna(0.0)
    action_source = "effective_action" if "effective_action" in working.columns else "action"
    action = pd.to_numeric(working[action_source], errors="coerce").fillna(-1).astype(int)
    inventory_delta = inventory_after - inventory_before

    buy_mask = action == ACTION_LABEL_TO_INDEX["buy"]
    sell_mask = action == ACTION_LABEL_TO_INDEX["sell"]
    executed_buy_mask = buy_mask & (inventory_delta > 0)
    executed_sell_mask = sell_mask & (inventory_delta < 0)

    diagnostics["submitted_buy_action_count"] = float(buy_mask.sum())
    diagnostics["submitted_sell_action_count"] = float(sell_mask.sum())
    diagnostics["buy_fill_rate"] = (
        float(executed_buy_mask.sum() / buy_mask.sum()) if bool(buy_mask.any()) else 0.0
    )
    diagnostics["sell_fill_rate"] = (
        float(executed_sell_mask.sum() / sell_mask.sum()) if bool(sell_mask.any()) else 0.0
    )
    diagnostics["average_buy_fill_delay_seconds"] = 0.0 if bool(executed_buy_mask.any()) else float("nan")
    diagnostics["average_sell_fill_delay_seconds"] = 0.0 if bool(executed_sell_mask.any()) else float("nan")
    diagnostics["inventory_transition_0_to_pos1"] = float(((inventory_before == 0.0) & (inventory_after == 1.0)).sum())
    diagnostics["inventory_transition_0_to_neg1"] = float(((inventory_before == 0.0) & (inventory_after == -1.0)).sum())
    diagnostics["inventory_transition_neg1_to_0"] = float(((inventory_before == -1.0) & (inventory_after == 0.0)).sum())
    diagnostics["inventory_transition_pos1_to_0"] = float(((inventory_before == 1.0) & (inventory_after == 0.0)).sum())

    transition_counts = (
        pd.DataFrame({"before": inventory_before, "after": inventory_after})
        .value_counts()
        .sort_index()
    )
    diagnostics["inventory_transition_counts"] = json.dumps(
        {
            f"{int(before)}->{int(after)}": int(count)
            for (before, after), count in transition_counts.items()
        },
        sort_keys=True,
    )

    for inventory_state, state_label in [(-1.0, "neg1"), (0.0, "0"), (1.0, "pos1")]:
        state_mask = inventory_before == inventory_state
        state_buy_mask = state_mask & buy_mask
        state_sell_mask = state_mask & sell_mask
        diagnostics[f"buy_chosen_at_inventory_{state_label}"] = float(state_buy_mask.sum())
        diagnostics[f"sell_chosen_at_inventory_{state_label}"] = float(state_sell_mask.sum())
        diagnostics[f"buy_fill_rate_at_inventory_{state_label}"] = (
            float((inventory_delta[state_buy_mask] > 0).mean()) if bool(state_buy_mask.any()) else float("nan")
        )
        diagnostics[f"sell_fill_rate_at_inventory_{state_label}"] = (
            float((inventory_delta[state_sell_mask] < 0).mean()) if bool(state_sell_mask.any()) else float("nan")
        )

    return diagnostics


def _linear_policy_output_diagnostics(
    policy: object | None,
    transition_frame: pd.DataFrame,
    *,
    action_labels: list[str],
    prefix: str = "",
) -> dict[str, float]:
    diagnostics = {
        f"{prefix}mean_logit_{label}": float("nan")
        for label in action_labels
    }
    diagnostics.update(
        {
            f"{prefix}mean_prob_{label}": float("nan")
            for label in action_labels
        }
    )
    diagnostics.update(
        {
            f"{prefix}deterministic_{label}_fraction": float("nan")
            for label in action_labels
        }
    )
    if policy is None or transition_frame.empty:
        return diagnostics
    if not hasattr(policy, "policy_weights") or not hasattr(policy, "policy_bias"):
        return diagnostics

    state_columns = sorted(column for column in transition_frame.columns if column.startswith("state_"))
    if not state_columns:
        return diagnostics

    states = transition_frame[state_columns].to_numpy(dtype=float)
    logits = states @ np.asarray(getattr(policy, "policy_weights"), dtype=float) + np.asarray(
        getattr(policy, "policy_bias"),
        dtype=float,
    )
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    greedy = probs.argmax(axis=1)

    for index, label in enumerate(action_labels):
        if index >= logits.shape[1]:
            break
        diagnostics[f"{prefix}mean_logit_{label}"] = float(logits[:, index].mean())
        diagnostics[f"{prefix}mean_prob_{label}"] = float(probs[:, index].mean())
        diagnostics[f"{prefix}deterministic_{label}_fraction"] = float((greedy == index).mean())
    return diagnostics


def diagnose_policy_outputs(policy: object | None, transition_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize logits, probabilities, and deterministic action collapse."""

    diagnostics = _linear_policy_output_diagnostics(
        None,
        pd.DataFrame(),
        action_labels=["sell", "hold", "buy"],
    )
    diagnostics.update(
        _linear_policy_output_diagnostics(
            None,
            pd.DataFrame(),
            action_labels=["hold", "quote_bid", "quote_ask", "quote_both"],
            prefix="quoter_",
        )
    )
    if policy is None or transition_frame.empty:
        return diagnostics

    if hasattr(policy, "policy_for_role"):
        role_series = _role_series(transition_frame)
        taker_frame = transition_frame.loc[role_series == "taker"].copy()
        quoter_frame = transition_frame.loc[role_series == "quoter"].copy()
        taker_policy = getattr(policy, "taker_policy", None)
        quoter_policy = getattr(policy, "quoter_policy", None)
        diagnostics.update(
            _linear_policy_output_diagnostics(
                taker_policy,
                taker_frame,
                action_labels=["sell", "hold", "buy"],
            )
        )
        diagnostics.update(
            _linear_policy_output_diagnostics(
                quoter_policy,
                quoter_frame,
                action_labels=["hold", "quote_bid", "quote_ask", "quote_both"],
                prefix="quoter_",
            )
        )
        return diagnostics

    diagnostics.update(
        _linear_policy_output_diagnostics(
            policy,
            transition_frame,
            action_labels=["sell", "hold", "buy"],
        )
    )
    return diagnostics


def diagnose_reward_shape(
    transition_frame: pd.DataFrame,
    *,
    inventory_alert_level: float = 6.0,
) -> dict[str, float]:
    """Check whether rewards at higher inventory levels look suspiciously favorable."""

    diagnostics = {
        "reward_mean_all_transitions": 0.0,
        "reward_mean_nonzero_inventory": 0.0,
        "reward_mean_alert_inventory": 0.0,
        "positive_reward_fraction_alert_inventory": 0.0,
        "wealth_delta_mean_all_transitions": 0.0,
        "wealth_delta_std_all_transitions": 0.0,
        "inventory_penalty_mean_all_transitions": 0.0,
        "inventory_penalty_std_all_transitions": 0.0,
        "flat_hold_penalty_mean_all_transitions": 0.0,
    }
    if transition_frame.empty:
        return diagnostics

    reward = pd.to_numeric(transition_frame["reward"], errors="coerce").fillna(0.0)
    wealth_delta = pd.to_numeric(
        transition_frame.get("wealth_delta", pd.Series(0.0, index=transition_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    inventory_penalty = pd.to_numeric(
        transition_frame.get("inventory_penalty", pd.Series(0.0, index=transition_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    flat_hold_penalty = pd.to_numeric(
        transition_frame.get("flat_hold_penalty", pd.Series(0.0, index=transition_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    inventory_after = pd.to_numeric(transition_frame["inventory_after"], errors="coerce").fillna(0.0).abs()
    alert_mask = inventory_after >= inventory_alert_level
    nonzero_mask = inventory_after > 0
    diagnostics["reward_mean_all_transitions"] = float(reward.mean())
    diagnostics["reward_mean_nonzero_inventory"] = float(reward[nonzero_mask].mean()) if bool(nonzero_mask.any()) else 0.0
    diagnostics["reward_mean_alert_inventory"] = float(reward[alert_mask].mean()) if bool(alert_mask.any()) else 0.0
    diagnostics["positive_reward_fraction_alert_inventory"] = (
        float((reward[alert_mask] > 0).mean()) if bool(alert_mask.any()) else 0.0
    )
    diagnostics["wealth_delta_mean_all_transitions"] = float(wealth_delta.mean())
    diagnostics["wealth_delta_std_all_transitions"] = float(wealth_delta.std(ddof=0))
    diagnostics["inventory_penalty_mean_all_transitions"] = float(inventory_penalty.mean())
    diagnostics["inventory_penalty_std_all_transitions"] = float(inventory_penalty.std(ddof=0))
    diagnostics["flat_hold_penalty_mean_all_transitions"] = float(flat_hold_penalty.mean())
    return diagnostics


def diagnose_action_reward_balance(transition_frame: pd.DataFrame) -> dict[str, float]:
    """Summarize reward components by chosen action type."""

    diagnostics: dict[str, float] = {}
    for label in ACTION_LABEL_TO_INDEX:
        diagnostics[f"{label}_reward_mean"] = 0.0
        diagnostics[f"{label}_wealth_delta_mean"] = 0.0
        diagnostics[f"{label}_inventory_penalty_mean"] = 0.0
        diagnostics[f"{label}_flat_hold_penalty_mean"] = 0.0
    diagnostics["hold_minus_best_trade_reward_gap"] = 0.0
    if transition_frame.empty or "action" not in transition_frame.columns:
        return diagnostics

    working = transition_frame.loc[_role_series(transition_frame) == "taker"].copy()
    if working.empty:
        return diagnostics
    working["action"] = pd.to_numeric(working["action"], errors="coerce").fillna(-1).astype(int)
    working["reward"] = pd.to_numeric(working.get("reward", 0.0), errors="coerce").fillna(0.0)
    working["wealth_delta"] = pd.to_numeric(
        working.get("wealth_delta", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    working["inventory_penalty"] = pd.to_numeric(
        working.get("inventory_penalty", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)
    working["flat_hold_penalty"] = pd.to_numeric(
        working.get("flat_hold_penalty", pd.Series(0.0, index=working.index)),
        errors="coerce",
    ).fillna(0.0)

    reward_by_action: dict[str, float] = {}
    for label, action_index in ACTION_LABEL_TO_INDEX.items():
        subset = working.loc[working["action"] == action_index]
        diagnostics[f"{label}_reward_mean"] = float(subset["reward"].mean()) if not subset.empty else 0.0
        diagnostics[f"{label}_wealth_delta_mean"] = (
            float(subset["wealth_delta"].mean()) if not subset.empty else 0.0
        )
        diagnostics[f"{label}_inventory_penalty_mean"] = (
            float(subset["inventory_penalty"].mean()) if not subset.empty else 0.0
        )
        diagnostics[f"{label}_flat_hold_penalty_mean"] = (
            float(subset["flat_hold_penalty"].mean()) if not subset.empty else 0.0
        )
        reward_by_action[label] = diagnostics[f"{label}_reward_mean"]
    diagnostics["hold_minus_best_trade_reward_gap"] = reward_by_action["hold"] - max(
        reward_by_action["buy"],
        reward_by_action["sell"],
    )
    return diagnostics


def compute_rl_run_diagnostics(frame: pd.DataFrame, rl_frame: pd.DataFrame) -> dict[str, float | str]:
    """Combine market and RL diagnostics for one completed run."""

    diagnostics: dict[str, float | str] = {}
    diagnostics.update(diagnose_spread_gaps(frame))
    diagnostics.update(diagnose_return_series(frame))
    diagnostics.update(diagnose_rl_behavior(rl_frame))
    diagnostics.update(diagnose_passive_provision(rl_frame))
    diagnostics.update(
        {
            "average_spread": float(average_spread(frame)),
            "average_top_of_book_depth": float(average_top_of_book_depth(frame)),
            "tail_exposure": float(tail_exposure(log_returns(frame["midprice"]))),
            "crash_rate": float(crash_rate(log_returns(frame["midprice"]))),
            "max_drawdown": float(max_drawdown(frame["midprice"])),
        }
    )
    return diagnostics


def compute_policy_evaluation_diagnostics(
    frame: pd.DataFrame,
    rl_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    *,
    policy: object | None = None,
    inventory_cap: float | None = None,
    inventory_alert_level: float = 6.0,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Return a richer diagnostic bundle for one evaluated policy run."""

    diagnostics = compute_rl_run_diagnostics(frame, rl_frame)
    role_series = _role_series(rl_frame) if not rl_frame.empty else pd.Series(dtype="string")
    has_quoter = bool((role_series == "quoter").any()) if not role_series.empty else False
    has_taker = bool((role_series == "taker").any()) if not role_series.empty else False
    resolved_inventory_cap = _resolve_inventory_cap(
        rl_frame,
        transition_frame=transition_frame,
        inventory_cap=inventory_cap,
    )
    agent_frame = summarize_rl_agents(
        rl_frame,
        inventory_alert_level=inventory_alert_level,
        inventory_cap=resolved_inventory_cap,
    )

    diagnostics.update(
        {
            "action_ordering": (
                "taker:0=sell,1=hold,2=buy; quoter:0=hold,1=quote_bid,2=quote_ask,3=quote_both"
                if has_quoter and has_taker
                else (
                    "quoter:0=hold,1=quote_bid,2=quote_ask,3=quote_both"
                    if has_quoter
                    else "0=sell,1=hold,2=buy"
                )
            ),
            "inventory_cap_present": "yes" if resolved_inventory_cap is not None else "no",
            "inventory_cap_value": float(resolved_inventory_cap) if resolved_inventory_cap is not None else float("nan"),
            "executed_buy_action_count": 0.0,
            "executed_sell_action_count": 0.0,
            "unfilled_buy_attempt_count": 0.0,
            "unfilled_sell_attempt_count": 0.0,
            "inventory_overall_min": 0.0,
            "inventory_overall_max": 0.0,
            "max_abs_inventory_reached": 0.0,
            "inventory_alert_level": float(inventory_alert_level),
            "inventory_alert_agent_count": 0.0,
        }
    )
    if not agent_frame.empty:
        diagnostics.update(
            {
                "executed_buy_action_count": float(agent_frame["executed_buy_actions"].sum()),
                "executed_sell_action_count": float(agent_frame["executed_sell_actions"].sum()),
                "unfilled_buy_attempt_count": float(agent_frame["unfilled_buy_attempts"].sum()),
                "unfilled_sell_attempt_count": float(agent_frame["unfilled_sell_attempts"].sum()),
                "inventory_overall_min": float(agent_frame["inventory_min"].min()),
                "inventory_overall_max": float(agent_frame["inventory_max"].max()),
                "max_abs_inventory_reached": float(agent_frame["max_abs_inventory"].max()),
                "inventory_alert_agent_count": float((agent_frame["inventory_alert_hits"] > 0).sum()),
            }
        )
    diagnostics.update(
        diagnose_inventory_cap_behavior(
            rl_frame,
            transition_frame,
            inventory_cap=resolved_inventory_cap,
        )
    )
    diagnostics.update(diagnose_passive_provision(rl_frame))
    diagnostics.update(diagnose_transition_dynamics(transition_frame))
    diagnostics.update(diagnose_policy_outputs(policy, transition_frame))
    diagnostics.update(diagnose_reward_shape(transition_frame, inventory_alert_level=inventory_alert_level))
    diagnostics.update(diagnose_action_reward_balance(transition_frame))
    return diagnostics, agent_frame


def build_rl_run_report(diagnostics: dict[str, float | str]) -> str:
    """Return a short markdown report for one RL-integrated run."""

    spread_cause = diagnostics["spread_gap_cause"]
    if spread_cause == "true_one_sided_book":
        spread_summary = "Undefined spread and midprice line up with true one-sided-book states rather than a resampling bug."
    elif spread_cause == "pipeline_issue":
        spread_summary = "Undefined spread or midprice remains inconsistent with the book state and indicates a pipeline bug."
    elif spread_cause == "missing_snapshot_or_empty_book":
        spread_summary = "Undefined spread and midprice coincide with missing-sided snapshots, including fully empty-book intervals."
    else:
        spread_summary = "No one-sided-book or undefined-midprice intervals were observed."

    kurtosis_summary = (
        "Full-series kurtosis is dominated by zero inflation at the 1-second sampling scale."
        if diagnostics["excess_kurtosis_full"] != diagnostics["excess_kurtosis_nonzero_only"]
        else "Full-series and nonzero-only kurtosis match closely."
    )

    net_flow = float(diagnostics["rl_net_signed_order_flow"])
    if abs(net_flow) <= max(1.0, 0.02 * (diagnostics["rl_executed_buy_volume"] + diagnostics["rl_executed_sell_volume"])):
        rl_summary = "Random-policy RL flow is directionally close to balanced."
    elif net_flow > 0:
        rl_summary = "Random-policy RL flow is buy-skewed."
    else:
        rl_summary = "Random-policy RL flow is sell-skewed."

    return "\n".join(
        [
            "# RL Run Diagnostics",
            "",
            "## Book State",
            f"- One-sided-book fraction (conditional on positive visible liquidity): {diagnostics['one_sided_book_fraction']:.6f}",
            f"- One-sided metric valid timestep count: {diagnostics.get('one_sided_metric_valid_timestep_count', float('nan')):.0f}",
            f"- Empty-book fraction: {diagnostics.get('empty_book_fraction', float('nan')):.6f}",
            f"- Average bid / ask volume: {diagnostics.get('average_bid_volume', float('nan')):.6f} / {diagnostics.get('average_ask_volume', float('nan')):.6f}",
            f"- Missing-quote one-sided fraction: {diagnostics['true_one_sided_book_fraction']:.6f}",
            f"- Both-sides-missing fraction: {diagnostics['both_sides_missing_fraction']:.6f}",
            f"- Missing best bid fraction: {diagnostics['missing_bid_fraction']:.6f}",
            f"- Missing best ask fraction: {diagnostics['missing_ask_fraction']:.6f}",
            f"- Undefined midprice fraction: {diagnostics['undefined_midprice_fraction']:.6f}",
            f"- Spread NaN fraction: {diagnostics['spread_nan_fraction']:.6f}",
            f"- Midprice NaN fraction: {diagnostics['midprice_nan_fraction']:.6f}",
            f"- Number of one-sided episodes: {int(diagnostics['num_one_sided_episodes'])}",
            f"- Max consecutive one-sided duration: {diagnostics['max_consecutive_one_sided_duration']:.6f} seconds ({diagnostics['max_consecutive_one_sided_duration_steps']:.0f} steps)",
            f"- Mean one-sided episode duration: {diagnostics['mean_one_sided_episode_duration']:.6f} seconds ({diagnostics['mean_one_sided_episode_duration_steps']:.2f} steps)",
            f"- P50 / P90 one-sided episode duration: {diagnostics['one_sided_episode_duration_p50_seconds']:.6f} / {diagnostics['one_sided_episode_duration_p90_seconds']:.6f} seconds",
            f"- Midprice gap without missing side fraction: {diagnostics['midprice_gap_without_missing_side_fraction']:.6f}",
            f"- Spread gap without missing side fraction: {diagnostics['spread_gap_without_missing_side_fraction']:.6f}",
            f"- Defined midprice despite missing side fraction: {diagnostics['defined_midprice_despite_missing_side_fraction']:.6f}",
            f"- Defined spread despite missing side fraction: {diagnostics['defined_spread_despite_missing_side_fraction']:.6f}",
            f"- Gap diagnosis: {spread_cause}",
            f"- Interpretation: {spread_summary}",
            "",
            "## Returns",
            f"- Return count: {int(diagnostics['return_count'])}",
            f"- Zero-return fraction: {diagnostics['zero_return_fraction']:.6f}",
            f"- Nonzero return count: {int(diagnostics['nonzero_return_count'])}",
            f"- Positive nonzero return fraction: {diagnostics['positive_nonzero_return_fraction']:.6f}",
            f"- Negative nonzero return fraction: {diagnostics['negative_nonzero_return_fraction']:.6f}",
            f"- Mean absolute nonzero return: {diagnostics['mean_abs_nonzero_return']:.8f}",
            f"- Max absolute nonzero return: {diagnostics['max_abs_nonzero_return']:.8f}",
            f"- Excess kurtosis (full): {diagnostics['excess_kurtosis_full']:.6f}",
            f"- Excess kurtosis (nonzero only): {diagnostics['excess_kurtosis_nonzero_only']:.6f}",
            f"- Interpretation: {kurtosis_summary}",
            "",
            "## RL Flow",
            f"- RL taker / quoter agents: {int(diagnostics.get('rl_taker_agent_count', 0.0))} / {int(diagnostics.get('rl_quoter_agent_count', 0.0))}",
            f"- Buy / hold / sell counts: {int(diagnostics['rl_buy_count'])} / {int(diagnostics['rl_hold_count'])} / {int(diagnostics['rl_sell_count'])}",
            f"- Quote hold / bid / ask / both counts: {int(diagnostics.get('rl_quote_hold_count', 0.0))} / {int(diagnostics.get('rl_quote_bid_count', 0.0))} / {int(diagnostics.get('rl_quote_ask_count', 0.0))} / {int(diagnostics.get('rl_quote_both_count', 0.0))}",
            f"- Executed buy volume: {diagnostics['rl_executed_buy_volume']:.0f}",
            f"- Executed sell volume: {diagnostics['rl_executed_sell_volume']:.0f}",
            f"- Net signed RL order flow: {diagnostics['rl_net_signed_order_flow']:.0f}",
            f"- Ending inventory mean / median: {diagnostics['rl_average_ending_inventory']:.3f} / {diagnostics['rl_inventory_median']:.3f}",
            f"- Ending inventory min / max: {diagnostics['rl_inventory_min']:.3f} / {diagnostics['rl_inventory_max']:.3f}",
            f"- Average reward per agent: {diagnostics['rl_average_reward_per_agent']:.6f}",
            f"- Passive bid / ask / both submission rate: {float(diagnostics.get('passive_bid_submission_rate', 0.0)):.6f} / {float(diagnostics.get('passive_ask_submission_rate', 0.0)):.6f} / {float(diagnostics.get('passive_both_quote_rate', 0.0)):.6f}",
            f"- Resting quote presence / quote fill rate: {float(diagnostics.get('resting_order_presence_fraction', 0.0)):.6f} / {float(diagnostics.get('quote_fill_rate', 0.0)):.6f}",
            f"- Interpretation: {rl_summary}",
            "",
            "## Market",
            f"- Average spread: {diagnostics['average_spread']:.6f}",
            f"- Average top-of-book depth: {diagnostics['average_top_of_book_depth']:.6f}",
            f"- Tail exposure: {diagnostics['tail_exposure']:.6f}",
            f"- Crash rate: {diagnostics['crash_rate']:.6f}",
            f"- Max drawdown: {diagnostics['max_drawdown']:.6f}",
        ]
    )


def build_policy_evaluation_report(
    diagnostics: dict[str, float | str],
    agent_frame: pd.DataFrame,
) -> str:
    """Return a markdown report for trained-policy debugging."""

    sell_chosen = float(diagnostics.get("rl_sell_count", 0.0))
    executed_sell = float(diagnostics.get("executed_sell_action_count", 0.0))
    deterministic_hold = float(diagnostics.get("deterministic_hold_fraction", 0.0))
    hold_fraction = float(diagnostics.get("rl_hold_count", 0.0)) / max(
        1.0,
        float(diagnostics.get("rl_buy_count", 0.0))
        + float(diagnostics.get("rl_hold_count", 0.0))
        + float(diagnostics.get("rl_sell_count", 0.0)),
    )

    if sell_chosen == 0.0:
        sell_diagnosis = "Sell actions are not broken in execution; the evaluated policy simply is not choosing them."
    elif executed_sell == 0.0:
        sell_diagnosis = "Sell actions are being chosen but not filling, which points to an execution-side issue."
    else:
        sell_diagnosis = "Sell actions are both chosen and executed."

    inventory_cap_present = str(diagnostics.get("inventory_cap_present", "no")) == "yes"
    inventory_cap_value = diagnostics.get("inventory_cap_value", float("nan"))

    if inventory_cap_present and float(diagnostics.get("blocked_action_count", 0.0)) > 0:
        inventory_diagnosis = (
            "Inventory limits are active and some actions were converted into holds at the cap. "
            "This run can be interpreted under explicit RL inventory control."
        )
    elif inventory_cap_present and float(diagnostics.get("inventory_at_cap_fraction", 0.0)) > 0:
        inventory_diagnosis = (
            "Inventory limits are active but not frequently binding. Inventory behavior still reflects the policy, "
            "with the cap acting only as a safety constraint."
        )
    elif float(diagnostics.get("inventory_alert_agent_count", 0.0)) > 0 and sell_chosen == 0.0:
        inventory_diagnosis = (
            "There is no inventory cap in code. The policy is saturating inventory and then stopping, "
            "which indicates policy degeneracy rather than a cap or logging bug."
        )
    elif deterministic_hold > 0.95 and hold_fraction > 0.95:
        inventory_diagnosis = (
            "There is no inventory cap in code. This checkpoint has collapsed to deterministic hold, "
            "so inventory saturation is not coming from execution logic."
        )
    else:
        inventory_diagnosis = (
            "There is no inventory cap in code. Inventory behavior is being driven by the policy, "
            "not by a hard-coded trading limit."
        )

    reward_diagnosis = (
        "Rewards do not indicate a bug in the inventory penalty path."
        if float(diagnostics.get("positive_reward_fraction_alert_inventory", 0.0)) <= 0.5
        else "High-inventory states are still often rewarded positively and deserve closer inspection."
    )
    hold_gap = float(diagnostics.get("hold_minus_best_trade_reward_gap", 0.0))
    if hold_gap > 0.0:
        action_reward_diagnosis = (
            "Hold has a higher mean reward than the best trading action, so greedy argmax is expected to collapse."
        )
    else:
        action_reward_diagnosis = (
            "At least one trading action matches or beats hold on mean reward, so pure-hold collapse is less justified."
        )

    lines = [
        "# Policy Evaluation Diagnostics",
        "",
        "## Action Mapping",
        f"- {diagnostics.get('action_ordering', '0=sell, 1=hold, 2=buy')}.",
        "",
        "## Book State",
        f"- One-sided-book fraction (conditional on positive visible liquidity): {float(diagnostics.get('one_sided_book_fraction', float('nan'))):.6f}",
        f"- One-sided metric valid timestep count: {float(diagnostics.get('one_sided_metric_valid_timestep_count', float('nan'))):.0f}",
        f"- Empty-book fraction: {float(diagnostics.get('empty_book_fraction', float('nan'))):.6f}",
        f"- Average bid / ask volume: {float(diagnostics.get('average_bid_volume', float('nan'))):.6f} / {float(diagnostics.get('average_ask_volume', float('nan'))):.6f}",
        f"- Missing-quote one-sided fraction: {float(diagnostics.get('true_one_sided_book_fraction', float('nan'))):.6f}",
        f"- Both-sides-missing fraction: {float(diagnostics.get('both_sides_missing_fraction', float('nan'))):.6f}",
        f"- Missing best bid / ask fraction: {float(diagnostics.get('missing_bid_fraction', float('nan'))):.6f} / {float(diagnostics.get('missing_ask_fraction', float('nan'))):.6f}",
        f"- Undefined midprice fraction: {float(diagnostics.get('undefined_midprice_fraction', float('nan'))):.6f}",
        f"- Max consecutive one-sided duration: {float(diagnostics.get('max_consecutive_one_sided_duration', float('nan'))):.6f} seconds ({float(diagnostics.get('max_consecutive_one_sided_duration_steps', float('nan'))):.0f} steps)",
        f"- One-sided episodes / mean duration: {int(float(diagnostics.get('num_one_sided_episodes', 0.0)))} / {float(diagnostics.get('mean_one_sided_episode_duration', float('nan'))):.6f} seconds",
        f"- Midprice gap without missing side fraction: {float(diagnostics.get('midprice_gap_without_missing_side_fraction', float('nan'))):.6f}",
        f"- Spread gap without missing side fraction: {float(diagnostics.get('spread_gap_without_missing_side_fraction', float('nan'))):.6f}",
        f"- Defined midprice despite missing side fraction: {float(diagnostics.get('defined_midprice_despite_missing_side_fraction', float('nan'))):.6f}",
        f"- Diagnosis: {diagnostics.get('spread_gap_cause', 'unknown')}",
        "",
        "## Inventory",
        f"- Inventory cap logic present: {diagnostics['inventory_cap_present']}",
        f"- Inventory cap value: {float(inventory_cap_value):.3f}" if inventory_cap_present else "- Inventory cap value: disabled",
        f"- Overall inventory min / max: {diagnostics['inventory_overall_min']:.3f} / {diagnostics['inventory_overall_max']:.3f}",
        f"- Max absolute inventory reached: {diagnostics['max_abs_inventory_reached']:.3f}",
        f"- Alert level: {diagnostics['inventory_alert_level']:.3f}",
        f"- Agents hitting alert level: {int(diagnostics['inventory_alert_agent_count'])}",
        f"- Fraction of RL decisions at cap: {float(diagnostics.get('inventory_at_cap_fraction', 0.0)):.6f}",
        f"- Fraction of RL decisions near cap: {float(diagnostics.get('inventory_near_cap_fraction', 0.0)):.6f}",
        f"- Diagnosis: {inventory_diagnosis}",
        "",
        "## Actions And Execution",
        f"- Chosen sell / hold / buy: {int(diagnostics['rl_sell_count'])} / {int(diagnostics['rl_hold_count'])} / {int(diagnostics['rl_buy_count'])}",
        f"- Quoter hold / bid / ask / both: {int(diagnostics.get('rl_quote_hold_count', 0.0))} / {int(diagnostics.get('rl_quote_bid_count', 0.0))} / {int(diagnostics.get('rl_quote_ask_count', 0.0))} / {int(diagnostics.get('rl_quote_both_count', 0.0))}",
        f"- Submitted sell / buy actions: {int(diagnostics['submitted_sell_action_count'])} / {int(diagnostics['submitted_buy_action_count'])}",
        f"- Executed sell / buy actions: {int(diagnostics['executed_sell_action_count'])} / {int(diagnostics['executed_buy_action_count'])}",
        f"- Blocked sell / buy actions at cap: {int(diagnostics.get('blocked_sell_action_count', 0.0))} / {int(diagnostics.get('blocked_buy_action_count', 0.0))}",
        f"- Unfilled sell / buy attempts: {int(diagnostics['unfilled_sell_attempt_count'])} / {int(diagnostics['unfilled_buy_attempt_count'])}",
        f"- Sell / buy fill rate: {diagnostics['sell_fill_rate']:.3f} / {diagnostics['buy_fill_rate']:.3f}",
        f"- Executed sell / buy volume: {diagnostics['rl_executed_sell_volume']:.0f} / {diagnostics['rl_executed_buy_volume']:.0f}",
        f"- Passive bid / ask / both submission rate: {float(diagnostics.get('passive_bid_submission_rate', 0.0)):.6f} / {float(diagnostics.get('passive_ask_submission_rate', 0.0)):.6f} / {float(diagnostics.get('passive_both_quote_rate', 0.0)):.6f}",
        f"- Resting quote presence / quote fill rate: {float(diagnostics.get('resting_order_presence_fraction', 0.0)):.6f} / {float(diagnostics.get('quote_fill_rate', 0.0)):.6f}",
        f"- Total RL passive / aggressive orders: {int(diagnostics.get('total_rl_passive_order_count', 0.0))} / {int(diagnostics.get('total_rl_aggressive_order_count', 0.0))}",
        f"- Diagnosis: {sell_diagnosis}",
        "",
        "## Policy Outputs",
        f"- Mean logits sell / hold / buy: {diagnostics['mean_logit_sell']:.6f} / {diagnostics['mean_logit_hold']:.6f} / {diagnostics['mean_logit_buy']:.6f}",
        f"- Mean probs sell / hold / buy: {diagnostics['mean_prob_sell']:.6f} / {diagnostics['mean_prob_hold']:.6f} / {diagnostics['mean_prob_buy']:.6f}",
        f"- Greedy action fractions sell / hold / buy: {diagnostics['deterministic_sell_fraction']:.6f} / {diagnostics['deterministic_hold_fraction']:.6f} / {diagnostics['deterministic_buy_fraction']:.6f}",
        "",
        "## Inventory Transitions",
        f"- 0 -> +1: {int(diagnostics['inventory_transition_0_to_pos1'])}",
        f"- 0 -> -1: {int(diagnostics['inventory_transition_0_to_neg1'])}",
        f"- -1 -> 0: {int(diagnostics['inventory_transition_neg1_to_0'])}",
        f"- +1 -> 0: {int(diagnostics['inventory_transition_pos1_to_0'])}",
        f"- All counted transitions: {diagnostics['inventory_transition_counts']}",
        "",
        "## Reward",
        f"- Mean reward all transitions: {diagnostics['reward_mean_all_transitions']:.6f}",
        f"- Mean reward with nonzero inventory: {diagnostics['reward_mean_nonzero_inventory']:.6f}",
        f"- Mean reward at alert inventory: {diagnostics['reward_mean_alert_inventory']:.6f}",
        f"- Positive reward fraction at alert inventory: {diagnostics['positive_reward_fraction_alert_inventory']:.6f}",
        f"- Mean wealth delta all transitions: {diagnostics['wealth_delta_mean_all_transitions']:.6f}",
        f"- Mean inventory penalty all transitions: {diagnostics['inventory_penalty_mean_all_transitions']:.6f}",
        f"- Mean flat-hold penalty all transitions: {diagnostics['flat_hold_penalty_mean_all_transitions']:.6f}",
        f"- Diagnosis: {reward_diagnosis}",
        "",
        "## Reward By Action",
        f"- Sell reward / wealth delta / inventory penalty / flat-hold penalty: {diagnostics['sell_reward_mean']:.6f} / {diagnostics['sell_wealth_delta_mean']:.6f} / {diagnostics['sell_inventory_penalty_mean']:.6f} / {diagnostics['sell_flat_hold_penalty_mean']:.6f}",
        f"- Hold reward / wealth delta / inventory penalty / flat-hold penalty: {diagnostics['hold_reward_mean']:.6f} / {diagnostics['hold_wealth_delta_mean']:.6f} / {diagnostics['hold_inventory_penalty_mean']:.6f} / {diagnostics['hold_flat_hold_penalty_mean']:.6f}",
        f"- Buy reward / wealth delta / inventory penalty / flat-hold penalty: {diagnostics['buy_reward_mean']:.6f} / {diagnostics['buy_wealth_delta_mean']:.6f} / {diagnostics['buy_inventory_penalty_mean']:.6f} / {diagnostics['buy_flat_hold_penalty_mean']:.6f}",
        f"- Hold minus best trade reward gap: {diagnostics['hold_minus_best_trade_reward_gap']:.6f}",
        f"- Diagnosis: {action_reward_diagnosis}",
        "",
        "## Per-Agent Ending Inventory",
    ]
    if agent_frame.empty:
        lines.append("- No RL agents were present.")
    else:
        for row in agent_frame.sort_values("agent_id").itertuples(index=False):
            lines.append(
                "- "
                f"agent {int(row.agent_id)} ({row.agent_role}): ending={row.ending_inventory:.3f}, "
                f"min={row.inventory_min:.3f}, max={row.inventory_max:.3f}, "
                f"blocked buys={int(row.blocked_buy_actions)}, blocked sells={int(row.blocked_sell_actions)}, "
                f"executed buys={int(row.executed_buy_actions)}, executed sells={int(row.executed_sell_actions)}"
            )
    return "\n".join(lines)


def save_rl_run_report(report: str, path: str | Path) -> None:
    """Write the RL diagnostics report to disk."""

    Path(path).write_text(report, encoding="utf-8")
