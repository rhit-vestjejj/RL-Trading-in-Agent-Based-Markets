"""Baseline market realism diagnostics for phi=0 debugging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analysis import average_spread, average_top_of_book_depth, log_returns


@dataclass(frozen=True)
class RealismThresholds:
    """Configurable thresholds for coarse realism flags."""

    max_price_drift_pct: float = 0.05
    max_fundamental_drift_pct: float = 0.03
    min_realized_volatility: float = 5e-4
    max_realized_volatility: float = 0.08
    max_price_fundamental_correlation: float = 0.995
    max_price_fundamental_mae: float = 0.02
    max_fraction_one_tick_spread: float = 0.80
    min_average_depth: float = 3.0
    max_fraction_near_zero_depth: float = 0.10
    max_fraction_high_imbalance: float = 0.40
    max_signed_flow_autocorr: float = 0.70
    min_signed_flow_std: float = 0.25
    max_signed_flow_same_sign_fraction: float = 0.80


def _safe_series(values: pd.Series | Sequence[float]) -> pd.Series:
    return pd.Series(values, dtype=float).dropna()


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    left_values = aligned.iloc[:, 0].to_numpy(dtype=float)
    right_values = aligned.iloc[:, 1].to_numpy(dtype=float)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    left_scale = float(np.sqrt(np.mean(left_centered**2)))
    right_scale = float(np.sqrt(np.mean(right_centered**2)))
    if left_scale == 0.0 or right_scale == 0.0:
        return float("nan")
    return float(np.mean(left_centered * right_centered) / (left_scale * right_scale))


def _lag1_autocorrelation(values: pd.Series) -> float:
    series = _safe_series(values)
    if len(series) < 3:
        return float("nan")
    leading = series.iloc[:-1].to_numpy(dtype=float)
    lagged = series.iloc[1:].to_numpy(dtype=float)
    leading_centered = leading - leading.mean()
    lagged_centered = lagged - lagged.mean()
    leading_scale = float(np.sqrt(np.mean(leading_centered**2)))
    lagged_scale = float(np.sqrt(np.mean(lagged_centered**2)))
    if leading_scale == 0.0 or lagged_scale == 0.0:
        return float("nan")
    return float(np.mean(leading_centered * lagged_centered) / (leading_scale * lagged_scale))


def _same_sign_fraction(values: pd.Series) -> float:
    series = _safe_series(values)
    if len(series) < 2:
        return float("nan")
    signs = np.sign(series.to_numpy(dtype=float))
    previous = signs[:-1]
    current = signs[1:]
    valid = (previous != 0) & (current != 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(previous[valid] == current[valid]))


def realized_volatility(prices: Sequence[float]) -> float:
    """Return a simple realized-volatility proxy for the session."""

    returns = log_returns(prices)
    if returns.empty:
        return float("nan")
    return float(np.sqrt(np.sum(np.square(returns))))


def _median_time_step_seconds(time_seconds: pd.Series) -> float:
    series = _safe_series(time_seconds)
    if len(series) < 2:
        return float("nan")
    deltas = np.diff(series.to_numpy(dtype=float))
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return float("nan")
    return float(np.median(deltas))


def _quote_run_lengths(best_bid: pd.Series, best_ask: pd.Series) -> np.ndarray:
    quote_frame = pd.DataFrame({"best_bid": best_bid, "best_ask": best_ask}).fillna(-1.0)
    if quote_frame.empty:
        return np.array([], dtype=float)
    quote_change = (
        quote_frame["best_bid"].ne(quote_frame["best_bid"].shift())
        | quote_frame["best_ask"].ne(quote_frame["best_ask"].shift())
    )
    run_ids = quote_change.cumsum()
    return run_ids.value_counts(sort=False).to_numpy(dtype=float)


def _run_lengths(values: pd.Series) -> pd.DataFrame:
    series = values.copy()
    if series.empty:
        return pd.DataFrame(columns=["value", "run_length"])
    change = series.ne(series.shift())
    run_ids = change.cumsum()
    grouped = pd.DataFrame({"value": series, "run_id": run_ids})
    return (
        grouped.groupby("run_id", dropna=False)
        .agg(value=("value", "first"), run_length=("value", "size"))
        .reset_index(drop=True)
    )


def _top_share(values: pd.Series) -> float:
    series = values.dropna()
    if series.empty:
        return float("nan")
    return float(series.value_counts(normalize=True).iloc[0])


def _mean_lifetime_for_type(agent_types: pd.Series, time_step_seconds: float, target_type: str) -> float:
    if np.isnan(time_step_seconds):
        return float("nan")
    runs = _run_lengths(agent_types.fillna("UNKNOWN"))
    target_runs = runs[runs["value"] == target_type]
    if target_runs.empty:
        return float("nan")
    return float((target_runs["run_length"] * time_step_seconds).mean())


def _mean_lifetime_for_non_type(agent_types: pd.Series, time_step_seconds: float, excluded_type: str) -> float:
    if np.isnan(time_step_seconds):
        return float("nan")
    runs = _run_lengths(agent_types.fillna("UNKNOWN"))
    target_runs = runs[(runs["value"] != excluded_type) & (runs["value"] != "UNKNOWN")]
    if target_runs.empty:
        return float("nan")
    return float((target_runs["run_length"] * time_step_seconds).mean())


def _slugify(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


def _share_for_type(agent_types: pd.Series, target_type: str) -> float:
    series = agent_types.dropna()
    if series.empty:
        return float("nan")
    return float((series == target_type).mean())


def compute_realism_diagnostics(
    frame: pd.DataFrame,
    *,
    tick_size: float,
    passive_orders: pd.DataFrame | None = None,
    trade_history: pd.DataFrame | None = None,
    near_zero_depth_threshold: float = 1.0,
    side_depth_threshold: float = 5.0,
    total_depth_threshold: float = 10.0,
    high_imbalance_threshold: float = 0.8,
) -> dict[str, float | str]:
    """Compute the requested realism diagnostics from one completed run."""

    if frame.empty:
        raise ValueError("Cannot compute realism diagnostics from an empty frame.")

    working = frame.copy()
    working["time_seconds"] = pd.to_numeric(working["time"], errors="coerce").fillna(0.0) / 1_000_000_000.0
    midprice = pd.to_numeric(working["midprice"], errors="coerce")
    fundamental = pd.to_numeric(working["fundamental_value"], errors="coerce")
    best_bid = pd.to_numeric(working.get("best_bid"), errors="coerce") if "best_bid" in working.columns else pd.Series(dtype=float)
    best_ask = pd.to_numeric(working.get("best_ask"), errors="coerce") if "best_ask" in working.columns else pd.Series(dtype=float)
    best_bid_agent_id = pd.Series(working.get("best_bid_agent_id")) if "best_bid_agent_id" in working.columns else pd.Series(dtype="object")
    best_ask_agent_id = pd.Series(working.get("best_ask_agent_id")) if "best_ask_agent_id" in working.columns else pd.Series(dtype="object")
    best_bid_agent_type = pd.Series(working.get("best_bid_agent_type")) if "best_bid_agent_type" in working.columns else pd.Series(dtype="object")
    best_ask_agent_type = pd.Series(working.get("best_ask_agent_type")) if "best_ask_agent_type" in working.columns else pd.Series(dtype="object")
    spread = pd.to_numeric(working["spread"], errors="coerce")
    bid_depth = pd.to_numeric(working["bid_depth"], errors="coerce")
    ask_depth = pd.to_numeric(working["ask_depth"], errors="coerce")
    traded_volume = pd.to_numeric(working.get("traded_volume"), errors="coerce") if "traded_volume" in working.columns else pd.Series(dtype=float)
    signed_flow = pd.to_numeric(working.get("signed_order_flow"), errors="coerce") if "signed_order_flow" in working.columns else pd.Series(dtype=float)
    imbalance = pd.to_numeric(working.get("imbalance"), errors="coerce") if "imbalance" in working.columns else pd.Series(dtype=float)
    total_depth = bid_depth + ask_depth
    time_step_seconds = _median_time_step_seconds(working["time_seconds"])
    quote_run_lengths = (
        _quote_run_lengths(best_bid, best_ask)
        if not best_bid.empty and not best_ask.empty
        else np.array([], dtype=float)
    )
    quote_lifetimes = (
        quote_run_lengths * time_step_seconds
        if not np.isnan(time_step_seconds)
        else np.array([], dtype=float)
    )
    session_minutes = float(working["time_seconds"].iloc[-1] - working["time_seconds"].iloc[0]) / 60.0

    price_start = float(midprice.iloc[0])
    price_end = float(midprice.iloc[-1])
    fundamental_start = float(fundamental.iloc[0]) if not fundamental.dropna().empty else float("nan")
    fundamental_end = float(fundamental.iloc[-1]) if not fundamental.dropna().empty else float("nan")

    price_change_abs = price_end - price_start
    price_change_pct = price_change_abs / price_start if price_start else float("nan")
    midprice_changes = midprice.diff().fillna(0.0)
    nonzero_midprice_changes = midprice_changes[midprice_changes != 0.0]
    fundamental_change_abs = fundamental_end - fundamental_start if not np.isnan(fundamental_start) else float("nan")
    fundamental_change_pct = (
        fundamental_change_abs / fundamental_start
        if not np.isnan(fundamental_start) and fundamental_start != 0
        else float("nan")
    )
    price_change_to_fundamental_change_ratio = (
        abs(price_change_abs) / abs(fundamental_change_abs)
        if not np.isnan(fundamental_change_abs) and abs(fundamental_change_abs) > 1e-12
        else float("nan")
    )

    diagnostics: dict[str, float | str] = {
        "session_duration_seconds": float(working["time_seconds"].iloc[-1] - working["time_seconds"].iloc[0]),
        "time_step_seconds": time_step_seconds,
        "price_start": price_start,
        "price_end": price_end,
        "price_change_abs": price_change_abs,
        "price_change_pct": price_change_pct,
        "fundamental_start": fundamental_start,
        "fundamental_end": fundamental_end,
        "fundamental_change_abs": fundamental_change_abs,
        "fundamental_change_pct": fundamental_change_pct,
        "price_change_to_fundamental_change_ratio": price_change_to_fundamental_change_ratio,
        "realized_volatility": realized_volatility(midprice),
        "average_spread": average_spread(working),
        "spread_variance": float(spread.var(ddof=0)) if not spread.dropna().empty else float("nan"),
        "average_depth": average_top_of_book_depth(working),
        "fraction_spread_one_tick": float(np.mean(np.isclose(spread, tick_size, atol=tick_size * 0.05))),
        "fraction_spread_two_ticks": float(np.mean(np.isclose(spread, 2.0 * tick_size, atol=tick_size * 0.05))),
        "fraction_spread_three_plus_ticks": float(np.mean(spread >= 3.0 * tick_size - tick_size * 0.05)),
        "fraction_near_zero_depth": float(np.mean((bid_depth <= near_zero_depth_threshold) | (ask_depth <= near_zero_depth_threshold))),
        "fraction_bid_depth_below_threshold": float(np.mean(bid_depth < side_depth_threshold)),
        "fraction_ask_depth_below_threshold": float(np.mean(ask_depth < side_depth_threshold)),
        "fraction_total_depth_below_threshold": float(np.mean(total_depth < total_depth_threshold)),
        "fraction_high_imbalance": float(np.mean(np.abs(imbalance) >= high_imbalance_threshold)) if not imbalance.dropna().empty else float("nan"),
        "depth_persistence_lag1": _lag1_autocorrelation(total_depth),
        "imbalance_persistence_lag1": _lag1_autocorrelation(imbalance),
        "mean_abs_depth_change": float(total_depth.diff().abs().dropna().mean()) if len(total_depth) > 1 else float("nan"),
        "mean_quote_lifetime_seconds": float(np.mean(quote_lifetimes)) if len(quote_lifetimes) > 0 else float("nan"),
        "median_quote_lifetime_seconds": float(np.median(quote_lifetimes)) if len(quote_lifetimes) > 0 else float("nan"),
        "quote_refresh_rate_per_minute": (
            float((len(quote_run_lengths) - 1) / session_minutes)
            if session_minutes > 0 and len(quote_run_lengths) > 0
            else float("nan")
        ),
        "best_bid_top_owner_share": _top_share(best_bid_agent_id),
        "best_ask_top_owner_share": _top_share(best_ask_agent_id),
        "best_bid_mm_share": float((best_bid_agent_type == "AdaptiveMarketMaker").mean()) if not best_bid_agent_type.empty else float("nan"),
        "best_ask_mm_share": float((best_ask_agent_type == "AdaptiveMarketMaker").mean()) if not best_ask_agent_type.empty else float("nan"),
        "best_bid_mm_quote_lifetime_seconds": _mean_lifetime_for_type(best_bid_agent_type, time_step_seconds, "AdaptiveMarketMaker"),
        "best_ask_mm_quote_lifetime_seconds": _mean_lifetime_for_type(best_ask_agent_type, time_step_seconds, "AdaptiveMarketMaker"),
        "best_bid_non_mm_quote_lifetime_seconds": _mean_lifetime_for_non_type(best_bid_agent_type, time_step_seconds, "AdaptiveMarketMaker"),
        "best_ask_non_mm_quote_lifetime_seconds": _mean_lifetime_for_non_type(best_ask_agent_type, time_step_seconds, "AdaptiveMarketMaker"),
        "signed_order_flow_autocorr_lag1": _lag1_autocorrelation(signed_flow),
        "signed_order_flow_same_sign_fraction": _same_sign_fraction(signed_flow),
        "signed_order_flow_std": float(_safe_series(signed_flow).std(ddof=0)) if not _safe_series(signed_flow).empty else float("nan"),
        "signed_order_flow_burstiness": (
            float(np.var(_safe_series(signed_flow), ddof=0) / max(_safe_series(signed_flow).abs().mean(), 1e-9))
            if not _safe_series(signed_flow).empty
            else float("nan")
        ),
        "traded_volume_burstiness": (
            float(np.var(_safe_series(traded_volume), ddof=0) / max(float(_safe_series(traded_volume).mean()), 1e-9))
            if not _safe_series(traded_volume).empty
            else float("nan")
        ),
        "fraction_zero_midprice_change": float((midprice_changes == 0.0).mean()),
        "average_nonzero_midprice_change": float(nonzero_midprice_changes.abs().mean()) if not nonzero_midprice_changes.empty else float("nan"),
        "midprice_fundamental_correlation": _safe_corr(midprice, fundamental),
        "midprice_fundamental_mae": float((midprice - fundamental).abs().mean()),
        "midprice_fundamental_max_abs_error": float((midprice - fundamental).abs().max()),
    }

    agent_types = sorted(
        {
            str(agent_type)
            for agent_type in pd.concat(
                [best_bid_agent_type.dropna(), best_ask_agent_type.dropna()],
                ignore_index=True,
            ).unique()
            if str(agent_type) != "UNKNOWN"
        }
    )
    for agent_type in agent_types:
        slug = _slugify(agent_type)
        diagnostics[f"best_bid_share_{slug}"] = _share_for_type(
            best_bid_agent_type,
            agent_type,
        )
        diagnostics[f"best_ask_share_{slug}"] = _share_for_type(
            best_ask_agent_type,
            agent_type,
        )
        diagnostics[f"best_bid_lifetime_seconds_{slug}"] = _mean_lifetime_for_type(
            best_bid_agent_type,
            time_step_seconds,
            agent_type,
        )
        diagnostics[f"best_ask_lifetime_seconds_{slug}"] = _mean_lifetime_for_type(
            best_ask_agent_type,
            time_step_seconds,
            agent_type,
        )

    diagnostics["fraction_non_mm_best_bid"] = (
        1.0 - float(diagnostics["best_bid_mm_share"])
        if not np.isnan(float(diagnostics["best_bid_mm_share"]))
        else float("nan")
    )
    diagnostics["fraction_non_mm_best_ask"] = (
        1.0 - float(diagnostics["best_ask_mm_share"])
        if not np.isnan(float(diagnostics["best_ask_mm_share"]))
        else float("nan")
    )

    if trade_history is not None and not trade_history.empty:
        passive_types = trade_history["passive_agent_type"].fillna("UNKNOWN")
        total_trade_count = len(trade_history)
        total_trade_volume = float(pd.to_numeric(trade_history["quantity"], errors="coerce").fillna(0.0).sum())
        mm_trade_volume = float(
            pd.to_numeric(
                trade_history.loc[passive_types == "AdaptiveMarketMaker", "quantity"],
                errors="coerce",
            ).fillna(0.0).sum()
        )
        diagnostics["fraction_trade_count_against_mm_quotes"] = float(
            (passive_types == "AdaptiveMarketMaker").mean()
        )
        diagnostics["fraction_trade_count_against_non_mm_quotes"] = (
            1.0 - float(diagnostics["fraction_trade_count_against_mm_quotes"])
        )
        diagnostics["fraction_traded_volume_against_mm_quotes"] = (
            mm_trade_volume / total_trade_volume if total_trade_volume > 0 else float("nan")
        )
        diagnostics["fraction_traded_volume_against_non_mm_quotes"] = (
            1.0 - float(diagnostics["fraction_traded_volume_against_mm_quotes"])
            if total_trade_volume > 0
            else float("nan")
        )
        diagnostics["trade_count"] = float(total_trade_count)
        diagnostics["traded_volume_total"] = total_trade_volume

    if passive_orders is not None and not passive_orders.empty:
        passive_only = passive_orders[passive_orders["is_passive"] == True].copy()
        session_minutes = max(float(diagnostics["session_duration_seconds"]) / 60.0, 1e-9)
        for agent_type in sorted(passive_only["agent_type"].dropna().astype(str).unique()):
            slug = _slugify(str(agent_type))
            subset = passive_only[passive_only["agent_type"] == agent_type].copy()
            diagnostics[f"passive_order_rate_per_minute_{slug}"] = float(len(subset) / session_minutes)
            diagnostics[f"passive_mean_order_size_{slug}"] = (
                float(pd.to_numeric(subset["quantity"], errors="coerce").mean())
                if "quantity" in subset.columns
                else float("nan")
            )
            diagnostics[f"passive_fill_rate_{slug}"] = (
                float(pd.to_numeric(subset["was_executed"], errors="coerce").mean())
                if "was_executed" in subset.columns
                else float("nan")
            )
            diagnostics[f"passive_mean_lifetime_seconds_{slug}"] = (
                float(pd.to_numeric(subset["time_to_terminal_event_seconds"], errors="coerce").mean())
                if "time_to_terminal_event_seconds" in subset.columns
                else float("nan")
            )
            if "rested_before_execution" in subset.columns:
                diagnostics[f"passive_rest_fraction_{slug}"] = float(
                    pd.to_numeric(subset["rested_before_execution"], errors="coerce").mean()
                )
            if "executed_quickly_after_rest" in subset.columns:
                diagnostics[f"passive_quick_execution_fraction_{slug}"] = float(
                    pd.to_numeric(
                        subset["executed_quickly_after_rest"],
                        errors="coerce",
                    ).mean()
                )
            diagnostics[f"passive_mean_distance_ticks_{slug}"] = float(
                pd.to_numeric(subset["same_side_distance_ticks"], errors="coerce").mean()
            )
            diagnostics[f"passive_join_share_{slug}"] = float(
                (subset["placement_bucket"] == "join").mean()
            )
            diagnostics[f"passive_improve_share_{slug}"] = float(
                (subset["placement_bucket"] == "improve").mean()
            )
            diagnostics[f"passive_behind_share_{slug}"] = float(
                (subset["placement_bucket"] == "behind").mean()
            )

        non_mm_passive = passive_only[
            passive_only["agent_type"] != "AdaptiveMarketMaker"
        ].copy()
        if not non_mm_passive.empty:
            diagnostics["fraction_non_mm_passive_join_inside"] = float(
                (non_mm_passive["placement_bucket"] == "join").mean()
            )
            diagnostics["fraction_non_mm_passive_improve_inside"] = float(
                (non_mm_passive["placement_bucket"] == "improve").mean()
            )
            diagnostics["fraction_non_mm_passive_behind_inside"] = float(
                (non_mm_passive["placement_bucket"] == "behind").mean()
            )
            diagnostics["mean_non_mm_passive_distance_ticks"] = float(
                pd.to_numeric(
                    non_mm_passive["same_side_distance_ticks"],
                    errors="coerce",
                ).mean()
            )
            rested_series = (
                pd.to_numeric(
                    non_mm_passive["rested_before_execution"],
                    errors="coerce",
                )
                if "rested_before_execution" in non_mm_passive.columns
                else pd.to_numeric(
                    non_mm_passive["was_accepted"],
                    errors="coerce",
                )
            )
            quick_series = (
                pd.to_numeric(
                    non_mm_passive["executed_quickly"],
                    errors="coerce",
                )
                if "executed_quickly" in non_mm_passive.columns
                else pd.to_numeric(
                    non_mm_passive["executed_quickly_after_rest"],
                    errors="coerce",
                )
            )
            diagnostics["fraction_non_mm_passive_orders_resting"] = float(rested_series.mean())
            diagnostics["fraction_non_mm_passive_orders_executed_quickly"] = float(
                quick_series.mean()
            )

    return diagnostics


def flag_realism_pathologies(
    diagnostics: Mapping[str, float | str],
    thresholds: RealismThresholds | None = None,
) -> dict[str, bool]:
    """Flag the pathologies requested in the debugging brief."""

    thresholds = thresholds or RealismThresholds()
    realized_vol = float(diagnostics["realized_volatility"])
    mid_corr = float(diagnostics["midprice_fundamental_correlation"])
    mid_mae = float(diagnostics["midprice_fundamental_mae"])
    spread_variance = float(diagnostics["spread_variance"])

    return {
        "excessive_price_drift": abs(float(diagnostics["price_change_pct"])) > thresholds.max_price_drift_pct,
        "excessive_fundamental_drift": abs(float(diagnostics["fundamental_change_pct"])) > thresholds.max_fundamental_drift_pct,
        "price_glued_to_fundamental": (
            not np.isnan(mid_corr)
            and mid_corr > thresholds.max_price_fundamental_correlation
            and mid_mae < thresholds.max_price_fundamental_mae
        ),
        "market_frozen_or_too_smooth": (
            np.isnan(realized_vol)
            or realized_vol < thresholds.min_realized_volatility
            or (not np.isnan(spread_variance) and spread_variance == 0.0)
        ),
        "spread_mechanically_stuck": float(diagnostics["fraction_spread_one_tick"]) > thresholds.max_fraction_one_tick_spread,
        "chronically_thin_book": (
            float(diagnostics["average_depth"]) < thresholds.min_average_depth
            or float(diagnostics["fraction_near_zero_depth"]) > thresholds.max_fraction_near_zero_depth
        ),
        "persistent_one_sided_depth_imbalance": float(diagnostics["fraction_high_imbalance"]) > thresholds.max_fraction_high_imbalance,
        "order_flow_too_blocky": (
            (not np.isnan(float(diagnostics["signed_order_flow_autocorr_lag1"])) and float(diagnostics["signed_order_flow_autocorr_lag1"]) > thresholds.max_signed_flow_autocorr)
            or (not np.isnan(float(diagnostics["signed_order_flow_std"])) and float(diagnostics["signed_order_flow_std"]) < thresholds.min_signed_flow_std)
            or (not np.isnan(float(diagnostics["signed_order_flow_same_sign_fraction"])) and float(diagnostics["signed_order_flow_same_sign_fraction"]) > thresholds.max_signed_flow_same_sign_fraction)
        ),
    }


def time_scale_assessment(
    diagnostics: Mapping[str, float | str],
    thresholds: RealismThresholds | None = None,
) -> tuple[str, str]:
    """Assess whether scale problems come from the clock, price process, or fundamentals."""

    thresholds = thresholds or RealismThresholds()
    session_seconds = float(diagnostics["session_duration_seconds"])
    price_drift = abs(float(diagnostics["price_change_pct"]))
    fundamental_drift = abs(float(diagnostics["fundamental_change_pct"]))
    realized_vol = float(diagnostics["realized_volatility"])

    if fundamental_drift > thresholds.max_fundamental_drift_pct and price_drift > thresholds.max_price_drift_pct:
        return (
            "both_scales_too_large",
            f"The session spans {session_seconds:.0f} literal simulator seconds, and both price and fundamental drift are too large for that horizon.",
        )
    if fundamental_drift > thresholds.max_fundamental_drift_pct:
        return (
            "fundamental_scale_too_large",
            f"The session spans {session_seconds:.0f} literal simulator seconds, and the fundamental is drifting too far over that horizon.",
        )
    if price_drift > thresholds.max_price_drift_pct or (not np.isnan(realized_vol) and realized_vol > thresholds.max_realized_volatility):
        return (
            "price_scale_or_trader_aggressiveness_too_large",
            f"The session spans {session_seconds:.0f} literal simulator seconds, so the price process or trader aggressiveness is too strong for a literal-second interpretation.",
        )
    return (
        "literal_second_interpretation_plausible",
        f"The session spans {session_seconds:.0f} literal simulator seconds and the aggregate price scale is plausible at that horizon.",
    )


def build_realism_report(
    diagnostics: Mapping[str, float | str],
    flags: Mapping[str, bool],
    *,
    thresholds: RealismThresholds | None = None,
) -> str:
    """Build a compact realism diagnostics report."""

    thresholds = thresholds or RealismThresholds()
    scale_label, scale_message = time_scale_assessment(diagnostics, thresholds=thresholds)
    flagged = [name for name, active in flags.items() if active]

    lines = [
        "# Realism Diagnostics Report",
        "",
        "## Session sanity",
        f"- Session duration (simulator seconds): {float(diagnostics['session_duration_seconds']):.1f}",
        f"- Logging step: {float(diagnostics['time_step_seconds']):.3f} seconds",
        f"- Price move: {float(diagnostics['price_change_abs']):.4f} ({100.0 * float(diagnostics['price_change_pct']):.2f}%)",
        f"- Fundamental move: {float(diagnostics['fundamental_change_abs']):.4f} ({100.0 * float(diagnostics['fundamental_change_pct']):.2f}%)",
        f"- Price/fundamental move ratio: {float(diagnostics['price_change_to_fundamental_change_ratio']):.4f}",
        f"- Realized volatility proxy: {float(diagnostics['realized_volatility']):.6f}",
        f"- Time-scale assessment: {scale_label}",
        f"- Interpretation: {scale_message}",
        "",
        "## Market quality",
        f"- Average spread: {float(diagnostics['average_spread']):.6f}",
        f"- Spread variance: {float(diagnostics['spread_variance']):.6f}",
        f"- Average depth: {float(diagnostics['average_depth']):.3f}",
        f"- Fraction one-tick spread: {100.0 * float(diagnostics['fraction_spread_one_tick']):.1f}%",
        f"- Fraction two-tick spread: {100.0 * float(diagnostics['fraction_spread_two_ticks']):.1f}%",
        f"- Fraction three-plus-tick spread: {100.0 * float(diagnostics['fraction_spread_three_plus_ticks']):.1f}%",
        f"- Fraction near-zero depth: {100.0 * float(diagnostics['fraction_near_zero_depth']):.1f}%",
        f"- Fraction bid depth below threshold: {100.0 * float(diagnostics['fraction_bid_depth_below_threshold']):.1f}%",
        f"- Fraction ask depth below threshold: {100.0 * float(diagnostics['fraction_ask_depth_below_threshold']):.1f}%",
        f"- Fraction total depth below threshold: {100.0 * float(diagnostics['fraction_total_depth_below_threshold']):.1f}%",
        f"- Fraction high imbalance: {100.0 * float(diagnostics['fraction_high_imbalance']):.1f}%",
        f"- Depth persistence lag-1: {float(diagnostics['depth_persistence_lag1']):.6f}",
        f"- Imbalance persistence lag-1: {float(diagnostics['imbalance_persistence_lag1']):.6f}",
        f"- Mean absolute depth change: {float(diagnostics['mean_abs_depth_change']):.6f}",
        f"- Mean top-quote lifetime: {float(diagnostics['mean_quote_lifetime_seconds']):.3f} seconds",
        f"- Median top-quote lifetime: {float(diagnostics['median_quote_lifetime_seconds']):.3f} seconds",
        f"- Quote refresh rate: {float(diagnostics['quote_refresh_rate_per_minute']):.3f} changes/minute",
        f"- Best bid top-owner share: {100.0 * float(diagnostics['best_bid_top_owner_share']):.1f}%",
        f"- Best ask top-owner share: {100.0 * float(diagnostics['best_ask_top_owner_share']):.1f}%",
        f"- Best bid MM share: {100.0 * float(diagnostics['best_bid_mm_share']):.1f}%",
        f"- Best ask MM share: {100.0 * float(diagnostics['best_ask_mm_share']):.1f}%",
        f"- Fraction non-MM best bid: {100.0 * float(diagnostics.get('fraction_non_mm_best_bid', float('nan'))):.1f}%",
        f"- Fraction non-MM best ask: {100.0 * float(diagnostics.get('fraction_non_mm_best_ask', float('nan'))):.1f}%",
        "",
        "## Price vs fundamental",
        f"- Correlation: {float(diagnostics['midprice_fundamental_correlation']):.6f}",
        f"- Mean absolute error: {float(diagnostics['midprice_fundamental_mae']):.6f}",
        f"- Max absolute error: {float(diagnostics['midprice_fundamental_max_abs_error']):.6f}",
        "",
        "## Order flow",
        f"- Signed flow lag-1 autocorrelation: {float(diagnostics['signed_order_flow_autocorr_lag1']):.6f}",
        f"- Signed flow same-sign fraction: {float(diagnostics['signed_order_flow_same_sign_fraction']):.6f}",
        f"- Signed flow standard deviation: {float(diagnostics['signed_order_flow_std']):.6f}",
        f"- Signed flow burstiness: {float(diagnostics['signed_order_flow_burstiness']):.6f}",
        f"- Traded-volume burstiness: {float(diagnostics['traded_volume_burstiness']):.6f}",
        f"- Fraction zero midprice change: {100.0 * float(diagnostics['fraction_zero_midprice_change']):.1f}%",
        f"- Average nonzero midprice change: {float(diagnostics['average_nonzero_midprice_change']):.6f}",
        f"- Fraction traded volume against MM quotes: {100.0 * float(diagnostics.get('fraction_traded_volume_against_mm_quotes', float('nan'))):.1f}%",
        f"- Fraction traded volume against non-MM quotes: {100.0 * float(diagnostics.get('fraction_traded_volume_against_non_mm_quotes', float('nan'))):.1f}%",
        "",
        "## Passive competition",
        f"- Fraction non-MM passive orders resting: {100.0 * float(diagnostics.get('fraction_non_mm_passive_orders_resting', float('nan'))):.1f}%",
        f"- Fraction non-MM passive orders executed quickly: {100.0 * float(diagnostics.get('fraction_non_mm_passive_orders_executed_quickly', float('nan'))):.1f}%",
        f"- Fraction non-MM passive orders joining inside: {100.0 * float(diagnostics.get('fraction_non_mm_passive_join_inside', float('nan'))):.1f}%",
        f"- Fraction non-MM passive orders improving inside: {100.0 * float(diagnostics.get('fraction_non_mm_passive_improve_inside', float('nan'))):.1f}%",
        f"- Fraction non-MM passive orders sitting behind inside: {100.0 * float(diagnostics.get('fraction_non_mm_passive_behind_inside', float('nan'))):.1f}%",
        "",
        "## Flagged pathologies",
        f"- {', '.join(flagged) if flagged else 'none'}",
    ]
    return "\n".join(lines) + "\n"


def save_realism_report(report: str, output_path: str | Path) -> Path:
    """Persist a realism report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
