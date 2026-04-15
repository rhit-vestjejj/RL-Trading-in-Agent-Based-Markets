"""Minimal analysis helpers for the market experiment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_fraction(mask: Iterable[bool]) -> float:
    """Return the mean of a boolean mask, or NaN when it is empty."""

    series = pd.Series(mask, dtype=bool)
    if series.empty:
        return float("nan")
    return float(series.mean())


def _conditional_fraction(mask: Iterable[bool], valid_mask: Iterable[bool]) -> float:
    """Return the mean of ``mask`` on the subset where ``valid_mask`` is ``True``."""

    valid_series = pd.Series(valid_mask, dtype=bool)
    if valid_series.empty or not bool(valid_series.any()):
        return float("nan")
    mask_series = pd.Series(mask, index=valid_series.index, dtype=bool)
    return float(mask_series.loc[valid_series].mean())


def _coerce_nonnegative_numeric(series: pd.Series) -> pd.Series:
    """Return a numeric series with negative values treated as missing."""

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0.0)


def _first_available_numeric_column(
    frame: pd.DataFrame,
    column_names: tuple[str, ...],
    *,
    default_value: float = float("nan"),
) -> pd.Series:
    """Return the first numeric column present in ``frame`` from ``column_names``."""

    index = frame.index
    for column_name in column_names:
        if column_name in frame.columns:
            return _coerce_nonnegative_numeric(frame[column_name])
    return pd.Series(default_value, index=index, dtype=float)


def _consecutive_true_lengths(mask: Iterable[bool]) -> np.ndarray:
    """Return run lengths for each consecutive ``True`` episode in ``mask``."""

    series = pd.Series(mask, dtype=bool)
    if series.empty or not bool(series.any()):
        return np.array([], dtype=int)

    group_ids = (series != series.shift(fill_value=False)).cumsum()
    lengths = [int(group.sum()) for _, group in series.groupby(group_ids) if bool(group.iloc[0])]
    return np.asarray(lengths, dtype=int)


def _infer_time_step_seconds(frame: pd.DataFrame) -> float:
    """Infer the fixed-grid sampling interval in seconds from the ``time`` column."""

    if "time" not in frame.columns:
        return float("nan")

    time_values = pd.to_numeric(frame["time"], errors="coerce").dropna().to_numpy(dtype=float)
    if time_values.size < 2:
        return float("nan")

    diffs = np.diff(np.sort(time_values))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return float("nan")
    return float(np.median(positive_diffs) / 1_000_000_000.0)


def one_sided_book_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize one-sided-book frequency, severity, and artifact checks.

    ``one_sided_book_fraction`` is defined only on timesteps with positive visible
    top-of-book liquidity, so fully empty-book states are excluded from the
    denominator rather than silently treated as one-sided or two-sided.
    """

    index = frame.index
    best_bid = pd.to_numeric(frame.get("best_bid", pd.Series(np.nan, index=index)), errors="coerce")
    best_ask = pd.to_numeric(frame.get("best_ask", pd.Series(np.nan, index=index)), errors="coerce")
    midprice = pd.to_numeric(frame.get("midprice", pd.Series(np.nan, index=index)), errors="coerce")
    spread = pd.to_numeric(frame.get("spread", pd.Series(np.nan, index=index)), errors="coerce")
    bid_volume = _first_available_numeric_column(frame, ("bid_depth", "bid_volume", "best_bid_volume", "best_bid_size"))
    ask_volume = _first_available_numeric_column(frame, ("ask_depth", "ask_volume", "best_ask_volume", "best_ask_size"))
    visible_liquidity_observed = bid_volume.notna() & ask_volume.notna()
    total_visible_liquidity = bid_volume + ask_volume
    valid_visible_liquidity = visible_liquidity_observed & (total_visible_liquidity > 0.0)
    empty_book = visible_liquidity_observed & (total_visible_liquidity == 0.0)
    visible_bid_present = bid_volume > 0.0
    visible_ask_present = ask_volume > 0.0
    visible_one_sided = valid_visible_liquidity & (visible_bid_present ^ visible_ask_present)
    visible_liquidity_imbalance = (
        (bid_volume - ask_volume).abs() / total_visible_liquidity.where(valid_visible_liquidity)
    )

    missing_bid = best_bid.isna()
    missing_ask = best_ask.isna()
    quoted_side_missing = missing_bid | missing_ask
    missing_quote_one_sided = missing_bid ^ missing_ask
    both_sides_missing = missing_bid & missing_ask

    undefined_midprice = midprice.isna()
    undefined_midprice_due_to_missing_side = undefined_midprice & quoted_side_missing
    midprice_gap_without_missing_side = undefined_midprice & ~quoted_side_missing
    defined_midprice_despite_missing_side = ~undefined_midprice & quoted_side_missing

    undefined_spread = spread.isna()
    undefined_spread_due_to_missing_side = undefined_spread & quoted_side_missing
    spread_gap_without_missing_side = undefined_spread & ~quoted_side_missing
    defined_spread_despite_missing_side = ~undefined_spread & quoted_side_missing

    episode_lengths_steps = _consecutive_true_lengths(visible_one_sided)
    episode_count = int(episode_lengths_steps.size)
    max_duration_steps = int(episode_lengths_steps.max()) if episode_count else 0
    mean_duration_steps = float(episode_lengths_steps.mean()) if episode_count else 0.0
    p50_duration_steps = float(np.percentile(episode_lengths_steps, 50)) if episode_count else 0.0
    p90_duration_steps = float(np.percentile(episode_lengths_steps, 90)) if episode_count else 0.0

    time_step_seconds = _infer_time_step_seconds(frame)
    if np.isnan(time_step_seconds):
        max_duration_seconds = float("nan") if episode_count else 0.0
        mean_duration_seconds = float("nan") if episode_count else 0.0
        p50_duration_seconds = float("nan") if episode_count else 0.0
        p90_duration_seconds = float("nan") if episode_count else 0.0
    else:
        max_duration_seconds = float(max_duration_steps * time_step_seconds)
        mean_duration_seconds = float(mean_duration_steps * time_step_seconds)
        p50_duration_seconds = float(p50_duration_steps * time_step_seconds)
        p90_duration_seconds = float(p90_duration_steps * time_step_seconds)

    return {
        "one_sided_book_fraction": _conditional_fraction(visible_one_sided, valid_visible_liquidity),
        "true_one_sided_book_fraction": _safe_fraction(missing_quote_one_sided),
        "missing_quote_one_sided_fraction": _safe_fraction(missing_quote_one_sided),
        "quoted_side_missing_fraction": _safe_fraction(quoted_side_missing),
        "both_sides_missing_fraction": _safe_fraction(both_sides_missing),
        "missing_bid_fraction": _safe_fraction(missing_bid),
        "missing_ask_fraction": _safe_fraction(missing_ask),
        "empty_book_fraction": _conditional_fraction(empty_book, visible_liquidity_observed),
        "empty_book_timestep_count": float(empty_book.sum()),
        "one_sided_metric_valid_timestep_count": float(valid_visible_liquidity.sum()),
        "one_sided_metric_observed_timestep_count": float(visible_liquidity_observed.sum()),
        "average_bid_volume": float(bid_volume.loc[visible_liquidity_observed].mean())
        if bool(visible_liquidity_observed.any())
        else float("nan"),
        "average_ask_volume": float(ask_volume.loc[visible_liquidity_observed].mean())
        if bool(visible_liquidity_observed.any())
        else float("nan"),
        "average_total_visible_liquidity": float(total_visible_liquidity.loc[visible_liquidity_observed].mean())
        if bool(visible_liquidity_observed.any())
        else float("nan"),
        "average_visible_liquidity_imbalance": float(visible_liquidity_imbalance.loc[valid_visible_liquidity].mean())
        if bool(valid_visible_liquidity.any())
        else float("nan"),
        "undefined_midprice_fraction": _safe_fraction(undefined_midprice_due_to_missing_side),
        "undefined_spread_fraction": _safe_fraction(undefined_spread_due_to_missing_side),
        "midprice_gap_without_missing_side_fraction": _safe_fraction(midprice_gap_without_missing_side),
        "spread_gap_without_missing_side_fraction": _safe_fraction(spread_gap_without_missing_side),
        "defined_midprice_despite_missing_side_fraction": _safe_fraction(defined_midprice_despite_missing_side),
        "defined_spread_despite_missing_side_fraction": _safe_fraction(defined_spread_despite_missing_side),
        "pipeline_issue_flag": float(bool(midprice_gap_without_missing_side.any() or defined_midprice_despite_missing_side.any())),
        "num_one_sided_episodes": float(episode_count),
        "max_consecutive_one_sided_duration": max_duration_seconds,
        "max_consecutive_one_sided_duration_seconds": max_duration_seconds,
        "max_consecutive_one_sided_duration_steps": float(max_duration_steps),
        "mean_one_sided_episode_duration": mean_duration_seconds,
        "mean_one_sided_episode_duration_seconds": mean_duration_seconds,
        "mean_one_sided_episode_duration_steps": mean_duration_steps,
        "one_sided_episode_duration_p50_seconds": p50_duration_seconds,
        "one_sided_episode_duration_p90_seconds": p90_duration_seconds,
        "one_sided_episode_duration_p50_steps": p50_duration_steps,
        "one_sided_episode_duration_p90_steps": p90_duration_steps,
    }


def log_returns(prices: Iterable[float]) -> pd.Series:
    """Compute log returns from a price series."""

    price_series = pd.Series(prices, dtype=float)
    safe_prices = price_series.where(price_series > 0)
    returns = np.log(safe_prices).diff()
    return returns.dropna()


def excess_kurtosis(returns: Iterable[float]) -> float:
    """Compute excess kurtosis using the centered fourth moment."""

    series = pd.Series(returns, dtype=float).dropna()
    if len(series) < 2:
        return float("nan")

    centered = series - series.mean()
    variance = float(np.mean(centered**2))
    if variance == 0.0:
        return 0.0

    fourth_moment = float(np.mean(centered**4))
    return fourth_moment / (variance**2) - 3.0


def squared_return_autocorrelation(returns: Iterable[float], max_lag: int) -> pd.Series:
    """Return the autocorrelation of squared returns up to ``max_lag``."""

    squared = pd.Series(returns, dtype=float).dropna() ** 2
    if squared.empty or float(squared.var(ddof=0)) == 0.0:
        return pd.Series({lag: float("nan") for lag in range(1, max_lag + 1)}, dtype=float)

    values = {}
    for lag in range(1, max_lag + 1):
        if lag >= len(squared):
            values[lag] = float("nan")
            continue
        leading = squared.iloc[:-lag].to_numpy(dtype=float)
        lagged = squared.iloc[lag:].to_numpy(dtype=float)
        if leading.size == 0 or lagged.size == 0:
            values[lag] = float("nan")
            continue
        leading_centered = leading - leading.mean()
        lagged_centered = lagged - lagged.mean()
        leading_scale = float(np.sqrt(np.mean(leading_centered**2)))
        lagged_scale = float(np.sqrt(np.mean(lagged_centered**2)))
        if leading_scale == 0.0 or lagged_scale == 0.0:
            values[lag] = float("nan")
            continue
        values[lag] = float(np.mean(leading_centered * lagged_centered) / (leading_scale * lagged_scale))
    return pd.Series(values, dtype=float)


def return_variance(returns: Iterable[float]) -> float:
    """Return the population variance of returns."""

    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return float("nan")
    return float(np.var(series, ddof=0))


def volatility_clustering_metric(returns: Iterable[float], max_lag: int) -> float:
    """Return a compact clustering proxy from squared-return autocorrelation."""

    autocorrelation = squared_return_autocorrelation(returns, max_lag=max_lag).dropna()
    if autocorrelation.empty:
        return float("nan")
    return float(autocorrelation.mean())


def tail_exposure(returns: Iterable[float], quantile: float = 0.05) -> float:
    """Return left-tail expected shortfall at the given quantile."""

    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return float("nan")
    threshold = float(series.quantile(quantile))
    tail = series[series <= threshold]
    if tail.empty:
        return float("nan")
    return float(tail.mean())


def crash_rate(returns: Iterable[float], threshold: float = -0.05) -> float:
    """Return the fraction of returns at or below the crash threshold."""

    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return float("nan")
    return float((series <= threshold).mean())


def drawdown_series(prices: Iterable[float]) -> pd.Series:
    """Return the drawdown series for a price path."""

    price_series = pd.Series(prices, dtype=float)
    running_peak = price_series.cummax()
    return price_series / running_peak - 1.0


def max_drawdown(prices: Iterable[float]) -> float:
    """Return the most negative drawdown in the series."""

    drawdowns = drawdown_series(prices).dropna()
    if drawdowns.empty:
        return float("nan")
    return float(drawdowns.min())


def average_spread(frame: pd.DataFrame) -> float:
    """Return the average spread from the simulation log."""

    return float(frame["spread"].mean())


def average_relative_spread(frame: pd.DataFrame) -> float:
    """Return the mean top-of-book spread relative to midprice."""

    valid = frame[["spread", "midprice"]].dropna()
    valid = valid[valid["midprice"] > 0]
    if valid.empty:
        return float("nan")
    return float((valid["spread"] / valid["midprice"]).mean())


def average_top_of_book_depth(frame: pd.DataFrame) -> float:
    """Return the average top-of-book depth across both sides."""

    return float(((frame["bid_depth"] + frame["ask_depth"]) / 2.0).mean())


def summarize_market_frame(
    frame: pd.DataFrame,
    *,
    squared_return_lags: int = 5,
    tail_quantile: float = 0.05,
    crash_threshold: float = -0.05,
) -> dict[str, float]:
    """Compute the core analysis metrics for one simulation frame."""

    returns = log_returns(frame["midprice"])
    clustering = squared_return_autocorrelation(returns, max_lag=squared_return_lags)
    summary = {
        "return_variance": return_variance(returns),
        "volatility_clustering": volatility_clustering_metric(
            returns,
            max_lag=squared_return_lags,
        ),
        "excess_kurtosis": excess_kurtosis(returns),
        "tail_exposure": tail_exposure(returns, quantile=tail_quantile),
        "crash_rate": crash_rate(returns, threshold=crash_threshold),
        "max_drawdown": max_drawdown(frame["midprice"]),
        "average_spread": average_spread(frame),
        "average_relative_spread": average_relative_spread(frame),
        "average_top_of_book_depth": average_top_of_book_depth(frame),
    }
    for lag, value in clustering.items():
        summary[f"squared_return_autocorr_lag_{int(lag)}"] = float(value)
    return summary
