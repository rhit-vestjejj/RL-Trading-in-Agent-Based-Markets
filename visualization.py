"""Minimal time-series visualization helpers for market logs."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


def _load_matplotlib(show_plot: bool):
    cache_root = Path(tempfile.gettempdir()) / "rl_market_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    if not show_plot and "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def load_market_frame(path: str | Path) -> pd.DataFrame:
    """Load a saved simulation log."""

    return pd.read_csv(path)


def time_in_seconds(frame: pd.DataFrame) -> pd.Series:
    """Convert nanosecond log time to seconds from market open."""

    return pd.to_numeric(frame["time"], errors="coerce").fillna(0.0) / 1_000_000_000.0


def rl_metric_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
    """Return RL metric columns with the requested suffix."""

    return [column for column in frame.columns if column.endswith(suffix)]


def one_sided_book_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a mask for timestamps where one side of the book is missing."""

    if "best_bid" not in frame.columns or "best_ask" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    best_bid = pd.to_numeric(frame["best_bid"], errors="coerce")
    best_ask = pd.to_numeric(frame["best_ask"], errors="coerce")
    return best_bid.isna() | best_ask.isna()


def rl_pnl_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return relative P&L from a raw RL wealth column."""

    wealth = pd.to_numeric(frame[column], errors="coerce")
    valid = wealth.dropna()
    if valid.empty:
        return wealth.fillna(0.0)
    nonzero = valid[valid != 0]
    baseline_index = nonzero.index[0] if not nonzero.empty else valid.index[0]
    baseline = float(wealth.loc[baseline_index])
    pnl = wealth - baseline
    pnl.loc[pnl.index < baseline_index] = np.nan
    return pnl


def rl_summary_series(frame: pd.DataFrame, columns: list[str], *, pnl: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return per-agent traces plus mean/min/max summaries."""

    if not columns:
        empty = pd.DataFrame(index=frame.index)
        nan_series = pd.Series(float("nan"), index=frame.index)
        return empty, nan_series, nan_series, nan_series

    traces = pd.DataFrame(index=frame.index)
    for column in columns:
        traces[column] = rl_pnl_series(frame, column) if pnl else pd.to_numeric(frame[column], errors="coerce")
    return traces, traces.mean(axis=1), traces.min(axis=1), traces.max(axis=1)


def _market_panel_count(frame: pd.DataFrame) -> int:
    has_rl_state = bool(rl_metric_columns(frame, "_inventory") or rl_metric_columns(frame, "_wealth"))
    return 6 if has_rl_state else 4


def _create_market_axes(frame: pd.DataFrame, *, title: str, show_plot: bool):
    plt = _load_matplotlib(show_plot=show_plot)
    num_panels = _market_panel_count(frame)
    fig, axes = plt.subplots(num_panels, 1, figsize=(11, 2.6 * num_panels), sharex=True)
    if num_panels == 1:
        axes = [axes]
    fig.suptitle(title)
    return plt, fig, axes


def create_market_figure(
    frame: pd.DataFrame,
    *,
    title: str = "Market Replay",
    show_plot: bool = False,
):
    """Create a compact multi-panel view of market evolution over time."""

    plt, fig, axes = _create_market_axes(frame, title=title, show_plot=show_plot)
    time_seconds = time_in_seconds(frame)
    has_rl_state = _market_panel_count(frame) == 6

    price_ax = axes[0]
    price_ax.plot(time_seconds, frame["midprice"], label="midprice", linewidth=1.6)
    if "fundamental_value" in frame.columns:
        price_ax.plot(
            time_seconds,
            frame["fundamental_value"],
            label="fundamental",
            linewidth=1.2,
            alpha=0.8,
        )
    price_ax.set_ylabel("Price")
    price_ax.legend(loc="upper left")
    price_ax.grid(True, alpha=0.3)

    spread_ax = axes[1]
    spread_ax.plot(time_seconds, frame["spread"], color="tab:red", linewidth=1.3)
    one_sided = one_sided_book_mask(frame)
    if bool(one_sided.any()):
        valid_spread = pd.to_numeric(frame["spread"], errors="coerce").dropna()
        marker_level = float(valid_spread.max()) if not valid_spread.empty else 0.0
        spread_ax.scatter(
            time_seconds[one_sided],
            pd.Series(marker_level, index=time_seconds[one_sided].index),
            color="tab:gray",
            marker="x",
            s=18,
            label="one-sided book",
        )
    spread_ax.set_ylabel("Spread")
    if bool(one_sided.any()):
        spread_ax.legend(loc="upper left")
    spread_ax.grid(True, alpha=0.3)

    depth_ax = axes[2]
    depth_ax.plot(time_seconds, frame["bid_depth"], label="bid depth", linewidth=1.2)
    depth_ax.plot(time_seconds, frame["ask_depth"], label="ask depth", linewidth=1.2)
    depth_ax.set_ylabel("Depth")
    depth_ax.legend(loc="upper left")
    depth_ax.grid(True, alpha=0.3)

    flow_ax = axes[3]
    flow_ax.plot(time_seconds, frame["traded_volume"], label="traded volume", linewidth=1.2)
    if "signed_order_flow" in frame.columns:
        flow_ax.plot(
            time_seconds,
            frame["signed_order_flow"],
            label="signed order flow",
            linewidth=1.2,
        )
    flow_ax.set_ylabel("Flow")
    flow_ax.legend(loc="upper left")
    flow_ax.grid(True, alpha=0.3)

    next_axis = 4
    inventory_columns = rl_metric_columns(frame, "_inventory")
    wealth_columns = rl_metric_columns(frame, "_wealth")

    if has_rl_state:
        inventory_ax = axes[next_axis]
        inventory_traces, inventory_mean, inventory_min, inventory_max = rl_summary_series(frame, inventory_columns, pnl=False)
        for column in inventory_traces.columns:
            inventory_ax.plot(time_seconds, inventory_traces[column], linewidth=0.9, alpha=0.18, color="tab:blue")
        if inventory_columns:
            inventory_ax.fill_between(time_seconds, inventory_min, inventory_max, color="tab:blue", alpha=0.12)
            inventory_ax.plot(time_seconds, inventory_mean, linewidth=1.8, color="tab:blue")
        inventory_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.45)
        inventory_ax.set_ylabel("Inventory")
        inventory_ax.grid(True, alpha=0.3)
        next_axis += 1

        wealth_ax = axes[next_axis]
        wealth_traces, wealth_mean, wealth_min, wealth_max = rl_summary_series(frame, wealth_columns, pnl=True)
        for column in wealth_traces.columns:
            wealth_ax.plot(time_seconds, wealth_traces[column], linewidth=0.9, alpha=0.18, color="tab:green")
        if wealth_columns:
            wealth_ax.fill_between(time_seconds, wealth_min, wealth_max, color="tab:green", alpha=0.12)
            wealth_ax.plot(time_seconds, wealth_mean, linewidth=1.8, color="tab:green")
        wealth_ax.set_ylabel("RL P&L")
        wealth_ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Seconds from market open")
    fig.tight_layout()
    return fig


def build_market_animation(
    frame: pd.DataFrame,
    *,
    title: str = "Market Replay",
    interval_ms: int = 40,
    tail_seconds: float | None = None,
    show_plot: bool = True,
):
    """Build a simple animated replay of the market state over time."""

    plt, fig, axes = _create_market_axes(frame, title=title, show_plot=show_plot)
    from matplotlib.animation import FuncAnimation

    time_seconds = time_in_seconds(frame)
    has_rl_state = _market_panel_count(frame) == 6

    price_ax = axes[0]
    (mid_line,) = price_ax.plot([], [], label="midprice", linewidth=1.8)
    fundamental_line = None
    if "fundamental_value" in frame.columns:
        (fundamental_line,) = price_ax.plot([], [], label="fundamental", linewidth=1.2, alpha=0.8)
    price_marker = price_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    status_text = price_ax.text(
        0.01,
        0.97,
        "",
        transform=price_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    price_ax.set_ylabel("Price")
    price_ax.legend(loc="upper left")
    price_ax.grid(True, alpha=0.3)

    spread_ax = axes[1]
    (spread_line,) = spread_ax.plot([], [], color="tab:red", linewidth=1.3)
    spread_marker = spread_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    spread_ax.set_ylabel("Spread")
    spread_ax.grid(True, alpha=0.3)

    depth_ax = axes[2]
    (bid_depth_line,) = depth_ax.plot([], [], label="bid depth", linewidth=1.2)
    (ask_depth_line,) = depth_ax.plot([], [], label="ask depth", linewidth=1.2)
    depth_marker = depth_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    depth_ax.set_ylabel("Depth")
    depth_ax.legend(loc="upper left")
    depth_ax.grid(True, alpha=0.3)

    flow_ax = axes[3]
    (volume_line,) = flow_ax.plot([], [], label="traded volume", linewidth=1.2)
    flow_lines = [volume_line]
    signed_flow_line = None
    if "signed_order_flow" in frame.columns:
        (signed_flow_line,) = flow_ax.plot([], [], label="signed order flow", linewidth=1.2)
        flow_lines.append(signed_flow_line)
    flow_marker = flow_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    flow_ax.set_ylabel("Flow")
    flow_ax.legend(loc="upper left")
    flow_ax.grid(True, alpha=0.3)

    inventory_lines: list[tuple[str, object]] = []
    wealth_lines: list[tuple[str, object]] = []
    inventory_mean_line = None
    inventory_min_line = None
    inventory_max_line = None
    wealth_mean_line = None
    inventory_traces = pd.DataFrame(index=frame.index)
    wealth_traces = pd.DataFrame(index=frame.index)
    if has_rl_state:
        inventory_ax = axes[4]
        inventory_columns = rl_metric_columns(frame, "_inventory")
        inventory_traces, inventory_mean, inventory_min, inventory_max = rl_summary_series(frame, inventory_columns, pnl=False)
        for column in inventory_columns:
            line = inventory_ax.plot([], [], linewidth=0.9, alpha=0.18, color="tab:blue")[0]
            inventory_lines.append((column, line))
        if inventory_columns:
            inventory_mean_line = inventory_ax.plot([], [], linewidth=1.8, color="tab:blue")[0]
            inventory_min_line = inventory_ax.plot([], [], linewidth=1.0, linestyle="--", alpha=0.45, color="tab:blue")[0]
            inventory_max_line = inventory_ax.plot([], [], linewidth=1.0, linestyle="--", alpha=0.45, color="tab:blue")[0]
        inventory_marker = inventory_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        inventory_ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.45)
        inventory_ax.set_ylabel("Inventory")
        inventory_ax.grid(True, alpha=0.3)

        wealth_ax = axes[5]
        wealth_columns = rl_metric_columns(frame, "_wealth")
        wealth_traces, wealth_mean, _, _ = rl_summary_series(frame, wealth_columns, pnl=True)
        for column in wealth_columns:
            line = wealth_ax.plot([], [], linewidth=0.9, alpha=0.18, color="tab:green")[0]
            wealth_lines.append((column, line))
        if wealth_columns:
            wealth_mean_line = wealth_ax.plot([], [], linewidth=1.8, color="tab:green")[0]
        wealth_marker = wealth_ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
        wealth_ax.set_ylabel("RL P&L")
        wealth_ax.grid(True, alpha=0.3)
    else:
        inventory_marker = None
        wealth_marker = None

    axes[-1].set_xlabel("Seconds from market open")

    numeric_columns = {
        "midprice": pd.to_numeric(frame["midprice"], errors="coerce"),
        "fundamental_value": pd.to_numeric(frame.get("fundamental_value"), errors="coerce") if "fundamental_value" in frame.columns else None,
        "spread": pd.to_numeric(frame["spread"], errors="coerce"),
        "bid_depth": pd.to_numeric(frame["bid_depth"], errors="coerce"),
        "ask_depth": pd.to_numeric(frame["ask_depth"], errors="coerce"),
        "traded_volume": pd.to_numeric(frame["traded_volume"], errors="coerce"),
        "signed_order_flow": pd.to_numeric(frame.get("signed_order_flow"), errors="coerce") if "signed_order_flow" in frame.columns else None,
    }
    for column in inventory_traces.columns:
        numeric_columns[column] = inventory_traces[column]
    for column in wealth_traces.columns:
        numeric_columns[column] = wealth_traces[column]
    if has_rl_state and not inventory_traces.empty:
        numeric_columns["inventory_mean"] = inventory_traces.mean(axis=1)
        numeric_columns["inventory_min"] = inventory_traces.min(axis=1)
        numeric_columns["inventory_max"] = inventory_traces.max(axis=1)
    if has_rl_state and not wealth_traces.empty:
        numeric_columns["wealth_mean"] = wealth_traces.mean(axis=1)

    def _axis_limits(series_list: list[pd.Series], padding_ratio: float = 0.05) -> tuple[float, float]:
        valid = pd.concat([series.dropna() for series in series_list if series is not None], ignore_index=True)
        if valid.empty:
            return -1.0, 1.0
        lower = float(valid.min())
        upper = float(valid.max())
        if lower == upper:
            delta = max(abs(lower) * padding_ratio, 1e-6)
            return lower - delta, upper + delta
        padding = (upper - lower) * padding_ratio
        return lower - padding, upper + padding

    price_ax.set_xlim(float(time_seconds.min()), float(time_seconds.max()) if len(time_seconds) > 1 else 1.0)
    price_ax.set_ylim(*_axis_limits([numeric_columns["midprice"], numeric_columns["fundamental_value"]]))
    spread_ax.set_ylim(*_axis_limits([numeric_columns["spread"]]))
    depth_ax.set_ylim(*_axis_limits([numeric_columns["bid_depth"], numeric_columns["ask_depth"]]))
    flow_ax.set_ylim(*_axis_limits([numeric_columns["traded_volume"], numeric_columns["signed_order_flow"]]))
    if has_rl_state:
        inventory_ax.set_ylim(*_axis_limits([numeric_columns[column] for column, _ in inventory_lines]))
        wealth_ax.set_ylim(*_axis_limits([numeric_columns[column] for column, _ in wealth_lines]))

    def _set_window(current_time: float) -> None:
        if tail_seconds is None or tail_seconds <= 0:
            return
        left = max(float(time_seconds.min()), current_time - tail_seconds)
        right = max(left + 1e-6, current_time)
        for axis in axes:
            axis.set_xlim(left, right)

    def _update(frame_index: int):
        end = frame_index + 1
        current_time = float(time_seconds.iloc[frame_index])
        x_values = time_seconds.iloc[:end]

        mid_line.set_data(x_values, numeric_columns["midprice"].iloc[:end])
        if fundamental_line is not None and numeric_columns["fundamental_value"] is not None:
            fundamental_line.set_data(x_values, numeric_columns["fundamental_value"].iloc[:end])
        spread_line.set_data(x_values, numeric_columns["spread"].iloc[:end])
        bid_depth_line.set_data(x_values, numeric_columns["bid_depth"].iloc[:end])
        ask_depth_line.set_data(x_values, numeric_columns["ask_depth"].iloc[:end])
        volume_line.set_data(x_values, numeric_columns["traded_volume"].iloc[:end])
        if signed_flow_line is not None and numeric_columns["signed_order_flow"] is not None:
            signed_flow_line.set_data(x_values, numeric_columns["signed_order_flow"].iloc[:end])

        for marker in [price_marker, spread_marker, depth_marker, flow_marker, inventory_marker, wealth_marker]:
            if marker is not None:
                marker.set_xdata([current_time, current_time])

        for column, line in inventory_lines:
            line.set_data(x_values, numeric_columns[column].iloc[:end])
        if inventory_mean_line is not None and "inventory_mean" in numeric_columns:
            inventory_mean_line.set_data(x_values, numeric_columns["inventory_mean"].iloc[:end])
        if inventory_min_line is not None and "inventory_min" in numeric_columns:
            inventory_min_line.set_data(x_values, numeric_columns["inventory_min"].iloc[:end])
        if inventory_max_line is not None and "inventory_max" in numeric_columns:
            inventory_max_line.set_data(x_values, numeric_columns["inventory_max"].iloc[:end])
        for column, line in wealth_lines:
            line.set_data(x_values, numeric_columns[column].iloc[:end])
        if wealth_mean_line is not None and "wealth_mean" in numeric_columns:
            wealth_mean_line.set_data(x_values, numeric_columns["wealth_mean"].iloc[:end])

        spread_value = float(numeric_columns["spread"].iloc[frame_index])
        midprice_value = float(numeric_columns["midprice"].iloc[frame_index])
        spread_status = "one-sided" if pd.isna(spread_value) else f"{spread_value:.4f}"
        status_text.set_text(
            f"t={current_time:.1f}s | mid={midprice_value:.4f} | spread={spread_status}"
        )
        _set_window(current_time)

        artists = [
            mid_line,
            spread_line,
            bid_depth_line,
            ask_depth_line,
            volume_line,
            price_marker,
            spread_marker,
            depth_marker,
            flow_marker,
            status_text,
        ]
        if fundamental_line is not None:
            artists.append(fundamental_line)
        if signed_flow_line is not None:
            artists.append(signed_flow_line)
        if inventory_marker is not None:
            artists.append(inventory_marker)
        if wealth_marker is not None:
            artists.append(wealth_marker)
        if inventory_mean_line is not None:
            artists.append(inventory_mean_line)
        if inventory_min_line is not None:
            artists.append(inventory_min_line)
        if inventory_max_line is not None:
            artists.append(inventory_max_line)
        if wealth_mean_line is not None:
            artists.append(wealth_mean_line)
        artists.extend(line for _, line in inventory_lines)
        artists.extend(line for _, line in wealth_lines)
        return artists

    animation = FuncAnimation(
        fig,
        _update,
        frames=len(frame),
        interval=max(int(interval_ms), 1),
        blit=False,
        repeat=False,
    )
    fig.tight_layout()
    return fig, animation


def save_market_plot(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Market Replay",
) -> Path:
    """Save a plot of the market over time."""

    fig = create_market_figure(frame, title=title, show_plot=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)

    plt = _load_matplotlib(show_plot=False)
    plt.close(fig)
    return path


def show_market_plot(
    frame: pd.DataFrame,
    *,
    title: str = "Market Replay",
) -> None:
    """Display the market time-series view in an interactive window."""

    plt = _load_matplotlib(show_plot=True)
    fig = create_market_figure(frame, title=title, show_plot=True)
    plt.show()
    plt.close(fig)


def show_market_animation(
    frame: pd.DataFrame,
    *,
    title: str = "Market Replay",
    interval_ms: int = 40,
    tail_seconds: float | None = None,
):
    """Display an animated replay window of the market over time."""

    plt = _load_matplotlib(show_plot=True)
    fig, animation = build_market_animation(
        frame,
        title=title,
        interval_ms=interval_ms,
        tail_seconds=tail_seconds,
        show_plot=True,
    )
    plt.show()
    plt.close(fig)
    return animation
