"""Compare diagnostics across multiple logging resolutions from the same run."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from analysis import summarize_market_frame
from visualization import save_market_plot

KEY_COMPARISON_COLUMNS = [
    "average_spread",
    "average_relative_spread",
    "average_top_of_book_depth",
    "return_variance",
    "excess_kurtosis",
    "volatility_clustering",
    "tail_exposure",
    "crash_rate",
    "max_drawdown",
    "traded_volume_total",
]


def parse_frequency_list(raw: str) -> list[str]:
    """Parse a comma-separated list of logging frequencies."""

    frequencies = [value.strip() for value in raw.split(",") if value.strip()]
    if not frequencies:
        raise ValueError("At least one logging frequency is required.")
    return frequencies


def summarize_sampling_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    squared_return_lags: int = 5,
    tail_quantile: float = 0.05,
    crash_threshold: float = -0.05,
) -> pd.DataFrame:
    """Summarize the same run at several logging resolutions."""

    records: list[dict[str, float | str]] = []
    for frequency, frame in frames.items():
        summary = summarize_market_frame(
            frame,
            squared_return_lags=squared_return_lags,
            tail_quantile=tail_quantile,
            crash_threshold=crash_threshold,
        )
        time_step_seconds = float(frame["time"].diff().dropna().median() / 1_000_000_000.0) if len(frame) > 1 else 0.0
        records.append(
            {
                "log_frequency": frequency,
                "num_rows": int(len(frame)),
                "time_step_seconds": time_step_seconds,
                "traded_volume_total": float(frame["traded_volume"].sum()) if "traded_volume" in frame.columns else 0.0,
                "mean_abs_signed_order_flow": float(frame["signed_order_flow"].abs().mean()) if "signed_order_flow" in frame.columns else 0.0,
                **summary,
            }
        )
    return pd.DataFrame(records)


def build_sampling_report(
    summary: pd.DataFrame,
    *,
    baseline_frequency: str,
    pct_change_threshold: float = 0.25,
) -> str:
    """Build a short report showing which conclusions are stable across sampling."""

    if summary.empty:
        raise ValueError("Cannot build a sampling report from an empty summary.")

    baseline = summary.loc[summary["log_frequency"] == baseline_frequency]
    if baseline.empty:
        raise ValueError(f"Baseline frequency '{baseline_frequency}' is not present in the summary.")
    baseline_row = baseline.iloc[0]

    sensitive_metrics: list[str] = []
    stable_metrics: list[str] = []
    for column in KEY_COMPARISON_COLUMNS:
        if column not in summary.columns:
            continue
        baseline_value = float(baseline_row[column])
        current_values = pd.to_numeric(summary[column], errors="coerce").dropna()
        if current_values.empty:
            continue
        if abs(baseline_value) < 1e-12:
            is_sensitive = current_values.abs().max() > pct_change_threshold
        else:
            relative_changes = ((current_values - baseline_value) / abs(baseline_value)).abs()
            is_sensitive = bool((relative_changes > pct_change_threshold).any())
        target_list = sensitive_metrics if is_sensitive else stable_metrics
        target_list.append(column)

    lines = [
        "# Sampling Comparison Report",
        "",
        f"- Baseline frequency for realism checks: {baseline_frequency}",
        f"- Frequencies compared: {', '.join(summary['log_frequency'].astype(str).tolist())}",
        f"- Stable metrics across these frequencies: {', '.join(stable_metrics) if stable_metrics else 'none'}",
        f"- Sampling-sensitive metrics across these frequencies: {', '.join(sensitive_metrics) if sensitive_metrics else 'none'}",
        "",
        "## Per-frequency diagnostics",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['log_frequency']}: rows={int(row['num_rows'])}, "
            f"spread={float(row['average_spread']):.6f}, "
            f"depth={float(row['average_top_of_book_depth']):.3f}, "
            f"return_variance={float(row['return_variance']):.6g}, "
            f"kurtosis={float(row['excess_kurtosis']):.4f}, "
            f"clustering={float(row['volatility_clustering']):.4f}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Use the finest frequency in this table as the main microstructure lens.",
            "- Treat coarser views as secondary presentation layers rather than primary realism evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_sampling_outputs(
    frames: Mapping[str, pd.DataFrame],
    *,
    output_dir: str | Path,
    title_prefix: str = "Sampling Comparison",
) -> list[Path]:
    """Save one CSV and one static market plot per logging frequency."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for frequency, frame in frames.items():
        safe_frequency = frequency.replace("/", "_")
        csv_path = output_path / f"market_{safe_frequency}.csv"
        plot_path = output_path / f"market_{safe_frequency}.png"
        frame.to_csv(csv_path, index=False)
        save_market_plot(frame, plot_path, title=f"{title_prefix} ({frequency})")
        saved_paths.extend([csv_path, plot_path])
    return saved_paths
