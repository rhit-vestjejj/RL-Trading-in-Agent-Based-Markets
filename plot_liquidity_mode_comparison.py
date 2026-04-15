"""Build a one-sided-book comparison plot across liquidity modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_summary_frame(folder: Path) -> tuple[pd.DataFrame, Path]:
    csv_path = folder / "phi_sweep_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path

    json_path = folder / "phi_sweep_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"No summary file found in {folder}")

    payload = json.loads(json_path.read_text())
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"Unsupported summary JSON structure in {json_path}")
    return pd.json_normalize(results, sep="_"), json_path


def _available_modes(frame: pd.DataFrame) -> set[str]:
    modes: set[str] = set()
    for mode in ("greedy", "stochastic"):
        if f"{mode}_one_sided_book_fraction_mean" in frame.columns:
            modes.add(mode)
    return modes


def _resolve_eval_mode(frames: list[pd.DataFrame], preferred: str = "greedy") -> str:
    common_modes = set.intersection(*(_available_modes(frame) for frame in frames))
    if not common_modes:
        raise ValueError("No common evaluation mode found across all supplied experiment summaries")
    if preferred in common_modes:
        return preferred
    if "stochastic" in common_modes:
        return "stochastic"
    return sorted(common_modes)[0]


def _extract_series(frame: pd.DataFrame, metric_column: str, extra_columns: list[str] | None = None) -> pd.DataFrame:
    if "phi" not in frame.columns:
        raise ValueError("Summary frame is missing the phi column")
    if metric_column not in frame.columns:
        raise ValueError(f"Summary frame is missing {metric_column}")
    columns = ["phi", metric_column]
    for column in extra_columns or []:
        if column in frame.columns and column not in columns:
            columns.append(column)
    series = frame[columns].copy()
    for column in columns:
        series[column] = pd.to_numeric(series[column], errors="coerce")
    series = series.dropna(subset=["phi", metric_column]).sort_values("phi")
    if series.empty:
        raise ValueError(f"No usable rows found for metric column {metric_column}")
    return series


def _resolve_error_columns(frame: pd.DataFrame, metric_column: str) -> tuple[str | None, str | None, str | None]:
    lower_column = metric_column.replace("_mean", "_ci95_lower")
    upper_column = metric_column.replace("_mean", "_ci95_upper")
    stderr_column = metric_column.replace("_mean", "_stderr")
    std_column = metric_column.replace("_mean", "_std")
    n_column = metric_column.replace("_mean", "_n")
    if lower_column in frame.columns and upper_column in frame.columns:
        return lower_column, upper_column, "ci95"
    if stderr_column in frame.columns:
        return stderr_column, stderr_column, "stderr"
    if std_column in frame.columns and n_column in frame.columns:
        return std_column, n_column, "std_over_sqrt_n"
    return None, None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taker-folder", type=Path, required=True)
    parser.add_argument("--mixed-folder", type=Path, required=True)
    parser.add_argument("--quoter-folder", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/comparison_plots/one_sided_book_fraction_by_liquidity_mode.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    folder_map: list[tuple[str, Path]] = [
        ("taker-only", args.taker_folder),
        ("mixed", args.mixed_folder),
    ]
    if args.quoter_folder is not None and args.quoter_folder.exists():
        folder_map.append(("quoter-only", args.quoter_folder))

    loaded: list[tuple[str, Path, pd.DataFrame, Path]] = []
    for label, folder in folder_map:
        frame, used_file = _load_summary_frame(folder)
        loaded.append((label, folder, frame, used_file))

    evaluation_mode = _resolve_eval_mode([frame for _, _, frame, _ in loaded], preferred="greedy")
    metric_column = f"{evaluation_mode}_one_sided_book_fraction_mean"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    color_map = {
        "taker-only": "#1f77b4",
        "mixed": "#d62728",
        "quoter-only": "#2ca02c",
    }

    for label, _, frame, _ in loaded:
        lower_column, upper_column, error_source = _resolve_error_columns(frame, metric_column)
        extra_columns = [column for column in (lower_column, upper_column) if column is not None]
        series = _extract_series(frame, metric_column, extra_columns=extra_columns)
        if error_source == "ci95":
            lower = pd.to_numeric(series[lower_column], errors="coerce").to_numpy(dtype=float)
            upper = pd.to_numeric(series[upper_column], errors="coerce").to_numpy(dtype=float)
            mean_values = pd.to_numeric(series[metric_column], errors="coerce").to_numpy(dtype=float)
            ax.errorbar(
                series["phi"],
                mean_values,
                yerr=[mean_values - lower, upper - mean_values],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                capsize=3.0,
                label=label,
                color=color_map[label],
            )
        elif error_source == "stderr":
            stderr = pd.to_numeric(series[lower_column], errors="coerce").to_numpy(dtype=float)
            ax.errorbar(
                series["phi"],
                series[metric_column],
                yerr=stderr,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                capsize=3.0,
                label=label,
                color=color_map[label],
            )
        elif error_source == "std_over_sqrt_n":
            std = pd.to_numeric(series[lower_column], errors="coerce").to_numpy(dtype=float)
            n_values = pd.to_numeric(series[upper_column], errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                stderr = std / np.sqrt(n_values)
            ax.errorbar(
                series["phi"],
                series[metric_column],
                yerr=stderr,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                capsize=3.0,
                label=label,
                color=color_map[label],
            )
        else:
            ax.plot(
                series["phi"],
                series[metric_column],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                label=label,
                color=color_map[label],
            )

    ax.set_title("One-Sided Order Book Fraction by Liquidity Mode")
    ax.set_xlabel("phi")
    ax.set_ylabel("One-sided order book fraction")
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Saved comparison figure to {args.output}")
    print(f"Evaluation mode used: {evaluation_mode}")
    print(f"Metric column used: {metric_column}")
    for label, folder, frame, used_file in loaded:
        print(f"{label}: folder={folder}")
        print(f"{label}: summary_file={used_file}")
        sample_size_column = metric_column.replace("_mean", "_n")
        if sample_size_column in frame.columns:
            sample_sizes = pd.to_numeric(frame[sample_size_column], errors="coerce").dropna()
            if not sample_sizes.empty:
                print(
                    f"{label}: metric_sample_size_range={int(sample_sizes.min())}..{int(sample_sizes.max())}"
                )
                if float(sample_sizes.min()) < 5.0:
                    print(
                        f"{label}: warning=small across-run sample for uncertainty estimates; "
                        "consider increasing evaluation seeds/runs."
                    )
    if not any(label == "quoter-only" for label, _, _, _ in loaded):
        print("quoter-only: not included (folder missing or not provided)")


if __name__ == "__main__":
    main()
