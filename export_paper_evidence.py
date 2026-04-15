"""Export paper evidence figures and a short support note from saved summaries."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


PHI_GRID = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
OUTPUT_DIR = Path("experiments/paper_evidence_exports")

os.environ.setdefault("MPLCONFIGDIR", str((OUTPUT_DIR / ".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((OUTPUT_DIR / ".cache").resolve()))

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentBundle:
    label: str
    summary_csv: Path


def _locate_experiments_root() -> Path:
    candidates = [
        Path("experiments"),
        Path("RL-Trading-in-Agent-Based-Markets/experiments"),
    ]
    for candidate in candidates:
        if (candidate / "paper_trained_nocap/phi_sweep_summary.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find a paper experiment bundle with paper_trained_nocap/phi_sweep_summary.csv.")


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("phi").reset_index(drop=True)
    frame["phi"] = pd.to_numeric(frame["phi"], errors="coerce")
    return frame


def _ci_bounds(frame: pd.DataFrame, column_prefix: str) -> tuple[pd.Series, pd.Series]:
    mean = pd.to_numeric(frame[f"{column_prefix}_mean"], errors="coerce")
    lower_col = f"{column_prefix}_ci95_lower"
    upper_col = f"{column_prefix}_ci95_upper"
    if lower_col in frame.columns and upper_col in frame.columns:
        lower = pd.to_numeric(frame[lower_col], errors="coerce")
        upper = pd.to_numeric(frame[upper_col], errors="coerce")
        return lower, upper
    std_col = f"{column_prefix}_std"
    if std_col in frame.columns:
        spread = pd.to_numeric(frame[std_col], errors="coerce")
        return mean - spread, mean + spread
    return mean, mean


def _plot_mode_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    output_path: Path,
    clip_min: float | None = None,
    clip_max: float | None = None,
    use_error_bars: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = {
        "greedy": ("#1f77b4", "o", "Greedy"),
        "stochastic": ("#d95f02", "s", "Stochastic"),
    }

    for mode, (color, marker, label) in styles.items():
        prefix = f"{mode}_{metric}"
        mean = pd.to_numeric(frame[f"{prefix}_mean"], errors="coerce")
        if clip_min is not None:
            mean = mean.clip(lower=clip_min)
        if clip_max is not None:
            mean = mean.clip(upper=clip_max)

        if use_error_bars:
            lower, upper = _ci_bounds(frame, prefix)
            if clip_min is not None:
                lower = lower.clip(lower=clip_min)
                upper = upper.clip(lower=clip_min)
            if clip_max is not None:
                lower = lower.clip(upper=clip_max)
                upper = upper.clip(upper=clip_max)
            yerr = [mean - lower, upper - mean]
            ax.errorbar(
                frame["phi"],
                mean,
                yerr=yerr,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2.0,
                capsize=4,
                label=label,
            )
        else:
            ax.plot(
                frame["phi"],
                mean,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2.0,
                label=label,
            )

    ax.set_title(title)
    ax.set_xlabel("RL participation fraction $\\phi$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PHI_GRID)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_inventory_cap_metric(
    summaries: dict[str, pd.DataFrame],
    metric: str,
    *,
    evaluation_mode: str,
    title: str,
    ylabel: str,
    output_path: Path,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    styles = {
        "no_cap": ("#1f77b4", "o", "No cap"),
        "cap_50": ("#e6ab02", "s", "Cap = 50"),
        "cap_20": ("#2ca25f", "^", "Cap = 20"),
    }

    for label, (color, marker, legend_label) in styles.items():
        frame = summaries[label]
        prefix = f"{evaluation_mode}_{metric}"
        mean = pd.to_numeric(frame[f"{prefix}_mean"], errors="coerce")
        lower, upper = _ci_bounds(frame, prefix)
        if clip_min is not None:
            mean = mean.clip(lower=clip_min)
            lower = lower.clip(lower=clip_min)
            upper = upper.clip(lower=clip_min)
        if clip_max is not None:
            mean = mean.clip(upper=clip_max)
            lower = lower.clip(upper=clip_max)
            upper = upper.clip(upper=clip_max)
        yerr = [mean - lower, upper - mean]
        ax.errorbar(
            frame["phi"],
            mean,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.0,
            capsize=4,
            label=legend_label,
        )

    ax.set_title(title)
    ax.set_xlabel("RL participation fraction $\\phi$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(PHI_GRID)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_shutdown_support(frame: pd.DataFrame, output_path: Path) -> None:
    specs = [
        ("one_sided_book_fraction", "One-Sided Book Fraction", (0.0, 1.0)),
        ("inactivity_fraction", "Inactivity Fraction", (0.0, 1.0)),
        ("quote_activity_fraction", "Quote Activity Fraction", (0.0, 1.0)),
        ("zero_return_fraction", "Zero-Return Fraction", (0.0, 1.0)),
    ]
    styles = {
        "greedy": ("#1f77b4", "o", "Greedy"),
        "stochastic": ("#d95f02", "s", "Stochastic"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for ax, (metric, title, (clip_min, clip_max)) in zip(axes.flat, specs):
        for mode, (color, marker, label) in styles.items():
            prefix = f"{mode}_{metric}"
            mean = pd.to_numeric(frame[f"{prefix}_mean"], errors="coerce").clip(lower=clip_min, upper=clip_max)
            lower, upper = _ci_bounds(frame, prefix)
            lower = lower.clip(lower=clip_min, upper=clip_max)
            upper = upper.clip(lower=clip_min, upper=clip_max)
            yerr = [mean - lower, upper - mean]
            ax.errorbar(
                frame["phi"],
                mean,
                yerr=yerr,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        ax.axvline(0.5, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_ylim(clip_min, clip_max)
        ax.set_xticks(PHI_GRID)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    for ax in axes[-1]:
        ax.set_xlabel("RL participation fraction $\\phi$")
    axes[0][0].legend(loc="best")
    fig.suptitle("$\\phi = 0.5$ is shutdown-like, not a liquidity recovery", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_int(value: float) -> str:
    return f"{int(round(float(value))):,}"


def _support_lines(summary: pd.DataFrame, mode: str) -> list[str]:
    row_04 = summary.loc[summary["phi"] == 0.40].iloc[0]
    row_05 = summary.loc[summary["phi"] == 0.50].iloc[0]

    submitted_04 = float(row_04[f"{mode}_evaluation_submitted_sell_count_total"]) + float(
        row_04[f"{mode}_evaluation_submitted_buy_count_total"]
    )
    submitted_05 = float(row_05[f"{mode}_evaluation_submitted_sell_count_total"]) + float(
        row_05[f"{mode}_evaluation_submitted_buy_count_total"]
    )
    executed_04 = float(row_04[f"{mode}_evaluation_executed_sell_count_total"]) + float(
        row_04[f"{mode}_evaluation_executed_buy_count_total"]
    )
    executed_05 = float(row_05[f"{mode}_evaluation_executed_sell_count_total"]) + float(
        row_05[f"{mode}_evaluation_executed_buy_count_total"]
    )

    return [
        (
            f"- `{mode}`: one-sided {_format_float(row_04[f'{mode}_one_sided_book_fraction_mean'])}"
            f" -> {_format_float(row_05[f'{mode}_one_sided_book_fraction_mean'])}, undefined-mid "
            f"{_format_float(row_04[f'{mode}_undefined_midprice_fraction_mean'])}"
            f" -> {_format_float(row_05[f'{mode}_undefined_midprice_fraction_mean'])}, max-run "
            f"{_format_float(row_04[f'{mode}_max_consecutive_one_sided_duration_mean'])}s"
            f" -> {_format_float(row_05[f'{mode}_max_consecutive_one_sided_duration_mean'])}s."
        ),
        (
            f"- `{mode}`: inactivity {_format_float(row_04[f'{mode}_inactivity_fraction_mean'])}"
            f" -> {_format_float(row_05[f'{mode}_inactivity_fraction_mean'])}, quote activity "
            f"{_format_float(row_04[f'{mode}_quote_activity_fraction_mean'])}"
            f" -> {_format_float(row_05[f'{mode}_quote_activity_fraction_mean'])}, zero-return "
            f"{_format_float(row_04[f'{mode}_zero_return_fraction_mean'])}"
            f" -> {_format_float(row_05[f'{mode}_zero_return_fraction_mean'])}."
        ),
        (
            f"- `{mode}`: submitted RL actions {_format_int(submitted_04)} -> {_format_int(submitted_05)},"
            f" executed RL actions {_format_int(executed_04)} -> {_format_int(executed_05)}."
        ),
    ]


def _write_note(
    output_path: Path,
    *,
    experiments_root: Path,
    nocap_bundle: ExperimentBundle,
    cap50_bundle: ExperimentBundle,
    cap20_bundle: ExperimentBundle,
    inventory_comparison_md: Path,
) -> None:
    note_lines = [
        "# Paper Evidence Note",
        "",
        "## Files Used",
        f"- {nocap_bundle.summary_csv}",
        f"- {cap50_bundle.summary_csv}",
        f"- {cap20_bundle.summary_csv}",
        f"- {inventory_comparison_md}",
        "",
        "## Metrics Available From Saved Outputs",
        "- one-sided book fraction with seed-level mean/std/stderr/95% CI",
        "- undefined mid-price fraction with seed-level mean/std/stderr/95% CI",
        "- max consecutive one-sided duration with seed-level mean/std/stderr/95% CI",
        "- inactivity fraction with seed-level mean/std/stderr/95% CI",
        "- quote activity fraction with seed-level mean/std/stderr/95% CI",
        "- zero-return fraction with seed-level mean/std/stderr/95% CI",
        "- average depth and average spread with seed-level mean/std/stderr/95% CI",
        "- true one-sided fraction and both-sides-missing fraction",
        "- RL submitted and executed action totals, plus executed RL buy/sell volume totals",
        "",
        "## Metrics Not Available In CI-Ready Form",
        "- market-wide executed trades per episode are not saved as a seed-level summary series in the paper bundle",
        "- market-wide trading volume is not saved as a seed-level summary series in the paper bundle",
        "- order submission rate is not saved directly; the closest saved proxies are quote-activity fraction and RL submitted/executed action totals",
        "",
        "## Strict Assessment Of `phi = 0.5`",
        "- Using the paper bundle, `phi = 0.5` is supported as a degenerate near-shutdown regime rather than a recovery in liquidity.",
        "- The key reason is that the drop in one-sided states at `phi = 0.5` coincides with very large increases in inactivity and zero-return behavior plus a collapse in quote activity and RL order submission/execution.",
    ]
    note_lines.extend(_support_lines(_load_summary(nocap_bundle.summary_csv), "greedy"))
    note_lines.extend(_support_lines(_load_summary(nocap_bundle.summary_csv), "stochastic"))
    note_lines.extend(
        [
            "- This is strong support for `paper_trained_nocap` specifically. The older top-level `experiments/phi_sweep_45min` bundle does not contain the same complete diagnostics and should not be treated as the authoritative source for this paper claim.",
            "",
            "## Figures Added",
            "- `one_sided_book_fraction_with_ci_vs_phi.png`",
            "- `undefined_midprice_fraction_with_ci_vs_phi.png`",
            "- `max_consecutive_one_sided_duration_with_ci_vs_phi.png`",
            "- `inventory_cap_one_sided_fraction_with_ci_vs_phi.png` (greedy comparison across no-cap / cap-50 / cap-20)",
            "- `phi_0_5_shutdown_support.png`",
            "",
            "## Recommendation On New Runs",
            "- No new sweeps are necessary if `paper_trained_nocap`, `paper_trained_cap50`, and `paper_trained_cap20` are the source-of-record experiment bundles for the paper.",
            "- If you instead need the paper to rest only on the older top-level `experiments/phi_sweep_45min` outputs, then targeted evaluation reruns at `phi = 0.4` and `phi = 0.5` would be necessary to recover the missing one-sided, undefined-mid, inactivity, and duration diagnostics.",
            "",
            f"Source experiments root: `{experiments_root}`",
        ]
    )
    output_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def main() -> None:
    experiments_root = _locate_experiments_root()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nocap_bundle = ExperimentBundle("no_cap", experiments_root / "paper_trained_nocap/phi_sweep_summary.csv")
    cap50_bundle = ExperimentBundle("cap_50", experiments_root / "paper_trained_cap50/phi_sweep_summary.csv")
    cap20_bundle = ExperimentBundle("cap_20", experiments_root / "paper_trained_cap20/phi_sweep_summary.csv")
    inventory_comparison_md = experiments_root / "paper_inventory_cap_comparison/inventory_cap_comparison.md"

    nocap_summary = _load_summary(nocap_bundle.summary_csv)
    cap_summaries = {
        "no_cap": nocap_summary,
        "cap_50": _load_summary(cap50_bundle.summary_csv),
        "cap_20": _load_summary(cap20_bundle.summary_csv),
    }

    _plot_mode_metric(
        nocap_summary,
        "one_sided_book_fraction",
        title="One-Sided Book Fraction vs $\\phi$",
        ylabel="Fraction",
        output_path=OUTPUT_DIR / "one_sided_book_fraction_with_ci_vs_phi.png",
        clip_min=0.0,
        clip_max=1.0,
    )
    _plot_mode_metric(
        nocap_summary,
        "undefined_midprice_fraction",
        title="Undefined Mid-Price Fraction vs $\\phi$",
        ylabel="Fraction",
        output_path=OUTPUT_DIR / "undefined_midprice_fraction_with_ci_vs_phi.png",
        clip_min=0.0,
        clip_max=1.0,
    )
    _plot_mode_metric(
        nocap_summary,
        "max_consecutive_one_sided_duration",
        title="Max Consecutive One-Sided Duration vs $\\phi$",
        ylabel="Seconds",
        output_path=OUTPUT_DIR / "max_consecutive_one_sided_duration_with_ci_vs_phi.png",
        clip_min=0.0,
    )
    _plot_inventory_cap_metric(
        cap_summaries,
        "one_sided_book_fraction",
        evaluation_mode="greedy",
        title="One-Sided Book Fraction vs $\\phi$ by Inventory Cap",
        ylabel="Fraction",
        output_path=OUTPUT_DIR / "inventory_cap_one_sided_fraction_with_ci_vs_phi.png",
        clip_min=0.0,
        clip_max=1.0,
    )
    _plot_shutdown_support(nocap_summary, OUTPUT_DIR / "phi_0_5_shutdown_support.png")

    _write_note(
        OUTPUT_DIR / "paper_evidence_note.md",
        experiments_root=experiments_root,
        nocap_bundle=nocap_bundle,
        cap50_bundle=cap50_bundle,
        cap20_bundle=cap20_bundle,
        inventory_comparison_md=inventory_comparison_md,
    )


if __name__ == "__main__":
    main()
