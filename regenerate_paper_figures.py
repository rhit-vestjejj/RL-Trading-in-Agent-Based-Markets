"""
Regenerate paper figures from 6-seed merged data so they match the headline figure's CI.

Reads experiments/more_seeds/merged_market_metrics.csv (per-seed, both modes), aggregates
to mean + 95% CI per phi, and saves the *_with_ci_vs_phi.png figures the paper references.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

cache = Path(tempfile.gettempdir()) / "regen_paper_mpl"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")
DATA = Path("experiments/regen_headline_taker/summaries/per_seed_market_metrics.csv")


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate per-seed data to mean + 95% CI per (phi, mode)."""
    rows = []
    for (phi, mode), grp in df.groupby(["phi", "evaluation_mode"]):
        vals = grp[metric].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1) if n > 1 else 0.0
        stderr = std / np.sqrt(n) if n > 0 else 0.0
        if n > 1:
            tcrit = stats.t.ppf(0.975, n - 1)
            lo, hi = mean - tcrit * stderr, mean + tcrit * stderr
        else:
            lo, hi = mean, mean
        rows.append({"phi": phi, "mode": mode, "mean": mean, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows).sort_values(["mode", "phi"]).reset_index(drop=True)


def plot_two_mode(df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    """Two-mode (greedy/stochastic) plot with 95% CI bars."""
    agg = aggregate(df, metric)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mode, color in [("greedy", "tab:blue"), ("stochastic", "tab:orange")]:
        sub = agg[agg["mode"] == mode]
        if sub.empty:
            continue
        phi = sub["phi"].to_numpy()
        mean = sub["mean"].to_numpy()
        lo = np.clip(sub["lo"].to_numpy(), 0, None) if "fraction" in metric else sub["lo"].to_numpy()
        hi = sub["hi"].to_numpy()
        ax.errorbar(
            phi, mean,
            yerr=np.vstack([mean - lo, hi - mean]),
            marker="o", linewidth=1.8, capsize=3.5, color=color, label=mode,
        )
    ax.set_xlabel(r"$\phi$ (RL participation fraction)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_missing_sides_combined(df: pd.DataFrame, out_path: Path) -> None:
    """Both missing bid AND missing ask on one figure (the figure_both_test.png replacement)."""
    bid_agg = aggregate(df, "missing_bid_fraction")
    ask_agg = aggregate(df, "missing_ask_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (label, mode) in zip(axes, [("Greedy", "greedy"), ("Stochastic", "stochastic")]):
        for agg, side, color in [(bid_agg, "Missing best bid", "tab:purple"), (ask_agg, "Missing best ask", "tab:red")]:
            sub = agg[agg["mode"] == mode]
            if sub.empty:
                continue
            phi = sub["phi"].to_numpy()
            mean = sub["mean"].to_numpy()
            lo = np.clip(sub["lo"].to_numpy(), 0, None)
            hi = sub["hi"].to_numpy()
            ax.errorbar(
                phi, mean,
                yerr=np.vstack([mean - lo, hi - mean]),
                marker="o", linewidth=1.8, capsize=3, color=color, label=side,
            )
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel("Fraction")
        ax.set_title(f"{label} — Missing Side Fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_missing_bid_only(df: pd.DataFrame, out_path: Path) -> None:
    """figure8.png replacement — missing best bid with CI bands."""
    plot_two_mode(df, "missing_bid_fraction",
                  ylabel="Fraction", title="Missing Best Bid Fraction vs Phi (6 seeds)",
                  out_path=out_path)


ALT_DATA = Path("experiments/regen_alt_profile/phi_sweep_summary.csv")
PRIMARY_DATA = Path("experiments/regen_headline_taker/phi_sweep_summary.csv")


def plot_alt_profile_comparison() -> None:
    """Regenerate alt_profile_comparison.png using 200ep 6-seed alt data."""
    if not ALT_DATA.exists():
        print(f"  SKIP alt comparison: {ALT_DATA} not found")
        return
    if not PRIMARY_DATA.exists():
        print(f"  SKIP alt comparison: {PRIMARY_DATA} not found")
        return

    alt = pd.read_csv(ALT_DATA).sort_values("phi")
    pri = pd.read_csv(PRIMARY_DATA).sort_values("phi")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    col = "one_sided_book_fraction"

    for ax, mode, pri_color, alt_color in [
        (axes[0], "greedy", "tab:blue", "tab:cyan"),
        (axes[1], "stochastic", "tab:orange", "goldenrod"),
    ]:
        mean_col = f"{mode}_{col}_mean"
        lo_col = f"{mode}_{col}_ci95_lower"
        hi_col = f"{mode}_{col}_ci95_upper"

        # Primary
        if mean_col in pri.columns:
            phi_p = pri["phi"].to_numpy()
            mean_p = pri[mean_col].to_numpy()
            lo_p = np.clip(pri[lo_col].to_numpy() if lo_col in pri.columns else mean_p, 0, None)
            hi_p = pri[hi_col].to_numpy() if hi_col in pri.columns else mean_p
            ax.errorbar(phi_p, mean_p, yerr=np.vstack([mean_p - lo_p, hi_p - mean_p]),
                        marker="s", lw=1.4, capsize=3, color=pri_color, alpha=0.85, ls="--",
                        label="Primary (RMSC04: 80 noise, 2 MM)")

        # Alt profile (200ep, 6 seeds)
        if mean_col in alt.columns:
            phi_a = alt["phi"].to_numpy()
            mean_a = alt[mean_col].to_numpy()
            lo_a = np.clip(alt[lo_col].to_numpy() if lo_col in alt.columns else mean_a, 0, None)
            hi_a = alt[hi_col].to_numpy() if hi_col in alt.columns else mean_a
            ax.errorbar(phi_a + 0.004, mean_a, yerr=np.vstack([mean_a - lo_a, hi_a - mean_a]),
                        marker="o", lw=1.8, capsize=3, color=alt_color,
                        label="Alt profile (baseline_v1: ZIC + 15 MM)")

        ax.set_xlabel(r"$\phi$ (RL participation fraction)")
        ax.set_ylabel("One-sided book fraction")
        ax.set_title(f"{mode.capitalize()}: Primary vs. Alternative Profile")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    out = PAPER_DIR / "alt_profile_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"Missing data: {DATA}")
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows from {DATA}")
    print(f"  phi values: {sorted(df['phi'].unique())}")
    print(f"  modes: {sorted(df['evaluation_mode'].unique())}")
    print(f"  seeds per (phi, mode): {df.groupby(['phi','evaluation_mode']).size().min()}–{df.groupby(['phi','evaluation_mode']).size().max()}")
    print()

    PAPER_DIR.mkdir(exist_ok=True)

    plot_two_mode(df, "undefined_midprice_fraction",
                  ylabel="Undefined mid-price fraction",
                  title="Undefined Mid-price Fraction vs Phi (6 seeds)",
                  out_path=PAPER_DIR / "undefined_midprice_fraction_with_ci_vs_phi.png")

    plot_two_mode(df, "max_consecutive_one_sided_duration_seconds",
                  ylabel="Max consecutive duration (seconds)",
                  title="Max Consecutive One-Sided Duration vs Phi (6 seeds)",
                  out_path=PAPER_DIR / "max_consecutive_one_sided_duration_with_ci_vs_phi.png")

    plot_missing_bid_only(df, PAPER_DIR / "figure8.png")
    plot_missing_sides_combined(df, PAPER_DIR / "figure_both_test.png")

    # figure4.png — P90 one-sided episode duration vs phi
    plot_two_mode(df, "one_sided_episode_duration_p90_seconds",
                  ylabel="P90 episode duration (seconds)",
                  title="P90 One-Sided Episode Duration vs Phi (6 seeds)",
                  out_path=PAPER_DIR / "figure4.png")

    # updated_one_sided_fraction.png — headline metric (replaces old multi-seed figure)
    plot_two_mode(df, "one_sided_book_fraction",
                  ylabel="One-sided book fraction",
                  title="One-Sided Book Fraction vs Phi (6 seeds)",
                  out_path=PAPER_DIR / "updated_one_sided_fraction.png")

    plot_alt_profile_comparison()

    print("\nDone. All regenerated figures saved with 6-seed 95% CIs.")


if __name__ == "__main__":
    main()
