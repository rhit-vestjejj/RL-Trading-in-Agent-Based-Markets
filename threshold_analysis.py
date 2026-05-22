"""
Analytical threshold argument for one-sided order book failures.

Derives φ* — the RL participation fraction at which aggregate liquidity
consumption exceeds the market's replenishment capacity — from simulation
parameters and low-φ empirical execution rates. Compares the prediction
against the empirically observed transition point.

Usage:
    python threshold_analysis.py [--experiment-dir EXPERIMENT_DIR]
                                 [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_DIR = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")
DEFAULT_EXPERIMENT = "experiments/paper_trained_nocap"

# ── Simulation parameters (from config.py / SimulationConfig defaults) ──────
NOISE_WAKE_SEC = 5.0       # noise_wake_up_frequency
NOISE_LIMIT_PROB = 0.45    # noise_limit_probability
MM_WAKE_SEC = 1.0          # mm_wake_up_frequency
MM_QUOTE_SIZE = 5          # mm_quote_size (units per side)
MM_SIDES = 2               # bid + ask
N_REPLACEABLE = 80         # noise trader pool (phi scales this pool)
N_MM = 2                   # number of market makers
SIM_SEC = 300              # simulation duration (09:30–09:35)

# ── Derived provision rates from parameters ──────────────────────────────────
# Noise trader: wakes every NOISE_WAKE_SEC seconds, submits a limit order
# with probability NOISE_LIMIT_PROB (size 1 each).
r_noise = NOISE_LIMIT_PROB / NOISE_WAKE_SEC  # limit orders/sec/noise trader

# Market maker: wakes every MM_WAKE_SEC, quotes MM_SIDES sides at MM_QUOTE_SIZE.
r_mm = MM_SIDES * MM_QUOTE_SIZE / MM_WAKE_SEC  # provision units/sec/MM


def compute_r_rl(diag: pd.DataFrame, phi: float, n_rl: int) -> float:
    """Estimate RL execution rate (executed market orders/sec/agent) at phi."""
    g = diag[np.isclose(diag["phi"], phi)]
    if g.empty:
        return float("nan")
    ex_buy = g["executed_buy_action_count"].mean()
    ex_sell = g["executed_sell_action_count"].mean()
    return float((ex_buy + ex_sell) / n_rl / SIM_SEC)


def phi_threshold(r_rl: float) -> float:
    """
    Solve for phi* where aggregate consumption equals aggregate provision.

    Consumption:  N_REPLACEABLE * phi * r_rl
    Provision:    N_REPLACEABLE * (1 - phi) * r_noise + N_MM * r_mm

    Setting equal and solving for phi:
        phi* = (N_REPLACEABLE * r_noise + N_MM * r_mm)
               / (N_REPLACEABLE * (r_noise + r_rl))
    """
    numerator = N_REPLACEABLE * r_noise + N_MM * r_mm
    denominator = N_REPLACEABLE * (r_noise + r_rl)
    return numerator / denominator


def run(experiment_dir: Path, output_dir: Path) -> None:
    import os
    import sys
    import tempfile

    # matplotlib setup (no display)
    cache = Path(tempfile.gettempdir()) / "threshold_analysis_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-seed diagnostics ────────────────────────────────────────────
    diag_path = experiment_dir / "summaries" / "per_seed_rl_diagnostics.csv"
    if not diag_path.exists():
        sys.exit(f"Diagnostics not found: {diag_path}")
    diag = pd.read_csv(diag_path)

    # ── Estimate r_rl from low-phi pre-breakdown regimes (phi=0.05 and 0.10) ─
    r_rl_05 = compute_r_rl(diag, 0.05, n_rl=4)
    r_rl_10 = compute_r_rl(diag, 0.10, n_rl=8)
    r_rl_mean = float(np.nanmean([r_rl_05, r_rl_10]))

    phi_star_mean = phi_threshold(r_rl_mean)
    phi_star_05 = phi_threshold(r_rl_05)
    phi_star_10 = phi_threshold(r_rl_10)
    phi_star_lo = min(phi_star_05, phi_star_10)
    phi_star_hi = max(phi_star_05, phi_star_10)

    # ── Load empirical summary ───────────────────────────────────────────────
    summary_path = experiment_dir / "phi_sweep_summary.csv"
    if not summary_path.exists():
        sys.exit(f"Summary not found: {summary_path}")
    summary = pd.read_csv(summary_path).sort_values("phi")
    phi_vals = summary["phi"].values

    one_sided_greedy = summary["greedy_one_sided_book_fraction_mean"].values
    one_sided_stoch = summary["stochastic_one_sided_book_fraction_mean"].values

    # CIs (clip negatives from narrow samples)
    def _ci(df: pd.DataFrame, col_prefix: str):
        lo = np.clip(df[f"{col_prefix}_ci95_lower"].values, 0, None)
        hi = df[f"{col_prefix}_ci95_upper"].values
        mean = df[f"{col_prefix}_mean"].values
        return mean - lo, hi - mean

    err_g_lo, err_g_hi = _ci(summary, "greedy_one_sided_book_fraction")
    err_s_lo, err_s_hi = _ci(summary, "stochastic_one_sided_book_fraction")

    # ── Rate curves ──────────────────────────────────────────────────────────
    phi_range = np.linspace(0.0, 0.55, 300)
    consumption = N_REPLACEABLE * phi_range * r_rl_mean
    provision = N_REPLACEABLE * (1 - phi_range) * r_noise + N_MM * r_mm

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: rate curves
    ax = axes[0]
    ax.plot(phi_range, consumption, color="tab:red", lw=2.0, label="Consumption $C(\\phi)$")
    ax.plot(phi_range, provision, color="tab:blue", lw=2.0, label="Provision $P(\\phi)$")
    ax.axvline(phi_star_mean, color="black", ls="--", lw=1.5,
               label=f"$\\phi^*$ = {phi_star_mean:.2f}")
    ax.axvspan(phi_star_lo, phi_star_hi, alpha=0.12, color="gray")
    ax.scatter([phi_star_mean], [N_REPLACEABLE * phi_star_mean * r_rl_mean],
               color="black", zorder=5, s=60)
    ax.set_xlabel("$\\phi$ (RL participation fraction)")
    ax.set_ylabel("Orders per second (aggregate)")
    ax.set_title("(a) Liquidity Flow Rates vs. $\\phi$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: empirical one-sided fraction + threshold
    ax2 = axes[1]
    ax2.errorbar(phi_vals, one_sided_greedy,
                 yerr=np.vstack([err_g_lo, err_g_hi]),
                 marker="o", lw=1.6, capsize=3, color="tab:blue",
                 label="Greedy")
    ax2.errorbar(phi_vals, one_sided_stoch,
                 yerr=np.vstack([err_s_lo, err_s_hi]),
                 marker="o", lw=1.6, capsize=3, color="tab:orange",
                 label="Stochastic")
    ax2.axvline(phi_star_mean, color="black", ls="--", lw=1.5,
                label=f"Predicted $\\phi^*$ = {phi_star_mean:.2f}")
    ax2.axvspan(phi_star_lo, phi_star_hi, alpha=0.12, color="gray",
                label="Threshold uncertainty")
    ax2.set_xlabel("$\\phi$ (RL participation fraction)")
    ax2.set_ylabel("One-sided book fraction")
    ax2.set_title("(b) Empirical Failure Rate with Predicted Threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-0.02)

    fig.tight_layout()
    out_path = output_dir / "threshold_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved threshold figure to {out_path}")

    # ── Print summary for paper text ─────────────────────────────────────────
    print()
    print("=== Threshold Analysis Results ===")
    print(f"r_noise (from config):         {r_noise:.4f} limit orders/sec/noise trader")
    print(f"r_mm    (from config):         {r_mm:.4f} provision units/sec/market maker")
    print(f"r_rl    (phi=0.05, empirical): {r_rl_05:.4f} market orders/sec/RL agent")
    print(f"r_rl    (phi=0.10, empirical): {r_rl_10:.4f} market orders/sec/RL agent")
    print(f"r_rl    (mean):                {r_rl_mean:.4f} market orders/sec/RL agent")
    print()
    print(f"Predicted threshold phi*:      {phi_star_mean:.3f}")
    print(f"Uncertainty range:             [{phi_star_lo:.3f}, {phi_star_hi:.3f}]")
    print()
    print("Consumption vs. provision at each phi:")
    for phi_v in phi_vals:
        if np.isclose(phi_v, 0.0):
            continue
        n_rl = int(round(N_REPLACEABLE * phi_v))
        n_noise = N_REPLACEABLE - n_rl
        c = n_rl * r_rl_mean
        p = n_noise * r_noise + N_MM * r_mm
        print(f"  phi={phi_v:.2f}: C={c:.2f}, P={p:.2f}, excess={c-p:+.2f}")

    # Save numerical results
    results_df = pd.DataFrame({
        "phi": phi_vals,
        "consumption_rate": [N_REPLACEABLE * phi_v * r_rl_mean for phi_v in phi_vals],
        "provision_rate": [N_REPLACEABLE * (1 - phi_v) * r_noise + N_MM * r_mm for phi_v in phi_vals],
        "excess_consumption": [N_REPLACEABLE * phi_v * r_rl_mean
                               - (N_REPLACEABLE * (1 - phi_v) * r_noise + N_MM * r_mm)
                               for phi_v in phi_vals],
        "greedy_one_sided_fraction": one_sided_greedy,
        "stochastic_one_sided_fraction": one_sided_stoch,
    })
    csv_path = output_dir / "threshold_analysis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved numerical results to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", default=str(PAPER_DIR))
    args = parser.parse_args()
    run(Path(args.experiment_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
