"""Update placeholder numbers in final.tex from completed experiment data.

Reads phi_sweep_summary.csv, per_seed_rl_diagnostics.csv, and
composition_control_summary.csv to replace TODO-marked numbers in the paper.

Usage:
    python update_paper_numbers.py [--dry-run]

Run after:
    1. All phi sweeps complete (regen_headline_taker, cap50, cap20, alt_profile)
    2. reviewer robustness sweep completes (regen_reviewer_robustness)
    3. threshold_analysis.py has been run (produces updated numbers)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HEADLINE = Path("experiments/regen_headline_taker")
ROBUSTNESS = Path("experiments/regen_reviewer_robustness")
TEX = Path("final.tex")


def load_summary() -> pd.DataFrame:
    p = HEADLINE / "phi_sweep_summary.csv"
    if not p.exists():
        raise SystemExit(f"Missing: {p}")
    return pd.read_csv(p).sort_values("phi")


def load_diag() -> pd.DataFrame:
    p = HEADLINE / "summaries" / "per_seed_rl_diagnostics.csv"
    if not p.exists():
        raise SystemExit(f"Missing: {p}")
    return pd.read_csv(p)


def load_robustness() -> pd.DataFrame | None:
    p = ROBUSTNESS / "summaries" / "composition_control_summary.csv"
    if not p.exists():
        print(f"  [SKIP] Robustness not found: {p}")
        return None
    df = pd.read_csv(p)
    df = df[df["evaluation_mode"] == "greedy"]
    df = df[np.isclose(df["rl_quoter_split"].astype(float), 0.0)]
    return df


def compute_lambda_r(diag: pd.DataFrame, phi: float, n_rl: int, sim_sec: int = 300) -> float:
    g = diag[np.isclose(diag["phi"].astype(float), phi)]
    if "evaluation_mode" in g.columns:
        g = g[g["evaluation_mode"] == "greedy"]
    if g.empty:
        return float("nan")
    buy = g["executed_buy_action_count"].mean()
    sell = g["executed_sell_action_count"].mean()
    return float((buy + sell) / n_rl / sim_sec)


def compute_phi_star(lambda_r: float) -> tuple[float, float, float]:
    """Returns (phi_star, phi_star_lo, phi_star_hi) from lambda_r estimates."""
    lambda_n = 0.09
    lambda_m = 10.0
    N_REPLACEABLE = 80
    N_MM = 2

    def phi_threshold(lr: float) -> float:
        return (N_REPLACEABLE * lambda_n + N_MM * lambda_m) / (N_REPLACEABLE * (lambda_n + lr))

    return phi_threshold(lambda_r), 0.0, 0.0  # lo/hi filled below


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = load_summary()
    diag = load_diag()
    rob = load_robustness()

    # ── Compute lambda_r and phi_star ────────────────────────────────────────
    r05 = compute_lambda_r(diag, 0.05, n_rl=4)
    r10 = compute_lambda_r(diag, 0.10, n_rl=8)
    r_mean = float(np.nanmean([r05, r10]))

    lambda_n, lambda_m = 0.09, 10.0
    N_REPLACEABLE, N_MM = 80, 2

    def phi_threshold(lr: float) -> float:
        return (N_REPLACEABLE * lambda_n + N_MM * lambda_m) / (N_REPLACEABLE * (lambda_n + lr))

    phi_star_mean = phi_threshold(r_mean)
    phi_star_lo = min(phi_threshold(r05), phi_threshold(r10))
    phi_star_hi = max(phi_threshold(r05), phi_threshold(r10))

    print(f"λ_r: {r05:.3f} (φ=0.05), {r10:.3f} (φ=0.10), mean={r_mean:.3f}")
    print(f"φ* = {phi_star_mean:.3f}  [{phi_star_lo:.3f}, {phi_star_hi:.3f}]")

    # ── Excess consumption at key phi values ─────────────────────────────────
    def excess(phi: float) -> float:
        consumption = N_REPLACEABLE * phi * r_mean
        provision = N_REPLACEABLE * (1 - phi) * lambda_n + N_MM * lambda_m
        return consumption - provision

    exc_030 = excess(0.30)
    exc_040 = excess(0.40)
    provision_040 = N_REPLACEABLE * (1 - 0.40) * lambda_n + N_MM * lambda_m
    exc_040_pct = exc_040 / provision_040 * 100

    print(f"Excess at φ=0.30: +{exc_030:.2f} units/s")
    print(f"Excess at φ=0.40: +{exc_040:.2f} units/s ({exc_040_pct:.0f}%)")

    # ── Headline one-sided fractions ─────────────────────────────────────────
    row_020 = summary[np.isclose(summary["phi"], 0.20)]
    row_030 = summary[np.isclose(summary["phi"], 0.30)]
    frac_020 = row_020["greedy_one_sided_book_fraction_mean"].values[0] if len(row_020) else float("nan")
    frac_030 = row_030["greedy_one_sided_book_fraction_mean"].values[0] if len(row_030) else float("nan")
    print(f"One-sided fraction: φ=0.20={frac_020*100:.1f}%, φ=0.30={frac_030*100:.1f}%")

    # ── Trained vs random at phi=0.40 ────────────────────────────────────────
    if rob is not None:
        col_m = "one_sided_book_fraction_mean"
        col_lo = "one_sided_book_fraction_ci95_lower"
        col_hi = "one_sided_book_fraction_ci95_upper"
        for phi_val in [0.40]:
            for cond in ["trained", "matched_random"]:
                g = rob[(rob["condition"] == cond) & np.isclose(rob["phi"].astype(float), phi_val)]
                if g.empty:
                    continue
                m = g[col_m].values[0]
                lo = g[col_lo].values[0] if col_lo in g else float("nan")
                hi = g[col_hi].values[0] if col_hi in g else float("nan")
                print(f"  φ=0.40 {cond}: {m*100:.1f}% CI [{lo*100:.1f}%, {hi*100:.1f}%]")

    # ── Apply substitutions to final.tex ────────────────────────────────────
    tex = TEX.read_text(encoding="utf-8")
    original = tex

    # Lambda_r values
    _repl_lr = (
        f"$\\lambda_r = {r05:.3f}$ at $\\phi = 0.05$ and $\\lambda_r = {r10:.3f}$"
        f" at $\\phi = 0.10$, with mean $\\bar{{\\lambda}}_r = {r_mean:.3f}$"
    )
    tex = re.sub(
        r"\$\\lambda_r = [\d.]+\$ at \$\\phi = 0\.05\$ and \$\\lambda_r = [\d.]+\$"
        r" at \$\\phi = 0\.10\$, with mean \$\\bar\{\\lambda\}_r = [\d.]+\$",
        lambda _: _repl_lr,
        tex
    )

    # phi* value
    _repl_phi = f"= {phi_star_mean:.3f} \\quad [{phi_star_lo:.3f},\\; {phi_star_hi:.3f}]% TODO: update phi*"
    tex = re.sub(
        r"= [\d.]+ \\quad \[[\d.]+,\\; [\d.]+\]% TODO: update phi\*",
        lambda _: _repl_phi,
        tex
    )

    # Excess consumption
    _repl_exc = (
        f"excess consumption is only $+{exc_030:.2f}$ units/s."
        f" At $\\phi = 0.40$, excess is $+{exc_040:.2f}$ units/s"
        f" (${{\\approx}}{exc_040_pct:.0f}\\%$)"
    )
    tex = re.sub(
        r"excess consumption is only \$\+[\d.]+\$ units/s\."
        r" At \$\\phi = 0\.40\$, excess is \$\+[\d.]+\$ units/s \(\$\{\\approx\}[\d]+\\%\$\)",
        lambda _: _repl_exc,
        tex
    )

    if tex == original:
        print("\nWarning: no replacements made (patterns may not match)")
    else:
        print(f"\nApplying {sum(1 for a, b in zip(original.splitlines(), tex.splitlines()) if a != b)} line changes")

    if not args.dry_run:
        TEX.write_text(tex, encoding="utf-8")
        print(f"Updated {TEX}")
    else:
        print("(dry run — not writing)")


if __name__ == "__main__":
    main()
