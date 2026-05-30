"""Extract key statistics from completed sweep results for paper text updates.

Run after all sweeps + reviewer robustness complete. Prints all the numbers
needed to update §3 of final.tex.

Usage:
    python extract_paper_numbers.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HEADLINE = Path("experiments/regen_headline_taker/phi_sweep_summary.csv")
CAP50 = Path("experiments/regen_cap50/phi_sweep_summary.csv")
CAP20 = Path("experiments/regen_cap20/phi_sweep_summary.csv")
ALT = Path("experiments/regen_alt_profile/phi_sweep_summary.csv")
ROBUSTNESS = Path("experiments/regen_reviewer_robustness/summaries/composition_control_summary.csv")
DIAG = Path("experiments/regen_headline_taker/summaries/per_seed_rl_diagnostics.csv")


def load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [SKIP] {label}: {path} not found")
        return None
    return pd.read_csv(path).sort_values("phi").reset_index(drop=True)


def fmt(val: float, pct: bool = False) -> str:
    if np.isnan(val):
        return "NaN"
    if pct:
        return f"{val * 100:.1f}\\%"
    return f"{val:.4f}"


def main() -> None:
    print("=" * 70)
    print("PAPER NUMBERS REPORT")
    print("=" * 70)

    # ── §3.1: headline one-sided fractions ──────────────────────────────────
    print("\n§3.1 Headline One-Sided Book Fraction")
    df = load(HEADLINE, "headline")
    if df is not None:
        for mode in ["greedy", "stochastic"]:
            print(f"\n  {mode.capitalize()}:")
            col_m = f"{mode}_one_sided_book_fraction_mean"
            col_lo = f"{mode}_one_sided_book_fraction_ci95_lower"
            col_hi = f"{mode}_one_sided_book_fraction_ci95_upper"
            for _, row in df.iterrows():
                phi = row["phi"]
                m = row.get(col_m, np.nan)
                lo = row.get(col_lo, np.nan)
                hi = row.get(col_hi, np.nan)
                print(f"    φ={phi:.2f}: {fmt(m, True)}  95% CI [{fmt(lo, True)}, {fmt(hi, True)}]")

    # ── §3.4: threshold analysis lambda estimates ────────────────────────────
    print("\n§3.4 Lambda_r estimates (from per-seed RL diagnostics)")
    diag = load(DIAG, "rl diagnostics")
    if diag is not None:
        SIM_SEC = 300  # 09:30–09:35
        for phi in [0.05, 0.10]:
            g = diag[np.isclose(diag["phi"].astype(float), phi)]
            if g.empty:
                print(f"  φ={phi:.2f}: no data")
                continue
            n_rl_col = next((c for c in g.columns if "n_rl" in c.lower() or "num_rl" in c.lower()), None)
            buy_col = next((c for c in g.columns if "executed_buy" in c.lower()), None)
            sell_col = next((c for c in g.columns if "executed_sell" in c.lower()), None)
            if buy_col and sell_col and n_rl_col:
                n_rl = g[n_rl_col].mean()
                r = (g[buy_col].mean() + g[sell_col].mean()) / n_rl / SIM_SEC
                print(f"  φ={phi:.2f}: λ_r = {r:.4f} (n_rl={n_rl:.1f})")
            else:
                print(f"  φ={phi:.2f}: cols={list(g.columns[:5])}...")

    # ── §3.7: inventory cap comparison ──────────────────────────────────────
    print("\n§3.7 Inventory Cap Comparison (greedy, one-sided fraction)")
    cap50_df = load(CAP50, "cap50")
    cap20_df = load(CAP20, "cap20")
    if df is not None and cap50_df is not None and cap20_df is not None:
        col = "greedy_one_sided_book_fraction_mean"
        print("\n  phi  | no-cap | cap-50 | cap-20")
        for phi_val in [0.30, 0.40, 0.50]:
            row0 = df[np.isclose(df["phi"], phi_val)]
            row50 = cap50_df[np.isclose(cap50_df["phi"], phi_val)]
            row20 = cap20_df[np.isclose(cap20_df["phi"], phi_val)]
            v0 = row0[col].values[0] if len(row0) else np.nan
            v50 = row50[col].values[0] if len(row50) else np.nan
            v20 = row20[col].values[0] if len(row20) else np.nan
            print(f"  {phi_val:.2f} | {fmt(v0, True)} | {fmt(v50, True)} | {fmt(v20, True)}")

    # ── §3.8: trained vs random ──────────────────────────────────────────────
    print("\n§3.8 Trained vs Random (greedy, one-sided fraction)")
    rob = load(ROBUSTNESS, "reviewer robustness")
    if rob is not None:
        rob = rob[rob["evaluation_mode"] == "greedy"]
        rob = rob[np.isclose(rob["rl_quoter_split"].astype(float), 0.0)]
        col_m = "one_sided_book_fraction_mean"
        col_lo = "one_sided_book_fraction_ci95_lower"
        col_hi = "one_sided_book_fraction_ci95_upper"
        for phi_val in [0.30, 0.40]:
            for cond in ["trained", "random"]:
                g = rob[(rob["condition"] == cond) & np.isclose(rob["phi"].astype(float), phi_val)]
                if g.empty:
                    print(f"  φ={phi_val:.2f} {cond}: no data")
                    continue
                m = g[col_m].values[0] if col_m in g else np.nan
                lo = g[col_lo].values[0] if col_lo in g else np.nan
                hi = g[col_hi].values[0] if col_hi in g else np.nan
                print(f"  φ={phi_val:.2f} {cond}: {fmt(m, True)} [{fmt(lo, True)}, {fmt(hi, True)}]")

    # ── alt profile ──────────────────────────────────────────────────────────
    print("\n§3.9 Alt Profile Comparison (greedy, one-sided fraction)")
    alt_df = load(ALT, "alt profile")
    if alt_df is not None and df is not None:
        col = "greedy_one_sided_book_fraction_mean"
        for phi_val in [0.30, 0.40]:
            row_p = df[np.isclose(df["phi"], phi_val)]
            row_a = alt_df[np.isclose(alt_df["phi"], phi_val)]
            vp = row_p[col].values[0] if len(row_p) and col in row_p else np.nan
            va = row_a[col].values[0] if len(row_a) and col in row_a else np.nan
            print(f"  φ={phi_val:.2f}: primary={fmt(vp, True)}, alt={fmt(va, True)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
