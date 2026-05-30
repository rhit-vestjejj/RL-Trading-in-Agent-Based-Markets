"""Generate figure11.png — one-sided book fraction by policy type (trained vs random).

Reads experiments/regen_reviewer_robustness/summaries/composition_control_summary.csv
and plots one-sided book fraction vs phi for the trained and random conditions
(taker-only quoter split = 0.0).

Output: Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller/figure11.png
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_DIR = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")
ROBUSTNESS_DIR = Path("experiments/regen_reviewer_robustness")
AGGREGATE_CSV = ROBUSTNESS_DIR / "summaries" / "composition_control_summary.csv"


def main() -> None:
    cache = Path(tempfile.gettempdir()) / "fig11_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not AGGREGATE_CSV.exists():
        raise SystemExit(f"Missing: {AGGREGATE_CSV}\nRun run_reviewer_robustness.py first.")

    df = pd.read_csv(AGGREGATE_CSV)
    df = df[df["evaluation_mode"] == "greedy"].copy()
    df = df[np.isclose(df["rl_quoter_split"].astype(float), 0.0)].copy()

    metric = "one_sided_book_fraction"
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_ci95_lower"
    hi_col = f"{metric}_ci95_upper"

    conditions = [c for c in ["trained", "random"] if c in df["condition"].values]
    colors = {"trained": "tab:blue", "random": "tab:orange"}
    labels = {"trained": "Trained RL policy", "random": "Random baseline"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond in conditions:
        sub = df[df["condition"] == cond].sort_values("phi")
        x = sub["phi"].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float) if mean_col in sub else np.full(len(x), np.nan)
        if lo_col in sub and hi_col in sub:
            lo = np.clip(sub[lo_col].to_numpy(dtype=float), 0, None)
            hi = sub[hi_col].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]),
                        marker="o", lw=1.8, capsize=3.5,
                        color=colors.get(cond, None),
                        label=labels.get(cond, cond))
        else:
            ax.plot(x, y, marker="o", lw=1.8,
                    color=colors.get(cond, None),
                    label=labels.get(cond, cond))

    ax.set_xlabel(r"$\phi$ (RL participation fraction)")
    ax.set_ylabel("One-sided book fraction")
    ax.set_title("One-Sided Book Fraction: Trained vs. Random Policy")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    fig.tight_layout()

    out = ROBUSTNESS_DIR / "figure11.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    if PAPER_DIR.exists():
        import shutil
        shutil.copy(out, PAPER_DIR / "figure11.png")
        print(f"Copied to {PAPER_DIR / 'figure11.png'}")

    plt.close(fig)


if __name__ == "__main__":
    main()
