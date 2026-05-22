"""
Run the phi-sweep experiment on an alternative market profile.

Uses the abides_baseline_v1 profile (50% ZIC traders, 15% noise, 20% value,
15% market makers) instead of the primary abides_rmsc04_small_v1 profile
(80% noise, 20% value, 2% market makers). This constitutes a substantially
different agent composition and serves as a robustness check: if the
one-sided order book failure persists in a different market ecology, it
strengthens the case that the effect is driven by learned behavior rather
than an artifact of the specific baseline configuration.

Key differences from the primary experiment:
  - ZIC traders replace noise traders as the dominant flow provider
  - 15 market makers (vs 2): much stronger passive liquidity provision
  - The theoretical threshold phi* shifts because r_mm and r_noise change

Usage:
    python run_alt_profile_experiment.py [--episodes N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phi_experiment import (
    DEFAULT_EVALUATION_SEEDS,
    parse_float_list,
    run_phi_experiment,
)
from ppo_training import PPOHyperparameters

ALT_PROFILE = "abides_baseline_v1"
DEFAULT_OUTPUT = "experiments/alt_profile"

# Use the same phi grid as the primary experiment.
# abides_baseline_v1 has max_phi = 0.65 so all values are in range.
PHI_GRID = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-grid", default=",".join(f"{p:.2f}" for p in PHI_GRID))
    parser.add_argument("--episodes", type=int, default=25,
                        help="Training episodes per phi (25 is enough for robustness check)")
    parser.add_argument("--evaluation-seeds", default=",".join(str(s) for s in DEFAULT_EVALUATION_SEEDS))
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--end-time", default="09:35:00")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    # Match primary experiment hyperparameters
    parser.add_argument("--lambda-q", type=float, default=0.01)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.02)
    parser.add_argument("--actor-lr", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phi_grid = parse_float_list(args.phi_grid)
    evaluation_seeds = [int(s) for s in args.evaluation_seeds.split(",") if s.strip()]
    output_dir = Path(args.output_dir)

    print(f"Running alternative profile experiment: {ALT_PROFILE}")
    print(f"  phi grid:         {phi_grid}")
    print(f"  episodes:         {args.episodes}")
    print(f"  evaluation seeds: {evaluation_seeds}")
    print(f"  output:           {output_dir}")
    print()
    print("Agent composition at phi=0 (abides_baseline_v1):")
    print("  ~51 ZIC traders, ~15 noise traders, ~20 value traders, ~15 market makers")
    print()

    result = run_phi_experiment(
        phi_grid=phi_grid,
        episodes=args.episodes,
        start_seed=7,
        evaluation_seeds=evaluation_seeds,
        output_dir=output_dir,
        end_time=args.end_time,
        log_frequency="1s",
        num_agents=args.num_agents,
        return_window=10,
        lambda_q=args.lambda_q,
        flat_hold_penalty=args.flat_hold_penalty,
        inventory_cap=None,
        rl_liquidity_mode="taker_only",
        market_profile=ALT_PROFILE,
        hyperparameters=PPOHyperparameters(
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
        ),
    )

    print(f"\nSaved experiment to {result['output_dir']}")
    print(f"Summary CSV:        {Path(result['output_dir']) / 'phi_sweep_summary.csv'}")

    # Generate comparison figure vs primary experiment
    _make_comparison_figure(
        alt_summary_path=Path(result["output_dir"]) / "phi_sweep_summary.csv",
        primary_summary_path=Path("experiments/paper_trained_nocap/phi_sweep_summary.csv"),
        output_dir=Path(result["output_dir"]),
    )


def _make_comparison_figure(
    alt_summary_path: Path,
    primary_summary_path: Path,
    output_dir: Path,
) -> None:
    import os
    import tempfile

    cache = Path(tempfile.gettempdir()) / "alt_profile_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    paper_dir = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")

    alt = pd.read_csv(alt_summary_path).sort_values("phi")
    if not primary_summary_path.exists():
        print("Primary summary not found; skipping comparison figure.")
        return
    pri = pd.read_csv(primary_summary_path).sort_values("phi")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, mode, pri_color, alt_color in [
        (axes[0], "greedy", "tab:blue", "tab:cyan"),
        (axes[1], "stochastic", "tab:orange", "goldenrod"),
    ]:
        col = f"{mode}_one_sided_book_fraction_mean"
        lo_col = f"{mode}_one_sided_book_fraction_ci95_lower"
        hi_col = f"{mode}_one_sided_book_fraction_ci95_upper"

        def _plot(df: pd.DataFrame, label: str, color: str, offset: float = 0.0) -> None:
            if col not in df.columns:
                return
            phi = df["phi"].values + offset
            mean = df[col].values
            lo = np.clip(df[lo_col].values if lo_col in df.columns else mean, 0, None)
            hi = df[hi_col].values if hi_col in df.columns else mean
            ax.errorbar(phi, mean, yerr=np.vstack([mean - lo, hi - mean]),
                        marker="o", lw=1.8, capsize=3, color=color, label=label)

        _plot(pri, "Primary (RMSC04: 80 noise, 2 MM)", pri_color)
        _plot(alt, f"Alt (baseline_v1: ZIC + 15 MM)", alt_color, offset=0.004)

        ax.set_xlabel("$\\phi$ (RL participation fraction)")
        ax.set_ylabel("One-sided book fraction")
        ax.set_title(f"{mode.capitalize()} — Primary vs. Alternative Profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    fig.savefig(output_dir / "alt_profile_comparison.png", dpi=150)
    if paper_dir.exists():
        import shutil
        shutil.copy(output_dir / "alt_profile_comparison.png",
                    paper_dir / "alt_profile_comparison.png")
        print(f"Copied comparison figure to {paper_dir / 'alt_profile_comparison.png'}")
    plt.close(fig)
    print(f"Saved comparison figure to {output_dir / 'alt_profile_comparison.png'}")


if __name__ == "__main__":
    main()
