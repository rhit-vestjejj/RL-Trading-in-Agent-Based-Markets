"""
Re-evaluate trained policies on additional seeds to tighten confidence intervals.

Loads the final checkpoint for each phi from an existing phi-sweep experiment
and evaluates on new seeds (default: 10, 11, 12). Merges with the original
per-seed data and produces updated summary CSVs and plots with narrower CIs.

Usage:
    python run_more_seeds.py [--experiment-dir EXPERIMENT_DIR]
                             [--additional-seeds 10,11,12]
                             [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DEFAULT_EXPERIMENT = "experiments/paper_trained_nocap"
DEFAULT_NEW_SEEDS = [10, 11, 12]
DEFAULT_OUTPUT = "experiments/more_seeds"
PAPER_DIR = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")

PHI_GRID = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
EVAL_MODES = ["greedy", "stochastic"]


def _load_matplotlib():
    cache = Path(tempfile.gettempdir()) / "more_seeds_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def evaluate_new_seeds(
    experiment_dir: Path,
    new_seeds: list[int],
    output_dir: Path,
) -> None:
    from baseline_configs import build_abides_rmsc04_small_v1_config
    from phi_experiment import compute_extended_market_metrics
    from ppo_training import load_policy_artifact, run_policy_episode, summarize_episode
    from rl_diagnostics import compute_policy_evaluation_diagnostics

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original experiment config for parameter reproducibility
    import json
    config_path = experiment_dir / "config" / "experiment_config.json"
    exp_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    end_time = exp_config.get("end_time", "09:35:00")
    log_frequency = exp_config.get("log_frequency", "1s")
    num_agents = exp_config.get("num_agents", 102)
    return_window = exp_config.get("return_window", 10)
    lambda_q = exp_config.get("lambda_q", 0.01)
    flat_hold_penalty = exp_config.get("flat_hold_penalty", 0.02)
    inventory_cap = exp_config.get("inventory_cap", None)

    new_diag_rows: list[dict] = []
    new_mkt_rows: list[dict] = []

    for phi in PHI_GRID:
        phi_label = f"{phi:.2f}"
        print(f"\n── phi={phi_label} ──────────────────────────")

        checkpoint_path = (
            experiment_dir / f"phi_{phi_label}" / "checkpoints" / "shared_ppo_policy_final.npz"
        )

        if phi == 0.0:
            policy = None
        elif not checkpoint_path.exists():
            print(f"  No checkpoint found, skipping: {checkpoint_path}")
            continue
        else:
            # We evaluate both greedy and stochastic using the same checkpoint
            policy = None  # loaded per-mode below

        for mode in EVAL_MODES:
            deterministic = mode == "greedy"

            if phi > 0.0:
                policy = load_policy_artifact(str(checkpoint_path), deterministic=deterministic)

            for seed in new_seeds:
                print(f"  {mode} seed={seed}...", end=" ", flush=True)
                config = build_abides_rmsc04_small_v1_config(
                    phi=phi,
                    seed=seed,
                    num_agents=num_agents,
                    end_time=end_time,
                    log_frequency=log_frequency,
                    return_window=return_window,
                    lambda_q=lambda_q,
                    flat_hold_penalty=flat_hold_penalty,
                    inventory_cap=inventory_cap,
                )
                market_frame, rl_frame, transition_frame = run_policy_episode(
                    config, shared_policy=policy
                )
                diag, _ = compute_policy_evaluation_diagnostics(
                    market_frame, rl_frame, transition_frame,
                    policy=policy, inventory_cap=inventory_cap,
                )
                diag = dict(diag)
                diag["phi"] = phi
                diag["seed"] = seed
                diag["evaluation_mode"] = mode
                new_diag_rows.append(diag)

                mkt = compute_extended_market_metrics(market_frame, tick_size=float(config.tick_size))
                mkt["phi"] = phi
                mkt["seed"] = seed
                mkt["evaluation_mode"] = mode
                new_mkt_rows.append(mkt)
                print("done")

    # ── Save new-seed raw rows ───────────────────────────────────────────────
    new_diag_df = pd.DataFrame(new_diag_rows)
    new_mkt_df = pd.DataFrame(new_mkt_rows)
    new_diag_df.to_csv(output_dir / "new_seeds_diagnostics.csv", index=False)
    new_mkt_df.to_csv(output_dir / "new_seeds_market_metrics.csv", index=False)

    # ── Merge with original per-seed data ────────────────────────────────────
    orig_diag_path = experiment_dir / "summaries" / "per_seed_rl_diagnostics.csv"
    orig_mkt_path = experiment_dir / "summaries" / "per_seed_market_metrics.csv"
    orig_summary_path = experiment_dir / "phi_sweep_summary.csv"

    orig_diag = pd.read_csv(orig_diag_path) if orig_diag_path.exists() else pd.DataFrame()
    orig_mkt = pd.read_csv(orig_mkt_path) if orig_mkt_path.exists() else pd.DataFrame()
    orig_summary = pd.read_csv(orig_summary_path) if orig_summary_path.exists() else pd.DataFrame()

    merged_diag = pd.concat([orig_diag, new_diag_df], ignore_index=True)
    merged_mkt = pd.concat([orig_mkt, new_mkt_df], ignore_index=True)
    merged_diag.to_csv(output_dir / "merged_diagnostics.csv", index=False)
    merged_mkt.to_csv(output_dir / "merged_market_metrics.csv", index=False)

    # ── Recompute summary with all seeds ─────────────────────────────────────
    summary_rows = []
    key_metrics = [
        "one_sided_book_fraction",
        "undefined_midprice_fraction",
        "max_consecutive_one_sided_duration",
        "zero_return_fraction",
        "inactivity_fraction",
        "average_top_of_book_depth",
    ]
    for phi in PHI_GRID:
        row = {"phi": phi}
        for mode in EVAL_MODES:
            sub = merged_mkt[(np.isclose(merged_mkt["phi"], phi)) & (merged_mkt["evaluation_mode"] == mode)]
            for metric in key_metrics:
                if metric not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                n = len(vals)
                if n == 0:
                    continue
                mean = float(vals.mean())
                std = float(vals.std(ddof=1)) if n > 1 else float("nan")
                stderr = float(std / np.sqrt(n)) if n > 1 else float("nan")
                ci_hw = 1.96 * stderr if not np.isnan(stderr) else float("nan")
                row[f"{mode}_{metric}_mean"] = mean
                row[f"{mode}_{metric}_std"] = std
                row[f"{mode}_{metric}_stderr"] = stderr
                row[f"{mode}_{metric}_ci95_lower"] = mean - ci_hw
                row[f"{mode}_{metric}_ci95_upper"] = mean + ci_hw
                row[f"{mode}_{metric}_n"] = float(n)
        summary_rows.append(row)

    updated_summary = pd.DataFrame(summary_rows).sort_values("phi").reset_index(drop=True)
    updated_summary.to_csv(output_dir / "updated_summary.csv", index=False)
    print(f"\nSaved updated summary to {output_dir / 'updated_summary.csv'}")

    # ── Generate updated one-sided book fraction plot ─────────────────────────
    _make_updated_plots(updated_summary, orig_summary, output_dir)


def _make_updated_plots(
    updated: pd.DataFrame,
    original: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt = _load_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    col = "one_sided_book_fraction"

    for ax, mode, color in [
        (axes[0], "greedy", "tab:blue"),
        (axes[1], "stochastic", "tab:orange"),
    ]:
        phi_u = updated["phi"].values
        mean_u = updated[f"{mode}_{col}_mean"].values
        lo_u = np.clip(updated[f"{mode}_{col}_ci95_lower"].values, 0, None)
        hi_u = updated[f"{mode}_{col}_ci95_upper"].values

        ax.errorbar(phi_u, mean_u,
                    yerr=np.vstack([mean_u - lo_u, hi_u - mean_u]),
                    marker="o", lw=1.8, capsize=3, color=color,
                    label="Updated (all seeds)")

        if not original.empty and f"{mode}_{col}_mean" in original.columns:
            phi_o = original["phi"].values
            mean_o = original[f"{mode}_{col}_mean"].values
            lo_o = np.clip(original[f"{mode}_{col}_ci95_lower"].values, 0, None)
            hi_o = original[f"{mode}_{col}_ci95_upper"].values
            ax.errorbar(phi_o + 0.003, mean_o,
                        yerr=np.vstack([mean_o - lo_o, hi_o - mean_o]),
                        marker="s", lw=1.2, capsize=3, color=color,
                        alpha=0.45, ls="--", label="Original (3 seeds)")

        ax.set_xlabel("$\\phi$")
        ax.set_ylabel("One-sided book fraction")
        ax.set_title(f"{mode.capitalize()} — One-Sided Book Fraction")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    plot_path = output_dir / "updated_one_sided_fraction.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved updated plot to {plot_path}")

    # Also copy to paper dir
    paper_copy = PAPER_DIR / "updated_one_sided_fraction.png"
    if PAPER_DIR.exists():
        import shutil
        shutil.copy(plot_path, paper_copy)
        print(f"Copied to {paper_copy}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--additional-seeds",
        default=",".join(str(s) for s in DEFAULT_NEW_SEEDS),
        help="Comma-separated list of new evaluation seeds",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    new_seeds = [int(s.strip()) for s in args.additional_seeds.split(",") if s.strip()]
    evaluate_new_seeds(
        experiment_dir=Path(args.experiment_dir),
        new_seeds=new_seeds,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
