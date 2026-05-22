"""
Run the alternative-profile phi-sweep in parallel — one subprocess per phi value.

Each phi trains and evaluates independently in its own directory. When all
subprocesses finish, results are merged into a single combined experiment
directory with a unified summary CSV and comparison figure.

Usage (on the big server):
    python3 -u run_alt_profile_parallel.py \
        --phi-grid 0.00,0.05,0.10,0.20,0.30,0.40,0.50,0.60 \
        --episodes 50 \
        --output-dir experiments/alt_profile_parallel \
        --workers 8

Workers defaults to the number of phi values (fully parallel).
Set --workers lower if you want to limit CPU usage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ALT_PROFILE = "abides_baseline_v1"
PAPER_DIR = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")


def run_single_phi(
    phi: float,
    episodes: int,
    output_dir: Path,
    python: str,
    evaluation_seeds: str,
) -> tuple[float, int]:
    phi_label = f"{phi:.2f}"
    phi_output = output_dir / f"chunk_phi_{phi_label}"
    log_path = output_dir / f"log_phi_{phi_label}.txt"

    # phi=0 has no RL agents — training is skipped automatically, use 1 episode
    ep = 1 if phi == 0.0 else episodes

    cmd = [
        python, "-u", "run_alt_profile_experiment.py",
        "--phi-grid", phi_label,
        "--episodes", str(ep),
        "--output-dir", str(phi_output),
        "--evaluation-seeds", evaluation_seeds,
    ]
    print(f"  [phi={phi_label}] starting → {phi_output}", flush=True)
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    status = "done" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"  [phi={phi_label}] {status}", flush=True)
    return phi, result.returncode


def merge_chunks(
    phi_grid: list[float],
    output_dir: Path,
    episodes: int,
    evaluation_seeds: list[int] | None = None,
) -> Path:
    """Merge per-phi chunk dirs into a single combined experiment directory."""
    import os, tempfile
    cache = Path(tempfile.gettempdir()) / "alt_parallel_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = output_dir / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    (combined / "summaries").mkdir(exist_ok=True)
    (combined / "plots").mkdir(exist_ok=True)

    summary_rows: list[pd.DataFrame] = []
    all_mkt_rows: list[pd.DataFrame] = []
    all_diag_rows: list[pd.DataFrame] = []

    for phi in phi_grid:
        phi_label = f"{phi:.2f}"
        chunk_dir = output_dir / f"chunk_phi_{phi_label}"
        phi_sweep_csv = chunk_dir / "phi_sweep_summary.csv"
        mkt_csv = chunk_dir / "summaries" / "per_seed_market_metrics.csv"
        diag_csv = chunk_dir / "summaries" / "per_seed_rl_diagnostics.csv"
        phi_subdir = chunk_dir / f"phi_{phi_label}"

        if not phi_sweep_csv.exists():
            print(f"  WARNING: no summary for phi={phi_label}, skipping")
            continue

        # Copy the phi_X.XX subdirectory into combined
        import shutil
        dest = combined / f"phi_{phi_label}"
        if dest.exists():
            shutil.rmtree(dest)
        if phi_subdir.exists():
            shutil.copytree(phi_subdir, dest)

        summary_rows.append(pd.read_csv(phi_sweep_csv))
        if mkt_csv.exists():
            all_mkt_rows.append(pd.read_csv(mkt_csv))
        if diag_csv.exists():
            all_diag_rows.append(pd.read_csv(diag_csv))

    if not summary_rows:
        print("ERROR: no chunk summaries found — check individual log files")
        return combined

    combined_summary = pd.concat(summary_rows, ignore_index=True).sort_values("phi").reset_index(drop=True)
    combined_summary.to_csv(combined / "phi_sweep_summary.csv", index=False)

    if all_mkt_rows:
        pd.concat(all_mkt_rows, ignore_index=True).to_csv(
            combined / "summaries" / "per_seed_market_metrics.csv", index=False
        )
    if all_diag_rows:
        pd.concat(all_diag_rows, ignore_index=True).to_csv(
            combined / "summaries" / "per_seed_rl_diagnostics.csv", index=False
        )

    # Write config
    config = {
        "market_profile": ALT_PROFILE,
        "phi_grid": phi_grid,
        "episodes": episodes,
        "evaluation_seeds": evaluation_seeds if evaluation_seeds is not None else [7, 8, 9],
    }
    (combined / "experiment_config.json").write_text(json.dumps(config, indent=2))

    # Generate cross-phi plots for the combined summary
    from phi_experiment import save_cross_phi_plots
    save_cross_phi_plots(combined_summary, combined / "plots")

    print(f"\nMerged {len(summary_rows)} phi chunks → {combined}")
    print(f"  Summary: {combined / 'phi_sweep_summary.csv'}")

    # Comparison figure vs primary
    _make_comparison_figure(combined_summary, combined)

    return combined


def _make_comparison_figure(alt_summary: pd.DataFrame, output_dir: Path) -> None:
    import os, tempfile, shutil
    cache = Path(tempfile.gettempdir()) / "alt_parallel_mpl2"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_path = Path("experiments/paper_trained_nocap/phi_sweep_summary.csv")
    has_primary = primary_path.exists()
    pri = pd.read_csv(primary_path).sort_values("phi") if has_primary else None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    col = "one_sided_book_fraction"

    for ax, mode, pri_color, alt_color in [
        (axes[0], "greedy", "tab:blue", "tab:cyan"),
        (axes[1], "stochastic", "tab:orange", "goldenrod"),
    ]:
        mean_col = f"{mode}_{col}_mean"
        lo_col = f"{mode}_{col}_ci95_lower"
        hi_col = f"{mode}_{col}_ci95_upper"

        if mean_col in alt_summary.columns:
            phi_a = alt_summary["phi"].values
            mean_a = alt_summary[mean_col].values
            lo_a = np.clip(alt_summary[lo_col].values if lo_col in alt_summary.columns else mean_a, 0, None)
            hi_a = alt_summary[hi_col].values if hi_col in alt_summary.columns else mean_a
            ax.errorbar(phi_a, mean_a, yerr=np.vstack([mean_a - lo_a, hi_a - mean_a]),
                        marker="o", lw=1.8, capsize=3, color=alt_color,
                        label="Alt profile (ZIC + 15 MM)")

        if has_primary and mean_col in pri.columns:
            phi_p = pri["phi"].values
            mean_p = pri[mean_col].values
            lo_p = np.clip(pri[lo_col].values if lo_col in pri.columns else mean_p, 0, None)
            hi_p = pri[hi_col].values if hi_col in pri.columns else mean_p
            ax.errorbar(phi_p + 0.004, mean_p, yerr=np.vstack([mean_p - lo_p, hi_p - mean_p]),
                        marker="s", lw=1.4, capsize=3, color=pri_color, alpha=0.6, ls="--",
                        label="Primary (RMSC04: noise + 2 MM)")

        ax.set_xlabel("$\\phi$ (RL participation fraction)")
        ax.set_ylabel("One-sided book fraction")
        ax.set_title(f"{mode.capitalize()}: Primary vs. Alternative Profile")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)

    fig.tight_layout()
    out = output_dir / "alt_profile_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Comparison figure: {out}")

    if PAPER_DIR.exists():
        import shutil
        shutil.copy(out, PAPER_DIR / "alt_profile_comparison.png")
        print(f"  Copied to paper dir")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-grid", default="0.00,0.05,0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output-dir", default="experiments/alt_profile_parallel")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = one per phi value)")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use for subprocesses")
    parser.add_argument("--evaluation-seeds", default="7,8,9,10,11,12",
                        help="Comma-separated evaluation seeds (more seeds = tighter CIs)")
    args = parser.parse_args()

    phi_grid = [float(p.strip()) for p in args.phi_grid.split(",") if p.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers > 0 else len(phi_grid)

    print(f"Alt profile parallel phi-sweep")
    print(f"  profile:  {ALT_PROFILE}")
    print(f"  phi grid: {phi_grid}")
    print(f"  episodes: {args.episodes}")
    print(f"  workers:  {workers} (of {len(phi_grid)} phi values)")
    print(f"  output:   {output_dir}")
    print(f"  python:   {args.python}")
    print()

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_single_phi, phi, args.episodes, output_dir, args.python, args.evaluation_seeds): phi
            for phi in phi_grid
        }
        for future in as_completed(futures):
            phi, code = future.result()
            if code != 0:
                failed.append(phi)

    if failed:
        print(f"\nWARNING: {len(failed)} phi values failed: {failed}")
        print("Check log files in", output_dir)
    else:
        print(f"\nAll {len(phi_grid)} phi values completed successfully.")

    print("\nMerging results...")
    seeds_list = [int(s) for s in args.evaluation_seeds.split(",") if s.strip()]
    combined = merge_chunks(phi_grid, output_dir, args.episodes, seeds_list)
    print(f"\nDone. Combined results at: {combined}")


if __name__ == "__main__":
    main()
