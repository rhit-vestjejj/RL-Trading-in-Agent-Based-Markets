"""Generate all paper figures from completed experiment data.

Run this after all four phi-sweep experiments have finished:
  - experiments/regen_headline_taker/phi_sweep_summary.csv
  - experiments/regen_cap50/phi_sweep_summary.csv
  - experiments/regen_cap20/phi_sweep_summary.csv
  - experiments/regen_alt_profile/phi_sweep_summary.csv

Then optionally after reviewer robustness sweep:
  - experiments/regen_reviewer_robustness/summaries/composition_control_summary.csv

Usage:
    python generate_all_figures.py [--skip-reviewer-robustness]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REQUIRED = [
    Path("experiments/regen_headline_taker/phi_sweep_summary.csv"),
    Path("experiments/regen_cap50/phi_sweep_summary.csv"),
    Path("experiments/regen_cap20/phi_sweep_summary.csv"),
    Path("experiments/regen_alt_profile/phi_sweep_summary.csv"),
]


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: {cmd[0]} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-reviewer-robustness", action="store_true")
    args = parser.parse_args()

    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        print("Missing required data files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    print("=== Step 1: regenerate per-seed figures (headline sweep) ===")
    run([sys.executable, "regenerate_paper_figures.py"])

    print("\n=== Step 2: generate evidence figures (shutdown, inventory caps) ===")
    run([sys.executable, "export_paper_evidence.py"])

    print("\n=== Step 3: generate threshold analysis figure ===")
    run([
        sys.executable, "threshold_analysis.py",
        "--experiment-dir", "experiments/regen_headline_taker",
        "--output-dir", "Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller",
    ])

    if not args.skip_reviewer_robustness:
        robustness_csv = Path(
            "experiments/regen_reviewer_robustness/summaries/composition_control_summary.csv"
        )
        if robustness_csv.exists():
            print("\n=== Step 4: generate figure11 (trained vs random) ===")
            run([sys.executable, "generate_figure11.py"])
        else:
            print(f"\nSkipping figure11: {robustness_csv} not found.")
            print("Run: python run_reviewer_robustness.py --trained-experiment-dir "
                  "experiments/regen_headline_taker --output-dir "
                  "experiments/regen_reviewer_robustness")

    print("\n=== Done ===")
    paper_dir = Path("Persistent_One_Sided_Order_Books_from_Learned_Trading_Behavior_smaller")
    if paper_dir.exists():
        pngs = sorted(paper_dir.glob("*.png"))
        print(f"Paper directory contains {len(pngs)} PNG files:")
        for p in pngs:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
