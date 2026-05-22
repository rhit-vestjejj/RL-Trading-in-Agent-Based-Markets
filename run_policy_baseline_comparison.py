"""Build a combined comparison between trained and random-baseline phi sweeps."""

from __future__ import annotations

import argparse

from policy_baseline_comparison import (
    DEFAULT_EVALUATION_MODE,
    DEFAULT_POLICY_COMPARISON_OUTPUT_DIR,
    build_policy_baseline_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-dir", type=str, required=True)
    parser.add_argument("--random-baseline-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_POLICY_COMPARISON_OUTPUT_DIR))
    parser.add_argument("--evaluation-mode", type=str, default=DEFAULT_EVALUATION_MODE, choices=("greedy", "stochastic"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_policy_baseline_comparison(
        {
            "trained": args.trained_dir,
            "random_baseline": args.random_baseline_dir,
        },
        output_dir=args.output_dir,
        evaluation_mode=args.evaluation_mode,
    )

    print(f"Saved comparison folder to {result['output_dir']}")
    print(f"Saved comparison CSV to {result['csv_path']}")
    print(f"Saved comparison JSON to {result['json_path']}")
    print(f"Saved comparison report to {result['md_path']}")
    print("Saved comparison plots:", [str(path) for path in result["plot_paths"]])


if __name__ == "__main__":
    main()
