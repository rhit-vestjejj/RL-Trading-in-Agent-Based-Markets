"""Build a comparison between trained baseline behavior and trained anti-degeneracy behavior."""

from __future__ import annotations

import argparse

from behavior_mode_comparison import (
    DEFAULT_BEHAVIOR_MODE_COMPARISON_OUTPUT_DIR,
    DEFAULT_EVALUATION_MODE,
    build_behavior_mode_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=str, required=True)
    parser.add_argument("--anti-degeneracy-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_BEHAVIOR_MODE_COMPARISON_OUTPUT_DIR))
    parser.add_argument("--evaluation-mode", type=str, default=DEFAULT_EVALUATION_MODE, choices=("greedy", "stochastic"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_behavior_mode_comparison(
        {
            "baseline": args.baseline_dir,
            "anti_degeneracy": args.anti_degeneracy_dir,
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
