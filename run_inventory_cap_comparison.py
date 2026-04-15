"""Build a combined robustness summary from completed inventory-cap phi sweeps."""

from __future__ import annotations

import argparse

from inventory_cap_comparison import (
    DEFAULT_COMPARISON_INPUTS,
    DEFAULT_COMPARISON_OUTPUT_DIR,
    DEFAULT_EVALUATION_MODE,
    build_inventory_cap_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cap-dir", type=str, default=str(DEFAULT_COMPARISON_INPUTS["no_cap"]))
    parser.add_argument("--cap50-dir", type=str, default=str(DEFAULT_COMPARISON_INPUTS["cap_50"]))
    parser.add_argument("--cap20-dir", type=str, default=str(DEFAULT_COMPARISON_INPUTS["cap_20"]))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_COMPARISON_OUTPUT_DIR))
    parser.add_argument("--evaluation-mode", type=str, default=DEFAULT_EVALUATION_MODE, choices=("greedy", "stochastic"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_inventory_cap_comparison(
        {
            "no_cap": args.no_cap_dir,
            "cap_50": args.cap50_dir,
            "cap_20": args.cap20_dir,
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
