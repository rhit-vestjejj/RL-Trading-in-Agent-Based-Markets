"""Run a narrow phi=0 ablation to diagnose price-discovery damping."""

from __future__ import annotations

import argparse

from ablation import (
    parse_ablation_specs,
    run_price_discovery_ablation,
    summarize_ablation_runs,
)
from calibration import consecutive_seeds
from config import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--end-time", type=str, default="09:45:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument(
        "--ablate",
        action="append",
        default=[],
        help="Ablation spec of the form name=v1,v2,... Repeat for Cartesian grids.",
    )
    parser.add_argument("--runs-output", type=str, default="price_discovery_ablation_runs.csv")
    parser.add_argument("--summary-output", type=str, default="price_discovery_ablation_summary.csv")
    return parser


def parse_seed_list(raw_seeds: str, start_seed: int, num_runs: int) -> list[int]:
    if raw_seeds.strip():
        return [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]
    return consecutive_seeds(start_seed, num_runs)


def main() -> None:
    args = build_parser().parse_args()
    parameter_columns, parameter_grid = parse_ablation_specs(args.ablate)
    seeds = parse_seed_list(args.seeds, args.start_seed, args.num_runs)

    base_config = SimulationConfig(
        num_agents=args.num_agents,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
        phi=0.0,
    )
    runs = run_price_discovery_ablation(
        base_config,
        seeds=seeds,
        parameter_grid=parameter_grid,
        parameter_columns=parameter_columns,
    )
    summary = summarize_ablation_runs(runs, parameter_columns=parameter_columns)

    runs.to_csv(args.runs_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"Saved ablation runs to {args.runs_output}")
    print(f"Saved ablation summary to {args.summary_output}")
    if not summary.empty:
        print(summary.head().to_string(index=False))


if __name__ == "__main__":
    main()
