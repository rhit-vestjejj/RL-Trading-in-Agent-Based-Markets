"""Run controlled baseline parameter sweeps for the ABIDES-backed prototype."""

from __future__ import annotations

import argparse

from calibration import (
    build_parameter_grid,
    consecutive_seeds,
    parse_sweep_spec,
    recommend_baseline_setting,
    run_parameter_sweep,
    save_sweep_plots,
)
from config import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="Sweep spec of the form name=v1,v2,... Repeat for Cartesian grids.",
    )
    parser.add_argument("--runs-output", type=str, default="parameter_sweep_runs.csv")
    parser.add_argument("--summary-output", type=str, default="parameter_sweep_summary.csv")
    parser.add_argument("--plot-dir", type=str, default="")
    return parser


def parse_seed_list(raw_seeds: str, start_seed: int, num_runs: int) -> list[int]:
    if raw_seeds.strip():
        return [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]
    return consecutive_seeds(start_seed, num_runs)


def main() -> None:
    args = build_parser().parse_args()
    sweep_specs = [parse_sweep_spec(spec) for spec in args.sweep]
    parameter_columns, parameter_grid = build_parameter_grid(sweep_specs)
    seeds = parse_seed_list(args.seeds, args.start_seed, args.num_runs)

    base_config = SimulationConfig(
        num_agents=args.num_agents,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
    )

    runs, summary = run_parameter_sweep(
        base_config,
        seeds=seeds,
        parameter_grid=parameter_grid,
        parameter_columns=parameter_columns,
    )
    runs.to_csv(args.runs_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    saved_plots = []
    if args.plot_dir:
        saved_plots = save_sweep_plots(
            summary,
            parameter_columns=parameter_columns,
            output_dir=args.plot_dir,
        )

    recommendation = recommend_baseline_setting(summary)
    print(f"Saved run-level sweep results to {args.runs_output}")
    print(f"Saved grouped sweep summary to {args.summary_output}")
    if saved_plots:
        print(f"Saved {len(saved_plots)} plots to {args.plot_dir}")
    print("Top-ranked baseline setting:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
