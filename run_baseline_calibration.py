"""Run a small baseline calibration sweep for the ABIDES-backed prototype."""

from __future__ import annotations

import argparse

from calibration import consecutive_seeds, recommend_baseline_setting, run_baseline_calibration
from config import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument("--runs-output", type=str, default="baseline_calibration_runs.csv")
    parser.add_argument("--summary-output", type=str, default="baseline_calibration_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig(
        num_agents=args.num_agents,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
    )
    runs, summary = run_baseline_calibration(
        config,
        seeds=consecutive_seeds(args.start_seed, args.num_runs),
    )
    runs.to_csv(args.runs_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    recommendation = recommend_baseline_setting(summary)
    print(f"Saved run-level calibration metrics to {args.runs_output}")
    print(f"Saved summary calibration metrics to {args.summary_output}")
    print(
        "Top baseline score:",
        f"{recommendation['ranking_score']:.3f}",
        f"classification share stable={recommendation.get('share_stable', 0.0):.3f}",
    )


if __name__ == "__main__":
    main()
