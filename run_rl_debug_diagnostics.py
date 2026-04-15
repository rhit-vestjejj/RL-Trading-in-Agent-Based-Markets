"""Run an RL-integrated simulation and export compact debugging diagnostics."""

from __future__ import annotations

import argparse

import pandas as pd

from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from config import MARKET_PROFILES, SimulationConfig
from market import MarketSimulator
from rl_diagnostics import (
    build_rl_run_report,
    compute_rl_run_diagnostics,
    save_rl_run_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--phi", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--market-profile",
        type=str,
        default=OFFICIAL_BASELINE_NAME,
        choices=sorted(MARKET_PROFILES),
    )
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument("--rl-policy", type=str, default="random")
    parser.add_argument("--output", type=str, default="rl_debug_market.csv")
    parser.add_argument("--rl-log-output", type=str, default="rl_debug_decisions.csv")
    parser.add_argument("--diagnostics-output", type=str, default="rl_debug_diagnostics.csv")
    parser.add_argument("--report-output", type=str, default="rl_debug_report.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_builder = (
        build_abides_rmsc04_small_v1_config
        if args.market_profile == OFFICIAL_BASELINE_NAME
        else SimulationConfig
    )
    config = config_builder(
        num_agents=args.num_agents,
        phi=args.phi,
        seed=args.seed,
        market_profile=args.market_profile,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
        rl_policy_name=args.rl_policy,
    )
    simulator = MarketSimulator(config)
    frame = simulator.run()
    rl_frame = simulator.extract_rl_frame()

    simulator.to_csv(args.output)
    rl_frame.to_csv(args.rl_log_output, index=False)

    diagnostics = compute_rl_run_diagnostics(frame, rl_frame)
    pd.DataFrame([diagnostics]).to_csv(args.diagnostics_output, index=False)
    report = build_rl_run_report(diagnostics)
    save_rl_run_report(report, args.report_output)

    print(f"Saved market log to {args.output}")
    print(f"Saved RL decision log to {args.rl_log_output}")
    print(f"Saved diagnostics table to {args.diagnostics_output}")
    print(f"Saved diagnostics report to {args.report_output}")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
