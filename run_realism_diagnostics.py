"""Run a phi=0 baseline simulation and generate a realism diagnostics report."""

from __future__ import annotations

import argparse

import pandas as pd

from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from config import MARKET_PROFILES, SimulationConfig
from logging_utils import (
    extract_limit_order_lifecycle_dataframe,
    extract_trade_history_dataframe,
)
from market import MarketSimulator
from realism_diagnostics import (
    build_realism_report,
    compute_realism_diagnostics,
    flag_realism_pathologies,
    save_realism_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--market-profile",
        type=str,
        default=OFFICIAL_BASELINE_NAME,
        choices=sorted(MARKET_PROFILES),
    )
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument("--output", type=str, default="realism_run.csv")
    parser.add_argument("--diagnostics-output", type=str, default="realism_diagnostics.csv")
    parser.add_argument("--report-output", type=str, default="realism_report.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_builder = build_abides_rmsc04_small_v1_config if args.market_profile == OFFICIAL_BASELINE_NAME else SimulationConfig
    config = config_builder(
        num_agents=args.num_agents,
        phi=args.phi,
        seed=args.seed,
        market_profile=args.market_profile,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
    )
    simulator = MarketSimulator(config)
    frame = simulator.run()
    simulator.to_csv(args.output)
    passive_orders = extract_limit_order_lifecycle_dataframe(simulator.end_state)
    trade_history = extract_trade_history_dataframe(simulator.end_state, config.ticker)

    diagnostics = compute_realism_diagnostics(
        frame,
        tick_size=config.tick_size,
        passive_orders=passive_orders,
        trade_history=trade_history,
    )
    flags = flag_realism_pathologies(diagnostics)
    report = build_realism_report(diagnostics, flags)

    pd.DataFrame([diagnostics | flags]).to_csv(args.diagnostics_output, index=False)
    save_realism_report(report, args.report_output)

    print(f"Saved run log to {args.output}")
    print(f"Saved diagnostics table to {args.diagnostics_output}")
    print(f"Saved realism report to {args.report_output}")
    for key, value in diagnostics.items():
        print(f"{key}: {value}")
    print("flags:", ", ".join(name for name, active in flags.items() if active) or "none")


if __name__ == "__main__":
    main()
