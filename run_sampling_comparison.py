"""Run one simulation and compare multiple logging resolutions from the same path."""

from __future__ import annotations

import argparse
from pathlib import Path

from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from config import MARKET_PROFILES, SimulationConfig
from market import MarketSimulator
from sampling_comparison import (
    build_sampling_report,
    parse_frequency_list,
    save_sampling_outputs,
    summarize_sampling_frames,
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
    parser.add_argument(
        "--frequencies",
        type=str,
        default="1s,5s,15s",
        help="Comma-separated logging frequencies to compare from the same run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sampling_comparison",
        help="Directory for per-frequency CSVs and plots.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="sampling_comparison_summary.csv",
        help="Path for the summary CSV.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="sampling_comparison_report.md",
        help="Path for the sampling comparison report.",
    )
    parser.add_argument(
        "--baseline-frequency",
        type=str,
        default="1s",
        help="Reference frequency for the stability report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frequencies = parse_frequency_list(args.frequencies)
    config_builder = build_abides_rmsc04_small_v1_config if args.market_profile == OFFICIAL_BASELINE_NAME else SimulationConfig
    config = config_builder(
        num_agents=args.num_agents,
        phi=args.phi,
        seed=args.seed,
        market_profile=args.market_profile,
        end_time=args.end_time,
        log_frequency=frequencies[0],
    )
    simulator = MarketSimulator(config)
    simulator.run()
    frames = simulator.extract_frames(frequencies)

    save_sampling_outputs(
        frames,
        output_dir=args.output_dir,
        title_prefix=f"Sampling Comparison (phi={args.phi}, seed={args.seed})",
    )
    summary = summarize_sampling_frames(frames)
    summary.to_csv(args.summary_output, index=False)

    report = build_sampling_report(
        summary,
        baseline_frequency=args.baseline_frequency,
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved per-frequency outputs to {args.output_dir}")
    print(f"Saved sampling summary to {args.summary_output}")
    print(f"Saved sampling report to {args.report_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
