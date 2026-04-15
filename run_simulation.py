"""Run the ABIDES-backed first-pass market prototype."""

from __future__ import annotations

import argparse

from analysis import (
    average_spread,
    average_top_of_book_depth,
    crash_rate,
    excess_kurtosis,
    log_returns,
    max_drawdown,
    squared_return_autocorrelation,
    tail_exposure,
)
from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from config import MARKET_PROFILES, SimulationConfig
from env import POLICY_BUILDERS
from market import MarketSimulator
from visualization import save_market_plot, show_market_animation, show_market_plot


def build_parser() -> argparse.ArgumentParser:
    """Create a small CLI for running the prototype."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--market-profile",
        type=str,
        default=OFFICIAL_BASELINE_NAME,
        choices=sorted(MARKET_PROFILES),
        help="Market ecology path to run. The official baseline is abides_rmsc04_small_v1.",
    )
    parser.add_argument("--end-time", type=str, default="10:00:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument(
        "--rl-policy",
        type=str,
        default="random",
        choices=sorted(POLICY_BUILDERS),
        help="Shared-policy placeholder used by RL traders when phi > 0.",
    )
    parser.add_argument("--output", type=str, default="simulation_log.csv")
    parser.add_argument(
        "--rl-log-output",
        type=str,
        default="",
        help="Optional path for the long-form per-decision RL agent log.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="",
        help="Optional path for a saved market-over-time plot image.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the market-over-time plot after the run finishes.",
    )
    parser.add_argument(
        "--animate-plot",
        action="store_true",
        help="Replay the market path in a Python window after the run finishes.",
    )
    parser.add_argument(
        "--animation-interval-ms",
        type=int,
        default=40,
        help="Animation frame interval in milliseconds for replay mode.",
    )
    parser.add_argument(
        "--animation-tail-seconds",
        type=float,
        default=0.0,
        help="Optional trailing time window. Use 0 to keep the axes fixed and just build the graph.",
    )
    return parser


def main() -> None:
    """Run a simulation and export the resulting log."""

    args = build_parser().parse_args()
    config_builder = build_abides_rmsc04_small_v1_config if args.market_profile == OFFICIAL_BASELINE_NAME else SimulationConfig
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
    simulator.to_csv(args.output)
    if args.rl_log_output:
        simulator.extract_rl_frame().to_csv(args.rl_log_output, index=False)
    saved_plot = None
    if args.animate_plot:
        show_market_animation(
            frame,
            title=f"Market Replay (phi={args.phi}, seed={args.seed})",
            interval_ms=args.animation_interval_ms,
            tail_seconds=args.animation_tail_seconds if args.animation_tail_seconds > 0 else None,
        )
    elif args.show_plot:
        show_market_plot(
            frame,
            title=f"Market Replay (phi={args.phi}, seed={args.seed})",
        )
    if args.plot_output:
        saved_plot = save_market_plot(
            frame,
            args.plot_output,
            title=f"Market Replay (phi={args.phi}, seed={args.seed})",
        )

    returns = log_returns(frame["midprice"])
    autocorr = squared_return_autocorrelation(returns, max_lag=5)

    print(f"Market profile: {config.market_profile}")
    print("Agent counts:", config.agent_counts())
    print(f"RL policy: {config.rl_policy_name}")
    print(f"Saved log to {args.output}")
    if args.rl_log_output:
        print(f"Saved RL decision log to {args.rl_log_output}")
    if saved_plot is not None:
        print(f"Saved market plot to {saved_plot}")
    print(f"Average spread: {average_spread(frame):.6f}")
    print(f"Average top-of-book depth: {average_top_of_book_depth(frame):.6f}")
    print(f"Excess kurtosis: {excess_kurtosis(returns):.6f}")
    print(f"Tail exposure (5% ES): {tail_exposure(returns):.6f}")
    print(f"Crash rate (-5% threshold): {crash_rate(returns):.6f}")
    print(f"Max drawdown: {max_drawdown(frame['midprice']):.6f}")
    print("Squared-return autocorrelation:")
    for lag, value in autocorr.items():
        print(f"  lag {int(lag)}: {value:.6f}")


if __name__ == "__main__":
    main()
