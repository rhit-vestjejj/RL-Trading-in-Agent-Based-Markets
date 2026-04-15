"""Evaluate a random, scripted, or trained shared policy on the frozen baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from env import build_policy
from ppo_training import evaluate_policy, load_policy_artifact
from rl_diagnostics import (
    build_policy_evaluation_report,
    compute_policy_evaluation_diagnostics,
)
from visualization import save_market_plot, show_market_animation, show_market_plot


def _parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=["random", "inventory_aware", "random_quoter", "inventory_aware_quoter", "trained"],
        default="random",
    )
    parser.add_argument("--evaluation-mode", choices=["greedy", "stochastic"], default="greedy")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--phi", type=float, default=0.10)
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--end-time", type=str, default="09:35:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument("--return-window", type=int, default=10)
    parser.add_argument("--lambda-q", type=float, default=0.01)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.02)
    parser.add_argument("--inventory-cap", type=int, default=None)
    parser.add_argument("--rl-liquidity-mode", choices=["taker_only", "mixed", "quoter_only"], default="taker_only")
    parser.add_argument("--rl-quoter-split", type=float, default=0.5)
    parser.add_argument("--rl-enable-passive-quotes", type=str, default="true")
    parser.add_argument("--rl-quote-mode", choices=["at_best", "one_tick_inside"], default="at_best")
    parser.add_argument("--rl-quote-offset-ticks", type=int, default=0)
    parser.add_argument("--rl-quote-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="policy_evaluation.csv")
    parser.add_argument("--summary-output", type=str, default="")
    parser.add_argument("--diagnostics-output", type=str, default="")
    parser.add_argument("--agent-diagnostics-output", type=str, default="")
    parser.add_argument("--report-output", type=str, default="")
    parser.add_argument("--plot-output-dir", type=str, default="")
    parser.add_argument("--show-market-plot", action="store_true")
    parser.add_argument("--animate-market", action="store_true")
    parser.add_argument("--visualize-seed", type=int, default=-1)
    parser.add_argument("--animation-interval-ms", type=int, default=40)
    parser.add_argument("--animation-tail-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    enable_passive_quotes = str(args.rl_enable_passive_quotes).strip().lower() in {"1", "true", "yes", "on"}
    parsed_seeds = _parse_int_list(args.seeds)
    target_visual_seed = args.visualize_seed if args.visualize_seed >= 0 else (parsed_seeds[0] if parsed_seeds else -1)
    captured_episode: dict[str, pd.DataFrame | int] = {}

    def handle_episode_visualization(
        phase: str,
        episode_index: int,
        seed: int,
        market_frame: pd.DataFrame,
        _rl_frame: pd.DataFrame,
        _transition_frame: pd.DataFrame,
    ) -> None:
        del episode_index
        if phase != "evaluation" or seed != target_visual_seed:
            return
        captured_episode["seed"] = seed
        captured_episode["market_frame"] = market_frame.copy()
        captured_episode["rl_frame"] = _rl_frame.copy()
        captured_episode["transition_frame"] = _transition_frame.copy()
        title = f"Policy Evaluation (phi={args.phi}, seed={seed})"
        if args.animate_market:
            show_market_animation(
                market_frame,
                title=title,
                interval_ms=args.animation_interval_ms,
                tail_seconds=args.animation_tail_seconds if args.animation_tail_seconds > 0 else None,
            )
        elif args.show_market_plot:
            show_market_plot(
                market_frame,
                title=title,
            )
        if args.plot_output_dir:
            output_dir = Path(args.plot_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_market_plot(
                market_frame,
                output_dir / f"evaluation_seed_{seed}.png",
                title=title,
            )

    if args.policy == "trained":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --policy trained")
        policy = load_policy_artifact(
            args.checkpoint,
            deterministic=(args.evaluation_mode == "greedy"),
        )
    else:
        policy = build_policy(args.policy)

    evaluation_frame, summary = evaluate_policy(
        policy=policy,
        phi=args.phi,
        seeds=parsed_seeds,
        deterministic=(args.evaluation_mode == "greedy"),
        config_factory=build_abides_rmsc04_small_v1_config,
        config_overrides={
            "num_agents": args.num_agents,
            "end_time": args.end_time,
            "log_frequency": args.log_frequency,
            "return_window": args.return_window,
            "lambda_q": args.lambda_q,
            "flat_hold_penalty": args.flat_hold_penalty,
            "inventory_cap": args.inventory_cap,
            "rl_liquidity_mode": args.rl_liquidity_mode,
            "rl_quoter_split": args.rl_quoter_split,
            "rl_enable_passive_quotes": enable_passive_quotes,
            "rl_quote_mode": args.rl_quote_mode,
            "rl_quote_offset_ticks": args.rl_quote_offset_ticks,
            "rl_quote_size": args.rl_quote_size,
        },
        episode_callback=handle_episode_visualization,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_frame.to_csv(output_path, index=False)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

    diagnostics_summary: dict[str, float | str] | None = None
    if captured_episode:
        diagnostics_summary, agent_frame = compute_policy_evaluation_diagnostics(
            captured_episode["market_frame"],
            captured_episode["rl_frame"],
            captured_episode["transition_frame"],
            policy=policy,
            inventory_cap=args.inventory_cap,
        )
        if args.diagnostics_output:
            diagnostics_path = Path(args.diagnostics_output)
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([diagnostics_summary]).to_csv(diagnostics_path, index=False)
        if args.agent_diagnostics_output:
            agent_path = Path(args.agent_diagnostics_output)
            agent_path.parent.mkdir(parents=True, exist_ok=True)
            agent_frame.to_csv(agent_path, index=False)
        if args.report_output:
            report_path = Path(args.report_output)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                build_policy_evaluation_report(diagnostics_summary, agent_frame),
                encoding="utf-8",
            )

    print(f"Saved evaluation episodes to {output_path}")
    if args.summary_output:
        print(f"Saved evaluation summary to {args.summary_output}")
    if args.diagnostics_output:
        print(f"Saved diagnostics summary to {args.diagnostics_output}")
    if args.agent_diagnostics_output:
        print(f"Saved per-agent diagnostics to {args.agent_diagnostics_output}")
    if args.report_output:
        print(f"Saved diagnostics report to {args.report_output}")
    print("Evaluation summary:", {"evaluation_mode": args.evaluation_mode, **summary})
    if diagnostics_summary is not None:
        print(
            "Selected-seed diagnostics:",
            {
                "action_ordering": str(diagnostics_summary.get("action_ordering", "0=sell, 1=hold, 2=buy")),
                "evaluation_mode": args.evaluation_mode,
                "inventory_cap_present": diagnostics_summary["inventory_cap_present"],
                "inventory_cap_value": diagnostics_summary["inventory_cap_value"],
                "chosen_actions": {
                    "sell": int(diagnostics_summary["rl_sell_count"]),
                    "hold": int(diagnostics_summary["rl_hold_count"]),
                    "buy": int(diagnostics_summary["rl_buy_count"]),
                },
                "submitted_actions": {
                    "sell": int(diagnostics_summary["submitted_sell_action_count"]),
                    "buy": int(diagnostics_summary["submitted_buy_action_count"]),
                },
                "blocked_actions": {
                    "sell": int(diagnostics_summary["blocked_sell_action_count"]),
                    "buy": int(diagnostics_summary["blocked_buy_action_count"]),
                },
                "executed_actions": {
                    "sell": int(diagnostics_summary["executed_sell_action_count"]),
                    "buy": int(diagnostics_summary["executed_buy_action_count"]),
                },
                "mean_probs": {
                    "sell": float(diagnostics_summary["mean_prob_sell"]),
                    "hold": float(diagnostics_summary["mean_prob_hold"]),
                    "buy": float(diagnostics_summary["mean_prob_buy"]),
                },
                "greedy_fractions": {
                    "sell": float(diagnostics_summary["deterministic_sell_fraction"]),
                    "hold": float(diagnostics_summary["deterministic_hold_fraction"]),
                    "buy": float(diagnostics_summary["deterministic_buy_fraction"]),
                },
                "passive_metrics": {
                    "bid_submission_rate": float(diagnostics_summary.get("passive_bid_submission_rate", 0.0)),
                    "ask_submission_rate": float(diagnostics_summary.get("passive_ask_submission_rate", 0.0)),
                    "both_submission_rate": float(diagnostics_summary.get("passive_both_quote_rate", 0.0)),
                    "quote_fill_rate": float(diagnostics_summary.get("quote_fill_rate", 0.0)),
                    "resting_presence_fraction": float(diagnostics_summary.get("resting_order_presence_fraction", 0.0)),
                },
                "inventory_range": (
                    float(diagnostics_summary["inventory_overall_min"]),
                    float(diagnostics_summary["inventory_overall_max"]),
                ),
                "max_abs_inventory_reached": float(diagnostics_summary["max_abs_inventory_reached"]),
            },
        )


if __name__ == "__main__":
    main()
