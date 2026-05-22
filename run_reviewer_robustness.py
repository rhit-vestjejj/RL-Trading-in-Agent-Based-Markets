"""Run reviewer-facing composition controls and diagnostic plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from reviewer_robustness import parse_float_list, parse_int_list, run_composition_control


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-grid", type=str, default="0.00,0.05,0.10,0.20,0.30,0.40,0.50")
    parser.add_argument(
        "--quoter-splits",
        type=str,
        default="0.00,0.25,0.50,0.75,1.00",
        help="RL quoter fractions to evaluate. 0=taker-only, 1=quoter-only.",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=str,
        default="7,8,9,10,11,12,13,14,15,16",
        help="Comma-separated evaluation seeds. Use 10-20 seeds for paper-facing runs.",
    )
    parser.add_argument("--evaluation-mode", choices=["greedy", "stochastic"], default="greedy")
    parser.add_argument(
        "--trained-experiment-dir",
        type=str,
        default="experiments/mixed_full_reward_shaped",
        help="Existing phi sweep containing phi_*/checkpoints/shared_ppo_policy_final.npz.",
    )
    parser.add_argument(
        "--no-trained",
        action="store_true",
        help="Only run the matched nonlearned control; do not load trained checkpoints.",
    )
    parser.add_argument(
        "--control-policy",
        choices=["random", "inventory_aware"],
        default="random",
        help="Nonlearned policy used for the matched-composition control.",
    )
    parser.add_argument("--num-agents", type=int, default=102)
    parser.add_argument("--end-time", type=str, default="09:35:00")
    parser.add_argument("--log-frequency", type=str, default="1s")
    parser.add_argument("--return-window", type=int, default=10)
    parser.add_argument("--lambda-q", type=float, default=0.01)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.02)
    parser.add_argument("--inventory-cap", type=int, default=None)
    parser.add_argument("--rl-quote-mode", choices=["at_best", "one_tick_inside"], default="at_best")
    parser.add_argument("--rl-quote-offset-ticks", type=int, default=0)
    parser.add_argument("--rl-quote-size", type=int, default=1)
    parser.add_argument("--event-window-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="experiments/reviewer_robustness")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_composition_control(
        phi_grid=parse_float_list(args.phi_grid),
        quoter_splits=parse_float_list(args.quoter_splits),
        evaluation_seeds=parse_int_list(args.evaluation_seeds),
        output_dir=Path(args.output_dir),
        end_time=args.end_time,
        log_frequency=args.log_frequency,
        num_agents=args.num_agents,
        return_window=args.return_window,
        lambda_q=args.lambda_q,
        flat_hold_penalty=args.flat_hold_penalty,
        inventory_cap=args.inventory_cap,
        evaluation_mode=args.evaluation_mode,
        trained_experiment_dir=args.trained_experiment_dir,
        include_trained=not args.no_trained,
        control_policy=args.control_policy,
        quote_mode=args.rl_quote_mode,
        quote_offset_ticks=args.rl_quote_offset_ticks,
        quote_size=args.rl_quote_size,
        event_window_steps=args.event_window_steps,
    )
    print(f"Saved reviewer robustness folder to {result['output_dir']}")
    print(f"Saved per-seed CSV to {result['per_seed_path']}")
    print(f"Saved aggregate CSV to {result['aggregate_path']}")
    print(f"Saved event-study CSV to {result['event_study_path']}")
    print(f"Saved report to {result['report_path']}")
    print("Saved plots:", [str(path) for path in result["saved_plots"]])


if __name__ == "__main__":
    main()
