"""Run the full shared-policy PPO phi-sweep experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from phi_experiment import (
    DEFAULT_EVALUATION_SEEDS,
    DEFAULT_PHI_GRID,
    default_experiment_output_dir,
    parse_float_list,
    parse_int_list,
    run_phi_experiment,
)
from ppo_training import PPOHyperparameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-grid", type=str, default=",".join(f"{phi:.2f}" for phi in DEFAULT_PHI_GRID))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--training-seeds", type=str, default="")
    parser.add_argument("--evaluation-seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_EVALUATION_SEEDS))
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=0.02)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--value-huber-delta", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phi_grid = parse_float_list(args.phi_grid)
    training_seeds = parse_int_list(args.training_seeds)
    evaluation_seeds = parse_int_list(args.evaluation_seeds)
    enable_passive_quotes = str(args.rl_enable_passive_quotes).strip().lower() in {"1", "true", "yes", "on"}
    output_dir = Path(args.output_dir) if args.output_dir else default_experiment_output_dir()

    result = run_phi_experiment(
        phi_grid=phi_grid,
        episodes=args.episodes,
        start_seed=args.start_seed,
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
        output_dir=output_dir,
        end_time=args.end_time,
        log_frequency=args.log_frequency,
        num_agents=args.num_agents,
        return_window=args.return_window,
        lambda_q=args.lambda_q,
        flat_hold_penalty=args.flat_hold_penalty,
        inventory_cap=args.inventory_cap,
        rl_liquidity_mode=args.rl_liquidity_mode,
        rl_quoter_split=args.rl_quoter_split,
        rl_enable_passive_quotes=enable_passive_quotes,
        rl_quote_mode=args.rl_quote_mode,
        rl_quote_offset_ticks=args.rl_quote_offset_ticks,
        rl_quote_size=args.rl_quote_size,
        evaluation_interval=args.evaluation_interval,
        checkpoint_interval=args.checkpoint_interval,
        hyperparameters=PPOHyperparameters(
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            actor_learning_rate=args.actor_lr,
            critic_learning_rate=args.critic_lr,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            reward_scale=args.reward_scale,
            entropy_coefficient=args.entropy_coefficient,
            gradient_clip_norm=args.gradient_clip_norm,
            value_huber_delta=args.value_huber_delta,
        ),
    )

    print(f"Saved experiment folder to {result['output_dir']}")
    print(f"Saved experiment config to {Path(result['output_dir']) / 'experiment_config.json'}")
    print(f"Saved summary CSV to {Path(result['output_dir']) / 'phi_sweep_summary.csv'}")
    print(f"Saved summary JSON to {Path(result['output_dir']) / 'phi_sweep_summary.json'}")
    print(f"Saved report to {Path(result['output_dir']) / 'phi_sweep_report.md'}")
    print("Saved top-level plots:", [str(path) for path in result["saved_top_level_plots"]])


if __name__ == "__main__":
    main()
