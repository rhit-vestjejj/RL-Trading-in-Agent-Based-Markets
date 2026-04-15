"""Train a shared PPO policy for RL agents in the frozen baseline market."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from ppo_training import PPOHyperparameters, save_policy_artifact, train_shared_policy
from training_reporting import (
    build_combined_progress_frame,
    build_training_progress_report,
    save_training_progress_plots,
)
from visualization import save_market_plot, show_market_animation, show_market_plot


def _parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi", type=float, default=0.10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--training-seeds", type=str, default="")
    parser.add_argument("--evaluation-seeds", type=str, default="")
    parser.add_argument("--evaluation-modes", type=str, default="greedy,stochastic")
    parser.add_argument("--evaluation-interval", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
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
    parser.add_argument("--training-log-output", type=str, default="ppo_training_log.csv")
    parser.add_argument("--evaluation-log-output", type=str, default="")
    parser.add_argument("--combined-log-output", type=str, default="ppo_progress_log.csv")
    parser.add_argument("--checkpoint-output", type=str, default="shared_ppo_policy.npz")
    parser.add_argument("--report-output", type=str, default="ppo_training_report.md")
    parser.add_argument("--plot-output-dir", type=str, default="")
    parser.add_argument("--show-market-plot", action="store_true")
    parser.add_argument("--animate-market", action="store_true")
    parser.add_argument("--visualize-episode", type=int, default=-1)
    parser.add_argument("--animation-interval-ms", type=int, default=40)
    parser.add_argument("--animation-tail-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_visual_episode = args.visualize_episode if args.visualize_episode >= 0 else args.episodes - 1
    training_seeds = _parse_int_list(args.training_seeds)
    evaluation_seeds = _parse_int_list(args.evaluation_seeds)
    evaluation_modes = _parse_str_list(args.evaluation_modes)

    enable_passive_quotes = str(args.rl_enable_passive_quotes).strip().lower() in {"1", "true", "yes", "on"}

    checkpoint_base_path = Path(args.checkpoint_output)

    def handle_checkpoint_snapshot(episode_index: int, seed: int, policy) -> None:
        if args.checkpoint_interval <= 0:
            return
        snapshot_path = checkpoint_base_path.with_name(
            f"{checkpoint_base_path.stem}_episode_{episode_index + 1:03d}_seed_{seed}{checkpoint_base_path.suffix}"
        )
        save_policy_artifact(policy, snapshot_path)

    def handle_episode_visualization(
        phase: str,
        episode_index: int,
        seed: int,
        market_frame: pd.DataFrame,
        _rl_frame: pd.DataFrame,
        _transition_frame: pd.DataFrame,
    ) -> None:
        if phase != "train" or episode_index != target_visual_episode:
            return
        title = f"PPO Training Episode {episode_index} (phi={args.phi}, seed={seed})"
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
                output_dir / f"train_episode_{episode_index:03d}_seed_{seed}.png",
                title=title,
            )

    hyperparameters = PPOHyperparameters(
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
    )

    policy, training_frame, evaluation_frame = train_shared_policy(
        phi=args.phi,
        episodes=args.episodes,
        start_seed=args.start_seed,
        training_seeds=training_seeds,
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
        hyperparameters=hyperparameters,
        evaluation_seeds=evaluation_seeds,
        evaluation_modes=evaluation_modes,
        evaluation_interval=args.evaluation_interval,
        episode_callback=handle_episode_visualization,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_callback=handle_checkpoint_snapshot,
    )

    training_path = Path(args.training_log_output)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_frame.to_csv(training_path, index=False)

    if args.evaluation_log_output:
        evaluation_path = Path(args.evaluation_log_output)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_frame.to_csv(evaluation_path, index=False)

    combined_frame = build_combined_progress_frame(training_frame, evaluation_frame)
    if args.combined_log_output:
        combined_path = Path(args.combined_log_output)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_frame.to_csv(combined_path, index=False)

    checkpoint_path = save_policy_artifact(policy, args.checkpoint_output)

    report_text = build_training_progress_report(training_frame, evaluation_frame)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

    print(f"Saved training log to {training_path}")
    if args.evaluation_log_output:
        print(f"Saved evaluation log to {args.evaluation_log_output}")
    if args.combined_log_output:
        print(f"Saved combined progress log to {args.combined_log_output}")
    if args.report_output:
        print(f"Saved training report to {args.report_output}")
    print(f"Saved checkpoint to {checkpoint_path}")
    if args.plot_output_dir:
        saved_plots = save_training_progress_plots(training_frame, evaluation_frame, args.plot_output_dir)
        if saved_plots:
            print("Saved progress plots:", [str(path) for path in saved_plots])

    if not training_frame.empty:
        latest = training_frame.iloc[-1]
        print(
            "Last episode:",
            {
                "episode": int(latest["episode"]),
                "total_training_reward": float(latest["total_training_reward"]),
                "average_reward_per_rl_agent": float(latest["average_reward_per_rl_agent"]),
                "average_abs_inventory": float(latest["average_abs_inventory"]),
                "average_abs_ending_inventory": float(latest["average_abs_ending_inventory"]),
                "max_abs_inventory": float(latest["max_abs_inventory"]),
                "blocked_buy_action_count": float(latest.get("blocked_buy_action_count", 0.0)),
                "blocked_sell_action_count": float(latest.get("blocked_sell_action_count", 0.0)),
                "buy_fraction": float(latest["buy_fraction"]),
                "hold_fraction": float(latest["hold_fraction"]),
                "sell_fraction": float(latest["sell_fraction"]),
                "quote_hold_fraction": float(latest.get("quote_hold_fraction", 0.0)),
                "quote_bid_fraction": float(latest.get("quote_bid_fraction", 0.0)),
                "quote_ask_fraction": float(latest.get("quote_ask_fraction", 0.0)),
                "quote_both_fraction": float(latest.get("quote_both_fraction", 0.0)),
                "collapse_state": str(latest["collapse_state"]),
                "policy_loss": float(latest["policy_loss"]),
                "value_loss": float(latest["value_loss"]),
                "entropy": float(latest["entropy"]),
                "reward_mean": float(latest["reward_mean"]),
                "reward_std": float(latest["reward_std"]),
                "wealth_delta_mean": float(latest["wealth_delta_mean"]),
                "inventory_penalty_mean": float(latest["inventory_penalty_mean"]),
                "flat_hold_penalty_mean": float(latest["flat_hold_penalty_mean"]),
                "num_transitions": float(latest["num_transitions"]),
            },
        )
        print(
            "Mean training reward:",
            float(training_frame["total_training_reward"].mean()),
        )

    if not evaluation_frame.empty:
        latest_rows = (
            evaluation_frame.sort_values(["episode", "evaluation_mode"])
            .groupby("evaluation_mode", as_index=False)
            .tail(1)
        )
        for row in latest_rows.itertuples(index=False):
            summary_columns = [column for column in evaluation_frame.columns if column.startswith("evaluation_")]
            summary: dict[str, object] = {}
            for column in summary_columns:
                if not hasattr(row, column):
                    continue
                value = getattr(row, column)
                if pd.isna(value):
                    continue
                if isinstance(value, str):
                    summary[column] = value
                else:
                    summary[column] = float(value)
            print(f"Latest evaluation summary ({row.evaluation_mode}):", summary)


if __name__ == "__main__":
    main()
