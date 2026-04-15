"""Tests for the minimal shared-policy PPO training path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from env import RLMarketEnvironment
from ppo_training import (
    SharedLinearPPOPolicy,
    SharedPolicyBundle,
    evaluate_policy,
    load_policy_artifact,
    run_policy_episode,
    save_policy_artifact,
    train_shared_policy,
)


class PPOTrainingTests(unittest.TestCase):
    """Focused regression coverage for PPO rollout collection and updates."""

    def test_policy_update_changes_parameters(self) -> None:
        policy = SharedLinearPPOPolicy(state_dim=4, seed=13)
        transitions = pd.DataFrame(
            {
                "action": [0, 1, 2, 0],
                "reward": [0.2, -0.1, 0.4, 0.1],
                "done": [False, False, True, False],
                "log_prob": [-1.0, -1.1, -1.2, -0.9],
                "value_estimate": [0.0, 0.1, -0.1, 0.05],
                "state_00": [0.0, 0.1, -0.2, 0.3],
                "state_01": [0.2, 0.0, 0.1, -0.1],
                "state_02": [0.3, -0.1, 0.0, 0.2],
                "state_03": [1.0, 0.0, -1.0, 0.5],
                "next_state_00": [0.1, -0.1, 0.0, 0.2],
                "next_state_01": [0.1, 0.2, -0.1, 0.0],
                "next_state_02": [0.2, 0.0, 0.2, -0.2],
                "next_state_03": [0.5, -0.5, 0.0, 1.0],
                "wealth_delta": [0.2, -0.1, 0.4, 0.1],
                "inventory_penalty": [0.0, 0.01, 0.04, 0.09],
            }
        )

        initial_policy_weights = policy.policy_weights.copy()
        initial_value_weights = policy.value_weights.copy()
        metrics = policy.update(transitions)

        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("num_transitions", metrics)
        self.assertFalse(np.allclose(initial_policy_weights, policy.policy_weights))
        self.assertFalse(np.allclose(initial_value_weights, policy.value_weights))

    def test_run_policy_episode_collects_transitions(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.10,
            seed=31,
            end_time="09:30:15",
            log_frequency="1s",
        )
        shared_policy = SharedLinearPPOPolicy(state_dim=config.return_window + 3, seed=31)

        _, rl_frame, transition_frame = run_policy_episode(config, shared_policy=shared_policy)

        self.assertGreater(len(rl_frame), 0)
        self.assertGreater(len(transition_frame), 0)
        self.assertTrue(
            {
                "agent_id",
                "decision_index",
                "action",
                "reward",
                "wealth_delta",
                "inventory_penalty",
                "done",
                "log_prob",
                "value_estimate",
                "state_00",
                "next_state_00",
            }.issubset(transition_frame.columns)
        )

    def test_training_and_evaluation_round_trip(self) -> None:
        seen_training_callbacks: list[tuple[str, int, int]] = []

        def on_training_episode(phase, episode_index, seed, market_frame, _rl_frame, _transition_frame):
            self.assertGreater(len(market_frame), 0)
            seen_training_callbacks.append((phase, episode_index, seed))

        policy, training_frame, evaluation_frame = train_shared_policy(
            phi=0.10,
            episodes=1,
            start_seed=41,
            config_factory=build_abides_rmsc04_small_v1_config,
            config_overrides={
                "end_time": "09:30:15",
                "log_frequency": "1s",
            },
            evaluation_seeds=[42],
            evaluation_modes=["greedy", "stochastic"],
            evaluation_interval=1,
            episode_callback=on_training_episode,
        )

        self.assertEqual(len(training_frame), 1)
        self.assertEqual(len(evaluation_frame), 2)
        self.assertEqual(
            seen_training_callbacks,
            [("train", 0, 41), ("evaluation", 0, 42), ("evaluation", 0, 42)],
        )
        self.assertIn("policy_loss", training_frame.columns)
        self.assertIn("collapse_state", training_frame.columns)
        self.assertIn("inventory_penalty_mean", training_frame.columns)
        self.assertIn("blocked_buy_action_count", training_frame.columns)
        self.assertIn("evaluation_total_reward_mean", evaluation_frame.columns)
        self.assertIn("evaluation_mode", evaluation_frame.columns)
        self.assertIn("evaluation_chosen_buy_count_total", evaluation_frame.columns)
        self.assertIn("evaluation_blocked_buy_action_count_total", evaluation_frame.columns)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "policy.npz"
            policy.save(checkpoint)
            loaded_policy = SharedLinearPPOPolicy.load(checkpoint)
            seen_eval_callbacks: list[tuple[str, int, int]] = []

            def on_eval_episode(phase, episode_index, seed, market_frame, _rl_frame, _transition_frame):
                self.assertGreater(len(market_frame), 0)
                seen_eval_callbacks.append((phase, episode_index, seed))

            evaluation_runs, summary = evaluate_policy(
                policy=loaded_policy,
                phi=0.10,
                seeds=[43],
                config_factory=build_abides_rmsc04_small_v1_config,
                config_overrides={
                    "end_time": "09:30:15",
                    "log_frequency": "1s",
                },
                episode_callback=on_eval_episode,
            )

        self.assertEqual(len(evaluation_runs), 1)
        self.assertEqual(seen_eval_callbacks, [("evaluation", 0, 43)])
        self.assertIn("evaluation_total_reward_mean", summary)

    def test_inventory_penalty_uses_dollar_lambda_scale(self) -> None:
        environment = RLMarketEnvironment(lambda_q=0.01)
        components = environment.compute_reward_components(
            previous_cash=100_000.0,
            previous_inventory=0,
            previous_midprice=10_000.0,
            current_cash=100_000.0,
            current_inventory=10,
            current_midprice=10_000.0,
        )

        self.assertAlmostEqual(components["wealth_delta"], 100_000.0)
        self.assertAlmostEqual(components["inventory_penalty"], 100.0)
        self.assertAlmostEqual(components["reward"], 99_900.0)

    def test_flat_hold_penalty_applies_only_to_flat_hold_states(self) -> None:
        environment = RLMarketEnvironment(lambda_q=0.01, flat_hold_penalty=0.02)
        flat_hold = environment.compute_reward_components(
            previous_cash=100_000.0,
            previous_inventory=0,
            previous_midprice=10_000.0,
            current_cash=100_000.0,
            current_inventory=0,
            current_midprice=10_000.0,
            previous_action=1,
        )
        executed_buy = environment.compute_reward_components(
            previous_cash=100_000.0,
            previous_inventory=0,
            previous_midprice=10_000.0,
            current_cash=99_000.0,
            current_inventory=1,
            current_midprice=10_000.0,
            previous_action=2,
        )

        self.assertAlmostEqual(flat_hold["flat_hold_penalty"], 2.0)
        self.assertAlmostEqual(flat_hold["reward"], -2.0)
        self.assertAlmostEqual(executed_buy["flat_hold_penalty"], 0.0)

    def test_policy_bundle_round_trip_preserves_both_roles(self) -> None:
        bundle = SharedPolicyBundle(
            taker_policy=SharedLinearPPOPolicy(state_dim=4, action_dim=3, seed=13),
            quoter_policy=SharedLinearPPOPolicy(state_dim=4, action_dim=4, seed=17),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "policy_bundle.npz"
            save_policy_artifact(bundle, checkpoint)
            loaded = load_policy_artifact(checkpoint)

        self.assertIsInstance(loaded, SharedPolicyBundle)
        self.assertIsNotNone(loaded.taker_policy)
        self.assertIsNotNone(loaded.quoter_policy)
        self.assertEqual(loaded.taker_policy.action_dim, 3)
        self.assertEqual(loaded.quoter_policy.action_dim, 4)


if __name__ == "__main__":
    unittest.main()
