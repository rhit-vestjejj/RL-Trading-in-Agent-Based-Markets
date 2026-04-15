"""Tests for RL-integrated run diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from rl_diagnostics import (
    build_policy_evaluation_report,
    build_rl_run_report,
    compute_policy_evaluation_diagnostics,
    compute_rl_run_diagnostics,
    diagnose_action_reward_balance,
    diagnose_inventory_cap_behavior,
    diagnose_passive_provision,
    diagnose_transition_dynamics,
    diagnose_return_series,
    diagnose_policy_outputs,
    diagnose_spread_gaps,
    summarize_rl_agents,
)


class RLRunDiagnosticsTests(unittest.TestCase):
    def test_spread_gap_diagnosis_identifies_true_one_sided_book(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "best_bid": [100.0, 100.0, float("nan")],
                "best_ask": [100.01, float("nan"), 100.02],
                "spread": [0.01, float("nan"), float("nan")],
                "midprice": [100.005, float("nan"), float("nan")],
                "bid_depth": [5.0, 5.0, 0.0],
                "ask_depth": [4.0, 0.0, 4.0],
            }
        )
        diagnostics = diagnose_spread_gaps(frame)
        self.assertEqual(diagnostics["spread_gap_cause"], "true_one_sided_book")
        self.assertEqual(diagnostics["spread_gap_without_missing_side_fraction"], 0.0)
        self.assertAlmostEqual(diagnostics["one_sided_book_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(diagnostics["one_sided_metric_valid_timestep_count"], 3.0)
        self.assertAlmostEqual(diagnostics["empty_book_fraction"], 0.0)
        self.assertAlmostEqual(diagnostics["max_consecutive_one_sided_duration"], 2.0)
        self.assertEqual(diagnostics["num_one_sided_episodes"], 1.0)

    def test_spread_gap_diagnosis_skips_empty_book_from_one_sided_fraction(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "best_bid": [100.0, float("nan"), 100.0],
                "best_ask": [100.01, float("nan"), float("nan")],
                "spread": [0.01, float("nan"), float("nan")],
                "midprice": [100.005, float("nan"), float("nan")],
                "bid_depth": [5.0, 0.0, 5.0],
                "ask_depth": [5.0, 0.0, 0.0],
            }
        )

        diagnostics = diagnose_spread_gaps(frame)

        self.assertAlmostEqual(diagnostics["one_sided_book_fraction"], 0.5)
        self.assertAlmostEqual(diagnostics["empty_book_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(diagnostics["one_sided_metric_valid_timestep_count"], 2.0)

    def test_spread_gap_diagnosis_flags_defined_midprice_despite_missing_side(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000],
                "best_bid": [100.0, 100.0],
                "best_ask": [100.01, float("nan")],
                "spread": [0.01, float("nan")],
                "midprice": [100.005, 100.0],
            }
        )
        diagnostics = diagnose_spread_gaps(frame)
        self.assertEqual(diagnostics["spread_gap_cause"], "pipeline_issue")
        self.assertGreater(diagnostics["defined_midprice_despite_missing_side_fraction"], 0.0)

    def test_return_diagnostics_separate_zero_inflation_from_nonzero_kurtosis(self) -> None:
        frame = pd.DataFrame({"midprice": [100.0, 100.0, 100.0, 100.01, 100.0, 100.02]})
        diagnostics = diagnose_return_series(frame)
        self.assertGreater(diagnostics["zero_return_fraction"], 0.0)
        self.assertGreater(diagnostics["excess_kurtosis_full"], diagnostics["excess_kurtosis_nonzero_only"])

    def test_rl_run_report_mentions_one_sided_book_and_rl_flow(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "best_bid": [100.0, 100.0, float("nan")],
                "best_ask": [100.01, float("nan"), 100.02],
                "spread": [0.01, float("nan"), float("nan")],
                "midprice": [100.005, float("nan"), float("nan")],
                "bid_depth": [5.0, 5.0, 0.0],
                "ask_depth": [5.0, 0.0, 4.0],
            }
        )
        rl_frame = pd.DataFrame(
            {
                "time_ns": [1, 2, 3],
                "agent_id": [1, 1, 2],
                "action": [2, 0, 1],
                "previous_action": [-1, 2, -1],
                "filled_quantity_since_last_decision": [0.0, 1.0, 0.0],
                "inventory": [0.0, 1.0, 0.0],
                "reward": [0.0, 1.0, 0.0],
            }
        )
        diagnostics = compute_rl_run_diagnostics(frame, rl_frame)
        report = build_rl_run_report(diagnostics)
        self.assertIn("true_one_sided_book", report)
        self.assertIn("One-sided-book fraction", report)
        self.assertIn("RL Run Diagnostics", report)

    def test_policy_diagnostics_identify_not_chosen_sells(self) -> None:
        class DummyPolicy:
            policy_weights = [[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]]
            policy_bias = [0.0, 0.3, 0.0]

        transition_frame = pd.DataFrame(
            {
                "state_00": [0.0, 1.0],
                "state_01": [1.0, 0.0],
                "reward": [0.0, 0.0],
                "inventory_after": [0.0, 0.0],
            }
        )
        policy_diag = diagnose_policy_outputs(DummyPolicy(), transition_frame)
        self.assertGreater(policy_diag["mean_prob_hold"], policy_diag["mean_prob_sell"])
        self.assertEqual(policy_diag["deterministic_sell_fraction"], 0.0)

    def test_transition_diagnostics_capture_inventory_transitions_and_fill_rates(self) -> None:
        transition_frame = pd.DataFrame(
            {
                "time": [0, 1, 2, 3],
                "time_next": [1, 2, 3, 4],
                "action": [0, 2, 2, 0],
                "effective_action": [0, 2, 1, 0],
                "inventory_before": [0.0, -1.0, 0.0, 1.0],
                "inventory_after": [-1.0, 0.0, 0.0, 0.0],
            }
        )
        diagnostics = diagnose_transition_dynamics(transition_frame)
        self.assertEqual(diagnostics["inventory_transition_0_to_neg1"], 1.0)
        self.assertEqual(diagnostics["inventory_transition_neg1_to_0"], 1.0)
        self.assertEqual(diagnostics["inventory_transition_pos1_to_0"], 1.0)
        self.assertEqual(diagnostics["submitted_buy_action_count"], 1.0)
        self.assertEqual(diagnostics["submitted_sell_action_count"], 2.0)
        self.assertEqual(diagnostics["buy_fill_rate"], 1.0)
        self.assertEqual(diagnostics["sell_fill_rate"], 1.0)
        self.assertEqual(diagnostics["buy_fill_rate_at_inventory_neg1"], 1.0)
        self.assertTrue(pd.isna(diagnostics["buy_fill_rate_at_inventory_0"]))

    def test_inventory_cap_diagnostics_report_blocked_actions_and_cap_contact(self) -> None:
        rl_frame = pd.DataFrame(
            {
                "time_ns": [1, 2, 3],
                "agent_id": [1, 1, 1],
                "inventory": [1.0, 2.0, 2.0],
                "inventory_cap": [2.0, 2.0, 2.0],
                "blocked_buy_due_to_cap": [False, True, False],
                "blocked_sell_due_to_cap": [False, False, True],
            }
        )
        transition_frame = pd.DataFrame({"inventory_cap": [2.0]})

        diagnostics = diagnose_inventory_cap_behavior(
            rl_frame,
            transition_frame,
        )

        self.assertEqual(diagnostics["inventory_cap_value"], 2.0)
        self.assertEqual(diagnostics["blocked_buy_action_count"], 1.0)
        self.assertEqual(diagnostics["blocked_sell_action_count"], 1.0)
        self.assertEqual(diagnostics["max_abs_inventory_reached"], 2.0)
        self.assertGreater(diagnostics["inventory_at_cap_fraction"], 0.0)

    def test_passive_provision_diagnostics_capture_resting_quote_activity(self) -> None:
        rl_frame = pd.DataFrame(
            {
                "agent_role": ["quoter", "quoter", "quoter"],
                "submitted_passive_bid_order_count": [1.0, 0.0, 1.0],
                "submitted_passive_ask_order_count": [1.0, 1.0, 0.0],
                "submitted_aggressive_order_count": [0.0, 0.0, 0.0],
                "passive_filled_quantity_since_last_decision": [1.0, 0.0, 0.0],
                "resting_bid": [True, True, False],
                "resting_ask": [True, False, True],
            }
        )

        diagnostics = diagnose_passive_provision(rl_frame)

        self.assertAlmostEqual(diagnostics["passive_bid_submission_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(diagnostics["passive_ask_submission_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(diagnostics["passive_both_quote_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(diagnostics["resting_order_presence_fraction"], 1.0)
        self.assertAlmostEqual(diagnostics["fraction_of_rl_agents_with_both_quotes"], 1.0 / 3.0)
        self.assertEqual(diagnostics["total_rl_passive_order_count"], 4.0)

    def test_action_reward_balance_reports_hold_trade_gap(self) -> None:
        transition_frame = pd.DataFrame(
            {
                "action": [0, 1, 2],
                "reward": [-0.03, -0.01, -0.02],
                "wealth_delta": [-0.01, 0.0, -0.01],
                "inventory_penalty": [0.02, 0.0, 0.01],
                "flat_hold_penalty": [0.0, 0.01, 0.0],
            }
        )
        diagnostics = diagnose_action_reward_balance(transition_frame)
        self.assertAlmostEqual(diagnostics["hold_reward_mean"], -0.01)
        self.assertAlmostEqual(diagnostics["buy_reward_mean"], -0.02)
        self.assertGreater(diagnostics["hold_minus_best_trade_reward_gap"], 0.0)

    def test_policy_evaluation_report_mentions_missing_cap_and_sell_choice(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000],
                "best_bid": [100.0, 100.0],
                "best_ask": [100.01, 100.01],
                "spread": [0.01, 0.01],
                "midprice": [100.0, 100.01],
                "bid_depth": [5.0, 5.0],
                "ask_depth": [5.0, 5.0],
            }
        )
        rl_frame = pd.DataFrame(
            {
                "time_ns": [1, 2, 1, 2],
                "agent_id": [1, 1, 2, 2],
                "action": [1, 1, 1, 1],
                "previous_action": [-1, 1, -1, 1],
                "filled_quantity_since_last_decision": [0.0, 0.0, 0.0, 0.0],
                "inventory": [0.0, 0.0, 0.0, 0.0],
                "reward": [0.0, 0.0, 0.0, 0.0],
            }
        )
        transition_frame = pd.DataFrame(
            {
                "state_00": [0.0, 0.1],
                "state_01": [0.2, 0.0],
                "reward": [0.0, 0.0],
                "inventory_after": [0.0, 0.0],
            }
        )

        diagnostics, agent_frame = compute_policy_evaluation_diagnostics(
            frame,
            rl_frame,
            transition_frame,
            policy=None,
        )
        report = build_policy_evaluation_report(diagnostics, agent_frame)
        self.assertIn("Inventory cap logic present: no", report)
        self.assertIn("not choosing", report)
        self.assertIn("One-sided-book fraction", report)
        self.assertFalse(agent_frame.empty)
        self.assertTrue({"inventory_min", "inventory_max", "ending_inventory"}.issubset(agent_frame.columns))


if __name__ == "__main__":
    unittest.main()
