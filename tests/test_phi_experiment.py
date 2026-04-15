"""Tests for the phi-sweep experiment helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from phi_experiment import (
    aggregate_mode_results,
    build_phi_sweep_report,
    compute_extended_market_metrics,
    run_phi_experiment,
    save_cross_phi_plots,
)


class PhiExperimentTests(unittest.TestCase):
    def test_compute_extended_market_metrics_includes_one_sided_book_stats(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                "best_bid": [100.00, float("nan"), 100.01, 100.02],
                "best_ask": [100.01, 100.02, float("nan"), 100.03],
                "midprice": [100.005, float("nan"), float("nan"), 100.025],
                "spread": [0.01, float("nan"), float("nan"), 0.01],
                "bid_depth": [5.0, 6.0, 7.0, 8.0],
                "ask_depth": [4.0, 0.0, 0.0, 7.0],
            }
        )
        metrics = compute_extended_market_metrics(frame, tick_size=0.01)
        self.assertIn("zero_return_fraction", metrics)
        self.assertIn("one_sided_book_fraction", metrics)
        self.assertIn("spread_share_1_tick", metrics)
        self.assertIn("average_depth", metrics)
        self.assertAlmostEqual(metrics["one_sided_book_fraction"], 0.5)
        self.assertAlmostEqual(metrics["empty_book_fraction"], 0.0)
        self.assertAlmostEqual(metrics["undefined_midprice_fraction"], 0.5)
        self.assertAlmostEqual(metrics["missing_bid_fraction"], 0.25)
        self.assertAlmostEqual(metrics["missing_ask_fraction"], 0.25)
        self.assertAlmostEqual(metrics["one_sided_metric_valid_timestep_count"], 4.0)
        self.assertAlmostEqual(metrics["average_bid_volume"], 6.5)
        self.assertAlmostEqual(metrics["average_ask_volume"], 2.75)
        self.assertAlmostEqual(metrics["max_consecutive_one_sided_duration"], 2.0)
        self.assertAlmostEqual(metrics["num_one_sided_episodes"], 1.0)

    def test_compute_extended_market_metrics_skips_empty_book_states(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "best_bid": [100.00, float("nan"), 100.01],
                "best_ask": [100.01, float("nan"), 100.02],
                "midprice": [100.005, float("nan"), 100.015],
                "spread": [0.01, float("nan"), 0.01],
                "bid_depth": [5.0, 0.0, 5.0],
                "ask_depth": [0.0, 0.0, 4.0],
            }
        )

        metrics = compute_extended_market_metrics(frame, tick_size=0.01)

        self.assertAlmostEqual(metrics["one_sided_book_fraction"], 0.5)
        self.assertAlmostEqual(metrics["empty_book_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["one_sided_metric_valid_timestep_count"], 2.0)
        self.assertAlmostEqual(metrics["both_sides_missing_fraction"], 1.0 / 3.0)

    def test_aggregate_mode_results_combines_episode_and_market_metrics(self) -> None:
        episode_frame = pd.DataFrame(
            {
                "total_training_reward": [-1.0, -2.0],
                "average_reward_per_rl_agent": [-0.1, -0.2],
                "average_abs_ending_inventory": [0.5, 1.0],
                "buy_fraction": [0.3, 0.4],
                "hold_fraction": [0.4, 0.3],
                "sell_fraction": [0.3, 0.3],
            }
        )
        diagnostics_frame = pd.DataFrame(
            {
                "inventory_overall_min": [-1.0, -2.0],
                "inventory_overall_max": [2.0, 3.0],
                "rl_sell_count": [10.0, 11.0],
                "rl_hold_count": [12.0, 13.0],
                "rl_buy_count": [14.0, 15.0],
                "executed_sell_action_count": [9.0, 10.0],
                "executed_buy_action_count": [13.0, 14.0],
                "rl_executed_sell_volume": [9.0, 10.0],
                "rl_executed_buy_volume": [13.0, 14.0],
                "submitted_sell_action_count": [10.0, 11.0],
                "submitted_buy_action_count": [14.0, 15.0],
            }
        )
        market_metrics_frame = pd.DataFrame(
            {
                "phi": [0.1, 0.1],
                "seed": [7, 8],
                "evaluation_mode": ["greedy", "greedy"],
                "average_spread": [0.01, 0.02],
                "average_depth": [5.0, 7.0],
                "zero_return_fraction": [0.6, 0.7],
                "one_sided_book_fraction": [0.1, float("nan")],
                "undefined_midprice_fraction": [0.1, 0.2],
                "max_consecutive_one_sided_duration": [2.0, 3.0],
            }
        )
        aggregate = aggregate_mode_results(
            phi=0.10,
            evaluation_mode="greedy",
            episode_frame=episode_frame,
            diagnostics_frame=diagnostics_frame,
            market_metrics_frame=market_metrics_frame,
            representative_seed=7,
        )
        self.assertAlmostEqual(aggregate["evaluation_total_reward_mean"], -1.5)
        self.assertAlmostEqual(aggregate["evaluation_buy_fraction_mean"], 0.35)
        self.assertEqual(aggregate["evaluation_inventory_min_global"], -2.0)
        self.assertAlmostEqual(aggregate["average_spread_mean"], 0.015)
        self.assertAlmostEqual(aggregate["average_depth_mean"], 6.0)
        self.assertAlmostEqual(aggregate["one_sided_book_fraction_mean"], 0.1)
        self.assertEqual(aggregate["one_sided_book_fraction_n"], 1.0)
        self.assertTrue(pd.isna(aggregate["one_sided_book_fraction_stderr"]))
        self.assertAlmostEqual(aggregate["max_consecutive_one_sided_duration_mean"], 2.5)

    def test_save_cross_phi_plots_writes_png_files(self) -> None:
        summary = pd.DataFrame(
            {
                "phi": [0.0, 0.1, 0.2],
                "greedy_volatility_mean": [0.01, 0.02, 0.03],
                "stochastic_volatility_mean": [0.01, 0.025, 0.035],
                "greedy_average_spread_mean": [0.01, 0.011, 0.012],
                "stochastic_average_spread_mean": [0.01, 0.012, 0.013],
                "greedy_average_depth_mean": [5.0, 5.5, 6.0],
                "stochastic_average_depth_mean": [5.0, 5.4, 5.8],
                "greedy_one_sided_book_fraction_mean": [0.0, 0.1, 0.2],
                "stochastic_one_sided_book_fraction_mean": [0.0, 0.12, 0.24],
                "greedy_undefined_midprice_fraction_mean": [0.0, 0.1, 0.2],
                "stochastic_undefined_midprice_fraction_mean": [0.0, 0.12, 0.24],
                "greedy_max_consecutive_one_sided_duration_mean": [0.0, 2.0, 4.0],
                "stochastic_max_consecutive_one_sided_duration_mean": [0.0, 3.0, 5.0],
                "greedy_missing_bid_fraction_mean": [0.0, 0.04, 0.08],
                "stochastic_missing_bid_fraction_mean": [0.0, 0.05, 0.09],
                "greedy_missing_ask_fraction_mean": [0.0, 0.06, 0.12],
                "stochastic_missing_ask_fraction_mean": [0.0, 0.07, 0.15],
                "greedy_zero_return_fraction_mean": [0.7, 0.6, 0.5],
                "stochastic_zero_return_fraction_mean": [0.7, 0.58, 0.48],
                "greedy_tail_exposure_mean": [-0.01, -0.015, -0.02],
                "stochastic_tail_exposure_mean": [-0.01, -0.016, -0.022],
                "greedy_crash_rate_mean": [0.0, 0.01, 0.02],
                "stochastic_crash_rate_mean": [0.0, 0.015, 0.025],
                "greedy_max_drawdown_mean": [-0.01, -0.02, -0.03],
                "stochastic_max_drawdown_mean": [-0.01, -0.025, -0.035],
                "greedy_evaluation_average_abs_ending_inventory_mean": [0.0, 0.5, 0.7],
                "stochastic_evaluation_average_abs_ending_inventory_mean": [0.0, 0.8, 1.0],
                "greedy_evaluation_buy_fraction_mean": [0.0, 0.3, 0.35],
                "greedy_evaluation_hold_fraction_mean": [1.0, 0.4, 0.3],
                "greedy_evaluation_sell_fraction_mean": [0.0, 0.3, 0.35],
                "stochastic_evaluation_buy_fraction_mean": [0.0, 0.32, 0.34],
                "stochastic_evaluation_hold_fraction_mean": [1.0, 0.36, 0.32],
                "stochastic_evaluation_sell_fraction_mean": [0.0, 0.32, 0.34],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_cross_phi_plots(summary, tmpdir)
            saved_names = {Path(path).name for path in saved}
            self.assertIn("one_sided_book_fraction_vs_phi.png", saved_names)
            self.assertIn("undefined_midprice_fraction_vs_phi.png", saved_names)
            self.assertIn("max_consecutive_one_sided_duration_vs_phi.png", saved_names)
            self.assertIn("missing_bid_fraction_vs_phi.png", saved_names)
            self.assertIn("missing_ask_fraction_vs_phi.png", saved_names)
            self.assertGreaterEqual(len(saved), 14)
            for path in saved:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_build_phi_sweep_report_mentions_phi_values(self) -> None:
        summary = pd.DataFrame(
            {
                "phi": [0.0, 0.1],
                "greedy_evaluation_total_reward_mean": [0.0, -1.0],
                "greedy_evaluation_hold_fraction_mean": [1.0, 0.5],
                "stochastic_evaluation_total_reward_mean": [0.0, -2.0],
                "stochastic_evaluation_hold_fraction_mean": [1.0, 0.4],
                "greedy_average_spread_mean": [0.01, 0.012],
                "greedy_average_depth_mean": [5.0, 6.0],
                "greedy_one_sided_book_fraction_mean": [0.0, 0.2],
                "greedy_undefined_midprice_fraction_mean": [0.0, 0.2],
                "greedy_max_consecutive_one_sided_duration_mean": [0.0, 3.0],
                "greedy_num_one_sided_episodes_mean": [0.0, 2.0],
                "greedy_midprice_gap_without_missing_side_fraction_mean": [0.0, 0.0],
                "greedy_spread_gap_without_missing_side_fraction_mean": [0.0, 0.0],
                "greedy_defined_midprice_despite_missing_side_fraction_mean": [0.0, 0.0],
                "greedy_defined_spread_despite_missing_side_fraction_mean": [0.0, 0.0],
                "stochastic_average_spread_mean": [0.01, 0.013],
                "stochastic_average_depth_mean": [5.0, 6.5],
                "stochastic_one_sided_book_fraction_mean": [0.0, 0.25],
                "stochastic_undefined_midprice_fraction_mean": [0.0, 0.25],
                "stochastic_max_consecutive_one_sided_duration_mean": [0.0, 4.0],
                "stochastic_num_one_sided_episodes_mean": [0.0, 3.0],
                "stochastic_midprice_gap_without_missing_side_fraction_mean": [0.0, 0.0],
                "stochastic_spread_gap_without_missing_side_fraction_mean": [0.0, 0.0],
                "stochastic_defined_midprice_despite_missing_side_fraction_mean": [0.0, 0.0],
                "stochastic_defined_spread_despite_missing_side_fraction_mean": [0.0, 0.0],
                "greedy_evaluation_average_abs_ending_inventory_mean": [0.0, 0.4],
                "stochastic_evaluation_average_abs_ending_inventory_mean": [0.0, 0.7],
            }
        )
        report = build_phi_sweep_report(
            summary,
            {
                "market_profile": "abides_rmsc04_small_v1",
                "phi_grid": [0.0, 0.1],
                "episodes": 10,
                "evaluation_seeds": [7, 8],
                "end_time": "09:35:00",
                "log_frequency": "1s",
                "lambda_q": 0.01,
                "flat_hold_penalty": 0.02,
                "inventory_cap": 20,
            },
        )
        self.assertIn("phi = 0.00", report)
        self.assertIn("phi = 0.10", report)
        self.assertIn("inventory_cap: 20", report)
        self.assertIn("Snapshot Integrity", report)
        self.assertIn("one-sided (valid-liquidity only) / undefined-mid / empty-book", report)

    def test_run_phi_experiment_saves_one_sided_metrics_and_pipeline_issue_flags(self) -> None:
        class DummyConfig:
            tick_size = 0.01

            @staticmethod
            def agent_counts() -> dict[str, int]:
                return {"RLTrader": 1}

        market_frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000],
                "best_bid": [100.00, 100.00, 100.01],
                "best_ask": [100.01, float("nan"), 100.02],
                "midprice": [100.005, 100.00, 100.015],
                "spread": [0.01, float("nan"), 0.01],
                "bid_depth": [5.0, 6.0, 5.0],
                "ask_depth": [5.0, 0.0, 6.0],
                "traded_volume": [1.0, 0.0, 2.0],
                "signed_order_flow": [1.0, 0.0, -1.0],
            }
        )
        rl_frame = pd.DataFrame(
            {
                "time_ns": [1, 2],
                "agent_id": [1, 1],
                "action": [1, 1],
                "previous_action": [-1, 1],
                "filled_quantity_since_last_decision": [0.0, 0.0],
                "inventory": [0.0, 0.0],
                "reward": [0.0, 0.0],
            }
        )
        transition_frame = pd.DataFrame()

        def fake_build_config(*args, **kwargs):
            return DummyConfig()

        def fake_run_policy_episode(config, shared_policy=None):
            return market_frame.copy(), rl_frame.copy(), transition_frame.copy()

        def fake_summarize_episode(*args, **kwargs):
            return {
                "total_training_reward": 0.0,
                "average_reward_per_rl_agent": 0.0,
                "average_abs_ending_inventory": 0.0,
                "buy_fraction": 0.0,
                "hold_fraction": 1.0,
                "sell_fraction": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "phi_validation"
            with mock.patch("phi_experiment.build_abides_rmsc04_small_v1_config", side_effect=fake_build_config), \
                mock.patch("phi_experiment.run_policy_episode", side_effect=fake_run_policy_episode), \
                mock.patch("phi_experiment.summarize_episode", side_effect=fake_summarize_episode), \
                mock.patch("builtins.print") as mocked_print:
                result = run_phi_experiment(
                    phi_grid=[0.0],
                    episodes=1,
                    start_seed=7,
                    evaluation_seeds=[7],
                    output_dir=output_dir,
                )

            self.assertEqual(Path(result["output_dir"]), output_dir)

            warning_messages = " ".join(
                " ".join(str(arg) for arg in call.args)
                for call in mocked_print.call_args_list
            )
            self.assertIn("WARNING: pipeline_issue detected", warning_messages)

            greedy_dir = output_dir / "phi_0.00" / "evaluation_greedy"
            stochastic_dir = output_dir / "phi_0.00" / "evaluation_stochastic"

            diagnostics_by_seed = pd.read_csv(greedy_dir / "diagnostics_by_seed.csv")
            market_metrics_by_seed = pd.read_csv(greedy_dir / "market_metrics_by_seed.csv")
            evaluation_summary = pd.read_csv(greedy_dir / "evaluation_summary.csv")
            phi_sweep_summary = pd.read_csv(output_dir / "phi_sweep_summary.csv")
            phi_sweep_report = (output_dir / "phi_sweep_report.md").read_text(encoding="utf-8")
            evaluation_report = (greedy_dir / "evaluation_report.md").read_text(encoding="utf-8")
            summary_json = json.loads((output_dir / "phi_sweep_summary.json").read_text(encoding="utf-8"))

            required_run_columns = {
                "one_sided_book_fraction",
                "missing_bid_fraction",
                "missing_ask_fraction",
                "undefined_midprice_fraction",
                "max_consecutive_one_sided_duration",
                "num_one_sided_episodes",
                "midprice_gap_without_missing_side_fraction",
                "defined_midprice_despite_missing_side_fraction",
                "pipeline_issue_flag",
            }
            self.assertTrue(required_run_columns.issubset(diagnostics_by_seed.columns))
            self.assertTrue(required_run_columns.issubset(market_metrics_by_seed.columns))
            self.assertTrue({"one_sided_book_fraction_mean", "consistency_status", "pipeline_issue_seed_count"}.issubset(evaluation_summary.columns))
            self.assertTrue({"greedy_one_sided_book_fraction_mean", "greedy_consistency_status", "stochastic_consistency_status"}.issubset(phi_sweep_summary.columns))

            self.assertEqual(evaluation_summary.loc[0, "consistency_status"], "pipeline_issue")
            self.assertEqual(phi_sweep_summary.loc[0, "greedy_consistency_status"], "pipeline_issue")
            self.assertEqual(summary_json["results"][0]["greedy"]["consistency_status"], "pipeline_issue")
            self.assertIn("pipeline_issue", evaluation_report)
            self.assertIn("pipeline_issue", phi_sweep_report)

            for plot_name in (
                "one_sided_book_fraction_vs_phi.png",
                "undefined_midprice_fraction_vs_phi.png",
                "max_consecutive_one_sided_duration_vs_phi.png",
                "missing_bid_fraction_vs_phi.png",
                "missing_ask_fraction_vs_phi.png",
            ):
                plot_path = output_dir / "plots" / plot_name
                self.assertTrue(plot_path.exists(), plot_name)
                self.assertGreater(plot_path.stat().st_size, 0)

            for mode_dir in (greedy_dir, stochastic_dir):
                self.assertTrue((mode_dir / "diagnostics_by_seed.csv").exists())
                self.assertTrue((mode_dir / "market_metrics_by_seed.csv").exists())
                self.assertTrue((mode_dir / "evaluation_summary.csv").exists())
                self.assertTrue((mode_dir / "evaluation_report.md").exists())


if __name__ == "__main__":
    unittest.main()
