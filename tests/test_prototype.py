"""Smoke tests for the first-pass market prototype."""

from __future__ import annotations

import unittest

import pandas as pd

from analysis import crash_rate, tail_exposure
from baseline_configs import OFFICIAL_BASELINE_NAME, build_abides_rmsc04_small_v1_config
from calibration import (
    DiagnosticThresholds,
    build_parameter_grid,
    classify_run,
    parse_sweep_spec,
    run_baseline_calibration,
    run_parameter_sweep,
)
from config import MAX_PHI, compute_agent_counts, compute_rl_role_counts
from market import MarketSimulator
from config import SimulationConfig


class PrototypeTests(unittest.TestCase):
    """Basic checks that the prototype builds and runs."""

    def test_agent_counts_sum_to_total(self) -> None:
        counts = compute_agent_counts(total_agents=102, phi=MAX_PHI)
        self.assertEqual(sum(counts.values()), 102)
        self.assertEqual(counts["rl"], 80)
        self.assertEqual(counts["noise"], 0)
        self.assertEqual(counts["value"], 20)
        self.assertEqual(counts["market_maker"], 2)
        self.assertEqual(counts["zic"], 0)
        self.assertEqual(counts["trend"], 0)

    def test_legacy_profile_keeps_trend_agents_available(self) -> None:
        counts = compute_agent_counts(total_agents=100, phi=0.0, market_profile="legacy_custom_v0")
        self.assertEqual(counts["trend"], 10)

    def test_rmsc04_profile_replaces_only_noise_with_rl(self) -> None:
        counts = compute_agent_counts(
            total_agents=102,
            phi=0.2,
            market_profile=OFFICIAL_BASELINE_NAME,
        )
        self.assertEqual(counts["value"], 20)
        self.assertEqual(counts["market_maker"], 2)
        self.assertEqual(counts["zic"], 0)
        self.assertEqual(counts["trend"], 0)
        self.assertLess(counts["noise"], 80)
        self.assertGreater(counts["rl"], 0)

    def test_mixed_rl_role_split_rounds_to_full_allocation(self) -> None:
        counts = compute_rl_role_counts(
            5,
            rl_liquidity_mode="mixed",
            rl_quoter_split=0.5,
        )
        self.assertEqual(counts["taker"] + counts["quoter"], 5)
        self.assertEqual(counts["taker"], 2)
        self.assertEqual(counts["quoter"], 3)

    def test_official_baseline_builder_sets_simple_profile(self) -> None:
        config = build_abides_rmsc04_small_v1_config(phi=0.0)
        self.assertEqual(config.market_profile, OFFICIAL_BASELINE_NAME)
        self.assertEqual(config.num_agents, 102)
        self.assertEqual(config.agent_counts()["trend"], 0)
        self.assertEqual(config.agent_counts()["noise"], 80)
        self.assertEqual(config.agent_counts()["value"], 20)
        self.assertEqual(config.agent_counts()["market_maker"], 2)

    def test_short_simulation_runs(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.13,
            seed=11,
            end_time="09:30:20",
            latency_type="no_latency",
            log_frequency="1s",
        )
        simulator = MarketSimulator(
            config
        )
        frame = simulator.run()
        self.assertGreater(len(frame), 0)
        required_columns = {
            "time",
            "best_bid",
            "best_ask",
            "best_bid_agent_id",
            "best_ask_agent_id",
            "best_bid_agent_type",
            "best_ask_agent_type",
            "midprice",
            "spread",
            "bid_depth",
            "ask_depth",
            "imbalance",
            "traded_volume",
            "signed_order_flow",
            "fundamental_value",
        }
        self.assertTrue(required_columns.issubset(frame.columns))
        self.assertEqual(int(frame["time"].diff().dropna().iloc[0]), int(1e9))
        self.assertTrue((frame["time"].diff().dropna() == int(1e9)).all())

    def test_official_baseline_path_omits_trend_traders(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.0,
            seed=13,
            end_time="09:30:05",
            latency_type="no_latency",
            log_frequency="1s",
        )
        simulator = MarketSimulator(config)
        agent_types = {getattr(agent, "type", "") for agent in simulator.config_state.kernel_config["agents"]}
        self.assertNotIn("TrendFollowerTrader", agent_types)

    def test_analysis_tail_and_crash_metrics(self) -> None:
        returns = pd.Series([-0.10, -0.06, -0.01, 0.01, 0.02])
        self.assertLessEqual(tail_exposure(returns, quantile=0.4), -0.06)
        self.assertAlmostEqual(crash_rate(returns, threshold=-0.05), 0.4)

    def test_baseline_calibration_runs(self) -> None:
        runs, summary = run_baseline_calibration(
            build_abides_rmsc04_small_v1_config(
                seed=31,
                end_time="09:30:10",
                latency_type="no_latency",
                log_frequency="1s",
            ),
            seeds=[31],
        )
        self.assertEqual(len(runs), 1)
        self.assertIn("average_spread", runs.columns)
        self.assertIn("ranking_score", summary.columns)

    def test_parse_sweep_and_grid(self) -> None:
        sweep_specs = [parse_sweep_spec("sigma_v=0.01,0.03"), parse_sweep_spec("alpha=0.001,0.005")]
        parameter_columns, grid = build_parameter_grid(sweep_specs)
        self.assertEqual(parameter_columns, ["sigma_v", "mm_alpha"])
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid[0]["sigma_v"], 0.01)

    def test_classification_logic(self) -> None:
        thresholds = DiagnosticThresholds(min_trade_count=10)
        frozen = classify_run(
            {
                "trade_count": 0,
                "average_top_of_book_depth": 2.0,
                "average_spread": 0.01,
                "average_relative_spread": 0.0001,
                "return_variance": 0.0,
                "crash_rate": 0.0,
                "max_drawdown": -0.01,
                "excess_kurtosis": 0.0,
            },
            thresholds,
        )
        self.assertEqual(frozen, "frozen")

    def test_parameter_sweep_runs(self) -> None:
        parameter_columns, grid = build_parameter_grid([parse_sweep_spec("sigma_v=0.03")])
        runs, summary = run_parameter_sweep(
            build_abides_rmsc04_small_v1_config(
                seed=19,
                end_time="09:30:10",
                latency_type="no_latency",
                log_frequency="1s",
            ),
            seeds=[19],
            parameter_grid=grid,
            parameter_columns=parameter_columns,
        )
        self.assertEqual(len(runs), 1)
        self.assertIn("baseline_quality_score", runs.columns)
        self.assertIn("classification", runs.columns)
        self.assertIn("ranking_score", summary.columns)


if __name__ == "__main__":
    unittest.main()
