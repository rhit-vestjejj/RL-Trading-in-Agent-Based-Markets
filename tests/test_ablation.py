"""Tests for the narrow price-discovery ablation helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from ablation import compute_ablation_metrics, parse_ablation_specs, summarize_ablation_runs


class AblationTests(unittest.TestCase):
    def test_compute_ablation_metrics_on_synthetic_inputs(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                "midprice": [100.0, 100.02, 100.01, 100.03],
                "fundamental_value": [100.0, 100.01, 100.00, 100.04],
                "best_bid": [99.99, 100.00, 99.99, 100.01],
                "best_ask": [100.01, 100.02, 100.03, 100.04],
                "spread": [0.02, 0.02, 0.04, 0.03],
                "bid_depth": [10.0, 12.0, 11.0, 9.0],
                "ask_depth": [9.0, 10.0, 8.0, 9.0],
                "imbalance": [0.05, 0.09, 0.16, 0.0],
                "traded_volume": [2.0, 3.0, 4.0, 5.0],
                "signed_order_flow": [1.0, -1.0, 2.0, -2.0],
                "best_bid_agent_type": [
                    "AdaptiveMarketMaker",
                    "TrendFollowerTrader",
                    "AdaptiveMarketMaker",
                    "ValueTrader",
                ],
                "best_ask_agent_type": [
                    "AdaptiveMarketMaker",
                    "AdaptiveMarketMaker",
                    "NoiseTrader",
                    "ValueTrader",
                ],
            }
        )
        passive_orders = pd.DataFrame(
            {
                "order_id": [1, 2, 3],
                "agent_type": [
                    "TrendFollowerTrader",
                    "ValueTrader",
                    "AdaptiveMarketMaker",
                ],
                "is_passive": [True, True, True],
                "placement_bucket": ["join", "improve", "join"],
                "quantity": [2, 2, 6],
                "was_executed": [True, True, False],
                "rested_before_execution": [True, False, True],
                "executed_quickly_after_rest": [False, True, False],
                "same_side_distance_ticks": [0.0, 1.0, 0.0],
                "side": ["bid", "ask", "bid"],
                "accepted_time_ns": [0.0, 1_000_000_000.0, 0.0],
                "time_submitted_ns": [0.0, 1_000_000_000.0, 0.0],
                "cancelled_time_ns": [float("nan"), float("nan"), 3_000_000_000.0],
                "executed_quantity": [2.0, 2.0, 0.0],
                "time_to_terminal_event_seconds": [3.0, 1.0, 5.0],
            }
        )
        trade_history = pd.DataFrame(
            {
                "quantity": [1, 2, 1, 3],
                "time_ns": [1_000_000_000, 2_000_000_000, 2_000_000_000, 3_000_000_000],
                "passive_order_id": [3, 2, 1, 3],
                "passive_agent_type": [
                    "AdaptiveMarketMaker",
                    "ValueTrader",
                    "TrendFollowerTrader",
                    "AdaptiveMarketMaker",
                ],
                "aggressor_agent_type": [
                    "TrendFollowerTrader",
                    "NoiseTrader",
                    "ValueTrader",
                    "TrendFollowerTrader",
                ],
            }
        )

        metrics = compute_ablation_metrics(
            frame,
            tick_size=0.01,
            passive_orders=passive_orders,
            trade_history=trade_history,
            market_open_ns=0,
            market_close_ns=3_000_000_000,
        )

        self.assertAlmostEqual(metrics["midprice_range"], 0.03)
        self.assertGreater(metrics["fraction_zero_midprice_change"], 0.0)
        self.assertAlmostEqual(metrics["passive_execution_share_adaptivemarketmaker"], 0.5)
        self.assertAlmostEqual(metrics["aggressive_execution_share_trendfollowertrader"], 0.5)
        self.assertAlmostEqual(metrics["passive_fill_rate_trendfollowertrader"], 1.0)
        self.assertAlmostEqual(metrics["passive_mean_lifetime_seconds_valuetrader"], 1.0)
        self.assertIn("mean_resting_depth_adaptivemarketmaker", metrics)

    def test_parse_and_summarize_ablation_specs(self) -> None:
        parameter_columns, parameter_grid = parse_ablation_specs(
            ["mm_quote_size=4,6", "trend_aggressive_probability=0.2,0.4"]
        )
        self.assertEqual(
            parameter_columns,
            ["mm_quote_size", "trend_aggressive_probability"],
        )
        self.assertEqual(len(parameter_grid), 4)

        runs = pd.DataFrame(
            {
                "mm_quote_size": [4, 4, 6, 6],
                "trend_aggressive_probability": [0.2, 0.2, 0.4, 0.4],
                "midprice_range": [0.5, 0.7, 0.9, 1.1],
                "fraction_zero_midprice_change": [0.6, 0.5, 0.4, 0.3],
                "average_nonzero_midprice_change": [0.01, 0.02, 0.03, 0.04],
                "midprice_fundamental_change_correlation": [0.2, 0.3, 0.4, 0.5],
                "average_spread": [0.02, 0.03, 0.03, 0.04],
                "average_depth": [10.0, 11.0, 12.0, 13.0],
                "trade_count": [100, 110, 120, 130],
                "best_bid_mm_share": [0.9, 0.8, 0.7, 0.6],
                "best_ask_mm_share": [0.9, 0.8, 0.7, 0.6],
                "fraction_trade_count_against_mm_quotes": [0.8, 0.7, 0.6, 0.5],
                "fraction_traded_volume_against_mm_quotes": [0.8, 0.7, 0.6, 0.5],
                "fraction_non_mm_passive_join_inside": [0.1, 0.2, 0.3, 0.4],
                "fraction_non_mm_passive_improve_inside": [0.2, 0.2, 0.2, 0.2],
                "fraction_spread_one_tick": [0.3, 0.4, 0.5, 0.6],
                "fraction_spread_two_ticks": [0.4, 0.3, 0.2, 0.1],
                "fraction_spread_three_plus_ticks": [0.3, 0.3, 0.3, 0.3],
            }
        )

        summary = summarize_ablation_runs(
            runs,
            parameter_columns=["mm_quote_size", "trend_aggressive_probability"],
        )
        self.assertIn("midprice_range_mean", summary.columns)
        self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
