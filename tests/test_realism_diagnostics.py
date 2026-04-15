"""Tests for realism diagnostics and pathology flags."""

from __future__ import annotations

import unittest

import pandas as pd

from realism_diagnostics import (
    build_realism_report,
    compute_realism_diagnostics,
    flag_realism_pathologies,
    time_scale_assessment,
)


class RealismDiagnosticsTests(unittest.TestCase):
    """Realism diagnostics should flag obvious pathologies on synthetic data."""

    def test_compute_and_flag_pathologies(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                "best_bid": [99.99, 99.98, 99.97, 99.96],
                "best_ask": [100.00, 100.00, 99.99, 99.98],
                "midprice": [100.0, 98.0, 92.0, 85.0],
                "fundamental_value": [100.0, 99.5, 99.0, 98.5],
                "spread": [0.01, 0.01, 0.01, 0.01],
                "bid_depth": [1.0, 0.0, 1.0, 0.0],
                "ask_depth": [1.0, 1.0, 0.0, 0.0],
                "imbalance": [0.0, 1.0, -1.0, 1.0],
                "traded_volume": [1.0, 1.0, 1.0, 1.0],
                "signed_order_flow": [2.0, 2.0, 2.0, 2.0],
            }
        )
        passive_orders = pd.DataFrame(
            {
                "agent_type": ["ValueTrader", "AdaptiveMarketMaker"],
                "is_passive": [True, True],
                "placement_bucket": ["join", "improve"],
                "was_accepted": [True, True],
                "was_executed": [True, False],
                "time_to_terminal_event_seconds": [2.0, 5.0],
                "executed_quickly_after_rest": [True, False],
                "same_side_distance_ticks": [0.0, 1.0],
            }
        )
        trade_history = pd.DataFrame(
            {
                "quantity": [1, 2, 3],
                "passive_agent_type": [
                    "AdaptiveMarketMaker",
                    "ValueTrader",
                    "AdaptiveMarketMaker",
                ],
            }
        )
        diagnostics = compute_realism_diagnostics(
            frame,
            tick_size=0.01,
            passive_orders=passive_orders,
            trade_history=trade_history,
        )
        flags = flag_realism_pathologies(diagnostics)
        label, _ = time_scale_assessment(diagnostics)
        report = build_realism_report(diagnostics, flags)

        self.assertEqual(label, "price_scale_or_trader_aggressiveness_too_large")
        self.assertTrue(flags["excessive_price_drift"])
        self.assertTrue(flags["spread_mechanically_stuck"])
        self.assertTrue(flags["chronically_thin_book"])
        self.assertTrue(flags["persistent_one_sided_depth_imbalance"])
        self.assertIn("Realism Diagnostics Report", report)
        self.assertIn("excessive_price_drift", report)
        self.assertAlmostEqual(float(diagnostics["fraction_spread_two_ticks"]), 0.0)
        self.assertAlmostEqual(float(diagnostics["fraction_spread_three_plus_ticks"]), 0.0)
        self.assertGreater(float(diagnostics["quote_refresh_rate_per_minute"]), 0.0)
        self.assertGreater(float(diagnostics["mean_quote_lifetime_seconds"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["fraction_bid_depth_below_threshold"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["fraction_ask_depth_below_threshold"]), 0.0)
        self.assertGreaterEqual(float(diagnostics["fraction_total_depth_below_threshold"]), 0.0)
        self.assertAlmostEqual(
            float(diagnostics["fraction_traded_volume_against_mm_quotes"]),
            4.0 / 6.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["fraction_non_mm_passive_join_inside"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["fraction_non_mm_passive_orders_resting"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["passive_fill_rate_valuetrader"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["passive_mean_lifetime_seconds_adaptivemarketmaker"]),
            5.0,
        )
        self.assertIn("Fraction traded volume against MM quotes", report)

    def test_signed_flow_lag1_uses_true_temporal_alignment(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                "best_bid": [99.99, 100.00, 99.99, 100.00],
                "best_ask": [100.00, 100.02, 100.00, 100.02],
                "midprice": [100.0, 100.1, 100.0, 100.1],
                "fundamental_value": [100.0, 100.0, 100.0, 100.0],
                "spread": [0.01, 0.02, 0.01, 0.02],
                "bid_depth": [5.0, 5.0, 5.0, 5.0],
                "ask_depth": [5.0, 5.0, 5.0, 5.0],
                "imbalance": [0.0, 0.0, 0.0, 0.0],
                "traded_volume": [1.0, 1.0, 1.0, 1.0],
                "signed_order_flow": [1.0, -1.0, 1.0, -1.0],
            }
        )

        diagnostics = compute_realism_diagnostics(frame, tick_size=0.01)

        self.assertLess(float(diagnostics["signed_order_flow_autocorr_lag1"]), -0.5)


if __name__ == "__main__":
    unittest.main()
