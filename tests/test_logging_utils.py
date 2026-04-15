"""Tests for logging and resampling helpers."""

from __future__ import annotations

import unittest
import warnings

import pandas as pd

from logging_utils import _resample_to_fixed_grid


class LoggingUtilsTests(unittest.TestCase):
    def test_resample_keeps_one_sided_book_midprice_and_spread_undefined(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "time_ns": 10,
                    "time": 10,
                    "best_bid": 10000.0,
                    "best_ask": 10001.0,
                    "bid_depth": 10.0,
                    "ask_depth": 5.0,
                    "midprice": 10000.5,
                    "spread": 1.0,
                    "imbalance": 0.333333,
                    "best_bid_agent_id": 1,
                    "best_ask_agent_id": 2,
                    "best_bid_agent_type": "MM",
                    "best_ask_agent_type": "Value",
                },
                {
                    "time_ns": 20,
                    "time": 20,
                    "best_bid": 10000.0,
                    "best_ask": float("nan"),
                    "bid_depth": 12.0,
                    "ask_depth": 0.0,
                    "midprice": float("nan"),
                    "spread": float("nan"),
                    "imbalance": 1.0,
                    "best_bid_agent_id": 1,
                    "best_ask_agent_id": pd.NA,
                    "best_bid_agent_type": "MM",
                    "best_ask_agent_type": pd.NA,
                },
            ]
        )
        grid = pd.DataFrame({"time_ns": [10, 20]})

        resampled = _resample_to_fixed_grid(grid, observations)

        self.assertEqual(float(resampled.loc[1, "best_bid"]), 10000.0)
        self.assertTrue(pd.isna(resampled.loc[1, "best_ask"]))
        self.assertTrue(pd.isna(resampled.loc[1, "spread"]))
        self.assertTrue(pd.isna(resampled.loc[1, "midprice"]))
        self.assertEqual(float(resampled.loc[1, "ask_depth"]), 0.0)

    def test_resample_fills_leading_gap_without_future_warning(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "time_ns": 10,
                    "time": 10,
                    "best_bid": 10000.0,
                    "best_ask": 10001.0,
                    "bid_depth": 10.0,
                    "ask_depth": 5.0,
                    "midprice": 10000.5,
                    "spread": 1.0,
                    "imbalance": 0.333333,
                    "best_bid_agent_id": 1,
                    "best_ask_agent_id": 2,
                    "best_bid_agent_type": "MM",
                    "best_ask_agent_type": "Value",
                }
            ]
        )
        grid = pd.DataFrame({"time_ns": [0, 10]})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resampled = _resample_to_fixed_grid(grid, observations)

        future_warnings = [warning for warning in caught if issubclass(warning.category, FutureWarning)]
        self.assertEqual(future_warnings, [])
        self.assertEqual(float(resampled.loc[0, "best_bid"]), 10000.0)
        self.assertEqual(float(resampled.loc[0, "best_ask"]), 10001.0)
        self.assertEqual(float(resampled.loc[0, "spread"]), 1.0)


if __name__ == "__main__":
    unittest.main()
