"""Tests for the fixed-clock random-walk oracle."""

from __future__ import annotations

import unittest

import numpy as np

from abides_oracle import RandomWalkOracle


class RandomWalkOracleTests(unittest.TestCase):
    """The oracle should evolve on a fixed simulator clock, not per query."""

    def test_fundamental_only_changes_when_crossing_update_boundaries(self) -> None:
        oracle = RandomWalkOracle(
            mkt_open=0,
            mkt_close=5_000_000_000,
            symbols={
                "ABM": {
                    "v0": 10_000,
                    "sigma_v": 5.0,
                    "fundamental_interval_ns": 1_000_000_000,
                    "random_state": np.random.RandomState(7),
                }
            },
        )

        half_second = oracle.advance_fundamental_value_series(500_000_000, "ABM")
        one_second = oracle.advance_fundamental_value_series(1_000_000_000, "ABM")
        one_point_nine_seconds = oracle.advance_fundamental_value_series(1_900_000_000, "ABM")
        two_seconds = oracle.advance_fundamental_value_series(2_000_000_000, "ABM")

        self.assertEqual(half_second, 10_000)
        self.assertEqual(one_second, one_point_nine_seconds)
        self.assertNotEqual(one_second, two_seconds)
        self.assertEqual(
            [row["FundamentalTime"] for row in oracle.f_log["ABM"]],
            [0, 1_000_000_000, 2_000_000_000],
        )


if __name__ == "__main__":
    unittest.main()
