"""Tests for same-run multi-resolution sampling comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from config import SimulationConfig
from market import MarketSimulator
from sampling_comparison import (
    build_sampling_report,
    parse_frequency_list,
    save_sampling_outputs,
    summarize_sampling_frames,
)


class SamplingComparisonTests(unittest.TestCase):
    """Sampling-comparison helpers should be deterministic and reusable."""

    @staticmethod
    def synthetic_frames() -> dict[str, pd.DataFrame]:
        return {
            "1s": pd.DataFrame(
                {
                    "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                    "midprice": [100.0, 100.1, 99.9, 100.2],
                    "spread": [0.02, 0.03, 0.02, 0.04],
                    "bid_depth": [5.0, 6.0, 4.0, 7.0],
                    "ask_depth": [4.0, 5.0, 5.0, 6.0],
                    "traded_volume": [1.0, 2.0, 3.0, 2.0],
                    "signed_order_flow": [1.0, -2.0, 1.0, 0.0],
                    "fundamental_value": [100.0, 100.02, 100.03, 100.05],
                }
            ),
            "5s": pd.DataFrame(
                {
                    "time": [0, 5_000_000_000],
                    "midprice": [100.0, 100.2],
                    "spread": [0.02, 0.04],
                    "bid_depth": [5.0, 7.0],
                    "ask_depth": [4.0, 6.0],
                    "traded_volume": [3.0, 5.0],
                    "signed_order_flow": [-1.0, 1.0],
                    "fundamental_value": [100.0, 100.05],
                }
            ),
        }

    def test_parse_frequency_list(self) -> None:
        self.assertEqual(parse_frequency_list("1s, 5s,15s"), ["1s", "5s", "15s"])

    def test_sampling_summary_and_report(self) -> None:
        frames = self.synthetic_frames()
        summary = summarize_sampling_frames(frames)
        self.assertEqual(set(summary["log_frequency"]), {"1s", "5s"})
        self.assertIn("average_spread", summary.columns)
        report = build_sampling_report(summary, baseline_frequency="1s")
        self.assertIn("Sampling Comparison Report", report)
        self.assertIn("1s", report)
        self.assertIn("5s", report)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_sampling_outputs(frames, output_dir=tmpdir, title_prefix="Test")
            self.assertEqual(len(saved), 4)
            for path in saved:
                self.assertTrue(Path(path).exists())

    def test_same_run_extracts_multiple_frequencies(self) -> None:
        simulator = MarketSimulator(
            build_abides_rmsc04_small_v1_config(
                seed=13,
                end_time="09:30:10",
                latency_type="no_latency",
                log_frequency="1s",
            )
        )
        simulator.run()
        frames = simulator.extract_frames(["1s", "5s"])
        self.assertIn("1s", frames)
        self.assertIn("5s", frames)
        self.assertGreaterEqual(len(frames["1s"]), len(frames["5s"]))
        self.assertEqual(int(frames["1s"]["time"].iloc[-1]), int(frames["5s"]["time"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
