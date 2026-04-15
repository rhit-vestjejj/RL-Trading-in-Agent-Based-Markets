"""Tests for longer-run PPO training reporting helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from training_reporting import (
    build_combined_progress_frame,
    build_training_progress_report,
    save_training_progress_plots,
)


class TrainingReportingTests(unittest.TestCase):
    def sample_training_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "episode": [0.0, 1.0, 2.0],
                "total_training_reward": [-10.0, -5.0, -1.0],
                "average_abs_ending_inventory": [4.0, 2.0, 1.0],
                "max_abs_inventory": [10.0, 7.0, 4.0],
                "entropy": [1.0, 0.9, 0.8],
                "value_loss": [0.4, 0.3, 0.2],
            }
        )

    def sample_evaluation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "episode": [0.0, 0.0, 2.0, 2.0],
                "evaluation_mode": ["greedy", "stochastic", "greedy", "stochastic"],
                "evaluation_total_reward_mean": [-3.0, -6.0, -1.0, -2.0],
                "evaluation_average_abs_ending_inventory_mean": [2.0, 2.5, 1.0, 1.2],
                "evaluation_buy_fraction_mean": [0.1, 0.3, 0.2, 0.31],
                "evaluation_hold_fraction_mean": [0.8, 0.4, 0.6, 0.39],
                "evaluation_sell_fraction_mean": [0.1, 0.3, 0.2, 0.30],
                "evaluation_inventory_min_global": [-1.0, -2.0, -1.0, -2.0],
                "evaluation_inventory_max_global": [0.0, 2.0, 1.0, 2.0],
            }
        )

    def test_build_combined_progress_frame_marks_phase(self) -> None:
        combined = build_combined_progress_frame(
            self.sample_training_frame(),
            self.sample_evaluation_frame(),
        )
        self.assertEqual(set(combined["phase"]), {"train", "evaluation"})
        self.assertIn("evaluation_mode", combined.columns)

    def test_build_training_progress_report_mentions_greedy_and_stochastic(self) -> None:
        report = build_training_progress_report(
            self.sample_training_frame(),
            self.sample_evaluation_frame(),
        )
        self.assertIn("Greedy Evaluation", report)
        self.assertIn("Stochastic Evaluation", report)
        self.assertIn("Mean reward last window", report)

    def test_save_training_progress_plots_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_training_progress_plots(
                self.sample_training_frame(),
                self.sample_evaluation_frame(),
                tmpdir,
            )
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
