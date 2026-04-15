"""Tests for baseline parameter selection and recommendation output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from baseline_selection import (
    RobustnessConfig,
    assess_nearby_robustness,
    build_recommendation_report,
    load_selection_inputs,
    rank_candidate_settings,
    recommend_baseline,
    save_recommendation_report,
)


def synthetic_summary() -> pd.DataFrame:
    """Return a deterministic grouped-summary table with one clear winner."""

    return pd.DataFrame(
        [
            {
                "sigma_v": 0.03,
                "mm_alpha": 0.005,
                "average_spread_mean": 0.018,
                "average_spread_std": 0.002,
                "average_relative_spread_mean": 0.00018,
                "average_relative_spread_std": 0.00002,
                "average_top_of_book_depth_mean": 7.0,
                "average_top_of_book_depth_std": 0.4,
                "return_variance_mean": 2.0e-6,
                "return_variance_std": 2.0e-7,
                "crash_rate_mean": 0.0,
                "crash_rate_std": 0.0,
                "tail_exposure_mean": -0.0010,
                "tail_exposure_std": 0.0002,
                "max_drawdown_mean": -0.020,
                "max_drawdown_std": 0.004,
                "trade_count_mean": 120.0,
                "trade_count_std": 6.0,
                "traded_volume_mean": 180.0,
                "traded_volume_std": 9.0,
                "mm_mean_abs_inventory_mean": 3.0,
                "mm_mean_abs_inventory_std": 0.5,
                "share_stable": 1.0,
                "share_thin": 0.0,
                "share_frozen": 0.0,
                "share_chaotic": 0.0,
                "share_crash-prone": 0.0,
                "num_runs": 4,
            },
            {
                "sigma_v": 0.02,
                "mm_alpha": 0.005,
                "average_spread_mean": 0.020,
                "average_spread_std": 0.003,
                "average_relative_spread_mean": 0.00020,
                "average_relative_spread_std": 0.00003,
                "average_top_of_book_depth_mean": 6.3,
                "average_top_of_book_depth_std": 0.8,
                "return_variance_mean": 1.7e-6,
                "return_variance_std": 3.0e-7,
                "crash_rate_mean": 0.0,
                "crash_rate_std": 0.0,
                "tail_exposure_mean": -0.0012,
                "tail_exposure_std": 0.0002,
                "max_drawdown_mean": -0.018,
                "max_drawdown_std": 0.005,
                "trade_count_mean": 110.0,
                "trade_count_std": 8.0,
                "traded_volume_mean": 165.0,
                "traded_volume_std": 10.0,
                "mm_mean_abs_inventory_mean": 2.5,
                "mm_mean_abs_inventory_std": 0.6,
                "share_stable": 0.75,
                "share_thin": 0.0,
                "share_frozen": 0.25,
                "share_chaotic": 0.0,
                "share_crash-prone": 0.0,
                "num_runs": 4,
            },
            {
                "sigma_v": 0.03,
                "mm_alpha": 0.003,
                "average_spread_mean": 0.021,
                "average_spread_std": 0.003,
                "average_relative_spread_mean": 0.00021,
                "average_relative_spread_std": 0.00003,
                "average_top_of_book_depth_mean": 6.0,
                "average_top_of_book_depth_std": 0.7,
                "return_variance_mean": 2.5e-6,
                "return_variance_std": 4.0e-7,
                "crash_rate_mean": 0.0,
                "crash_rate_std": 0.0,
                "tail_exposure_mean": -0.0013,
                "tail_exposure_std": 0.0002,
                "max_drawdown_mean": -0.022,
                "max_drawdown_std": 0.006,
                "trade_count_mean": 115.0,
                "trade_count_std": 8.0,
                "traded_volume_mean": 170.0,
                "traded_volume_std": 11.0,
                "mm_mean_abs_inventory_mean": 3.3,
                "mm_mean_abs_inventory_std": 0.6,
                "share_stable": 0.75,
                "share_thin": 0.25,
                "share_frozen": 0.0,
                "share_chaotic": 0.0,
                "share_crash-prone": 0.0,
                "num_runs": 4,
            },
            {
                "sigma_v": 0.02,
                "mm_alpha": 0.003,
                "average_spread_mean": 0.012,
                "average_spread_std": 0.006,
                "average_relative_spread_mean": 0.00012,
                "average_relative_spread_std": 0.00005,
                "average_top_of_book_depth_mean": 1.2,
                "average_top_of_book_depth_std": 0.4,
                "return_variance_mean": 2.0e-9,
                "return_variance_std": 1.0e-9,
                "crash_rate_mean": 0.0,
                "crash_rate_std": 0.0,
                "tail_exposure_mean": -0.0002,
                "tail_exposure_std": 0.0001,
                "max_drawdown_mean": -0.004,
                "max_drawdown_std": 0.002,
                "trade_count_mean": 15.0,
                "trade_count_std": 4.0,
                "traded_volume_mean": 22.0,
                "traded_volume_std": 6.0,
                "mm_mean_abs_inventory_mean": 8.0,
                "mm_mean_abs_inventory_std": 2.0,
                "share_stable": 0.0,
                "share_thin": 0.5,
                "share_frozen": 0.5,
                "share_chaotic": 0.0,
                "share_crash-prone": 0.0,
                "num_runs": 4,
            },
        ]
    )


def synthetic_runs() -> pd.DataFrame:
    """Return a deterministic run-level table matching the grouped ranking."""

    rows = [
        {
            "seed": 1,
            "sigma_v": 0.03,
            "mm_alpha": 0.005,
            "average_spread": 0.017,
            "average_relative_spread": 0.00017,
            "average_top_of_book_depth": 7.2,
            "return_variance": 1.8e-6,
            "excess_kurtosis": 1.0,
            "volatility_clustering": 0.15,
            "max_drawdown": -0.018,
            "crash_rate": 0.0,
            "tail_exposure": -0.0009,
            "trade_count": 124,
            "traded_volume": 184.0,
            "baseline_quality_score": 100.0,
            "mm_mean_inventory": 0.5,
            "mm_mean_abs_inventory": 2.8,
            "mm_max_abs_inventory": 5.0,
            "classification": "stable",
        },
        {
            "seed": 2,
            "sigma_v": 0.03,
            "mm_alpha": 0.005,
            "average_spread": 0.019,
            "average_relative_spread": 0.00019,
            "average_top_of_book_depth": 6.8,
            "return_variance": 2.2e-6,
            "excess_kurtosis": 0.8,
            "volatility_clustering": 0.12,
            "max_drawdown": -0.022,
            "crash_rate": 0.0,
            "tail_exposure": -0.0011,
            "trade_count": 116,
            "traded_volume": 176.0,
            "baseline_quality_score": 100.0,
            "mm_mean_inventory": -0.4,
            "mm_mean_abs_inventory": 3.2,
            "mm_max_abs_inventory": 5.5,
            "classification": "stable",
        },
        {
            "seed": 1,
            "sigma_v": 0.02,
            "mm_alpha": 0.005,
            "average_spread": 0.019,
            "average_relative_spread": 0.00019,
            "average_top_of_book_depth": 6.9,
            "return_variance": 1.6e-6,
            "excess_kurtosis": 0.7,
            "volatility_clustering": 0.10,
            "max_drawdown": -0.016,
            "crash_rate": 0.0,
            "tail_exposure": -0.0010,
            "trade_count": 118,
            "traded_volume": 170.0,
            "baseline_quality_score": 96.0,
            "mm_mean_inventory": 0.2,
            "mm_mean_abs_inventory": 2.1,
            "mm_max_abs_inventory": 4.5,
            "classification": "stable",
        },
        {
            "seed": 2,
            "sigma_v": 0.02,
            "mm_alpha": 0.005,
            "average_spread": 0.021,
            "average_relative_spread": 0.00021,
            "average_top_of_book_depth": 5.7,
            "return_variance": 1.8e-6,
            "excess_kurtosis": 1.3,
            "volatility_clustering": 0.08,
            "max_drawdown": -0.020,
            "crash_rate": 0.0,
            "tail_exposure": -0.0014,
            "trade_count": 102,
            "traded_volume": 160.0,
            "baseline_quality_score": 82.0,
            "mm_mean_inventory": -0.1,
            "mm_mean_abs_inventory": 2.9,
            "mm_max_abs_inventory": 5.2,
            "classification": "frozen",
        },
        {
            "seed": 1,
            "sigma_v": 0.03,
            "mm_alpha": 0.003,
            "average_spread": 0.020,
            "average_relative_spread": 0.00020,
            "average_top_of_book_depth": 6.5,
            "return_variance": 2.1e-6,
            "excess_kurtosis": 1.1,
            "volatility_clustering": 0.11,
            "max_drawdown": -0.020,
            "crash_rate": 0.0,
            "tail_exposure": -0.0012,
            "trade_count": 120,
            "traded_volume": 174.0,
            "baseline_quality_score": 92.0,
            "mm_mean_inventory": 0.7,
            "mm_mean_abs_inventory": 3.0,
            "mm_max_abs_inventory": 5.5,
            "classification": "stable",
        },
        {
            "seed": 2,
            "sigma_v": 0.03,
            "mm_alpha": 0.003,
            "average_spread": 0.022,
            "average_relative_spread": 0.00022,
            "average_top_of_book_depth": 5.5,
            "return_variance": 2.9e-6,
            "excess_kurtosis": 1.4,
            "volatility_clustering": 0.10,
            "max_drawdown": -0.024,
            "crash_rate": 0.0,
            "tail_exposure": -0.0014,
            "trade_count": 110,
            "traded_volume": 166.0,
            "baseline_quality_score": 88.0,
            "mm_mean_inventory": -0.9,
            "mm_mean_abs_inventory": 3.6,
            "mm_max_abs_inventory": 6.0,
            "classification": "thin",
        },
        {
            "seed": 1,
            "sigma_v": 0.02,
            "mm_alpha": 0.003,
            "average_spread": 0.010,
            "average_relative_spread": 0.00010,
            "average_top_of_book_depth": 1.5,
            "return_variance": 1.0e-9,
            "excess_kurtosis": 0.1,
            "volatility_clustering": 0.02,
            "max_drawdown": -0.003,
            "crash_rate": 0.0,
            "tail_exposure": -0.0001,
            "trade_count": 18,
            "traded_volume": 24.0,
            "baseline_quality_score": 40.0,
            "mm_mean_inventory": 1.8,
            "mm_mean_abs_inventory": 7.0,
            "mm_max_abs_inventory": 11.0,
            "classification": "thin",
        },
        {
            "seed": 2,
            "sigma_v": 0.02,
            "mm_alpha": 0.003,
            "average_spread": 0.014,
            "average_relative_spread": 0.00014,
            "average_top_of_book_depth": 0.9,
            "return_variance": 3.0e-9,
            "excess_kurtosis": 0.2,
            "volatility_clustering": 0.01,
            "max_drawdown": -0.005,
            "crash_rate": 0.0,
            "tail_exposure": -0.0003,
            "trade_count": 12,
            "traded_volume": 20.0,
            "baseline_quality_score": 30.0,
            "mm_mean_inventory": -2.0,
            "mm_mean_abs_inventory": 9.0,
            "mm_max_abs_inventory": 13.0,
            "classification": "frozen",
        },
    ]
    return pd.DataFrame(rows)


class BaselineSelectionTests(unittest.TestCase):
    """Selection logic should be deterministic on synthetic sweep data."""

    def test_rank_candidates_selects_expected_baseline(self) -> None:
        ranked = rank_candidate_settings(
            synthetic_summary(),
            parameter_columns=["sigma_v", "mm_alpha"],
        )
        recommended = recommend_baseline(ranked)
        self.assertAlmostEqual(float(recommended["sigma_v"]), 0.03)
        self.assertAlmostEqual(float(recommended["mm_alpha"]), 0.005)
        self.assertEqual(int(recommended["candidate_rank"]), 1)
        self.assertGreater(float(recommended["selection_score"]), float(ranked.iloc[1]["selection_score"]))

    def test_report_is_written_and_mentions_recommendation(self) -> None:
        ranked = rank_candidate_settings(
            synthetic_summary(),
            parameter_columns=["sigma_v", "mm_alpha"],
        )
        recommended = recommend_baseline(ranked)
        robustness = assess_nearby_robustness(
            ranked,
            parameter_columns=["sigma_v", "mm_alpha"],
            recommended=recommended,
            config=RobustnessConfig(
                score_gap=20.0,
                min_stable_share=0.5,
                min_similar_fraction=0.5,
            ),
        )
        self.assertEqual(robustness["label"], "part of a stable region")

        report = build_recommendation_report(
            ranked,
            parameter_columns=["sigma_v", "mm_alpha"],
            recommended=recommended,
            robustness=robustness,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = save_recommendation_report(report, Path(tmpdir) / "recommended_baseline.md")
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Recommended Phi=0 Baseline", text)
            self.assertIn("- sigma_v: 0.03", text)
            self.assertIn("- mm_alpha: 0.005", text)

    def test_runs_input_reconstructs_summary_and_same_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_path = Path(tmpdir) / "synthetic_runs.csv"
            synthetic_runs().to_csv(runs_path, index=False)
            _, summary, parameter_columns = load_selection_inputs(runs_input=runs_path)

        ranked = rank_candidate_settings(summary, parameter_columns=parameter_columns)
        recommended = recommend_baseline(ranked)
        self.assertEqual(parameter_columns, ["sigma_v", "mm_alpha"])
        self.assertAlmostEqual(float(recommended["sigma_v"]), 0.03)
        self.assertAlmostEqual(float(recommended["mm_alpha"]), 0.005)


if __name__ == "__main__":
    unittest.main()
