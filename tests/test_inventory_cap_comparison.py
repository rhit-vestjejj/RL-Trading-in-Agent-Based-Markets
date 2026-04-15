"""Tests for the inventory-cap robustness comparison utility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from inventory_cap_comparison import (
    build_inventory_cap_comparison,
    load_inventory_cap_experiment,
)


class InventoryCapComparisonTests(unittest.TestCase):
    @staticmethod
    def _write_synthetic_experiment(root: Path, *, inventory_cap: int | None, cap_label: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "summaries").mkdir(parents=True, exist_ok=True)
        config = {
            "market_profile": "abides_rmsc04_small_v1",
            "phi_grid": [0.0, 0.2, 0.3, 0.5],
            "episodes": 50,
            "start_seed": 7,
            "evaluation_seeds": [7, 8, 9],
            "evaluation_modes": ["greedy", "stochastic"],
            "evaluation_interval": 5,
            "checkpoint_interval": 5,
            "end_time": "09:35:00",
            "log_frequency": "1s",
            "lambda_q": 0.01,
            "flat_hold_penalty": 0.02,
            "inventory_cap": inventory_cap,
        }
        (root / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "phi_sweep_summary.json").write_text(json.dumps({"experiment_config": config, "results": []}), encoding="utf-8")
        (root / "phi_sweep_report.md").write_text(f"# {cap_label}\n", encoding="utf-8")
        pd.DataFrame({"phi": [0.0], "seed": [7]}).to_csv(root / "summaries" / "per_seed_market_metrics.csv", index=False)
        pd.DataFrame({"phi": [0.0], "seed": [7]}).to_csv(root / "summaries" / "per_seed_rl_diagnostics.csv", index=False)

        if cap_label == "no_cap":
            volatility = [0.010, 0.012, 0.020, 0.030]
            spread = [0.010, 0.011, 0.015, 0.020]
            depth = [10.0, 9.0, 6.5, 5.0]
            one_sided = [0.00, 0.02, 0.10, 0.25]
            tail = [-0.010, -0.012, -0.020, -0.030]
            avg_inventory = [12.0, 18.0, 26.0, 32.0]
            max_inventory = [40.0, 55.0, 70.0, 88.0]
        elif cap_label == "cap_50":
            volatility = [0.010, 0.011, 0.017, 0.024]
            spread = [0.010, 0.011, 0.014, 0.018]
            depth = [10.0, 9.2, 7.2, 6.0]
            one_sided = [0.00, 0.02, 0.08, 0.18]
            tail = [-0.010, -0.011, -0.017, -0.024]
            avg_inventory = [10.0, 14.0, 18.0, 22.0]
            max_inventory = [28.0, 34.0, 42.0, 50.0]
        else:
            volatility = [0.010, 0.0105, 0.013, 0.016]
            spread = [0.010, 0.0105, 0.012, 0.014]
            depth = [10.0, 9.6, 8.8, 8.0]
            one_sided = [0.00, 0.01, 0.04, 0.09]
            tail = [-0.010, -0.0105, -0.013, -0.016]
            avg_inventory = [6.0, 8.0, 10.0, 12.0]
            max_inventory = [12.0, 16.0, 20.0, 20.0]

        summary = pd.DataFrame(
            {
                "phi": [0.0, 0.2, 0.3, 0.5],
                "greedy_average_spread_mean": spread,
                "greedy_average_depth_mean": depth,
                "greedy_zero_return_fraction_mean": [0.70, 0.68, 0.60, 0.55],
                "greedy_volatility_mean": volatility,
                "greedy_tail_exposure_mean": tail,
                "greedy_max_drawdown_mean": [-0.02, -0.03, -0.05, -0.08],
                "greedy_one_sided_book_fraction_mean": one_sided,
                "greedy_undefined_midprice_fraction_mean": one_sided,
                "greedy_max_consecutive_one_sided_duration_mean": [0.0, 2.0, 6.0, 10.0],
                "greedy_evaluation_average_abs_ending_inventory_mean": avg_inventory,
                "greedy_evaluation_max_abs_inventory_reached_max": max_inventory,
                "greedy_pipeline_issue_seed_fraction": [0.0, 0.0, 0.0, 0.0],
                "greedy_midprice_gap_without_missing_side_fraction_mean": [0.0, 0.0, 0.0, 0.0],
                "greedy_defined_midprice_despite_missing_side_fraction_mean": [0.0, 0.0, 0.0, 0.0],
                "greedy_consistency_status": ["consistent", "consistent", "consistent", "consistent"],
                "stochastic_average_spread_mean": spread,
                "stochastic_average_depth_mean": depth,
                "stochastic_zero_return_fraction_mean": [0.70, 0.68, 0.60, 0.55],
                "stochastic_volatility_mean": volatility,
                "stochastic_tail_exposure_mean": tail,
                "stochastic_max_drawdown_mean": [-0.02, -0.03, -0.05, -0.08],
                "stochastic_one_sided_book_fraction_mean": one_sided,
                "stochastic_undefined_midprice_fraction_mean": one_sided,
                "stochastic_max_consecutive_one_sided_duration_mean": [0.0, 2.0, 6.0, 10.0],
                "stochastic_evaluation_average_abs_ending_inventory_mean": avg_inventory,
                "stochastic_evaluation_max_abs_inventory_reached_max": max_inventory,
                "stochastic_pipeline_issue_seed_fraction": [0.0, 0.0, 0.0, 0.0],
                "stochastic_midprice_gap_without_missing_side_fraction_mean": [0.0, 0.0, 0.0, 0.0],
                "stochastic_defined_midprice_despite_missing_side_fraction_mean": [0.0, 0.0, 0.0, 0.0],
                "stochastic_consistency_status": ["consistent", "consistent", "consistent", "consistent"],
            }
        )
        summary.to_csv(root / "phi_sweep_summary.csv", index=False)

    def test_load_inventory_cap_experiment_reads_frozen_summary_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "phi_sweep_cap50"
            self._write_synthetic_experiment(root, inventory_cap=50, cap_label="cap_50")
            frame = load_inventory_cap_experiment(root, cap_label="cap_50")
            self.assertEqual(set(frame["inventory_cap_label"]), {"cap_50"})
            self.assertIn("average_spread", frame.columns)
            self.assertIn("one_sided_book_fraction", frame.columns)
            self.assertIn("pipeline_issue_seed_fraction", frame.columns)
            self.assertAlmostEqual(frame.loc[frame["phi"] == 0.3, "max_abs_inventory"].iloc[0], 42.0)

    def test_build_inventory_cap_comparison_writes_combined_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            no_cap_dir = tmp_root / "phi_sweep_nocap"
            cap50_dir = tmp_root / "phi_sweep_cap50"
            cap20_dir = tmp_root / "phi_sweep_cap20"
            output_dir = tmp_root / "inventory_cap_comparison"

            self._write_synthetic_experiment(no_cap_dir, inventory_cap=None, cap_label="no_cap")
            self._write_synthetic_experiment(cap50_dir, inventory_cap=50, cap_label="cap_50")
            self._write_synthetic_experiment(cap20_dir, inventory_cap=20, cap_label="cap_20")

            result = build_inventory_cap_comparison(
                {
                    "no_cap": no_cap_dir,
                    "cap_50": cap50_dir,
                    "cap_20": cap20_dir,
                },
                output_dir=output_dir,
            )

            comparison_frame = pd.read_csv(result["csv_path"])
            report = result["md_path"].read_text(encoding="utf-8")
            payload = json.loads(result["json_path"].read_text(encoding="utf-8"))

            self.assertEqual(len(comparison_frame), 12)
            self.assertTrue(
                {
                    "inventory_cap_label",
                    "phi",
                    "average_spread",
                    "average_depth",
                    "tail_exposure",
                    "one_sided_book_fraction",
                    "average_abs_ending_inventory",
                    "max_abs_inventory",
                    "pipeline_issue_seed_fraction",
                }.issubset(comparison_frame.columns)
            )
            self.assertIn("Does the deterioration around phi ≈ 0.30 still exist with cap 50", report)
            self.assertIn("Does one-sided-book fraction still rise with phi under caps", report)
            self.assertIn("robust", report)
            self.assertEqual(payload["evaluation_mode"], "greedy")

            expected_plots = {
                "volatility_vs_phi_by_inventory_cap.png",
                "spread_vs_phi_by_inventory_cap.png",
                "average_depth_vs_phi_by_inventory_cap.png",
                "tail_exposure_vs_phi_by_inventory_cap.png",
                "one_sided_book_fraction_vs_phi_by_inventory_cap.png",
                "undefined_midprice_fraction_vs_phi_by_inventory_cap.png",
                "max_consecutive_one_sided_duration_vs_phi_by_inventory_cap.png",
                "average_abs_inventory_vs_phi_by_inventory_cap.png",
            }
            saved_plot_names = {path.name for path in result["plot_paths"]}
            self.assertEqual(saved_plot_names, expected_plots)
            for path in result["plot_paths"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
