"""Tests for trained-vs-baseline comparison outputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from policy_baseline_comparison import build_policy_baseline_comparison


class PolicyBaselineComparisonTests(unittest.TestCase):
    @staticmethod
    def _write_synthetic_experiment(root: Path, *, policy_type: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        config = {
            "market_profile": "abides_rmsc04_small_v1",
            "phi_grid": [0.0, 0.2, 0.3, 0.5],
            "episodes": 50,
            "evaluation_seeds": [7, 8, 9],
            "evaluation_modes": ["greedy", "stochastic"],
            "policy_type": policy_type,
        }
        (root / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "phi_sweep_report.md").write_text(f"# {policy_type}\n", encoding="utf-8")

        if policy_type == "trained":
            spread = [0.010, 0.011, 0.016, 0.021]
            depth = [10.0, 9.0, 6.0, 5.0]
            one_sided = [0.00, 0.03, 0.12, 0.28]
            ask_bias = [0.00, 0.01, 0.03, 0.05]
            quote_activity = [0.80, 0.77, 0.68, 0.58]
            inactivity = [0.20, 0.23, 0.32, 0.42]
            breakdown = [-2.0, -0.8, 1.2, 2.6]
            transition_phi = 0.30
        else:
            spread = [0.010, 0.0105, 0.012, 0.014]
            depth = [10.0, 9.7, 8.7, 8.1]
            one_sided = [0.00, 0.01, 0.04, 0.08]
            ask_bias = [0.00, 0.00, 0.01, 0.02]
            quote_activity = [0.80, 0.79, 0.77, 0.75]
            inactivity = [0.20, 0.21, 0.23, 0.25]
            breakdown = [-2.0, -1.2, -0.2, 0.6]
            transition_phi = float("nan")

        summary = pd.DataFrame(
            {
                "phi": [0.0, 0.2, 0.3, 0.5],
                "greedy_average_spread_mean": spread,
                "greedy_average_depth_mean": depth,
                "greedy_one_sided_book_fraction_mean": one_sided,
                "greedy_undefined_midprice_fraction_mean": one_sided,
                "greedy_ask_side_failure_bias_mean": ask_bias,
                "greedy_mean_one_sided_episode_duration_mean": [0.0, 1.0, 3.0, 5.0],
                "greedy_p90_one_sided_episode_duration_mean": [0.0, 1.5, 4.5, 7.0],
                "greedy_quote_activity_fraction_mean": quote_activity,
                "greedy_inactivity_fraction_mean": inactivity,
                "greedy_market_breakdown_score": breakdown,
                "greedy_average_abs_ending_inventory_mean": [0.0, 1.0, 2.0, 3.0],
                "greedy_transition_phi_one_sided": [transition_phi] * 4,
                "greedy_transition_phi_undefined_midprice": [transition_phi] * 4,
                "greedy_transition_phi_depth_collapse": [0.3 if policy_type == 'trained' else float('nan')] * 4,
                "greedy_transition_phi_spread_widening": [0.3 if policy_type == 'trained' else 0.5] * 4,
                "stochastic_average_spread_mean": spread,
                "stochastic_average_depth_mean": depth,
                "stochastic_one_sided_book_fraction_mean": one_sided,
                "stochastic_undefined_midprice_fraction_mean": one_sided,
                "stochastic_ask_side_failure_bias_mean": ask_bias,
                "stochastic_mean_one_sided_episode_duration_mean": [0.0, 1.0, 3.0, 5.0],
                "stochastic_p90_one_sided_episode_duration_mean": [0.0, 1.5, 4.5, 7.0],
                "stochastic_quote_activity_fraction_mean": quote_activity,
                "stochastic_inactivity_fraction_mean": inactivity,
                "stochastic_market_breakdown_score": breakdown,
                "stochastic_average_abs_ending_inventory_mean": [0.0, 1.0, 2.0, 3.0],
                "stochastic_transition_phi_one_sided": [transition_phi] * 4,
                "stochastic_transition_phi_undefined_midprice": [transition_phi] * 4,
                "stochastic_transition_phi_depth_collapse": [0.3 if policy_type == 'trained' else float('nan')] * 4,
                "stochastic_transition_phi_spread_widening": [0.3 if policy_type == 'trained' else 0.5] * 4,
            }
        )
        summary.to_csv(root / "phi_sweep_summary.csv", index=False)

    def test_build_policy_baseline_comparison_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            trained_dir = tmp_root / "trained"
            baseline_dir = tmp_root / "baseline"
            output_dir = tmp_root / "policy_baseline_comparison"

            self._write_synthetic_experiment(trained_dir, policy_type="trained")
            self._write_synthetic_experiment(baseline_dir, policy_type="random_baseline")

            result = build_policy_baseline_comparison(
                {
                    "trained": trained_dir,
                    "random_baseline": baseline_dir,
                },
                output_dir=output_dir,
            )

            comparison_frame = pd.read_csv(result["csv_path"])
            report = result["md_path"].read_text(encoding="utf-8")
            payload = json.loads(result["json_path"].read_text(encoding="utf-8"))

            self.assertEqual(len(comparison_frame), 8)
            self.assertTrue(
                {
                    "policy_type",
                    "phi",
                    "average_spread",
                    "average_depth",
                    "one_sided_book_fraction",
                    "quote_activity_fraction",
                    "inactivity_fraction",
                    "market_breakdown_score",
                    "transition_phi_one_sided",
                }.issubset(comparison_frame.columns)
            )
            self.assertIn("Does the learned policy deteriorate more sharply than the random baseline", report)
            self.assertEqual(payload["evaluation_mode"], "greedy")

            expected_plots = {
                "average_spread_vs_phi_by_policy_type.png",
                "average_depth_vs_phi_by_policy_type.png",
                "one_sided_book_fraction_vs_phi_by_policy_type.png",
                "ask_side_failure_bias_vs_phi_by_policy_type.png",
                "quote_activity_fraction_vs_phi_by_policy_type.png",
                "inactivity_fraction_vs_phi_by_policy_type.png",
                "market_breakdown_score_vs_phi_by_policy_type.png",
                "p90_one_sided_episode_duration_vs_phi_by_policy_type.png",
            }
            saved_plot_names = {path.name for path in result["plot_paths"]}
            self.assertEqual(saved_plot_names, expected_plots)
            for path in result["plot_paths"]:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
