"""Tests for baseline-vs-anti-degeneracy behavior comparison."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from behavior_mode_comparison import build_behavior_mode_comparison


def _write_experiment(root: Path, *, anti_degeneracy_mode: str, hold_streak_penalty: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_config.json").write_text(
        json.dumps(
            {
                "policy_type": "trained",
                "anti_degeneracy_mode": anti_degeneracy_mode,
                "hold_streak_penalty": hold_streak_penalty,
                "hold_streak_grace": 2,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "phi": [0.0, 0.2, 0.3],
            "stochastic_quote_activity_fraction_mean": [0.0, 0.10, 0.12] if anti_degeneracy_mode == "off" else [0.0, 0.30, 0.35],
            "stochastic_inactivity_fraction_mean": [1.0, 0.90, 0.88] if anti_degeneracy_mode == "off" else [1.0, 0.70, 0.65],
            "stochastic_one_sided_book_fraction_mean": [0.0, 0.18, 0.22] if anti_degeneracy_mode == "off" else [0.0, 0.12, 0.16],
            "stochastic_ask_side_failure_bias_mean": [0.0, 0.05, 0.08] if anti_degeneracy_mode == "off" else [0.0, 0.04, 0.06],
            "stochastic_p90_one_sided_episode_duration_mean": [0.0, 8.0, 10.0] if anti_degeneracy_mode == "off" else [0.0, 6.0, 8.0],
            "stochastic_mean_hold_streak_mean": [1.0, 9.0, 10.0] if anti_degeneracy_mode == "off" else [1.0, 4.0, 5.0],
            "stochastic_max_hold_streak_mean": [1.0, 18.0, 20.0] if anti_degeneracy_mode == "off" else [1.0, 8.0, 9.0],
            "stochastic_fraction_of_agents_with_near_total_inactivity_mean": [1.0, 0.80, 0.85] if anti_degeneracy_mode == "off" else [1.0, 0.30, 0.35],
            "stochastic_market_breakdown_score": [-2.0, 1.5, 2.5] if anti_degeneracy_mode == "off" else [-2.0, 1.0, 1.8],
            "stochastic_anti_degeneracy_penalty_mean_mean": [0.0, 0.0, 0.0] if anti_degeneracy_mode == "off" else [0.0, 0.02, 0.03],
            "stochastic_average_abs_ending_inventory_mean": [0.0, 0.9, 1.2] if anti_degeneracy_mode == "off" else [0.0, 0.8, 1.0],
            "stochastic_evaluation_inventory_at_cap_fraction_mean": [0.0, 0.20, 0.22] if anti_degeneracy_mode == "off" else [0.0, 0.16, 0.18],
            "stochastic_transition_phi_one_sided": [0.2, 0.2, 0.2],
            "stochastic_fraction_of_decisions_in_hold_streak_ge_3_mean": [0.0, 0.70, 0.75] if anti_degeneracy_mode == "off" else [0.0, 0.25, 0.30],
            "stochastic_fraction_of_decisions_in_hold_streak_ge_5_mean": [0.0, 0.55, 0.60] if anti_degeneracy_mode == "off" else [0.0, 0.10, 0.12],
        }
    ).to_csv(root / "phi_sweep_summary.csv", index=False)
    (root / "phi_sweep_report.md").write_text("# placeholder\n", encoding="utf-8")


class BehaviorModeComparisonTests(unittest.TestCase):
    def test_build_behavior_mode_comparison_writes_expected_plots_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            baseline_dir = tmp_root / "baseline"
            anti_dir = tmp_root / "anti"
            output_dir = tmp_root / "comparison"
            _write_experiment(baseline_dir, anti_degeneracy_mode="off", hold_streak_penalty=0.0)
            _write_experiment(anti_dir, anti_degeneracy_mode="hold_streak_penalty", hold_streak_penalty=0.01)

            result = build_behavior_mode_comparison(
                {
                    "baseline": baseline_dir,
                    "anti_degeneracy": anti_dir,
                },
                output_dir=output_dir,
                evaluation_mode="stochastic",
            )

            report = Path(result["md_path"]).read_text(encoding="utf-8")
            plot_names = {path.name for path in result["plot_paths"]}

            self.assertIn("survives anti-degeneracy intervention", report)
            self.assertIn("inactivity_fraction_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("quote_activity_fraction_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("one_sided_book_fraction_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("ask_side_failure_bias_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("p90_one_sided_episode_duration_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("mean_hold_streak_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("max_hold_streak_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("fraction_agents_near_total_inactivity_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("market_breakdown_score_vs_phi_by_behavior_mode.png", plot_names)
            self.assertIn("anti_degeneracy_penalty_mean_vs_phi.png", plot_names)


if __name__ == "__main__":
    unittest.main()
