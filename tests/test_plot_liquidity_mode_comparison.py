"""Tests for the liquidity-mode comparison plotter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import plot_liquidity_mode_comparison


class PlotLiquidityModeComparisonTests(unittest.TestCase):
    def test_main_builds_plot_with_ci_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            taker_dir = root / "taker"
            mixed_dir = root / "mixed"
            output_path = root / "one_sided_book_fraction_by_liquidity_mode.png"
            taker_dir.mkdir()
            mixed_dir.mkdir()

            taker_summary = pd.DataFrame(
                {
                    "phi": [0.0, 0.1],
                    "greedy_one_sided_book_fraction_mean": [0.02, 0.03],
                    "greedy_one_sided_book_fraction_ci95_lower": [0.01, 0.02],
                    "greedy_one_sided_book_fraction_ci95_upper": [0.03, 0.04],
                    "greedy_one_sided_book_fraction_n": [3.0, 3.0],
                }
            )
            mixed_summary = pd.DataFrame(
                {
                    "phi": [0.0, 0.1],
                    "greedy_one_sided_book_fraction_mean": [0.01, 0.015],
                    "greedy_one_sided_book_fraction_ci95_lower": [0.005, 0.010],
                    "greedy_one_sided_book_fraction_ci95_upper": [0.015, 0.020],
                    "greedy_one_sided_book_fraction_n": [3.0, 3.0],
                }
            )
            taker_summary.to_csv(taker_dir / "phi_sweep_summary.csv", index=False)
            mixed_summary.to_csv(mixed_dir / "phi_sweep_summary.csv", index=False)

            with mock.patch(
                "sys.argv",
                [
                    "plot_liquidity_mode_comparison.py",
                    "--taker-folder",
                    str(taker_dir),
                    "--mixed-folder",
                    str(mixed_dir),
                    "--output",
                    str(output_path),
                ],
            ):
                plot_liquidity_mode_comparison.main()

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
