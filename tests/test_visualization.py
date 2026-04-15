"""Tests for market log visualization helpers."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import pandas as pd

from visualization import build_market_animation, create_market_figure, rl_pnl_series, save_market_plot


class VisualizationTests(unittest.TestCase):
    """Visualization should save a figure from a simple market frame."""

    @staticmethod
    def sample_frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": [0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                "midprice": [100.0, 100.1, 100.05, 100.2],
                "fundamental_value": [100.0, 100.02, 100.04, 100.06],
                "spread": [0.02, 0.02, 0.03, 0.02],
                "bid_depth": [5.0, 6.0, 7.0, 6.0],
                "ask_depth": [4.0, 5.0, 5.0, 6.0],
                "traded_volume": [1.0, 2.0, 1.0, 3.0],
                "signed_order_flow": [1.0, -1.0, 0.0, 2.0],
                "rl_trader_0_inventory": [0.0, 1.0, 1.0, 0.0],
                "rl_trader_0_wealth": [0.0, 0.1, 0.05, 0.2],
                "rl_trader_1_inventory": [0.0, -1.0, -2.0, -1.0],
                "rl_trader_1_wealth": [0.0, -0.05, -0.02, 0.01],
            }
        )

    def test_save_market_plot_creates_image(self) -> None:
        frame = self.sample_frame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "market_replay.png"
            saved = save_market_plot(frame, output_path, title="Test Replay")
            self.assertEqual(saved, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_build_market_animation_creates_animation(self) -> None:
        frame = self.sample_frame()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Animation was deleted without rendering anything.*",
                category=UserWarning,
            )
            fig, animation = build_market_animation(
                frame,
                title="Animated Replay",
                interval_ms=50,
                tail_seconds=2.0,
                show_plot=False,
            )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(animation)
        self.assertEqual(animation._save_count, len(frame))
        animation._draw_was_started = True

    def test_rl_pnl_series_uses_first_valid_wealth_as_baseline(self) -> None:
        frame = self.sample_frame().copy()
        frame["rl_trader_0_wealth"] = [0.0, 100.1, 100.05, 100.2]
        pnl = rl_pnl_series(frame, "rl_trader_0_wealth")
        self.assertTrue(pd.isna(pnl.iloc[0]))
        self.assertEqual(list(pnl.iloc[1:].round(6)), [0.0, -0.05, 0.1])

    def test_rl_panels_do_not_create_per_agent_legend_spam(self) -> None:
        frame = self.sample_frame()
        fig = create_market_figure(frame, title="Legend Test", show_plot=False)
        inventory_ax = fig.axes[4]
        wealth_ax = fig.axes[5]
        self.assertIsNone(inventory_ax.get_legend())
        self.assertIsNone(wealth_ax.get_legend())

    def test_evaluation_cli_animates_before_saving_plot(self) -> None:
        import evaluate_trained_policy

        frame = self.sample_frame()
        call_order: list[str] = []

        def fake_evaluate_policy(*args, **kwargs):
            callback = kwargs["episode_callback"]
            callback("evaluation", 0, 7, frame, pd.DataFrame(), pd.DataFrame())
            return pd.DataFrame([{"episode": 0.0, "buy_fraction": 0.0, "hold_fraction": 1.0, "sell_fraction": 0.0}]), {}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "evaluate_trained_policy.py",
                "--policy",
                "random",
                "--seeds",
                "7",
                "--output",
                str(Path(tmpdir) / "eval.csv"),
                "--plot-output-dir",
                str(Path(tmpdir) / "plots"),
                "--animate-market",
                "--visualize-seed",
                "7",
            ]
            with mock.patch("sys.argv", argv), \
                mock.patch.object(evaluate_trained_policy, "evaluate_policy", side_effect=fake_evaluate_policy), \
                mock.patch.object(evaluate_trained_policy, "show_market_animation", side_effect=lambda *a, **k: call_order.append("animate")), \
                mock.patch.object(evaluate_trained_policy, "save_market_plot", side_effect=lambda *a, **k: call_order.append("save")):
                evaluate_trained_policy.main()

        self.assertEqual(call_order, ["animate", "save"])

    def test_training_cli_animates_before_saving_plot(self) -> None:
        import train_shared_ppo

        frame = self.sample_frame()
        training_frame = pd.DataFrame(
            [
                {
                    "episode": 0.0,
                    "total_training_reward": -1.0,
                    "average_reward_per_rl_agent": -0.1,
                    "average_abs_inventory": 0.0,
                    "average_abs_ending_inventory": 0.0,
                    "max_abs_inventory": 0.0,
                    "buy_fraction": 0.0,
                    "hold_fraction": 1.0,
                    "sell_fraction": 0.0,
                    "collapse_state": "hold_only",
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "reward_mean": 0.0,
                    "reward_std": 0.0,
                    "wealth_delta_mean": 0.0,
                    "inventory_penalty_mean": 0.0,
                    "flat_hold_penalty_mean": 0.0,
                    "num_transitions": 0.0,
                }
            ]
        )
        call_order: list[str] = []

        class DummyPolicy:
            def save(self, path):
                return Path(path)

        def fake_train_shared_policy(*args, **kwargs):
            callback = kwargs["episode_callback"]
            callback("train", 0, 7, frame, pd.DataFrame(), pd.DataFrame())
            return DummyPolicy(), training_frame, pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "train_shared_ppo.py",
                "--episodes",
                "1",
                "--training-log-output",
                str(Path(tmpdir) / "train.csv"),
                "--checkpoint-output",
                str(Path(tmpdir) / "policy.npz"),
                "--plot-output-dir",
                str(Path(tmpdir) / "plots"),
                "--animate-market",
                "--visualize-episode",
                "0",
            ]
            with mock.patch("sys.argv", argv), \
                mock.patch.object(train_shared_ppo, "train_shared_policy", side_effect=fake_train_shared_policy), \
                mock.patch.object(train_shared_ppo, "show_market_animation", side_effect=lambda *a, **k: call_order.append("animate")), \
                mock.patch.object(train_shared_ppo, "save_market_plot", side_effect=lambda *a, **k: call_order.append("save")), \
                mock.patch.object(train_shared_ppo, "save_training_progress_plots", return_value=[]):
                train_shared_ppo.main()

        self.assertEqual(call_order[:2], ["animate", "save"])


if __name__ == "__main__":
    unittest.main()
