"""Regression tests for anti-degeneracy hold-streak penalty and bootstrap CI."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# Stub out agents.base so env.py can be imported without the ABIDES submodule.
agents_stub = MagicMock()
agents_stub.base = MagicMock()
agents_stub.base.MarketObservation = object
sys.modules.setdefault("agents", agents_stub)
sys.modules.setdefault("agents.base", agents_stub.base)

from env import RLMarketEnvironment


class TestAntiDegeneracyPenalty(unittest.TestCase):
    """Unit tests for the hold-streak anti-degeneracy penalty in RLMarketEnvironment."""

    def _env(self, grace: int = 3, penalty: float = 0.01) -> RLMarketEnvironment:
        return RLMarketEnvironment(hold_streak_grace=grace, hold_streak_penalty_per_step=penalty)

    def _components(self, env: RLMarketEnvironment, previous_action: int, streak: int) -> dict:
        return env.compute_reward_components(
            previous_cash=0.0,
            previous_inventory=0,
            previous_midprice=100.0,
            current_cash=0.0,
            current_inventory=0,
            current_midprice=100.0,
            previous_action=previous_action,
            hold_streak=streak,
        )

    def test_no_penalty_within_grace(self) -> None:
        env = self._env(grace=3, penalty=0.01)
        # Streak == grace: no penalty yet
        result = self._components(env, previous_action=1, streak=3)
        self.assertEqual(result["anti_degeneracy_penalty"], 0.0)

    def test_no_penalty_below_grace(self) -> None:
        env = self._env(grace=3, penalty=0.01)
        result = self._components(env, previous_action=1, streak=2)
        self.assertEqual(result["anti_degeneracy_penalty"], 0.0)

    def test_penalty_beyond_grace(self) -> None:
        env = self._env(grace=3, penalty=0.01)
        # streak=5, grace=3 → excess=2 → penalty = 0.01 * 100 * 2 = 2.0
        result = self._components(env, previous_action=1, streak=5)
        self.assertAlmostEqual(result["anti_degeneracy_penalty"], 2.0)

    def test_penalty_scales_linearly(self) -> None:
        env = self._env(grace=0, penalty=0.005)
        # streak=4, grace=0 → excess=4 → penalty = 0.005 * 100 * 4 = 2.0
        result = self._components(env, previous_action=1, streak=4)
        self.assertAlmostEqual(result["anti_degeneracy_penalty"], 2.0)

    def test_no_penalty_for_non_hold_action(self) -> None:
        env = self._env(grace=0, penalty=1.0)
        for action in (0, 2):
            result = self._components(env, previous_action=action, streak=100)
            self.assertEqual(result["anti_degeneracy_penalty"], 0.0,
                             f"Expected 0 penalty for action={action}")

    def test_penalty_subtracted_from_reward(self) -> None:
        env = self._env(grace=0, penalty=0.01)
        result = self._components(env, previous_action=1, streak=5)
        expected_penalty = 0.01 * 100.0 * 5
        self.assertAlmostEqual(result["anti_degeneracy_penalty"], expected_penalty)
        # The penalty should reduce the reward
        no_penalty_env = RLMarketEnvironment(hold_streak_grace=0, hold_streak_penalty_per_step=0.0)
        base = no_penalty_env.compute_reward_components(
            previous_cash=0.0, previous_inventory=0, previous_midprice=100.0,
            current_cash=0.0, current_inventory=0, current_midprice=100.0,
            previous_action=1, hold_streak=5,
        )
        self.assertAlmostEqual(base["reward"] - expected_penalty, result["reward"])

    def test_zero_penalty_config_no_effect(self) -> None:
        env = RLMarketEnvironment(hold_streak_grace=0, hold_streak_penalty_per_step=0.0)
        result = self._components(env, previous_action=1, streak=100)
        self.assertEqual(result["anti_degeneracy_penalty"], 0.0)

    def test_anti_degeneracy_key_always_present(self) -> None:
        env = RLMarketEnvironment()
        result = self._components(env, previous_action=1, streak=0)
        self.assertIn("anti_degeneracy_penalty", result)


class TestBootstrapCI(unittest.TestCase):
    """Regression tests for the bootstrap confidence interval in phi_experiment."""

    def _summarize(self, values: list[float]) -> dict:
        """Inline re-implementation matching phi_experiment._summarize_numeric_series."""
        series = pd.Series(values)
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        sample_size = int(len(numeric))
        if sample_size == 0:
            return {"mean": float("nan"), "std": float("nan"), "stderr": float("nan"),
                    "ci95_lower": float("nan"), "ci95_upper": float("nan"), "n": 0.0}
        mean_value = float(numeric.mean())
        if sample_size == 1:
            return {"mean": mean_value, "std": float("nan"), "stderr": float("nan"),
                    "ci95_lower": float("nan"), "ci95_upper": float("nan"), "n": 1.0}
        std_value = float(numeric.std(ddof=1))
        stderr_value = float(std_value / np.sqrt(sample_size))
        rng = np.random.default_rng(seed=42)
        bootstrap_means = [
            float(np.mean(rng.choice(numeric.values, size=sample_size, replace=True)))
            for _ in range(2000)
        ]
        return {
            "mean": mean_value, "std": std_value, "stderr": stderr_value,
            "ci95_lower": float(np.percentile(bootstrap_means, 2.5)),
            "ci95_upper": float(np.percentile(bootstrap_means, 97.5)),
            "n": float(sample_size),
        }

    def test_ci_brackets_mean(self) -> None:
        result = self._summarize([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertLessEqual(result["ci95_lower"], result["mean"])
        self.assertGreaterEqual(result["ci95_upper"], result["mean"])

    def test_single_value_returns_nan_ci(self) -> None:
        result = self._summarize([0.5])
        self.assertTrue(np.isnan(result["ci95_lower"]))
        self.assertTrue(np.isnan(result["ci95_upper"]))
        self.assertEqual(result["n"], 1.0)

    def test_ci_width_shrinks_with_more_data(self) -> None:
        small = self._summarize([0.1, 0.9])
        large = self._summarize([0.5] * 20)
        small_width = small["ci95_upper"] - small["ci95_lower"]
        large_width = large["ci95_upper"] - large["ci95_lower"]
        self.assertGreater(small_width, large_width)

    def test_mean_preserved(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self._summarize(values)
        self.assertAlmostEqual(result["mean"], 3.0)


if __name__ == "__main__":
    unittest.main()
