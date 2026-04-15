"""Tests for the shared-policy RL-agent integration."""

from __future__ import annotations

import unittest

from baseline_configs import build_abides_rmsc04_small_v1_config
from env import InventoryAwarePolicy, build_policy
from market import MarketSimulator


class RLIntegrationTests(unittest.TestCase):
    """Smoke tests for RL insertion into the frozen baseline profile."""

    def test_inventory_aware_policy_is_available(self) -> None:
        policy = build_policy("inventory_aware")
        self.assertIsInstance(policy, InventoryAwarePolicy)
        self.assertEqual(policy.act(state=[0.0, 0.0, 2.0], rng=None), 0)
        self.assertEqual(policy.act(state=[0.0, 0.0, -2.0], rng=None), 2)
        self.assertEqual(policy.act(state=[0.0, 0.0, 0.0], rng=None), 1)

    def test_market_simulator_accepts_custom_rl_policy_factory(self) -> None:
        config = build_abides_rmsc04_small_v1_config(phi=0.10, end_time="09:30:05")
        simulator = MarketSimulator(
            config,
            rl_policy_factory=lambda agent_id, rng: InventoryAwarePolicy(float(agent_id % 2 + 1)),
        )

        rl_agents = [
            agent
            for agent in simulator.config_state.kernel_config["agents"]
            if getattr(agent, "type", "") == "RLTrader"
        ]

        self.assertGreater(len(rl_agents), 0)
        self.assertTrue(all(isinstance(agent.policy, InventoryAwarePolicy) for agent in rl_agents))

    def test_mixed_liquidity_mode_builds_takers_and_quoters(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.10,
            end_time="09:30:05",
            rl_liquidity_mode="mixed",
            rl_quoter_split=0.5,
        )
        simulator = MarketSimulator(config)

        agent_types = [
            getattr(agent, "type", "")
            for agent in simulator.config_state.kernel_config["agents"]
            if getattr(agent, "type", "") in {"RLTrader", "RLQuotingTrader"}
        ]

        self.assertIn("RLTrader", agent_types)
        self.assertIn("RLQuotingTrader", agent_types)

    def test_rl_decision_log_contains_state_action_reward_fields(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.10,
            seed=23,
            end_time="09:30:15",
            log_frequency="1s",
            latency_type="no_latency",
        )
        simulator = MarketSimulator(config)
        frame = simulator.run()
        rl_frame = simulator.extract_rl_frame()

        self.assertGreater(len(frame), 0)
        self.assertGreater(len(rl_frame), 0)
        required_columns = {
            "time",
            "time_ns",
            "agent_id",
            "agent_name",
            "decision_index",
            "action",
            "action_label",
            "previous_action",
            "inventory",
            "cash",
            "wealth",
            "reward",
            "spread",
            "imbalance",
            "midprice",
            "executed_since_last_decision",
            "filled_quantity_since_last_decision",
            "return_00",
        }
        self.assertTrue(required_columns.issubset(rl_frame.columns))
        self.assertTrue((rl_frame["action"].isin([0, 1, 2])).all())

        prefixed_columns = {
            column
            for column in frame.columns
            if column.endswith("_inventory")
            or column.endswith("_cash")
            or column.endswith("_wealth")
            or column.endswith("_reward")
            or column.endswith("_action")
        }
        self.assertTrue(prefixed_columns)

    def test_rl_agents_do_not_all_act_in_lockstep(self) -> None:
        config = build_abides_rmsc04_small_v1_config(
            phi=0.10,
            seed=29,
            end_time="09:30:20",
            log_frequency="1s",
            latency_type="no_latency",
        )
        simulator = MarketSimulator(config)
        simulator.run()
        rl_frame = simulator.extract_rl_frame()

        rl_count = config.agent_counts()["rl"]
        decisions_per_time = rl_frame.groupby("time_ns")["agent_id"].nunique()

        self.assertGreater(rl_count, 0)
        self.assertGreater(rl_frame["time_ns"].nunique(), 1)
        self.assertLess(int(decisions_per_time.max()), rl_count)


if __name__ == "__main__":
    unittest.main()
