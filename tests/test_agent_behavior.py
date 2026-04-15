"""Tests for targeted baseline-agent behavior adjustments."""

from __future__ import annotations

from collections import deque
import unittest

import numpy as np

from abides_agents import (
    ABIDESObservation,
    AdaptiveMarketMaker,
    NoiseTrader,
    RLQuotingTrader,
    RLTrader,
    TrendFollowerTrader,
    ValueTrader,
    ZICTrader,
)
from abides_markets.orders import LimitOrder, Side
from env import RLMarketEnvironment


class StubRandomState:
    def __init__(self, rand_values, randint_values=None, choice_values=None):
        self._rand_values = list(rand_values)
        self._randint_values = list(randint_values or [])
        self._choice_values = list(choice_values or [])

    def rand(self):
        return float(self._rand_values.pop(0))

    def randint(self, low, high=None):
        del low, high
        return int(self._randint_values.pop(0))

    def choice(self, values):
        del values
        return self._choice_values.pop(0)

    def normal(self, loc=0.0, scale=1.0):
        del loc, scale
        return 0.0


class FixedActionPolicy:
    def __init__(self, action: int):
        self.action = int(action)

    def sample_action(self, state, rng):
        del state, rng
        return {"action": self.action, "log_prob": -0.5, "value": 0.0}


class AgentBehaviorTests(unittest.TestCase):
    """Small unit tests for calibration-oriented agent behavior changes."""

    def test_trend_signal_uses_history_and_noise_hook(self) -> None:
        trader = TrendFollowerTrader(
            id=1,
            name="trend",
            type="TrendFollowerTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=1,
            short_window=2,
            long_window=4,
            signal_noise=0.0,
            signal_threshold=1.0,
            trade_probability=1.0,
            aggressive_probability=0.0,
            aggressive_order_size=1,
            passive_order_size=2,
            random_state=np.random.RandomState(7),
            log_orders=False,
        )
        trader.midprice_history = deque([100.0, 101.0, 102.0, 104.0], maxlen=256)
        self.assertAlmostEqual(trader._trend_signal(), 1.25)

    def test_market_maker_only_refreshes_when_quote_shift_is_large_enough(self) -> None:
        market_maker = AdaptiveMarketMaker(
            id=2,
            name="mm",
            type="AdaptiveMarketMaker",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=1,
            target_spread=2,
            alpha_cents=0.2,
            quote_size=5,
            reprice_threshold=2,
            min_refresh_interval=1,
            random_state=np.random.RandomState(11),
            log_orders=False,
        )
        bid_order = LimitOrder(
            agent_id=market_maker.id,
            time_placed=0,
            symbol="ABM",
            quantity=5,
            side=Side.BID,
            limit_price=100,
        )
        ask_order = LimitOrder(
            agent_id=market_maker.id,
            time_placed=0,
            symbol="ABM",
            quantity=5,
            side=Side.ASK,
            limit_price=102,
        )
        market_maker.orders = {1: bid_order, 2: ask_order}
        market_maker.last_quoted_bid = 100
        market_maker.last_quoted_ask = 102
        market_maker.last_refresh_time = 0

        self.assertFalse(
            market_maker._needs_refresh(
                side=Side.BID,
                desired_price=101,
                last_quoted_price=100,
            )
        )
        self.assertTrue(
            market_maker._needs_refresh(
                side=Side.BID,
                desired_price=102,
                last_quoted_price=100,
            )
        )

    def test_market_makers_sample_diverse_quote_biases(self) -> None:
        sampled_biases = set()
        for seed in range(5, 11):
            market_maker = AdaptiveMarketMaker(
                id=seed,
                name=f"mm_{seed}",
                type="AdaptiveMarketMaker",
                symbol="ABM",
                starting_cash=1_000_000,
                wake_up_freq=10,
            target_spread=3,
            alpha_cents=0.2,
            quote_size=6,
            reprice_threshold=2,
            min_refresh_interval=1,
                random_state=np.random.RandomState(seed),
                log_orders=False,
            )
            sampled_biases.add(
                (market_maker.agent_spread_offset, market_maker.agent_center_bias)
            )

        self.assertGreater(len(sampled_biases), 1)

    def test_market_maker_joins_best_when_step_ahead_is_only_one_tick(self) -> None:
        market_maker = AdaptiveMarketMaker(
            id=12,
            name="mm_soft_join",
            type="AdaptiveMarketMaker",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            target_spread=3,
            alpha_cents=0.0,
            quote_size=6,
            reprice_threshold=2,
            min_refresh_interval=1,
            random_state=np.random.RandomState(31),
            log_orders=False,
        )
        softened_bid, softened_ask = market_maker._soften_inside_step_ahead(
            ABIDESObservation(
                time=0,
                best_bid=100,
                best_ask=102,
                bid_depth=5,
                ask_depth=5,
                midprice=101.0,
                spread=2.0,
                imbalance=0.0,
            ),
            bid_price=101,
            ask_price=101,
        )

        self.assertEqual(softened_bid, 100)
        self.assertEqual(softened_ask, 102)

    def test_trend_passive_orders_use_larger_quantity(self) -> None:
        trader = TrendFollowerTrader(
            id=7,
            name="trend_passive_size",
            type="TrendFollowerTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            short_window=2,
            long_window=4,
            signal_noise=0.0,
            signal_threshold=1.0,
            trade_probability=1.0,
            aggressive_probability=0.0,
            aggressive_order_size=1,
            passive_order_size=2,
            random_state=np.random.RandomState(13),
            log_orders=False,
        )
        trader.midprice_history = deque([100.0, 101.0, 102.0, 104.0], maxlen=256)

        observation = ABIDESObservation(
            time=10,
            best_bid=100,
            best_ask=102,
            bid_depth=6,
            ask_depth=6,
            midprice=101,
            spread=2.0,
            imbalance=0.0,
        )

        placed = {}

        def record_near_touch(*args, **kwargs):
            placed.update(kwargs)

        trader.place_near_touch_limit = record_near_touch
        trader.on_observation(observation)

        self.assertEqual(placed["side"], Side.BID)
        self.assertEqual(placed["quantity"], 2)

    def test_value_passive_orders_use_minimum_competitive_size(self) -> None:
        trader = ValueTrader(
            id=8,
            name="value_passive_size",
            type="ValueTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            sigma_eta=0.0,
            delta=2,
            aggressive_probability=0.0,
            aggressive_order_size=1,
            passive_order_size=2,
            random_state=np.random.RandomState(17),
            log_orders=False,
        )
        trader.observe_private_value = lambda sigma_n: 104

        observation = ABIDESObservation(
            time=10,
            best_bid=100,
            best_ask=102,
            bid_depth=6,
            ask_depth=6,
            midprice=101,
            spread=2.0,
            imbalance=0.0,
        )

        placed = {}

        def record_near_touch(*args, **kwargs):
            placed.update(kwargs)

        trader.place_near_touch_limit = record_near_touch
        trader.on_observation(observation)

        self.assertEqual(placed["side"], Side.BID)
        self.assertEqual(placed["quantity"], 2)

    def test_rl_trader_blocks_buy_at_inventory_cap_and_logs_it(self) -> None:
        trader = RLTrader(
            id=21,
            name="rl_cap",
            type="RLTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            environment=RLMarketEnvironment(return_window=2, lambda_q=0.01),
            order_size=1,
            inventory_cap=2,
            policy=FixedActionPolicy(2),
            random_state=np.random.RandomState(5),
            log_orders=False,
        )
        trader._build_state = lambda observation: np.zeros(5, dtype=float)
        trader.inventory = lambda: 2
        trader.cash_position = lambda: 1_000_000.0
        trader.wealth = lambda midprice: 1_000_000.0
        submitted_orders = []
        trader.place_market_order = lambda symbol, quantity, side: submitted_orders.append(
            {"symbol": symbol, "quantity": quantity, "side": side}
        )

        trader.on_observation(
            ABIDESObservation(
                time=10,
                best_bid=100,
                best_ask=102,
                bid_depth=6,
                ask_depth=6,
                midprice=101.0,
                spread=2.0,
                imbalance=0.0,
            )
        )

        self.assertEqual(submitted_orders, [])
        self.assertEqual(trader.last_effective_action, 1)
        self.assertTrue(trader.metrics_log[-1]["blocked_by_inventory_cap"])
        self.assertTrue(trader.metrics_log[-1]["blocked_buy_due_to_cap"])
        self.assertEqual(trader.metrics_log[-1]["effective_action"], 1)
        self.assertEqual(trader.metrics_log[-1]["action"], 2)

    def test_rl_quoter_places_both_sided_passive_quotes(self) -> None:
        trader = RLQuotingTrader(
            id=31,
            name="rl_quoter",
            type="RLQuotingTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            environment=RLMarketEnvironment(return_window=2, lambda_q=0.01),
            quote_size=1,
            inventory_cap=2,
            policy=FixedActionPolicy(3),
            random_state=np.random.RandomState(7),
            log_orders=False,
        )
        trader._build_state = lambda observation: np.zeros(5, dtype=float)
        trader.inventory = lambda: 0
        trader.cash_position = lambda: 1_000_000.0
        trader.wealth = lambda midprice: 1_000_000.0
        trader._has_same_or_better_passive_order = lambda side, price: False
        trader._cancel_side_orders = lambda side: None
        placed_orders = []

        def submit_passive_limit_order(*, side, quantity, limit_price, tag):
            placed_orders.append(
                {"side": side, "quantity": quantity, "limit_price": limit_price, "tag": tag}
            )
            if side.is_bid():
                trader.total_passive_bid_orders_submitted += 1
            else:
                trader.total_passive_ask_orders_submitted += 1
            return True

        trader._submit_passive_limit_order = submit_passive_limit_order

        trader.on_observation(
            ABIDESObservation(
                time=10,
                best_bid=100,
                best_ask=102,
                bid_depth=6,
                ask_depth=6,
                midprice=101.0,
                spread=2.0,
                imbalance=0.0,
            )
        )

        self.assertEqual(len(placed_orders), 2)
        self.assertEqual({order["side"] for order in placed_orders}, {Side.BID, Side.ASK})
        self.assertEqual({order["limit_price"] for order in placed_orders}, {100, 102})
        self.assertEqual(trader.metrics_log[-1]["effective_action"], 3)
        self.assertEqual(trader.metrics_log[-1]["submitted_passive_order_count"], 2.0)

    def test_rl_quoter_degrades_both_sided_quote_at_long_cap(self) -> None:
        trader = RLQuotingTrader(
            id=32,
            name="rl_quoter_cap",
            type="RLQuotingTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            environment=RLMarketEnvironment(return_window=2, lambda_q=0.01),
            quote_size=1,
            inventory_cap=2,
            policy=FixedActionPolicy(3),
            random_state=np.random.RandomState(11),
            log_orders=False,
        )
        trader._build_state = lambda observation: np.zeros(5, dtype=float)
        trader.inventory = lambda: 2
        trader.cash_position = lambda: 1_000_000.0
        trader.wealth = lambda midprice: 1_000_000.0
        trader._has_same_or_better_passive_order = lambda side, price: False
        cancelled_sides = []
        trader._cancel_side_orders = lambda side: cancelled_sides.append(side)
        placed_orders = []

        def submit_passive_limit_order(*, side, quantity, limit_price, tag):
            placed_orders.append({"side": side, "limit_price": limit_price})
            if side.is_bid():
                trader.total_passive_bid_orders_submitted += 1
            else:
                trader.total_passive_ask_orders_submitted += 1
            return True

        trader._submit_passive_limit_order = submit_passive_limit_order

        trader.on_observation(
            ABIDESObservation(
                time=10,
                best_bid=100,
                best_ask=102,
                bid_depth=6,
                ask_depth=6,
                midprice=101.0,
                spread=2.0,
                imbalance=0.0,
            )
        )

        self.assertEqual([order["side"] for order in placed_orders], [Side.ASK])
        self.assertIn(Side.BID, cancelled_sides)
        self.assertEqual(trader.metrics_log[-1]["effective_action"], 2)
        self.assertTrue(trader.metrics_log[-1]["blocked_buy_due_to_cap"])
        self.assertEqual(trader.metrics_log[-1]["submitted_passive_order_count"], 1.0)

    def test_zic_can_switch_deep_passive_orders_to_near_touch_competition(self) -> None:
        trader = ZICTrader(
            id=9,
            name="zic_inside_competition",
            type="ZICTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            sigma_eta=0.0,
            surplus_min_ticks=1,
            surplus_max_ticks=1,
            inside_competition_probability=1.0,
            random_state=np.random.RandomState(23),
            log_orders=False,
        )
        trader.observe_private_value = lambda sigma_n: 100
        trader.random_state = np.random.RandomState(1)

        observation = ABIDESObservation(
            time=10,
            best_bid=100,
            best_ask=102,
            bid_depth=6,
            ask_depth=6,
            midprice=101,
            spread=2.0,
            imbalance=0.0,
        )

        placed = {}

        def record_near_touch(*args, **kwargs):
            placed.update(kwargs)

        trader.place_near_touch_limit = record_near_touch
        trader.place_limit_order = lambda *args, **kwargs: self.fail("expected near-touch competition")
        trader.on_observation(observation)

        self.assertEqual(placed["side"], Side.BID)
        self.assertEqual(placed["offset_ticks"], 1)
        self.assertEqual(placed["tag"], "zic_near_touch")

    def test_market_maker_wake_frequency_supports_jitter(self) -> None:
        market_maker = AdaptiveMarketMaker(
            id=5,
            name="mm_jitter",
            type="AdaptiveMarketMaker",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            wake_up_jitter=3,
            min_refresh_interval=1,
            target_spread=3,
            alpha_cents=0.2,
            quote_size=6,
            reprice_threshold=2,
            random_state=np.random.RandomState(13),
            log_orders=False,
        )

        frequencies = {market_maker.get_wake_frequency() for _ in range(8)}
        self.assertGreater(len(frequencies), 1)
        self.assertTrue(all(value >= 1 for value in frequencies))

    def test_near_touch_limit_is_passive_not_crossing(self) -> None:
        trader = NoiseTrader(
            id=6,
            name="noise",
            type="NoiseTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            limit_order_probability=1.0,
            random_state=np.random.RandomState(21),
            log_orders=False,
        )
        placed = {}

        def record_limit_order(symbol, quantity, side, limit_price, **kwargs):
            placed["symbol"] = symbol
            placed["quantity"] = quantity
            placed["side"] = side
            placed["limit_price"] = limit_price

        trader.place_limit_order = record_limit_order
        observation = ABIDESObservation(
            time=0,
            best_bid=100,
            best_ask=102,
            bid_depth=5,
            ask_depth=5,
            midprice=101,
            spread=2.0,
            imbalance=0.0,
        )

        trader.place_near_touch_limit(observation, side=Side.BID, quantity=1, offset_ticks=1)
        self.assertEqual(placed["limit_price"], 101)

        trader.place_near_touch_limit(observation, side=Side.ASK, quantity=1, offset_ticks=1)
        self.assertEqual(placed["limit_price"], 101)

    def test_noise_passive_orders_can_rest_behind_inside_in_tight_spread(self) -> None:
        trader = NoiseTrader(
            id=14,
            name="noise_tight",
            type="NoiseTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            limit_order_probability=1.0,
            random_state=np.random.RandomState(1),
            log_orders=False,
        )
        trader.random_state = StubRandomState(rand_values=[0.2, 0.2, 0.2], randint_values=[1])
        observation = ABIDESObservation(
            time=0,
            best_bid=100,
            best_ask=101,
            bid_depth=5,
            ask_depth=5,
            midprice=100.5,
            spread=1.0,
            imbalance=0.0,
        )
        placed = {}

        def record_near_touch(*args, **kwargs):
            placed.update(kwargs)

        trader.place_near_touch_limit = record_near_touch
        trader.on_observation(observation)

        self.assertEqual(placed["offset_ticks"], -1)

    def test_market_maker_can_widen_under_high_imbalance(self) -> None:
        market_maker = AdaptiveMarketMaker(
            id=15,
            name="mm_widen",
            type="AdaptiveMarketMaker",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            target_spread=2,
            alpha_cents=0.0,
            quote_size=5,
            reprice_threshold=2,
            min_refresh_interval=1,
            random_state=np.random.RandomState(17),
            log_orders=False,
        )
        market_maker.agent_spread_offset = 0
        market_maker.agent_center_bias = 0
        market_maker.random_state = StubRandomState(rand_values=[0.2], choice_values=[0])
        market_maker.holdings["ABM"] = 0

        bid_price, ask_price = market_maker._desired_quotes(
            ABIDESObservation(
                time=0,
                best_bid=99,
                best_ask=101,
                bid_depth=20,
                ask_depth=2,
                midprice=100,
                spread=2.0,
                imbalance=0.8181818181818182,
            )
        )

        self.assertGreaterEqual(ask_price - bid_price, 3)

    def test_non_mm_passive_order_maintenance_cancels_stale_and_excess_quotes(self) -> None:
        trader = TrendFollowerTrader(
            id=10,
            name="trend_maintenance",
            type="TrendFollowerTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            short_window=2,
            long_window=4,
            signal_noise=0.0,
            signal_threshold=1.0,
            trade_probability=1.0,
            aggressive_probability=0.0,
            aggressive_order_size=1,
            passive_order_size=2,
            random_state=np.random.RandomState(29),
            log_orders=False,
        )
        trader.current_time = 10_000_000_000
        stale_bid = LimitOrder(
            agent_id=trader.id,
            time_placed=0,
            symbol="ABM",
            quantity=2,
            side=Side.BID,
            limit_price=100,
        )
        fresh_bid = LimitOrder(
            agent_id=trader.id,
            time_placed=0,
            symbol="ABM",
            quantity=2,
            side=Side.BID,
            limit_price=101,
        )
        trader.orders = {1: stale_bid, 2: fresh_bid}
        trader.limit_order_lifecycle = {
            stale_bid.order_id: {
                "is_passive": True,
                "accepted_time_ns": 0,
                "time_submitted_ns": 0,
            },
            fresh_bid.order_id: {
                "is_passive": True,
                "accepted_time_ns": 9_500_000_000,
                "time_submitted_ns": 9_500_000_000,
            },
        }
        cancelled_ids = []
        trader.cancel_order = lambda order, **kwargs: cancelled_ids.append(order.order_id)

        trader._maintain_passive_orders()

        self.assertIn(stale_bid.order_id, cancelled_ids)

    def test_same_or_better_passive_order_blocks_redundant_stack(self) -> None:
        trader = ValueTrader(
            id=11,
            name="value_stack_guard",
            type="ValueTrader",
            symbol="ABM",
            starting_cash=1_000_000,
            wake_up_freq=10,
            sigma_eta=0.0,
            delta=2,
            aggressive_probability=0.0,
            aggressive_order_size=1,
            passive_order_size=2,
            random_state=np.random.RandomState(41),
            log_orders=False,
        )
        bid_order = LimitOrder(
            agent_id=trader.id,
            time_placed=0,
            symbol="ABM",
            quantity=2,
            side=Side.BID,
            limit_price=100,
        )
        trader.orders = {1: bid_order}
        trader.limit_order_lifecycle = {
            bid_order.order_id: {
                "is_passive": True,
                "accepted_time_ns": 0,
                "time_submitted_ns": 0,
            }
        }

        self.assertTrue(trader._has_same_or_better_passive_order(Side.BID, 100))
        self.assertTrue(trader._has_same_or_better_passive_order(Side.BID, 99))
        self.assertFalse(trader._has_same_or_better_passive_order(Side.BID, 101))


if __name__ == "__main__":
    unittest.main()
