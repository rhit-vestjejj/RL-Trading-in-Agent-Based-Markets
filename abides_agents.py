"""ABIDES-backed implementations of the experiment's trader classes."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Protocol

import numpy as np

from abides_support import bootstrap_abides_paths
from env import RLMarketEnvironment, RLPolicy, RandomPolicy, RandomQuoterPolicy

bootstrap_abides_paths()

from abides_core import Message, NanosecondTime
from abides_markets.messages.order import LimitOrderMsg
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import LimitOrder, Side


def _safe_midprice(
    bid: Optional[int],
    ask: Optional[int],
    fallback: Optional[int],
) -> Optional[int]:
    if bid is not None and ask is not None:
        return int(round((bid + ask) / 2))
    if fallback is not None:
        return fallback
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


def _safe_imbalance(bid_depth: int, ask_depth: int) -> float:
    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return 0.0
    return (bid_depth - ask_depth) / total_depth


def _choose_passive_offset_ticks(
    random_state: np.random.RandomState,
    spread_ticks: float,
    *,
    two_tick_improve_probability: float,
    wide_spread_improve_probability: float,
    tight_spread_behind_probability: float = 0.18,
    behind_inside_probability: float = 0.08,
) -> int:
    if spread_ticks <= 1.0:
        return -1 if random_state.rand() < tight_spread_behind_probability else 0
    if spread_ticks >= 3.0:
        draw = random_state.rand()
        if draw < wide_spread_improve_probability:
            return 1
        if draw < wide_spread_improve_probability + behind_inside_probability:
            return -1
        return 0
    if spread_ticks >= 2.0:
        draw = random_state.rand()
        if draw < two_tick_improve_probability:
            return 1
        if draw < two_tick_improve_probability + behind_inside_probability:
            return -1
        return 0
    return 0


def _maybe_nanoseconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


@dataclass(frozen=True)
class ABIDESObservation:
    """Compact market observation shared across custom ABIDES agents."""

    time: NanosecondTime
    best_bid: Optional[int]
    best_ask: Optional[int]
    bid_depth: int
    ask_depth: int
    midprice: Optional[int]
    spread: float
    imbalance: float


class ABIDESPollingTrader(TradingAgent):
    """Small helper base for polling-style ABIDES trading agents."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        poisson_arrival: bool = True,
        wake_up_jitter: NanosecondTime = 0,
        max_live_passive_orders_per_side: Optional[int] = None,
        passive_order_max_age: Optional[NanosecondTime] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(id, name, type, random_state, starting_cash, log_orders)
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival
        self.arrival_rate = wake_up_freq
        self.wake_up_jitter = max(0, int(wake_up_jitter))
        self.max_live_passive_orders_per_side = (
            None
            if max_live_passive_orders_per_side is None
            else max(1, int(max_live_passive_orders_per_side))
        )
        self.passive_order_max_age = (
            None if passive_order_max_age is None else max(1, int(passive_order_max_age))
        )
        self.state = "AWAITING_WAKEUP"
        self.oracle = None
        self.midprice_history: Deque[float] = deque(maxlen=256)
        self.return_history: Deque[float] = deque(maxlen=256)
        self.last_observed_midprice: Optional[int] = None
        self.limit_order_lifecycle: Dict[int, Dict[str, Any]] = {}

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        self.oracle = self.kernel.oracle

    def wakeup(self, current_time: NanosecondTime):
        can_trade = super().wakeup(current_time)
        self.state = "AWAITING_WAKEUP"
        if not can_trade:
            return

        self._maintain_passive_orders()
        self.on_before_query()
        self.get_current_spread(self.symbol, depth=1)
        self.state = "AWAITING_SPREAD"

    def receive_message(
        self,
        current_time: NanosecondTime,
        sender_id: int,
        message: Message,
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if (
            self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
            and not self.mkt_closed
        ):
            observation = self._build_observation(current_time)
            if observation is not None:
                self.on_observation(observation)
            self.state = "AWAITING_WAKEUP"
            if not self.mkt_closed:
                self.set_wakeup(current_time + self.get_wake_frequency())

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            delta_time = float(self.wake_up_freq)
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
        if self.wake_up_jitter > 0:
            delta_time += int(
                self.random_state.randint(
                    low=-self.wake_up_jitter,
                    high=self.wake_up_jitter + 1,
                )
            )
        return max(1, int(round(delta_time)))

    def inventory(self) -> int:
        return int(self.get_holdings(self.symbol))

    def cash_position(self) -> int:
        return int(self.holdings["CASH"])

    def wealth(self, midprice: Optional[int]) -> int:
        if midprice is None:
            midprice = self.last_trade.get(self.symbol, 0)
        return self.cash_position() + self.inventory() * int(midprice or 0)

    def observe_private_value(self, sigma_n: float) -> int:
        return int(
            self.oracle.observe_price(
                self.symbol,
                self.current_time,
                random_state=self.random_state,
                sigma_n=sigma_n,
            )
        )

    def _live_limit_orders(self, side: Optional[Side] = None) -> List[LimitOrder]:
        return [
            order
            for order in self.orders.values()
            if isinstance(order, LimitOrder) and (side is None or order.side == side)
        ]

    def _is_live_passive_order(self, order: LimitOrder) -> bool:
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        return bool(lifecycle is not None and lifecycle.get("is_passive", False))

    def _passive_order_event_time(self, order: LimitOrder) -> Optional[int]:
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        if lifecycle is None:
            return None
        accepted_time = _maybe_nanoseconds(lifecycle.get("accepted_time_ns"))
        submitted_time = _maybe_nanoseconds(lifecycle.get("time_submitted_ns"))
        return accepted_time if accepted_time is not None else submitted_time

    def _has_same_or_better_passive_order(self, side: Side, limit_price: int) -> bool:
        for order in self._live_limit_orders(side):
            if not self._is_live_passive_order(order):
                continue
            if side.is_bid() and order.limit_price >= limit_price:
                return True
            if side.is_ask() and order.limit_price <= limit_price:
                return True
        return False

    def _passive_cancel_priority(self, order: LimitOrder) -> tuple[float, int]:
        competitiveness = float(order.limit_price) if order.side.is_bid() else float(-order.limit_price)
        event_time = self._passive_order_event_time(order) or 0
        return competitiveness, -event_time

    def _has_live_order(self, side: Side) -> bool:
        return any(
            isinstance(order, LimitOrder) and order.side == side
            for order in self.orders.values()
        )

    def _cancel_side_orders(self, side: Side) -> None:
        for order in list(self.orders.values()):
            if isinstance(order, LimitOrder) and order.side == side:
                self.cancel_order(order)

    def _maintain_passive_orders(self) -> None:
        if self.passive_order_max_age is not None:
            for order in list(self._live_limit_orders()):
                if not self._is_live_passive_order(order):
                    continue
                event_time = self._passive_order_event_time(order)
                if event_time is None:
                    continue
                if int(self.current_time) - event_time >= self.passive_order_max_age:
                    self.cancel_order(order)

        if self.max_live_passive_orders_per_side is None:
            return

        for side in (Side.BID, Side.ASK):
            passive_orders = [
                order
                for order in self._live_limit_orders(side)
                if self._is_live_passive_order(order)
            ]
            overflow = len(passive_orders) - self.max_live_passive_orders_per_side
            if overflow <= 0:
                continue
            passive_orders.sort(key=self._passive_cancel_priority)
            for order in passive_orders[:overflow]:
                self.cancel_order(order)

    def _limit_order_snapshot(
        self,
        order: LimitOrder,
    ) -> Dict[str, Any]:
        best_bid, bid_depth, best_ask, ask_depth = self.get_known_bid_ask(order.symbol)
        if order.side.is_bid():
            same_side_best = best_bid
            opposite_touch = best_ask
            same_side_distance = (
                float(order.limit_price - best_bid) if best_bid is not None else float("nan")
            )
            opposite_distance = (
                float(best_ask - order.limit_price)
                if best_ask is not None
                else float("nan")
            )
            is_passive = best_ask is None or order.limit_price < best_ask
        else:
            same_side_best = best_ask
            opposite_touch = best_bid
            same_side_distance = (
                float(best_ask - order.limit_price) if best_ask is not None else float("nan")
            )
            opposite_distance = (
                float(order.limit_price - best_bid)
                if best_bid is not None
                else float("nan")
            )
            is_passive = best_bid is None or order.limit_price > best_bid

        if not is_passive:
            placement_bucket = "cross"
        elif np.isnan(same_side_distance):
            placement_bucket = "unknown"
        elif same_side_distance > 0:
            placement_bucket = "improve"
        elif same_side_distance == 0:
            placement_bucket = "join"
        else:
            placement_bucket = "behind"

        return {
            "order_id": int(order.order_id),
            "agent_id": int(self.id),
            "agent_type": str(self.type),
            "source": str(order.tag) if order.tag is not None else "unlabeled",
            "symbol": str(order.symbol),
            "side": "bid" if order.side.is_bid() else "ask",
            "quantity": int(order.quantity),
            "limit_price": int(order.limit_price),
            "tag": str(order.tag) if order.tag is not None else None,
            "time_submitted_ns": int(self.current_time),
            "best_bid_at_submit": (
                int(best_bid) if best_bid is not None else None
            ),
            "best_ask_at_submit": (
                int(best_ask) if best_ask is not None else None
            ),
            "bid_depth_at_submit": int(bid_depth),
            "ask_depth_at_submit": int(ask_depth),
            "same_side_best_at_submit": (
                int(same_side_best) if same_side_best is not None else None
            ),
            "opposite_touch_at_submit": (
                int(opposite_touch) if opposite_touch is not None else None
            ),
            "same_side_distance_ticks": same_side_distance,
            "opposite_touch_distance_ticks": opposite_distance,
            "is_passive": bool(is_passive),
            "placement_bucket": placement_bucket,
            "accepted_time_ns": None,
            "first_execution_time_ns": None,
            "cancelled_time_ns": None,
            "executed_quantity": 0,
        }

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: Side,
        limit_price: int,
        order_id: Optional[int] = None,
        is_hidden: bool = False,
        is_price_to_comply: bool = False,
        insert_by_id: bool = False,
        is_post_only: bool = False,
        ignore_risk: bool = True,
        tag: Any = None,
    ) -> None:
        order = self.create_limit_order(
            symbol,
            quantity,
            side,
            limit_price,
            order_id,
            is_hidden,
            is_price_to_comply,
            insert_by_id,
            is_post_only,
            ignore_risk,
            tag,
        )
        if order is None:
            return

        snapshot = self._limit_order_snapshot(order)
        passive_controls_enabled = (
            self.max_live_passive_orders_per_side is not None
            or self.passive_order_max_age is not None
        )
        if (
            passive_controls_enabled
            and snapshot["is_passive"]
            and self._has_same_or_better_passive_order(side, limit_price)
        ):
            return

        self.limit_order_lifecycle[order.order_id] = snapshot
        self.orders[order.order_id] = deepcopy(order)
        self.send_message(self.exchange_id, LimitOrderMsg(order))
        if passive_controls_enabled and snapshot["is_passive"]:
            self._maintain_passive_orders()

        if self.log_orders:
            self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

    def place_near_touch_limit(
        self,
        observation: ABIDESObservation,
        side: Side,
        quantity: int,
        *,
        offset_ticks: int = 0,
        tag: str = "near_touch",
    ) -> None:
        limit_price = self._near_touch_limit_price(
            observation,
            side=side,
            offset_ticks=offset_ticks,
        )
        if limit_price is None:
            return

        self.place_limit_order(
            self.symbol,
            quantity=quantity,
            side=side,
            limit_price=int(limit_price),
            tag=tag,
        )

    def _near_touch_limit_price(
        self,
        observation: ABIDESObservation,
        *,
        side: Side,
        offset_ticks: int = 0,
    ) -> Optional[int]:
        if side.is_bid():
            if observation.best_bid is not None:
                limit_price = observation.best_bid + int(offset_ticks)
                if observation.best_ask is not None:
                    limit_price = min(limit_price, observation.best_ask - 1)
            elif observation.best_ask is not None:
                limit_price = observation.best_ask - 1 - int(offset_ticks)
            else:
                return None
        else:
            if observation.best_ask is not None:
                limit_price = observation.best_ask - int(offset_ticks)
                if observation.best_bid is not None:
                    limit_price = max(limit_price, observation.best_bid + 1)
            elif observation.best_bid is not None:
                limit_price = observation.best_bid + 1 + int(offset_ticks)
            else:
                return None
        return max(1, int(limit_price))

    def order_accepted(self, order: LimitOrder) -> None:
        super().order_accepted(order)
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        if lifecycle is not None and lifecycle["accepted_time_ns"] is None:
            lifecycle["accepted_time_ns"] = int(self.current_time)

    def order_executed(self, order) -> None:
        super().order_executed(order)
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        if lifecycle is None:
            return
        if lifecycle["first_execution_time_ns"] is None:
            lifecycle["first_execution_time_ns"] = int(self.current_time)
        lifecycle["executed_quantity"] = int(lifecycle["executed_quantity"]) + int(order.quantity)

    def order_cancelled(self, order: LimitOrder) -> None:
        super().order_cancelled(order)
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        if lifecycle is not None and lifecycle["cancelled_time_ns"] is None:
            lifecycle["cancelled_time_ns"] = int(self.current_time)

    def order_partial_cancelled(self, order: LimitOrder) -> None:
        super().order_partial_cancelled(order)
        lifecycle = self.limit_order_lifecycle.get(order.order_id)
        if lifecycle is not None:
            lifecycle["cancelled_time_ns"] = int(self.current_time)

    def order_replaced(self, old_order: LimitOrder, new_order: LimitOrder) -> None:
        super().order_replaced(old_order, new_order)
        lifecycle = self.limit_order_lifecycle.pop(old_order.order_id, None)
        if lifecycle is None:
            return
        lifecycle["cancelled_time_ns"] = int(self.current_time)
        replacement = dict(lifecycle)
        replacement["order_id"] = int(new_order.order_id)
        replacement["quantity"] = int(new_order.quantity)
        replacement["limit_price"] = int(new_order.limit_price)
        replacement["time_submitted_ns"] = int(self.current_time)
        replacement["accepted_time_ns"] = None
        replacement["first_execution_time_ns"] = None
        replacement["cancelled_time_ns"] = None
        replacement["executed_quantity"] = 0
        self.limit_order_lifecycle[new_order.order_id] = replacement

    def on_before_query(self) -> None:
        """Hook called before the agent requests the next spread snapshot."""

    def on_observation(self, observation: ABIDESObservation) -> None:
        raise NotImplementedError

    def _build_observation(
        self,
        current_time: NanosecondTime,
    ) -> Optional[ABIDESObservation]:
        bid, bid_depth, ask, ask_depth = self.get_known_bid_ask(self.symbol)
        fallback = self.last_trade.get(self.symbol)
        midprice = _safe_midprice(bid=bid, ask=ask, fallback=fallback)
        if midprice is None:
            return None

        if self.last_observed_midprice is not None and self.last_observed_midprice > 0:
            log_return = float(np.log(midprice / self.last_observed_midprice))
            self.return_history.append(log_return)

        self.midprice_history.append(float(midprice))
        self.last_observed_midprice = midprice

        spread = float(ask - bid) if bid is not None and ask is not None else 0.0
        return ABIDESObservation(
            time=current_time,
            best_bid=bid,
            best_ask=ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            midprice=midprice,
            spread=spread,
            imbalance=_safe_imbalance(bid_depth=bid_depth, ask_depth=ask_depth),
        )


class NoiseTrader(ABIDESPollingTrader):
    """Background random order-flow trader."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        limit_order_probability: float = 0.5,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            max_live_passive_orders_per_side=1,
            passive_order_max_age=int(8 * 1_000_000_000),
            log_orders=log_orders,
        )
        self.limit_order_probability = limit_order_probability

    def on_observation(self, observation: ABIDESObservation) -> None:
        side = Side.BID if self.random_state.rand() < 0.5 else Side.ASK
        if self.random_state.rand() < self.limit_order_probability:
            quantity = int(self.random_state.randint(1, 3))
            spread_ticks = float(observation.spread)
            if spread_ticks <= 1.0:
                offset_ticks = -1 if self.random_state.rand() < 0.25 else 0
            elif spread_ticks >= 2.0:
                draw = self.random_state.rand()
                if draw < 0.12:
                    offset_ticks = 1
                elif draw < 0.25:
                    offset_ticks = -1
                else:
                    offset_ticks = 0
            else:
                offset_ticks = 0
            self.place_near_touch_limit(
                observation,
                side=side,
                quantity=quantity,
                offset_ticks=offset_ticks,
            )
        else:
            quantity = int(self.random_state.randint(1, 4))
            self.place_market_order(self.symbol, quantity=quantity, side=side)


class ZICTrader(ABIDESPollingTrader):
    """Zero-intelligence constrained trader with a noisy private valuation."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        sigma_eta: float,
        surplus_min_ticks: int,
        surplus_max_ticks: int,
        inside_competition_probability: float = 0.0,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            max_live_passive_orders_per_side=2,
            passive_order_max_age=int(15 * 1_000_000_000),
            log_orders=log_orders,
        )
        self.sigma_eta = sigma_eta
        self.surplus_min_ticks = surplus_min_ticks
        self.surplus_max_ticks = surplus_max_ticks
        self.inside_competition_probability = inside_competition_probability

    def on_observation(self, observation: ABIDESObservation) -> None:
        side = Side.BID if self.random_state.rand() < 0.5 else Side.ASK
        quantity = int(self.random_state.randint(1, 4))
        private_value = self.observe_private_value(self.sigma_eta)
        surplus = self.random_state.uniform(
            low=self.surplus_min_ticks,
            high=self.surplus_max_ticks,
        )
        surplus = int(round(surplus))
        limit_price = (
            private_value - surplus if side.is_bid() else private_value + surplus
        )
        same_side_best = observation.best_bid if side.is_bid() else observation.best_ask
        behind_inside = (
            same_side_best is not None
            and (
                limit_price < same_side_best
                if side.is_bid()
                else limit_price > same_side_best
            )
        )
        within_inside_competition_band = (
            behind_inside
            and abs(limit_price - int(same_side_best)) <= self.surplus_max_ticks
        )
        if (
            within_inside_competition_band
            and self.random_state.rand() < self.inside_competition_probability
        ):
            self.place_near_touch_limit(
                observation,
                side=side,
                quantity=quantity,
                offset_ticks=1,
                tag="zic_near_touch",
            )
            return
        self.place_limit_order(
            self.symbol,
            quantity=quantity,
            side=side,
            limit_price=max(1, int(limit_price)),
            tag="zic_limit",
        )


class TrendFollowerTrader(ABIDESPollingTrader):
    """Trader driven by short/long moving-average crossover."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        short_window: int,
        long_window: int,
        signal_noise: float,
        signal_threshold: float,
        trade_probability: float,
        aggressive_probability: float,
        aggressive_order_size: int,
        passive_order_size: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            max_live_passive_orders_per_side=1,
            passive_order_max_age=int(8 * 1_000_000_000),
            log_orders=log_orders,
        )
        self.short_window = short_window
        self.long_window = long_window
        self.signal_noise = signal_noise
        self.signal_threshold = signal_threshold
        self.trade_probability = trade_probability
        self.aggressive_probability = float(np.clip(aggressive_probability, 0.0, 1.0))
        self.aggressive_order_size = max(1, int(aggressive_order_size))
        self.passive_order_size = max(1, int(passive_order_size))

    def _trend_signal(self) -> float:
        series = np.array(self.midprice_history, dtype=float)
        short_ma = float(np.mean(series[-self.short_window :]))
        long_ma = float(np.mean(series[-self.long_window :]))
        signal = short_ma - long_ma
        if self.signal_noise > 0:
            signal += float(self.random_state.normal(loc=0.0, scale=self.signal_noise))
        return signal

    def on_observation(self, observation: ABIDESObservation) -> None:
        if len(self.midprice_history) < self.long_window:
            return

        if self.random_state.rand() > self.trade_probability:
            return

        signal = self._trend_signal()
        threshold_noise = 0.0
        if self.signal_noise > 0:
            threshold_noise = float(
                self.random_state.normal(loc=0.0, scale=self.signal_noise * 0.5)
            )
        effective_threshold = max(0.0, self.signal_threshold + threshold_noise)
        if signal > effective_threshold:
            if self.random_state.rand() < self.aggressive_probability:
                self.place_market_order(
                    self.symbol,
                    quantity=self.aggressive_order_size,
                    side=Side.BID,
                )
            else:
                offset_ticks = _choose_passive_offset_ticks(
                    self.random_state,
                    observation.spread,
                    two_tick_improve_probability=0.35,
                    wide_spread_improve_probability=0.75,
                )
                self.place_near_touch_limit(
                    observation,
                    side=Side.BID,
                    quantity=self.passive_order_size,
                    offset_ticks=offset_ticks,
                )
        elif signal < -effective_threshold:
            if self.random_state.rand() < self.aggressive_probability:
                self.place_market_order(
                    self.symbol,
                    quantity=self.aggressive_order_size,
                    side=Side.ASK,
                )
            else:
                offset_ticks = _choose_passive_offset_ticks(
                    self.random_state,
                    observation.spread,
                    two_tick_improve_probability=0.35,
                    wide_spread_improve_probability=0.75,
                )
                self.place_near_touch_limit(
                    observation,
                    side=Side.ASK,
                    quantity=self.passive_order_size,
                    offset_ticks=offset_ticks,
                )


class ValueTrader(ABIDESPollingTrader):
    """Trader that reacts to deviations between private value and market midprice."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        sigma_eta: float,
        delta: int,
        aggressive_probability: float,
        aggressive_order_size: int,
        passive_order_size: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            max_live_passive_orders_per_side=1,
            passive_order_max_age=int(8 * 1_000_000_000),
            log_orders=log_orders,
        )
        self.sigma_eta = sigma_eta
        self.delta = delta
        self.aggressive_probability = float(np.clip(aggressive_probability, 0.0, 1.0))
        self.aggressive_order_size = max(1, int(aggressive_order_size))
        self.passive_order_size = max(1, int(passive_order_size))

    def on_observation(self, observation: ABIDESObservation) -> None:
        if observation.midprice is None:
            return

        private_value = self.observe_private_value(self.sigma_eta)
        mispricing = private_value - observation.midprice
        if mispricing > self.delta:
            if self.random_state.rand() < self.aggressive_probability:
                self.place_market_order(
                    self.symbol,
                    quantity=self.aggressive_order_size,
                    side=Side.BID,
                )
            else:
                offset_ticks = _choose_passive_offset_ticks(
                    self.random_state,
                    observation.spread,
                    two_tick_improve_probability=0.40,
                    wide_spread_improve_probability=0.80,
                )
                self.place_near_touch_limit(
                    observation,
                    side=Side.BID,
                    quantity=self.passive_order_size,
                    offset_ticks=offset_ticks,
                )
        elif mispricing < -self.delta:
            if self.random_state.rand() < self.aggressive_probability:
                self.place_market_order(
                    self.symbol,
                    quantity=self.aggressive_order_size,
                    side=Side.ASK,
                )
            else:
                offset_ticks = _choose_passive_offset_ticks(
                    self.random_state,
                    observation.spread,
                    two_tick_improve_probability=0.40,
                    wide_spread_improve_probability=0.80,
                )
                self.place_near_touch_limit(
                    observation,
                    side=Side.ASK,
                    quantity=self.passive_order_size,
                    offset_ticks=offset_ticks,
                )


class AdaptiveMarketMaker(ABIDESPollingTrader):
    """Simple two-sided market maker with linear inventory skew."""

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        wake_up_jitter: NanosecondTime = 0,
        min_refresh_interval: NanosecondTime = 1,
        target_spread: int,
        alpha_cents: float,
        quote_size: int,
        reprice_threshold: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            poisson_arrival=True,
            wake_up_jitter=wake_up_jitter,
            log_orders=log_orders,
        )
        self.target_spread = target_spread
        self.alpha_cents = alpha_cents
        self.quote_size = max(1, int(quote_size))
        self.reprice_threshold = max(1, int(reprice_threshold))
        self.agent_spread_offset = 0
        self.agent_center_bias = int(self.random_state.choice([-1, 0, 1]))
        self.min_refresh_interval = max(1, int(min_refresh_interval))
        self.last_refresh_time: Optional[NanosecondTime] = None
        self.next_refresh_side = Side.BID if self.random_state.rand() < 0.5 else Side.ASK
        self.last_quoted_bid: Optional[int] = None
        self.last_quoted_ask: Optional[int] = None

    def on_before_query(self) -> None:
        return

    def _desired_quotes(self, observation: ABIDESObservation) -> tuple[int, int]:
        effective_spread = max(1, self.target_spread + self.agent_spread_offset)
        if abs(observation.imbalance) >= 0.60 or abs(self.inventory()) >= self.quote_size * 2:
            if self.random_state.rand() < 0.35:
                effective_spread = min(3, effective_spread + 1)
        elif self.random_state.rand() < 0.04:
            effective_spread = min(3, effective_spread + 1)
        epsilon = float(self.random_state.normal(loc=0.0, scale=0.2))
        capped_skew = np.clip(
            self.alpha_cents * self.inventory(),
            -effective_spread / 2,
            effective_spread / 2,
        )
        bid_price = int(
            round(
                observation.midprice
                - effective_spread / 2
                - capped_skew
                + self.agent_center_bias
                + epsilon
            )
        )
        ask_price = int(
            round(
                observation.midprice
                + effective_spread / 2
                - capped_skew
                + self.agent_center_bias
                + epsilon
            )
        )
        bid_price, ask_price = self._soften_inside_step_ahead(
            observation,
            bid_price=bid_price,
            ask_price=ask_price,
        )
        bid_price = max(1, bid_price)
        ask_price = max(bid_price + 1, ask_price)
        return bid_price, ask_price

    def _soften_inside_step_ahead(
        self,
        observation: ABIDESObservation,
        *,
        bid_price: int,
        ask_price: int,
    ) -> tuple[int, int]:
        """Join an already-adequate inside quote instead of stepping one tick ahead."""

        if (
            observation.best_bid is not None
            and observation.bid_depth > 0
            and bid_price > observation.best_bid
            and (bid_price - observation.best_bid) <= 1
        ):
            bid_price = int(observation.best_bid)
        if (
            observation.best_ask is not None
            and observation.ask_depth > 0
            and ask_price < observation.best_ask
            and (observation.best_ask - ask_price) <= 1
        ):
            ask_price = int(observation.best_ask)
        return bid_price, ask_price

    def _has_live_order(self, side: Side) -> bool:
        return any(
            isinstance(order, LimitOrder) and order.side == side
            for order in self.orders.values()
        )

    def _cancel_side_orders(self, side: Side) -> None:
        for order in list(self.orders.values()):
            if isinstance(order, LimitOrder) and order.side == side:
                self.cancel_order(order)

    def _needs_refresh(
        self,
        *,
        side: Side,
        desired_price: int,
        last_quoted_price: Optional[int],
    ) -> bool:
        if not self._has_live_order(side):
            return True
        if last_quoted_price is None:
            return True
        return abs(desired_price - last_quoted_price) >= self.reprice_threshold

    def _place_side_quote(self, side: Side, price: int) -> None:
        self.place_limit_order(
            self.symbol,
            quantity=self.quote_size,
            side=side,
            limit_price=price,
            tag="mm_quote",
        )
        if side.is_bid():
            self.last_quoted_bid = price
        else:
            self.last_quoted_ask = price

    def on_observation(self, observation: ABIDESObservation) -> None:
        if observation.midprice is None:
            return

        bid_price, ask_price = self._desired_quotes(observation)
        refresh_bid = self._needs_refresh(
            side=Side.BID,
            desired_price=bid_price,
            last_quoted_price=self.last_quoted_bid,
        )
        refresh_ask = self._needs_refresh(
            side=Side.ASK,
            desired_price=ask_price,
            last_quoted_price=self.last_quoted_ask,
        )

        if not refresh_bid and not refresh_ask:
            return

        if (
            self.last_refresh_time is not None
            and observation.time - self.last_refresh_time < self.min_refresh_interval
            and self._has_live_order(Side.BID)
            and self._has_live_order(Side.ASK)
        ):
            return

        if not self._has_live_order(Side.BID) and not self._has_live_order(Side.ASK):
            self._place_side_quote(Side.BID, bid_price)
            self._place_side_quote(Side.ASK, ask_price)
            self.next_refresh_side = Side.ASK if self.next_refresh_side.is_bid() else Side.BID
            self.last_refresh_time = observation.time
            return

        side_to_refresh = self.next_refresh_side
        alternate_side = Side.ASK if side_to_refresh.is_bid() else Side.BID
        refresh_plan = (
            (side_to_refresh, refresh_bid if side_to_refresh.is_bid() else refresh_ask),
            (alternate_side, refresh_bid if alternate_side.is_bid() else refresh_ask),
        )
        for side, should_refresh in refresh_plan:
            if not should_refresh:
                continue
            self._cancel_side_orders(side)
            self._place_side_quote(
                side,
                bid_price if side.is_bid() else ask_price,
            )
            self.next_refresh_side = alternate_side if side == side_to_refresh else side_to_refresh
            self.last_refresh_time = observation.time
            return

        self.last_refresh_time = observation.time


class BaseRLTrader(ABIDESPollingTrader):
    """Shared reward/logging path for directional and quoting RL agents."""

    ACTION_LABELS: Dict[int, str] = {}
    AGENT_ROLE = "taker"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        environment: RLMarketEnvironment,
        inventory_cap: int | None = None,
        policy: Optional[RLPolicy] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        max_live_passive_orders_per_side: Optional[int] = None,
        passive_order_max_age: Optional[NanosecondTime] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            random_state=random_state,
            max_live_passive_orders_per_side=max_live_passive_orders_per_side,
            passive_order_max_age=passive_order_max_age,
            log_orders=log_orders,
        )
        self.environment = environment
        self.inventory_cap = int(inventory_cap) if inventory_cap is not None else None
        self.policy = policy if policy is not None else RandomPolicy()
        self.metrics_log: List[Dict[str, object]] = []
        self.transition_log: List[Dict[str, object]] = []
        self.pending_snapshot: Optional[Dict[str, float]] = None
        self.pending_transition: Optional[Dict[str, object]] = None
        self.last_state: Optional[np.ndarray] = None
        self.last_effective_action: Optional[int] = None
        self.decision_count = 0
        self.total_aggressive_orders_submitted = 0
        self.total_passive_bid_orders_submitted = 0
        self.total_passive_ask_orders_submitted = 0
        self.total_passive_executed_quantity = 0

    def on_observation(self, observation: ABIDESObservation) -> None:
        if observation.midprice is None:
            return

        current_cash = self.cash_position()
        current_inventory = self.inventory()
        current_midprice = observation.midprice
        resting_bid_before, resting_ask_before = self._current_resting_quote_flags()

        reward_cents = 0.0
        reward_components = {
            "wealth_delta": 0.0,
            "inventory_penalty": 0.0,
            "flat_hold_penalty": 0.0,
            "passive_fill_bonus": 0.0,
            "two_sided_quote_reward": 0.0,
            "missing_quote_penalty": 0.0,
            "reward": 0.0,
        }
        previous_action = self.last_effective_action
        inventory_delta = 0.0
        cash_delta = 0.0
        passive_fill_quantity = 0.0
        if self.pending_snapshot is not None:
            inventory_delta = float(current_inventory - int(self.pending_snapshot["inventory"]))
            cash_delta = float(current_cash - float(self.pending_snapshot["cash"]))
            passive_fill_quantity = max(
                0.0,
                float(self.total_passive_executed_quantity) - float(self.pending_snapshot["passive_fill_total"]),
            )
            reward_components = self.environment.compute_reward_components(
                previous_cash=float(self.pending_snapshot["cash"]),
                previous_inventory=int(self.pending_snapshot["inventory"]),
                previous_midprice=float(self.pending_snapshot["midprice"]),
                current_cash=float(current_cash),
                current_inventory=current_inventory,
                current_midprice=float(current_midprice),
                previous_action=previous_action,
                passive_fill_quantity=passive_fill_quantity,
                has_resting_bid=resting_bid_before,
                has_resting_ask=resting_ask_before,
                quote_reward_eligible=self._quote_reward_eligible(),
            )
            reward_cents = float(reward_components["reward"])

        state = self._build_state(observation)
        self.last_state = state
        if self.pending_transition is not None:
            self._append_transition(
                reward_cents=reward_cents,
                reward_components=reward_components,
                next_state=state,
                done=False,
                event_time=int(observation.time),
            )

        policy_decision = self._sample_policy_action(state)
        action = int(policy_decision["action"])
        effective_action, cap_block = self._apply_inventory_cap(action, current_inventory)

        aggressive_before = int(self.total_aggressive_orders_submitted)
        passive_bid_before = int(self.total_passive_bid_orders_submitted)
        passive_ask_before = int(self.total_passive_ask_orders_submitted)
        self._submit_effective_action(effective_action, observation)
        submitted_aggressive = int(self.total_aggressive_orders_submitted) - aggressive_before
        submitted_passive_bid = int(self.total_passive_bid_orders_submitted) - passive_bid_before
        submitted_passive_ask = int(self.total_passive_ask_orders_submitted) - passive_ask_before
        resting_bid_after, resting_ask_after = self._current_resting_quote_flags()

        inventory_at_cap = self._inventory_at_cap(current_inventory)
        inventory_near_cap = self._inventory_near_cap(current_inventory)
        metrics = {
            "time": int(observation.time),
            "agent_id": int(self.id),
            "agent_name": str(self.name),
            "agent_role": str(self.AGENT_ROLE),
            "decision_index": int(self.decision_count),
            "action": int(action),
            "action_label": self.ACTION_LABELS.get(action, "unknown"),
            "effective_action": int(effective_action),
            "effective_action_label": self.ACTION_LABELS.get(effective_action, "unknown"),
            "previous_action": int(previous_action) if previous_action is not None else -1,
            "previous_action_label": (
                self.ACTION_LABELS.get(previous_action, "none")
                if previous_action is not None
                else "none"
            ),
            "inventory_cap": float(self.inventory_cap) if self.inventory_cap is not None else np.nan,
            "blocked_by_inventory_cap": bool(cap_block["blocked"]),
            "blocked_buy_due_to_cap": bool(cap_block["blocked_buy"]),
            "blocked_sell_due_to_cap": bool(cap_block["blocked_sell"]),
            "blocked_action_label": str(cap_block["blocked_action_label"]),
            "blocked_action_reason": str(cap_block["reason"]),
            "inventory_at_cap": bool(inventory_at_cap),
            "inventory_near_cap": bool(inventory_near_cap),
            "inventory": float(current_inventory),
            "cash": float(current_cash),
            "wealth": float(self.wealth(current_midprice)),
            "reward": float(reward_cents),
            "spread": float(observation.spread),
            "imbalance": float(observation.imbalance),
            "midprice": float(current_midprice),
            "best_bid": float(observation.best_bid) if observation.best_bid is not None else np.nan,
            "best_ask": float(observation.best_ask) if observation.best_ask is not None else np.nan,
            "log_prob": float(policy_decision.get("log_prob", np.nan)),
            "value_estimate": float(policy_decision.get("value", np.nan)),
            "executed_since_last_decision": float(abs(inventory_delta) > 0.0),
            "filled_quantity_since_last_decision": float(abs(inventory_delta)),
            "passive_filled_quantity_since_last_decision": float(passive_fill_quantity),
            "inventory_delta_since_last_decision": float(inventory_delta),
            "cash_delta_since_last_decision": float(cash_delta),
            "wealth_delta_since_last_decision": float(reward_components["wealth_delta"]),
            "inventory_penalty_since_last_decision": float(reward_components["inventory_penalty"]),
            "flat_hold_penalty_since_last_decision": float(reward_components["flat_hold_penalty"]),
            "passive_fill_bonus_since_last_decision": float(reward_components["passive_fill_bonus"]),
            "two_sided_quote_reward_since_last_decision": float(
                reward_components["two_sided_quote_reward"]
            ),
            "missing_quote_penalty_since_last_decision": float(
                reward_components["missing_quote_penalty"]
            ),
            "submitted_passive_bid_order_count": float(submitted_passive_bid),
            "submitted_passive_ask_order_count": float(submitted_passive_ask),
            "submitted_passive_order_count": float(submitted_passive_bid + submitted_passive_ask),
            "submitted_aggressive_order_count": float(submitted_aggressive),
            "total_rl_passive_order_count": float(
                self.total_passive_bid_orders_submitted + self.total_passive_ask_orders_submitted
            ),
            "total_rl_aggressive_order_count": float(self.total_aggressive_orders_submitted),
            "resting_bid_before_action": bool(resting_bid_before),
            "resting_ask_before_action": bool(resting_ask_before),
            "resting_bid": bool(resting_bid_after),
            "resting_ask": bool(resting_ask_after),
            "resting_both_quotes": bool(resting_bid_after and resting_ask_after),
        }
        for index, value in enumerate(state[: self.environment.return_window]):
            metrics[f"return_{index:02d}"] = float(value)
        self.metrics_log.append(metrics)

        self.pending_snapshot = {
            "cash": float(current_cash),
            "inventory": float(current_inventory),
            "midprice": float(current_midprice),
            "passive_fill_total": float(self.total_passive_executed_quantity),
        }
        self.pending_transition = {
            "time": int(observation.time),
            "agent_id": int(self.id),
            "agent_name": str(self.name),
            "agent_role": str(self.AGENT_ROLE),
            "decision_index": int(self.decision_count),
            "state": np.array(state, dtype=float),
            "action": int(action),
            "action_label": self.ACTION_LABELS.get(action, "unknown"),
            "effective_action": int(effective_action),
            "effective_action_label": self.ACTION_LABELS.get(effective_action, "unknown"),
            "inventory_cap": float(self.inventory_cap) if self.inventory_cap is not None else np.nan,
            "blocked_by_inventory_cap": bool(cap_block["blocked"]),
            "blocked_buy_due_to_cap": bool(cap_block["blocked_buy"]),
            "blocked_sell_due_to_cap": bool(cap_block["blocked_sell"]),
            "blocked_action_reason": str(cap_block["reason"]),
            "log_prob": float(policy_decision.get("log_prob", np.nan)),
            "value_estimate": float(policy_decision.get("value", np.nan)),
            "inventory_before": float(current_inventory),
            "cash_before": float(current_cash),
            "midprice_before": float(current_midprice),
            "submitted_passive_bid_order_count": float(submitted_passive_bid),
            "submitted_passive_ask_order_count": float(submitted_passive_ask),
            "submitted_passive_order_count": float(submitted_passive_bid + submitted_passive_ask),
            "submitted_aggressive_order_count": float(submitted_aggressive),
            "resting_bid_after_action": bool(resting_bid_after),
            "resting_ask_after_action": bool(resting_ask_after),
        }
        self.last_effective_action = effective_action
        self.decision_count += 1

    def kernel_stopping(self) -> None:
        if self.pending_transition is not None and self.pending_snapshot is not None:
            terminal_midprice = self.last_observed_midprice
            if terminal_midprice is None:
                terminal_midprice = self.last_trade.get(self.symbol)
            if terminal_midprice is None:
                terminal_midprice = int(self.pending_snapshot["midprice"])
            resting_bid, resting_ask = self._current_resting_quote_flags()
            reward_components = self.environment.compute_reward_components(
                previous_cash=float(self.pending_snapshot["cash"]),
                previous_inventory=int(self.pending_snapshot["inventory"]),
                previous_midprice=float(self.pending_snapshot["midprice"]),
                current_cash=float(self.cash_position()),
                current_inventory=self.inventory(),
                current_midprice=float(terminal_midprice),
                previous_action=self.last_effective_action,
                passive_fill_quantity=max(
                    0.0,
                    float(self.total_passive_executed_quantity)
                    - float(self.pending_snapshot["passive_fill_total"]),
                ),
                has_resting_bid=resting_bid,
                has_resting_ask=resting_ask,
                quote_reward_eligible=self._quote_reward_eligible(),
            )
            reward_cents = float(reward_components["reward"])
            terminal_state = (
                np.array(self.last_state, dtype=float)
                if self.last_state is not None
                else np.array(self.pending_transition["state"], dtype=float)
            )
            self._append_transition(
                reward_cents=reward_cents,
                reward_components=reward_components,
                next_state=terminal_state,
                done=True,
                event_time=int(self.current_time),
            )
        super().kernel_stopping()

    def order_executed(self, order) -> None:
        super().order_executed(order)
        if isinstance(order, LimitOrder):
            lifecycle = self.limit_order_lifecycle.get(order.order_id)
            if lifecycle is not None and lifecycle.get("is_passive", False):
                self.total_passive_executed_quantity += int(order.quantity)

    def _sample_policy_action(self, state: np.ndarray) -> Dict[str, float]:
        if hasattr(self.policy, "sample_action"):
            decision = getattr(self.policy, "sample_action")(state, self.random_state)
            return {
                "action": int(decision["action"]),
                "log_prob": float(decision.get("log_prob", np.nan)),
                "value": float(decision.get("value", np.nan)),
            }
        return {
            "action": int(self.policy.act(state, self.random_state)),
            "log_prob": float("nan"),
            "value": float("nan"),
        }

    def _append_transition(
        self,
        *,
        reward_cents: float,
        reward_components: Dict[str, float],
        next_state: np.ndarray,
        done: bool,
        event_time: int,
    ) -> None:
        if self.pending_transition is None:
            return
        transition = dict(self.pending_transition)
        transition["time_next"] = int(event_time)
        transition["reward"] = float(reward_cents)
        transition["wealth_delta"] = float(reward_components["wealth_delta"])
        transition["inventory_penalty"] = float(reward_components["inventory_penalty"])
        transition["flat_hold_penalty"] = float(reward_components["flat_hold_penalty"])
        transition["passive_fill_bonus"] = float(reward_components["passive_fill_bonus"])
        transition["two_sided_quote_reward"] = float(reward_components["two_sided_quote_reward"])
        transition["missing_quote_penalty"] = float(reward_components["missing_quote_penalty"])
        transition["next_state"] = np.array(next_state, dtype=float)
        transition["done"] = bool(done)
        transition["inventory_after"] = float(self.inventory())
        transition["cash_after"] = float(self.cash_position())
        transition["midprice_after"] = float(
            self.last_observed_midprice
            if self.last_observed_midprice is not None
            else transition["midprice_before"]
        )
        self.transition_log.append(transition)
        self.pending_transition = None

    def _apply_inventory_cap(self, action: int, current_inventory: int) -> tuple[int, Dict[str, object]]:
        if self.inventory_cap is None:
            return action, {
                "blocked": False,
                "blocked_buy": False,
                "blocked_sell": False,
                "blocked_action_label": "none",
                "reason": "",
            }

        if action == 2 and current_inventory >= self.inventory_cap:
            return 1, {
                "blocked": True,
                "blocked_buy": True,
                "blocked_sell": False,
                "blocked_action_label": "buy",
                "reason": "buy_blocked_at_long_cap",
            }
        if action == 0 and current_inventory <= -self.inventory_cap:
            return 1, {
                "blocked": True,
                "blocked_buy": False,
                "blocked_sell": True,
                "blocked_action_label": "sell",
                "reason": "sell_blocked_at_short_cap",
            }
        return action, {
            "blocked": False,
            "blocked_buy": False,
            "blocked_sell": False,
            "blocked_action_label": "none",
            "reason": "",
        }

    def _inventory_at_cap(self, inventory: int) -> bool:
        if self.inventory_cap is None or self.inventory_cap <= 0:
            return False
        return abs(int(inventory)) >= int(self.inventory_cap)

    def _inventory_near_cap(self, inventory: int) -> bool:
        if self.inventory_cap is None or self.inventory_cap <= 0:
            return False
        threshold = max(int(self.inventory_cap) - 1, int(np.ceil(0.9 * float(self.inventory_cap))))
        return abs(int(inventory)) >= threshold

    def _quote_reward_eligible(self) -> bool:
        return False

    def _current_resting_quote_flags(self) -> tuple[bool, bool]:
        return self._has_live_order(Side.BID), self._has_live_order(Side.ASK)

    def _submit_aggressive_market_order(self, *, side: Side, quantity: int) -> None:
        self.place_market_order(self.symbol, quantity=quantity, side=side)
        self.total_aggressive_orders_submitted += 1

    def _submit_passive_limit_order(
        self,
        *,
        side: Side,
        quantity: int,
        limit_price: int,
        tag: str,
    ) -> bool:
        before_ids = set(self.orders)
        self.place_limit_order(
            self.symbol,
            quantity=quantity,
            side=side,
            limit_price=limit_price,
            tag=tag,
        )
        submitted = bool(set(self.orders) - before_ids)
        if submitted:
            if side.is_bid():
                self.total_passive_bid_orders_submitted += 1
            else:
                self.total_passive_ask_orders_submitted += 1
        return submitted

    def _submit_effective_action(self, effective_action: int, observation: ABIDESObservation) -> None:
        raise NotImplementedError

    def _build_state(self, observation: ABIDESObservation) -> np.ndarray:
        from agents.base import MarketObservation as PrototypeObservation

        history_length = max(self.environment.return_window, 1)
        return_history = np.array(list(self.return_history)[-history_length:], dtype=float)
        mid_history = np.array(list(self.midprice_history)[-history_length:], dtype=float)

        prototype_observation = PrototypeObservation(
            time=int(observation.time),
            best_bid=float(observation.best_bid or 0),
            best_ask=float(observation.best_ask or 0),
            bid_depth=int(observation.bid_depth),
            ask_depth=int(observation.ask_depth),
            midprice=float(observation.midprice or 0),
            spread=float(observation.spread),
            imbalance=float(observation.imbalance),
            fundamental_value=float(self.last_trade.get(self.symbol, observation.midprice or 0)),
            midprice_history=mid_history / 100.0,
            return_history=return_history,
            tick_size=0.01,
        )
        current_inventory = self.inventory()
        return self.environment.build_state(
            observation=prototype_observation,
            inventory=current_inventory,
        )


class RLTrader(BaseRLTrader):
    """Legacy directional RL trader that only submits aggressive orders."""

    ACTION_LABELS = {
        0: "sell",
        1: "hold",
        2: "buy",
    }
    AGENT_ROLE = "taker"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        environment: RLMarketEnvironment,
        order_size: int,
        inventory_cap: int | None = None,
        policy: Optional[RLPolicy] = None,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            environment=environment,
            inventory_cap=inventory_cap,
            policy=policy,
            random_state=random_state,
            log_orders=log_orders,
        )
        self.order_size = max(1, int(order_size))

    def _submit_effective_action(self, effective_action: int, observation: ABIDESObservation) -> None:
        del observation
        if effective_action == 0:
            self._submit_aggressive_market_order(side=Side.ASK, quantity=self.order_size)
        elif effective_action == 2:
            self._submit_aggressive_market_order(side=Side.BID, quantity=self.order_size)


class RLQuotingTrader(BaseRLTrader):
    """Passive RL quoter with a small four-action quoting interface."""

    ACTION_LABELS = {
        0: "hold",
        1: "quote_bid",
        2: "quote_ask",
        3: "quote_both",
    }
    AGENT_ROLE = "quoter"

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        wake_up_freq: NanosecondTime,
        *,
        environment: RLMarketEnvironment,
        quote_size: int,
        inventory_cap: int | None = None,
        policy: Optional[RLPolicy] = None,
        quote_mode: str = "at_best",
        quote_offset_ticks: int = 0,
        enable_passive_quotes: bool = True,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            type=type,
            symbol=symbol,
            starting_cash=starting_cash,
            wake_up_freq=wake_up_freq,
            environment=environment,
            inventory_cap=inventory_cap,
            policy=policy if policy is not None else RandomQuoterPolicy(),
            random_state=random_state,
            max_live_passive_orders_per_side=1,
            passive_order_max_age=int(8 * 1_000_000_000),
            log_orders=log_orders,
        )
        normalized_quote_mode = str(quote_mode).strip().lower()
        if normalized_quote_mode not in {"at_best", "one_tick_inside"}:
            raise ValueError("quote_mode must be 'at_best' or 'one_tick_inside'.")
        self.quote_size = max(1, int(quote_size))
        self.quote_mode = normalized_quote_mode
        self.quote_offset_ticks = int(quote_offset_ticks)
        self.enable_passive_quotes = bool(enable_passive_quotes)

    def _quote_reward_eligible(self) -> bool:
        return bool(self.enable_passive_quotes)

    def _apply_inventory_cap(self, action: int, current_inventory: int) -> tuple[int, Dict[str, object]]:
        if self.inventory_cap is None:
            return action, {
                "blocked": False,
                "blocked_buy": False,
                "blocked_sell": False,
                "blocked_action_label": "none",
                "reason": "",
            }

        blocked_buy = current_inventory >= self.inventory_cap
        blocked_sell = current_inventory <= -self.inventory_cap
        effective_action = int(action)
        reason = ""
        if action == 1 and blocked_buy:
            effective_action = 0
            reason = "quote_bid_blocked_at_long_cap"
        elif action == 2 and blocked_sell:
            effective_action = 0
            reason = "quote_ask_blocked_at_short_cap"
        elif action == 3 and blocked_buy and not blocked_sell:
            effective_action = 2
            reason = "quote_bid_leg_blocked_at_long_cap"
        elif action == 3 and blocked_sell and not blocked_buy:
            effective_action = 1
            reason = "quote_ask_leg_blocked_at_short_cap"
        elif action == 3 and blocked_buy and blocked_sell:
            effective_action = 0
            reason = "quote_both_blocked_by_inventory_cap"
        return effective_action, {
            "blocked": bool(reason),
            "blocked_buy": bool(blocked_buy and action in {1, 3}),
            "blocked_sell": bool(blocked_sell and action in {2, 3}),
            "blocked_action_label": "quote" if bool(reason) else "none",
            "reason": reason,
        }

    def _desired_quote_price(
        self,
        observation: ABIDESObservation,
        *,
        side: Side,
    ) -> Optional[int]:
        base_offset = 1 if self.quote_mode == "one_tick_inside" else 0
        return self._near_touch_limit_price(
            observation,
            side=side,
            offset_ticks=base_offset + self.quote_offset_ticks,
        )

    def _refresh_side_quote(self, side: Side, observation: ABIDESObservation) -> None:
        desired_price = self._desired_quote_price(observation, side=side)
        if desired_price is None:
            self._cancel_side_orders(side)
            return
        if self._has_same_or_better_passive_order(side, desired_price):
            return
        self._cancel_side_orders(side)
        self._submit_passive_limit_order(
            side=side,
            quantity=self.quote_size,
            limit_price=int(desired_price),
            tag="rl_quote",
        )

    def _submit_effective_action(self, effective_action: int, observation: ABIDESObservation) -> None:
        if not self.enable_passive_quotes:
            self._cancel_side_orders(Side.BID)
            self._cancel_side_orders(Side.ASK)
            return

        if effective_action == 0:
            self._cancel_side_orders(Side.BID)
            self._cancel_side_orders(Side.ASK)
            return

        if effective_action == 1:
            self._cancel_side_orders(Side.ASK)
            self._refresh_side_quote(Side.BID, observation)
            return

        if effective_action == 2:
            self._cancel_side_orders(Side.BID)
            self._refresh_side_quote(Side.ASK, observation)
            return

        self._refresh_side_quote(Side.BID, observation)
        self._refresh_side_quote(Side.ASK, observation)
