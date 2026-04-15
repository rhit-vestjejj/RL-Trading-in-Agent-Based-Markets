"""A lightweight limit order book used by the research prototype."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Deque, Dict, List, Literal, Mapping, Optional


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


def safe_imbalance(bid_depth: int, ask_depth: int) -> float:
    """Return the top-of-book imbalance with zero-division protection."""

    total_depth = bid_depth + ask_depth
    if total_depth <= 0:
        return 0.0
    return (bid_depth - ask_depth) / total_depth


@dataclass(frozen=True)
class TopOfBook:
    """Top-of-book state used by the agents, logger, and analysis code."""

    best_bid: float
    best_ask: float
    bid_depth: int
    ask_depth: int
    midprice: float
    spread: float
    imbalance: float


@dataclass(frozen=True)
class OrderRequest:
    """Incoming order submitted by an agent."""

    agent_id: str
    side: Side
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    timestamp: int = 0


@dataclass
class RestingOrder:
    """Order resting in the book until fully matched or cancelled."""

    order_id: int
    agent_id: str
    side: Side
    quantity: int
    price: float
    timestamp: int


@dataclass(frozen=True)
class Trade:
    """Executed trade between an aggressor and a resting order."""

    timestamp: int
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    aggressor_side: Side


class LimitOrderBook:
    """Simple price-time-priority order book for a single asset.

    This is intentionally lightweight. It only models the top of book, order
    matching, resting limit orders, and order cancellation by agent. The class
    is kept separate from the agent and environment logic so it can later be
    swapped for ABIDES infrastructure with minimal changes.
    """

    def __init__(self, tick_size: float, initial_price: float) -> None:
        self.tick_size = tick_size
        self.initial_price = initial_price
        self.last_trade_price = initial_price
        self.last_midprice = initial_price
        self._next_order_id = 1
        self._bids: Dict[float, Deque[RestingOrder]] = defaultdict(deque)
        self._asks: Dict[float, Deque[RestingOrder]] = defaultdict(deque)
        self._order_lookup: Dict[int, RestingOrder] = {}
        self._agent_orders: Dict[str, set[int]] = defaultdict(set)

    def round_to_tick(self, price: float) -> float:
        """Round a price to the configured tick size."""

        rounded = round(price / self.tick_size) * self.tick_size
        return round(rounded, 10)

    def top_of_book(self) -> TopOfBook:
        """Return the current top-of-book state.

        If one side is empty, the method falls back to the last known midprice
        to keep the market state well-defined for the prototype.
        """

        best_bid = max(self._bids) if self._bids else round(self.last_midprice - self.tick_size, 10)
        best_ask = min(self._asks) if self._asks else round(self.last_midprice + self.tick_size, 10)
        bid_depth = sum(order.quantity for order in self._bids.get(best_bid, ()))
        ask_depth = sum(order.quantity for order in self._asks.get(best_ask, ()))
        midprice = (best_bid + best_ask) / 2.0
        spread = max(best_ask - best_bid, 0.0)
        self.last_midprice = midprice

        return TopOfBook(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            midprice=midprice,
            spread=spread,
            imbalance=safe_imbalance(bid_depth, ask_depth),
        )

    def cancel_agent_orders(self, agent_id: str) -> None:
        """Cancel every currently resting order for ``agent_id``."""

        for order_id in list(self._agent_orders.get(agent_id, set())):
            order = self._order_lookup.pop(order_id, None)
            if order is None:
                continue
            book = self._bids if order.side == "buy" else self._asks
            queue = book.get(order.price)
            if queue is None:
                continue
            try:
                queue.remove(order)
            except ValueError:
                pass
            if not queue:
                book.pop(order.price, None)
        self._agent_orders.pop(agent_id, None)

    def process_order(
        self,
        request: OrderRequest,
        agents: Mapping[str, object],
    ) -> List[Trade]:
        """Match an order and optionally leave unfilled volume resting in the book."""

        if request.quantity <= 0:
            return []
        if request.order_type == "limit" and request.price is None:
            raise ValueError("Limit orders require a price.")

        remaining = request.quantity
        price = self.round_to_tick(request.price) if request.price is not None else None
        normalized_request = replace(request, price=price)
        trades: List[Trade] = []

        if normalized_request.side == "buy":
            trades, remaining = self._match_buy_order(normalized_request, remaining, agents)
        else:
            trades, remaining = self._match_sell_order(normalized_request, remaining, agents)

        if remaining > 0 and normalized_request.order_type == "limit" and normalized_request.price is not None:
            self._add_resting_order(
                agent_id=normalized_request.agent_id,
                side=normalized_request.side,
                quantity=remaining,
                price=normalized_request.price,
                timestamp=normalized_request.timestamp,
            )

        return trades

    def _match_buy_order(
        self,
        request: OrderRequest,
        remaining: int,
        agents: Mapping[str, object],
    ) -> tuple[List[Trade], int]:
        trades: List[Trade] = []

        while remaining > 0 and self._asks:
            best_ask = min(self._asks)
            if request.order_type == "limit" and request.price is not None and request.price < best_ask:
                break

            queue = self._asks[best_ask]
            while queue and remaining > 0:
                resting = queue[0]
                quantity = min(remaining, resting.quantity)
                trade_price = resting.price
                self._apply_fill(
                    buyer=agents[request.agent_id],
                    seller=agents[resting.agent_id],
                    price=trade_price,
                    quantity=quantity,
                )
                trades.append(
                    Trade(
                        timestamp=request.timestamp,
                        price=trade_price,
                        quantity=quantity,
                        buyer_id=request.agent_id,
                        seller_id=resting.agent_id,
                        aggressor_side="buy",
                    )
                )
                resting.quantity -= quantity
                remaining -= quantity
                self.last_trade_price = trade_price
                if resting.quantity == 0:
                    queue.popleft()
                    self._remove_resting_order(resting)

            if not queue:
                self._asks.pop(best_ask, None)

        return trades, remaining

    def _match_sell_order(
        self,
        request: OrderRequest,
        remaining: int,
        agents: Mapping[str, object],
    ) -> tuple[List[Trade], int]:
        trades: List[Trade] = []

        while remaining > 0 and self._bids:
            best_bid = max(self._bids)
            if request.order_type == "limit" and request.price is not None and request.price > best_bid:
                break

            queue = self._bids[best_bid]
            while queue and remaining > 0:
                resting = queue[0]
                quantity = min(remaining, resting.quantity)
                trade_price = resting.price
                self._apply_fill(
                    buyer=agents[resting.agent_id],
                    seller=agents[request.agent_id],
                    price=trade_price,
                    quantity=quantity,
                )
                trades.append(
                    Trade(
                        timestamp=request.timestamp,
                        price=trade_price,
                        quantity=quantity,
                        buyer_id=resting.agent_id,
                        seller_id=request.agent_id,
                        aggressor_side="sell",
                    )
                )
                resting.quantity -= quantity
                remaining -= quantity
                self.last_trade_price = trade_price
                if resting.quantity == 0:
                    queue.popleft()
                    self._remove_resting_order(resting)

            if not queue:
                self._bids.pop(best_bid, None)

        return trades, remaining

    def _add_resting_order(
        self,
        agent_id: str,
        side: Side,
        quantity: int,
        price: float,
        timestamp: int,
    ) -> None:
        resting = RestingOrder(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )
        self._next_order_id += 1
        book = self._bids if side == "buy" else self._asks
        book[price].append(resting)
        self._order_lookup[resting.order_id] = resting
        self._agent_orders[agent_id].add(resting.order_id)

    def _remove_resting_order(self, order: RestingOrder) -> None:
        self._order_lookup.pop(order.order_id, None)
        agent_orders = self._agent_orders.get(order.agent_id)
        if agent_orders is not None:
            agent_orders.discard(order.order_id)
            if not agent_orders:
                self._agent_orders.pop(order.agent_id, None)

    @staticmethod
    def _apply_fill(buyer: object, seller: object, price: float, quantity: int) -> None:
        buyer.cash -= price * quantity
        buyer.inventory += quantity
        seller.cash += price * quantity
        seller.inventory -= quantity
