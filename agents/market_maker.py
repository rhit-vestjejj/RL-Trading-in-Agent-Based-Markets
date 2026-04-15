"""Adaptive market maker implementation."""

from __future__ import annotations

from typing import List

from agents.base import BaseTrader, MarketObservation, OrderIntent


class AdaptiveMarketMaker(BaseTrader):
    """Two-sided liquidity provider with an inventory skew."""

    def __init__(
        self,
        agent_id: str,
        rng,
        target_spread: float,
        alpha: float,
        quote_size: int,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.target_spread = target_spread
        self.alpha = alpha
        self.quote_size = quote_size

    def decide(self, observation: MarketObservation) -> List[OrderIntent]:
        half_spread = self.target_spread / 2.0
        inventory_skew = self.alpha * self.inventory
        bid_price = observation.midprice - half_spread - inventory_skew
        ask_price = observation.midprice + half_spread - inventory_skew

        bid_price = max(bid_price, observation.tick_size)
        ask_price = max(ask_price, bid_price + observation.tick_size)

        return [
            OrderIntent(side="buy", quantity=self.quote_size, order_type="limit", price=bid_price),
            OrderIntent(side="sell", quantity=self.quote_size, order_type="limit", price=ask_price),
        ]
