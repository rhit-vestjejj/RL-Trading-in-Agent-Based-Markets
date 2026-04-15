"""Noise trader implementation."""

from __future__ import annotations

from typing import List

from agents.base import BaseTrader, MarketObservation, OrderIntent


class NoiseTrader(BaseTrader):
    """Background random order-flow trader."""

    def __init__(
        self,
        agent_id: str,
        rng,
        limit_order_probability: float = 0.5,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.limit_order_probability = limit_order_probability

    def decide(self, observation: MarketObservation) -> List[OrderIntent]:
        side = "buy" if self.rng.random() < 0.5 else "sell"
        quantity = int(self.rng.integers(1, 4))

        if self.rng.random() < self.limit_order_probability:
            if side == "buy":
                price = min(observation.best_ask, observation.best_bid + observation.tick_size)
            else:
                price = max(observation.best_bid, observation.best_ask - observation.tick_size)
            return [OrderIntent(side=side, quantity=quantity, order_type="limit", price=price)]

        return [OrderIntent(side=side, quantity=quantity, order_type="market")]
