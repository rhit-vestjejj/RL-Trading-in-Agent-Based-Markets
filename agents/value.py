"""Value trader implementation."""

from __future__ import annotations

from typing import Dict, List

from agents.base import BaseTrader, MarketObservation, OrderIntent


class ValueTrader(BaseTrader):
    """Trader that buys or sells when price deviates from private value."""

    def __init__(
        self,
        agent_id: str,
        rng,
        sigma_eta: float,
        delta: float,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.sigma_eta = sigma_eta
        self.delta = delta

    def observe(self, observation: MarketObservation) -> Dict[str, float | MarketObservation]:
        private_value = observation.fundamental_value + self.rng.normal(0.0, self.sigma_eta)
        return {"market": observation, "v_hat": private_value}

    def decide(self, observation: Dict[str, float | MarketObservation]) -> List[OrderIntent]:
        market = observation["market"]
        if not isinstance(market, MarketObservation):
            return []

        mispricing = float(observation["v_hat"]) - market.midprice
        if mispricing > self.delta:
            quantity = int(self.rng.integers(1, 3))
            return [OrderIntent(side="buy", quantity=quantity, order_type="market")]
        if mispricing < -self.delta:
            quantity = int(self.rng.integers(1, 3))
            return [OrderIntent(side="sell", quantity=quantity, order_type="market")]
        return []
