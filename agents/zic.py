"""Zero-intelligence constrained trader implementation."""

from __future__ import annotations

from typing import Dict, List

from agents.base import BaseTrader, MarketObservation, OrderIntent


class ZICTrader(BaseTrader):
    """Trader that quotes around a noisy private valuation."""

    def __init__(
        self,
        agent_id: str,
        rng,
        sigma_eta: float,
        surplus_min_ticks: int,
        surplus_max_ticks: int,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.sigma_eta = sigma_eta
        self.surplus_min_ticks = surplus_min_ticks
        self.surplus_max_ticks = surplus_max_ticks

    def observe(self, observation: MarketObservation) -> Dict[str, float | MarketObservation]:
        private_value = observation.fundamental_value + self.rng.normal(0.0, self.sigma_eta)
        return {"market": observation, "v_hat": private_value}

    def decide(self, observation: Dict[str, float | MarketObservation]) -> List[OrderIntent]:
        market = observation["market"]
        if not isinstance(market, MarketObservation):
            return []

        side = "buy" if self.rng.random() < 0.5 else "sell"
        quantity = int(self.rng.integers(1, 4))
        surplus_ticks = self.rng.uniform(self.surplus_min_ticks, self.surplus_max_ticks)
        surplus = surplus_ticks * market.tick_size
        private_value = float(observation["v_hat"])
        price = private_value - surplus if side == "buy" else private_value + surplus
        price = max(price, market.tick_size)
        return [OrderIntent(side=side, quantity=quantity, order_type="limit", price=price)]
