"""Trend-following trader implementation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from agents.base import BaseTrader, MarketObservation, OrderIntent


class TrendFollowerTrader(BaseTrader):
    """Trader that follows a simple moving-average crossover signal."""

    def __init__(
        self,
        agent_id: str,
        rng,
        short_window: int,
        long_window: int,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.short_window = short_window
        self.long_window = long_window

    def observe(self, observation: MarketObservation) -> Dict[str, float | MarketObservation]:
        if len(observation.midprice_history) < self.long_window:
            signal = 0.0
        else:
            short_ma = float(np.mean(observation.midprice_history[-self.short_window :]))
            long_ma = float(np.mean(observation.midprice_history[-self.long_window :]))
            signal = short_ma - long_ma
        return {"market": observation, "signal": signal}

    def decide(self, observation: Dict[str, float | MarketObservation]) -> List[OrderIntent]:
        market = observation["market"]
        if not isinstance(market, MarketObservation):
            return []

        signal = float(observation["signal"])
        if signal > 0:
            return [OrderIntent(side="buy", quantity=1, order_type="market")]
        if signal < 0:
            return [OrderIntent(side="sell", quantity=1, order_type="market")]
        return []
