"""Shared agent interfaces and observation structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from lob import OrderRequest, OrderType, Side


@dataclass(frozen=True)
class MarketObservation:
    """State snapshot shared with agents at each decision point."""

    time: int
    best_bid: float
    best_ask: float
    bid_depth: int
    ask_depth: int
    midprice: float
    spread: float
    imbalance: float
    fundamental_value: float
    midprice_history: np.ndarray
    return_history: np.ndarray
    tick_size: float


@dataclass(frozen=True)
class OrderIntent:
    """Agent decision that can be turned into a book order."""

    side: Side
    quantity: int
    order_type: OrderType
    price: Optional[float] = None

    def to_request(self, agent_id: str, timestamp: int) -> OrderRequest:
        """Convert the intent into an order request for the order book."""

        return OrderRequest(
            agent_id=agent_id,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            price=self.price,
            timestamp=timestamp,
        )


class BaseTrader:
    """Base class for all traders in the prototype market."""

    def __init__(self, agent_id: str, rng: np.random.Generator) -> None:
        self.agent_id = agent_id
        self.rng = rng
        self.cash = 0.0
        self.inventory = 0

    def observe(self, observation: MarketObservation) -> Any:
        """Return the observation or a trader-specific derived signal."""

        return observation

    def decide(self, observation: Any) -> List[OrderIntent]:
        """Return a list of desired orders."""

        raise NotImplementedError

    def submit_orders(self, observation: MarketObservation) -> List[OrderIntent]:
        """Convenience wrapper that runs the observe/decide pipeline."""

        interpreted_observation = self.observe(observation)
        return self.decide(interpreted_observation)

    def wealth(self, midprice: float) -> float:
        """Return marked-to-market wealth at the provided midprice."""

        return self.cash + self.inventory * midprice
