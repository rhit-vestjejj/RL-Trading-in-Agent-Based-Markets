"""RL-compatible trader implementation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from agents.base import BaseTrader, MarketObservation, OrderIntent
from env import RLMarketEnvironment, RLPolicy, RandomPolicy


class RLTrader(BaseTrader):
    """Trader with a compact RL-facing state and discrete action space."""

    def __init__(
        self,
        agent_id: str,
        rng: np.random.Generator,
        environment: RLMarketEnvironment,
        order_size: int = 1,
        policy: Optional[RLPolicy] = None,
    ) -> None:
        super().__init__(agent_id=agent_id, rng=rng)
        self.environment = environment
        self.order_size = order_size
        self.policy = policy if policy is not None else RandomPolicy()
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def observe(self, observation: MarketObservation) -> np.ndarray:
        state = self.environment.build_state(observation=observation, inventory=self.inventory)
        self.last_state = state
        return state

    def decide(self, observation: np.ndarray) -> List[OrderIntent]:
        action = int(self.policy.act(observation, self.rng))
        self.last_action = action

        if action == 0:
            return [OrderIntent(side="sell", quantity=self.order_size, order_type="market")]
        if action == 2:
            return [OrderIntent(side="buy", quantity=self.order_size, order_type="market")]
        return []
