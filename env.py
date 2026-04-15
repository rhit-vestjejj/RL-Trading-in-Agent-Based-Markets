"""RL-facing state, reward, and policy helpers for the prototype market."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np

from agents.base import MarketObservation


class RLPolicy(Protocol):
    """Minimal policy interface used by ``RLTrader``."""

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        """Return one of the discrete actions: 0=sell, 1=hold, 2=buy."""


class RandomPolicy:
    """Placeholder policy that samples actions uniformly."""

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        del state
        if hasattr(rng, "integers"):
            return int(rng.integers(0, 3))
        return int(rng.randint(0, 3))


class RandomQuoterPolicy:
    """Placeholder policy for the passive RL quoter action set."""

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        del state
        if hasattr(rng, "integers"):
            return int(rng.integers(0, 4))
        return int(rng.randint(0, 4))


class InventoryAwarePolicy:
    """Simple scripted placeholder that mean-reverts inventory."""

    def __init__(self, inventory_band: float = 1.0) -> None:
        self.inventory_band = float(inventory_band)

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        inventory = float(state[-1])
        if inventory > self.inventory_band:
            return 0
        if inventory < -self.inventory_band:
            return 2
        return 1


class InventoryAwareQuoterPolicy:
    """Simple scripted quoter that leans against inventory."""

    def __init__(self, inventory_band: float = 1.0) -> None:
        self.inventory_band = float(inventory_band)

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        del rng
        inventory = float(state[-1])
        if inventory > self.inventory_band:
            return 2
        if inventory < -self.inventory_band:
            return 1
        return 3


POLICY_BUILDERS: Dict[str, type] = {
    "random": RandomPolicy,
    "random_quoter": RandomQuoterPolicy,
    "inventory_aware": InventoryAwarePolicy,
    "inventory_aware_quoter": InventoryAwareQuoterPolicy,
}


def build_policy(policy_name: str) -> RLPolicy:
    """Instantiate a named placeholder policy."""

    try:
        policy_cls = POLICY_BUILDERS[policy_name]
    except KeyError as exc:
        available = ", ".join(sorted(POLICY_BUILDERS))
        raise ValueError(f"unknown rl_policy_name={policy_name!r}. Available: {available}") from exc
    return policy_cls()


@dataclass(frozen=True)
class RLMarketEnvironment:
    """Helper that builds RL states and computes rewards.

    This is not a full Gym environment. It only provides the compact state
    encoding and reward function needed for the first-pass market prototype.
    """

    return_window: int = 10
    lambda_q: float = 0.01
    currency_scale: float = 100.0
    flat_hold_penalty: float = 0.0
    passive_fill_reward: float = 0.0
    two_sided_quote_reward: float = 0.0
    missing_quote_penalty: float = 0.0

    def build_state(self, observation: MarketObservation, inventory: int) -> np.ndarray:
        """Return the fixed-length RL state vector.

        State layout:
        [recent_returns..., spread, imbalance, inventory]
        """

        recent_returns = np.zeros(self.return_window, dtype=float)
        observed_returns = observation.return_history[-self.return_window :]
        if len(observed_returns) > 0:
            recent_returns[-len(observed_returns) :] = observed_returns

        return np.concatenate(
            [
                recent_returns,
                np.array([observation.spread, observation.imbalance, float(inventory)], dtype=float),
            ]
        )

    def compute_reward(
        self,
        previous_cash: float,
        previous_inventory: int,
        previous_midprice: float,
        current_cash: float,
        current_inventory: int,
        current_midprice: float,
    ) -> float:
        """Compute the inventory-penalized marked-to-market reward."""

        return self.compute_reward_components(
            previous_cash=previous_cash,
            previous_inventory=previous_inventory,
            previous_midprice=previous_midprice,
            current_cash=current_cash,
            current_inventory=current_inventory,
            current_midprice=current_midprice,
        )["reward"]

    def compute_reward_components(
        self,
        previous_cash: float,
        previous_inventory: int,
        previous_midprice: float,
        current_cash: float,
        current_inventory: int,
        current_midprice: float,
        previous_action: int | None = None,
        passive_fill_quantity: float = 0.0,
        has_resting_bid: bool = False,
        has_resting_ask: bool = False,
        quote_reward_eligible: bool = False,
    ) -> dict[str, float]:
        """Return reward components in the simulator's native currency units.

        ABIDES cash and prices are represented in cents. The agreed reward uses
        ``lambda_q`` in dollar terms, so the inventory penalty must be scaled
        into cents to stay consistent with the wealth-delta units here.
        """

        wealth_t = previous_cash + previous_inventory * previous_midprice
        wealth_t_plus_1 = current_cash + current_inventory * current_midprice
        wealth_delta = wealth_t_plus_1 - wealth_t
        inventory_penalty = self.lambda_q * self.currency_scale * float(current_inventory**2)
        flat_hold_penalty = (
            float(self.flat_hold_penalty) * self.currency_scale
            if previous_action == 1 and int(current_inventory) == 0
            else 0.0
        )
        passive_fill_bonus = (
            float(self.passive_fill_reward)
            * self.currency_scale
            * max(float(passive_fill_quantity), 0.0)
        )
        two_sided_quote_reward = (
            float(self.two_sided_quote_reward) * self.currency_scale
            if quote_reward_eligible and bool(has_resting_bid) and bool(has_resting_ask)
            else 0.0
        )
        missing_quote_penalty = (
            float(self.missing_quote_penalty) * self.currency_scale
            if quote_reward_eligible and not (bool(has_resting_bid) and bool(has_resting_ask))
            else 0.0
        )
        reward = (
            wealth_delta
            - inventory_penalty
            - flat_hold_penalty
            + passive_fill_bonus
            + two_sided_quote_reward
            - missing_quote_penalty
        )
        return {
            "wealth_delta": float(wealth_delta),
            "inventory_penalty": float(inventory_penalty),
            "flat_hold_penalty": float(flat_hold_penalty),
            "passive_fill_bonus": float(passive_fill_bonus),
            "two_sided_quote_reward": float(two_sided_quote_reward),
            "missing_quote_penalty": float(missing_quote_penalty),
            "reward": float(reward),
        }
