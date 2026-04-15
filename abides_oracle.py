"""ABIDES oracle implementing the prototype's latent random-walk fundamental."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from abides_support import bootstrap_abides_paths

bootstrap_abides_paths()

from abides_core import NanosecondTime
from abides_markets.oracles.oracle import Oracle


class RandomWalkOracle(Oracle):
    """Latent fundamental process with Gaussian random-walk innovations.

    This matches the prototype requirement more closely than ABIDES' default
    mean-reverting oracle: for each symbol, the fundamental evolves as

    ``v_{t+1} = v_t + epsilon_t`` with ``epsilon_t ~ Normal(0, sigma_v^2)``.

    The process evolves on a fixed simulator clock so its scale depends on
    elapsed simulator time rather than on how frequently agents query the
    oracle. That keeps the latent path comparable across different market
    ecologies and wake-up rates.
    """

    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: Dict[str, Dict[str, Any]],
    ) -> None:
        self.mkt_open = mkt_open
        self.mkt_close = mkt_close
        self.symbols = symbols
        self.f_log: Dict[str, List[Dict[str, int]]] = {}
        self._state: Dict[str, Tuple[int, int]] = {}
        self._update_interval: Dict[str, NanosecondTime] = {}

        for symbol, parameters in symbols.items():
            initial_value = int(parameters["v0"])
            update_interval = max(1, int(parameters.get("fundamental_interval_ns", 1_000_000_000)))
            self._state[symbol] = (0, initial_value)
            self._update_interval[symbol] = update_interval
            self.f_log[symbol] = [
                {"FundamentalTime": mkt_open, "FundamentalValue": initial_value}
            ]

    def get_daily_open_price(
        self,
        symbol: str,
        mkt_open: NanosecondTime,
        cents: bool = True,
    ) -> int:
        del mkt_open, cents
        return int(self.symbols[symbol]["v0"])

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: float = 0.0,
    ) -> int:
        value = self.advance_fundamental_value_series(current_time=current_time, symbol=symbol)
        noisy_observation = value + random_state.normal(loc=0.0, scale=sigma_n)
        return max(1, int(round(noisy_observation)))

    def advance_fundamental_value_series(
        self,
        current_time: NanosecondTime,
        symbol: str,
    ) -> int:
        clamped_time = min(max(current_time, self.mkt_open), self.mkt_close)
        update_interval = self._update_interval[symbol]
        current_step = int((clamped_time - self.mkt_open) // update_interval)

        last_step, last_value = self._state[symbol]
        if current_step <= last_step:
            return last_value

        parameters = self.symbols[symbol]
        random_state = parameters["random_state"]
        sigma_v = float(parameters["sigma_v"])
        next_value = last_value

        for step in range(last_step + 1, current_step + 1):
            innovation = random_state.normal(loc=0.0, scale=sigma_v)
            next_value = max(1, int(round(next_value + innovation)))
            step_time = min(self.mkt_open + step * update_interval, self.mkt_close)
            self.f_log[symbol].append(
                {"FundamentalTime": step_time, "FundamentalValue": next_value}
            )

        self._state[symbol] = (current_step, next_value)
        return next_value
