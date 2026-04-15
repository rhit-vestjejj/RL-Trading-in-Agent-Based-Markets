"""ABIDES-backed market simulator for the research prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from abides_agents import (
    AdaptiveMarketMaker,
    NoiseTrader,
    RLQuotingTrader,
    RLTrader,
    TrendFollowerTrader,
    ValueTrader,
    ZICTrader,
)
from abides_oracle import RandomWalkOracle
from abides_support import bootstrap_abides_paths
from config import SimulationConfig
from env import RLMarketEnvironment, RLPolicy, build_policy
from logging_utils import (
    extract_abides_dataframe,
    extract_rl_decision_dataframe,
    extract_rl_transition_dataframe,
)

bootstrap_abides_paths()

from abides_core import Kernel
from abides_core.utils import str_to_ns, subdict
from abides_markets.agents import ExchangeAgent
from abides_markets.utils import generate_latency_model


@dataclass(frozen=True)
class ABIDESConfigState:
    """Kernel configuration plus a few experiment-specific metadata values."""

    kernel_config: Dict[str, object]
    ticker: str


class MarketSimulator:
    """Run the market experiment on top of the ABIDES kernel."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        rl_policy_factory: Optional[
            Callable[..., RLPolicy]
        ] = None,
    ) -> None:
        self.config = config
        self.rl_policy_factory = rl_policy_factory
        self.environment = RLMarketEnvironment(
            return_window=config.return_window,
            lambda_q=config.lambda_q,
            flat_hold_penalty=config.flat_hold_penalty,
            passive_fill_reward=config.rl_passive_fill_reward,
            two_sided_quote_reward=config.rl_two_sided_quote_reward,
            missing_quote_penalty=config.rl_missing_quote_penalty,
        )
        self.config_state = self._build_config_state()
        self.end_state: Dict[str, object] | None = None
        self.frame: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """Run the ABIDES simulation and return the extracted market log."""

        kernel_seed = np.random.RandomState(seed=self.config.seed + 1)
        kernel = Kernel(
            random_state=kernel_seed,
            skip_log=True,
            **subdict(
                self.config_state.kernel_config,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
        )
        self.end_state = kernel.run()
        self.frame = self.extract_frame(self.config.log_frequency)
        return self.frame

    def to_csv(self, path: str, index: bool = False) -> None:
        """Export the most recent simulation log to CSV."""

        if self.frame is None:
            raise RuntimeError("Simulation has not been run yet.")
        self.frame.to_csv(path, index=index)

    def extract_frame(self, log_frequency: str) -> pd.DataFrame:
        """Extract a DataFrame at a requested logging frequency from the same run."""

        if self.end_state is None:
            raise RuntimeError("Simulation has not been run yet.")
        return extract_abides_dataframe(
            end_state=self.end_state,
            ticker=self.config_state.ticker,
            log_frequency=log_frequency,
        )

    def extract_frames(self, log_frequencies: Sequence[str]) -> Dict[str, pd.DataFrame]:
        """Extract multiple logging resolutions from the same completed run."""

        return {
            frequency: self.extract_frame(frequency)
            for frequency in log_frequencies
        }

    def extract_rl_frame(self) -> pd.DataFrame:
        """Extract the long-form RL decision log from the same completed run."""

        if self.end_state is None:
            raise RuntimeError("Simulation has not been run yet.")
        return extract_rl_decision_dataframe(self.end_state)

    def extract_rl_transition_frame(self) -> pd.DataFrame:
        """Extract the long-form RL transition log from the same completed run."""

        if self.end_state is None:
            raise RuntimeError("Simulation has not been run yet.")
        return extract_rl_transition_dataframe(self.end_state)

    def _build_config_state(self) -> ABIDESConfigState:
        rng = np.random.RandomState(seed=self.config.seed)
        profile = self.config.profile()
        counts = self._adjust_agent_counts(self.config.agent_counts())
        rl_role_counts = self.config.rl_role_counts()

        date_ns = int(pd.to_datetime(self.config.date).to_datetime64())
        market_open = date_ns + str_to_ns(self.config.market_open_time)
        market_close = date_ns + str_to_ns(self.config.end_time)
        tick_size_cents = max(1, int(round(self.config.tick_size * 100)))

        symbol_config = {
            self.config.ticker: {
                "v0": int(round(self.config.initial_fundamental * 100)),
                "sigma_v": float(self.config.sigma_v * 100),
                "fundamental_interval_ns": str_to_ns(self.config.fundamental_update_frequency),
                "random_state": np.random.RandomState(
                    seed=int(rng.randint(low=0, high=2**32 - 1))
                ),
            }
        }
        oracle = RandomWalkOracle(
            mkt_open=market_open,
            mkt_close=market_close,
            symbols=symbol_config,
        )

        agents: List[object] = [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=market_open,
                mkt_close=market_close,
                symbols=[self.config.ticker],
                book_logging=True,
                book_log_depth=self.config.book_log_depth,
                log_orders=False,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=500,
                random_state=self._spawn_random_state(rng),
            )
        ]

        next_id = 1

        noise_wake_ns = str_to_ns(self.config.noise_wake_up_frequency)
        zic_wake_ns = str_to_ns(self.config.zic_wake_up_frequency)
        trend_wake_ns = str_to_ns(self.config.trend_wake_up_frequency)
        value_wake_ns = str_to_ns(self.config.value_wake_up_frequency)
        mm_wake_ns = str_to_ns(self.config.mm_wake_up_frequency)
        rl_wake_ns = str_to_ns(self.config.rl_wake_up_frequency)

        def sample_wake_frequency(
            base_frequency_ns: int,
            *,
            low_scale: float = 0.8,
            high_scale: float = 1.25,
        ) -> int:
            scale = float(rng.uniform(low=low_scale, high=high_scale))
            return max(1, int(round(base_frequency_ns * scale)))

        def extend_agents(
            count: int,
            builder: Callable[[int], object],
        ) -> None:
            nonlocal next_id
            for _ in range(count):
                agents.append(builder(next_id))
                next_id += 1

        extend_agents(
            counts["noise"],
            lambda agent_id: NoiseTrader(
                id=agent_id,
                name=f"NoiseTrader_{agent_id}",
                type="NoiseTrader",
                symbol=self.config.ticker,
                starting_cash=self.config.starting_cash,
                wake_up_freq=sample_wake_frequency(noise_wake_ns),
                limit_order_probability=self.config.noise_limit_probability,
                random_state=self._spawn_random_state(rng),
                log_orders=False,
            ),
        )
        extend_agents(
            counts["zic"],
            lambda agent_id: ZICTrader(
                id=agent_id,
                name=f"ZICTrader_{agent_id}",
                type="ZICTrader",
                symbol=self.config.ticker,
                starting_cash=self.config.starting_cash,
                wake_up_freq=sample_wake_frequency(zic_wake_ns),
                sigma_eta=self.config.zic_sigma_eta * 100,
                surplus_min_ticks=self.config.zic_surplus_min_ticks * tick_size_cents,
                surplus_max_ticks=self.config.zic_surplus_max_ticks * tick_size_cents,
                random_state=self._spawn_random_state(rng),
                log_orders=False,
            ),
        )
        if counts["trend"] > 0:
            extend_agents(
                counts["trend"],
                lambda agent_id: TrendFollowerTrader(
                    id=agent_id,
                    name=f"TrendFollowerTrader_{agent_id}",
                    type="TrendFollowerTrader",
                    symbol=self.config.ticker,
                    starting_cash=self.config.starting_cash,
                    wake_up_freq=sample_wake_frequency(trend_wake_ns),
                    short_window=self.config.trend_short_window,
                    long_window=self.config.trend_long_window,
                    signal_noise=self.config.trend_signal_noise_ticks * tick_size_cents,
                    signal_threshold=self.config.trend_signal_threshold_ticks * tick_size_cents,
                    trade_probability=self.config.trend_trade_probability,
                    aggressive_probability=self.config.trend_aggressive_probability,
                    aggressive_order_size=self.config.trend_order_size,
                    passive_order_size=self.config.trend_passive_order_size,
                    random_state=self._spawn_random_state(rng),
                    log_orders=False,
                ),
            )
        extend_agents(
            counts["value"],
            lambda agent_id: ValueTrader(
                id=agent_id,
                name=f"ValueTrader_{agent_id}",
                type="ValueTrader",
                symbol=self.config.ticker,
                starting_cash=self.config.starting_cash,
                wake_up_freq=sample_wake_frequency(value_wake_ns),
                sigma_eta=self.config.value_sigma_eta * 100,
                delta=self.config.value_delta_ticks * tick_size_cents,
                aggressive_probability=self.config.value_aggressive_probability,
                aggressive_order_size=self.config.value_order_size,
                passive_order_size=self.config.value_passive_order_size,
                random_state=self._spawn_random_state(rng),
                log_orders=False,
            ),
        )
        extend_agents(
            counts["market_maker"],
            lambda agent_id: (
                lambda mm_wake_freq: AdaptiveMarketMaker(
                    id=agent_id,
                    name=f"AdaptiveMarketMaker_{agent_id}",
                    type="AdaptiveMarketMaker",
                    symbol=self.config.ticker,
                    starting_cash=self.config.starting_cash,
                    wake_up_freq=mm_wake_freq,
                    wake_up_jitter=0,
                    min_refresh_interval=int(round(rng.uniform(1.0, 3.0) * str_to_ns("1s"))),
                    target_spread=max(
                        1,
                        int(
                            round(
                                rng.uniform(
                                    low=max(1.5, float(self.config.mm_spread_ticks) - 1.0),
                                    high=float(self.config.mm_spread_ticks),
                                )
                                * tick_size_cents
                            )
                        ),
                    ),
                    alpha_cents=float(rng.uniform(0.002, 0.01) * 100),
                    quote_size=max(
                        2,
                        int(
                            round(
                                rng.lognormal(
                                    mean=float(np.log(max(10, self.config.mm_quote_size))),
                                    sigma=0.30,
                                )
                            )
                        ),
                    ),
                    reprice_threshold=max(1, int(round(rng.uniform(1.0, 2.0) * tick_size_cents))),
                    random_state=self._spawn_random_state(rng),
                    log_orders=False,
                )
            )(int(round(rng.uniform(0.5, 3.0) * str_to_ns("1s")))),
        )
        extend_agents(
            rl_role_counts["taker"],
            lambda agent_id: RLTrader(
                id=agent_id,
                name=f"RLTrader_{agent_id}",
                type="RLTrader",
                symbol=self.config.ticker,
                starting_cash=self.config.starting_cash,
                wake_up_freq=sample_wake_frequency(rl_wake_ns),
                environment=self.environment,
                order_size=self.config.rl_order_size,
                inventory_cap=self.config.inventory_cap,
                policy=self._build_rl_policy(agent_id, rng, role="taker"),
                random_state=self._spawn_random_state(rng),
                log_orders=False,
            ),
        )
        extend_agents(
            rl_role_counts["quoter"],
            lambda agent_id: RLQuotingTrader(
                id=agent_id,
                name=f"RLQuotingTrader_{agent_id}",
                type="RLQuotingTrader",
                symbol=self.config.ticker,
                starting_cash=self.config.starting_cash,
                wake_up_freq=sample_wake_frequency(rl_wake_ns),
                environment=self.environment,
                quote_size=self.config.rl_quote_size,
                inventory_cap=self.config.inventory_cap,
                policy=self._build_rl_policy(agent_id, rng, role="quoter"),
                quote_mode=self.config.rl_quote_mode,
                quote_offset_ticks=self.config.rl_quote_offset_ticks,
                enable_passive_quotes=self.config.rl_enable_passive_quotes,
                random_state=self._spawn_random_state(rng),
                log_orders=False,
            ),
        )

        kernel_config = {
            "start_time": date_ns,
            "stop_time": market_close + str_to_ns("1s"),
            "agents": agents,
            "agent_latency_model": generate_latency_model(
                len(agents),
                latency_type=self.config.latency_type,
            ),
            "default_computation_delay": self.config.default_computation_delay,
            "custom_properties": {
                "oracle": oracle,
                "market_profile": profile.name,
                "market_profile_description": profile.description,
                "rl_liquidity_mode": self.config.rl_liquidity_mode,
            },
            "stdout_log_level": self.config.stdout_log_level,
        }
        return ABIDESConfigState(kernel_config=kernel_config, ticker=self.config.ticker)

    @staticmethod
    def _spawn_random_state(rng: np.random.RandomState) -> np.random.RandomState:
        return np.random.RandomState(seed=int(rng.randint(low=0, high=2**32 - 1)))

    def _build_rl_policy(
        self,
        agent_id: int,
        rng: np.random.RandomState,
        *,
        role: str,
    ) -> RLPolicy:
        if self.rl_policy_factory is not None:
            policy_rng = self._spawn_random_state(rng)
            try:
                return self.rl_policy_factory(agent_id, policy_rng, role)
            except TypeError:
                return self.rl_policy_factory(agent_id, policy_rng)
        policy_name = (
            self.config.rl_policy_name
            if role == "taker"
            else self.config.rl_quoter_policy_name
        )
        return build_policy(policy_name)

    def _adjust_agent_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        adjusted = dict(counts)
        target_mm_count = max(
            0,
            int(round(adjusted["market_maker"] * float(self.config.market_maker_count_scale))),
        )
        delta = target_mm_count - adjusted["market_maker"]
        if delta == 0:
            return adjusted

        adjusted["market_maker"] = target_mm_count
        adjusted["zic"] = max(0, adjusted["zic"] - delta)

        imbalance = self.config.num_agents - sum(adjusted.values())
        if imbalance != 0:
            adjusted["zic"] = max(0, adjusted["zic"] + imbalance)
        return adjusted
