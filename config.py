"""Configuration objects and agent-count utilities for the prototype market."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Dict, Mapping


@dataclass(frozen=True)
class MarketProfile:
    """Simple named market ecology used to build agent counts."""

    name: str
    fractions: Mapping[str, float]
    replaceable_fractions: Mapping[str, float]
    description: str

    @property
    def max_phi(self) -> float:
        return float(sum(self.replaceable_fractions.values()))


MARKET_PROFILES: Dict[str, MarketProfile] = {
    "abides_rmsc04_small_v1": MarketProfile(
        name="abides_rmsc04_small_v1",
        fractions={
            "zic": 0.00,
            "trend": 0.00,
            "noise": 80.0 / 102.0,
            "value": 20.0 / 102.0,
            "market_maker": 2.0 / 102.0,
        },
        replaceable_fractions={
            "noise": 80.0 / 102.0,
        },
        description=(
            "Official small ABIDES-style RMSC04 baseline: 80 noise, 20 value, and 2 market makers."
        ),
    ),
    "abides_baseline_v1": MarketProfile(
        name="abides_baseline_v1",
        fractions={
            "zic": 0.50,
            "trend": 0.00,
            "noise": 0.15,
            "value": 0.20,
            "market_maker": 0.15,
        },
        replaceable_fractions={
            "zic": 0.50,
            "noise": 0.15,
        },
        description=(
            "Older simple ABIDES-style baseline kept for reference: ZI/noise flow, value traders, and market makers."
        ),
    ),
    "legacy_custom_v0": MarketProfile(
        name="legacy_custom_v0",
        fractions={
            "zic": 0.45,
            "trend": 0.10,
            "noise": 0.10,
            "value": 0.20,
            "market_maker": 0.15,
        },
        replaceable_fractions={
            "zic": 0.45,
            "trend": 0.10,
            "noise": 0.10,
        },
        description=(
            "Legacy custom ecology kept for reference: ZIC, trend, noise, value, and market makers."
        ),
    ),
}

DEFAULT_MARKET_PROFILE = "abides_rmsc04_small_v1"
BASELINE_FRACTIONS: Dict[str, float] = dict(MARKET_PROFILES[DEFAULT_MARKET_PROFILE].fractions)
REPLACEABLE_FRACTIONS: Dict[str, float] = dict(
    MARKET_PROFILES[DEFAULT_MARKET_PROFILE].replaceable_fractions
)
MAX_PHI = MARKET_PROFILES[DEFAULT_MARKET_PROFILE].max_phi


def get_market_profile(profile_name: str) -> MarketProfile:
    """Return a named market profile."""

    try:
        return MARKET_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(MARKET_PROFILES))
        raise ValueError(f"unknown market_profile={profile_name!r}. Available: {available}") from exc


def compute_agent_counts(
    total_agents: int,
    phi: float,
    *,
    market_profile: str = DEFAULT_MARKET_PROFILE,
) -> Dict[str, int]:
    """Return integer trader counts for a named market profile.

    Counts are allocated by largest remainder so they sum exactly to ``total_agents``
    while staying close to the target fractions. ``phi`` always replaces only the
    replaceable pool defined by the chosen market profile.
    """

    profile = get_market_profile(market_profile)
    if total_agents <= 0:
        raise ValueError("total_agents must be positive.")
    if not 0.0 <= phi <= profile.max_phi:
        raise ValueError(f"phi must be between 0.0 and {profile.max_phi:.2f}.")

    replacement_scale = 1.0 - (phi / profile.max_phi if profile.max_phi > 0 else 0.0)
    target_fractions = {
        "zic": profile.replaceable_fractions.get("zic", 0.0) * replacement_scale,
        "trend": profile.replaceable_fractions.get("trend", 0.0) * replacement_scale,
        "noise": profile.replaceable_fractions.get("noise", 0.0) * replacement_scale,
        "value": float(profile.fractions.get("value", 0.0)),
        "market_maker": float(profile.fractions.get("market_maker", 0.0)),
        "rl": phi,
    }

    raw_counts = {name: target_fractions[name] * total_agents for name in target_fractions}
    counts = {name: floor(value) for name, value in raw_counts.items()}
    remainder = total_agents - sum(counts.values())

    ranked_remainders = sorted(
        (
            (raw_counts[name] - counts[name], target_fractions[name], name)
            for name in target_fractions
        ),
        reverse=True,
    )

    for _, _, name in ranked_remainders[:remainder]:
        counts[name] += 1

    return counts


def compute_rl_role_counts(
    total_rl_agents: int,
    *,
    rl_liquidity_mode: str,
    rl_quoter_split: float = 0.5,
) -> Dict[str, int]:
    """Split total RL participation across directional and quoting roles."""

    total_rl_agents = int(total_rl_agents)
    if total_rl_agents < 0:
        raise ValueError("total_rl_agents must be non-negative.")
    if not 0.0 <= float(rl_quoter_split) <= 1.0:
        raise ValueError("rl_quoter_split must be between 0.0 and 1.0.")

    mode = str(rl_liquidity_mode).strip().lower()
    if mode == "taker_only":
        return {"taker": total_rl_agents, "quoter": 0}
    if mode == "quoter_only":
        return {"taker": 0, "quoter": total_rl_agents}
    if mode != "mixed":
        raise ValueError(
            "rl_liquidity_mode must be one of taker_only, mixed, or quoter_only."
        )

    quoter_count = min(
        total_rl_agents,
        max(0, int(floor(total_rl_agents * float(rl_quoter_split) + 0.5))),
    )
    taker_count = total_rl_agents - quoter_count
    return {"taker": taker_count, "quoter": quoter_count}


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration for a single simulation run."""

    num_agents: int = 102
    phi: float = 0.0
    seed: int = 7
    market_profile: str = DEFAULT_MARKET_PROFILE
    date: str = "20210205"
    market_open_time: str = "09:30:00"
    end_time: str = "10:00:00"
    ticker: str = "ABM"
    starting_cash: int = 10_000_000
    stdout_log_level: str = "WARNING"
    latency_type: str = "no_latency"
    default_computation_delay: int = 0
    book_log_depth: int = 1
    log_frequency: str = "1s"
    tick_size: float = 0.01
    initial_price: float = 100.0
    initial_fundamental: float = 100.0
    sigma_v: float = 0.03
    fundamental_update_frequency: str = "1s"
    lambda_q: float = 0.01
    flat_hold_penalty: float = 0.0
    inventory_cap: int | None = None
    return_window: int = 10
    rl_order_size: int = 1
    rl_quote_size: int = 1
    rl_policy_name: str = "random"
    rl_quoter_policy_name: str = "random_quoter"
    rl_liquidity_mode: str = "taker_only"
    rl_quoter_split: float = 0.5
    rl_enable_passive_quotes: bool = True
    rl_quote_mode: str = "at_best"
    rl_quote_offset_ticks: int = 0
    rl_passive_fill_reward: float = 0.0
    rl_two_sided_quote_reward: float = 0.0
    rl_missing_quote_penalty: float = 0.0
    trend_short_window: int = 5
    trend_long_window: int = 20
    zic_surplus_min_ticks: int = 0
    zic_surplus_max_ticks: int = 3
    zic_sigma_eta: float = 1.0
    value_sigma_eta: float = 1.0
    value_delta_ticks: int = 2
    mm_spread_ticks: int = 3
    mm_alpha: float = 0.002
    mm_quote_size: int = 5
    market_maker_count_scale: float = 1.0
    mm_requote_threshold_ticks: int = 2
    noise_limit_probability: float = 0.45
    noise_wake_up_frequency: str = "5s"
    zic_wake_up_frequency: str = "5s"
    trend_wake_up_frequency: str = "2s"
    trend_signal_noise_ticks: float = 1.0
    trend_signal_threshold_ticks: int = 1
    trend_trade_probability: float = 0.75
    trend_order_size: int = 1
    trend_passive_order_size: int = 2
    trend_aggressive_probability: float = 0.40
    value_wake_up_frequency: str = "1s"
    value_order_size: int = 1
    value_passive_order_size: int = 2
    value_aggressive_probability: float = 0.42
    mm_wake_up_frequency: str = "1s"
    rl_wake_up_frequency: str = "1s"

    def profile(self) -> MarketProfile:
        """Return the selected market profile."""

        return get_market_profile(self.market_profile)

    def agent_counts(self) -> Dict[str, int]:
        """Return the integer trader counts implied by this configuration."""

        return compute_agent_counts(
            total_agents=self.num_agents,
            phi=self.phi,
            market_profile=self.market_profile,
        )

    def rl_role_counts(self) -> Dict[str, int]:
        """Return the taker/quoter split implied by the configured RL mode."""

        return compute_rl_role_counts(
            self.agent_counts()["rl"],
            rl_liquidity_mode=self.rl_liquidity_mode,
            rl_quoter_split=self.rl_quoter_split,
        )
