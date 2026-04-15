"""Official baseline configuration builders."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from config import DEFAULT_MARKET_PROFILE, SimulationConfig, get_market_profile

OFFICIAL_BASELINE_NAME = DEFAULT_MARKET_PROFILE


def build_abides_rmsc04_small_v1_config(**overrides: Any) -> SimulationConfig:
    """Return the official small RMSC04-style baseline configuration."""

    config = SimulationConfig(
        market_profile=OFFICIAL_BASELINE_NAME,
        num_agents=102,
    )
    return replace(config, **overrides)


def build_abides_baseline_v1_config(**overrides: Any) -> SimulationConfig:
    """Backward-compatible alias for the current official baseline builder."""

    return build_abides_rmsc04_small_v1_config(**overrides)


def official_baseline_summary() -> str:
    """Return a short human-readable description of the official baseline."""

    profile = get_market_profile(OFFICIAL_BASELINE_NAME)
    fractions = profile.fractions
    return (
        f"{profile.name}: "
        f"{fractions['noise'] * 102:.0f} noise, "
        f"{fractions['value'] * 102:.0f} value, "
        f"{fractions['market_maker'] * 102:.0f} market makers."
    )
