"""Minimal shared-policy PPO training utilities for the ABIDES prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from baseline_configs import build_abides_rmsc04_small_v1_config
from config import SimulationConfig
from env import RLPolicy
from market import MarketSimulator
from rl_diagnostics import compute_policy_evaluation_diagnostics

EpisodeCallback = Callable[[str, int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame], None]
CheckpointCallback = Callable[[int, int, object], None]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class _AdamUpdater:
    def __init__(self, shape: tuple[int, ...], beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, parameter: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (gradient**2)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return parameter - learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass(frozen=True)
class PPOHyperparameters:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_learning_rate: float = 0.01
    critic_learning_rate: float = 0.02
    update_epochs: int = 4
    minibatch_size: int = 256
    reward_scale: float = 1.0
    entropy_coefficient: float = 0.01
    gradient_clip_norm: float = 1.0
    value_huber_delta: float = 5.0
    normalize_advantages: bool = True


def _clip_gradient(gradient: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return gradient
    norm = float(np.linalg.norm(gradient))
    if norm <= max_norm or norm <= 1e-12:
        return gradient
    return gradient * (max_norm / norm)


def _huber_loss(errors: np.ndarray, delta: float) -> np.ndarray:
    abs_errors = np.abs(errors)
    quadratic = np.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _action_collapse_state(
    buy_fraction: float,
    hold_fraction: float,
    sell_fraction: float,
    *,
    threshold: float = 0.9,
) -> str:
    if buy_fraction >= threshold:
        return "buy_only"
    if sell_fraction >= threshold:
        return "sell_only"
    if hold_fraction >= threshold:
        return "hold_only"
    return "mixed"


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    agent_ids: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=float)
    returns = np.zeros_like(rewards, dtype=float)
    unique_agent_ids = pd.unique(agent_ids)

    for agent_id in unique_agent_ids:
        agent_mask = agent_ids == agent_id
        indices = np.flatnonzero(agent_mask)
        gae = 0.0
        for idx in indices[::-1]:
            not_done = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * next_values[idx] * not_done - values[idx]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[idx] = gae
            returns[idx] = gae + values[idx]
    return advantages, returns


class SharedLinearPPOPolicy(RLPolicy):
    """Shared linear softmax actor with a linear value baseline."""

    def __init__(
        self,
        state_dim: int,
        *,
        action_dim: int = 3,
        seed: int = 7,
        deterministic: bool = False,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.deterministic = bool(deterministic)
        self.hyperparameters = hyperparameters or PPOHyperparameters()
        rng = np.random.RandomState(seed)
        self.update_random_state = np.random.RandomState(seed + 1_000)
        self.policy_weights = rng.normal(loc=0.0, scale=0.01, size=(self.state_dim, self.action_dim))
        self.policy_bias = np.zeros(self.action_dim, dtype=float)
        self.value_weights = rng.normal(loc=0.0, scale=0.01, size=self.state_dim)
        self.value_bias = np.array(0.0, dtype=float)
        self.policy_weight_optimizer = _AdamUpdater(self.policy_weights.shape)
        self.policy_bias_optimizer = _AdamUpdater(self.policy_bias.shape)
        self.value_weight_optimizer = _AdamUpdater(self.value_weights.shape)
        self.value_bias_optimizer = _AdamUpdater(self.value_bias.shape)

    def act(self, state: np.ndarray, rng: np.random.Generator) -> int:
        return int(self.sample_action(state, rng)["action"])

    def sample_action(self, state: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        state_vector = np.asarray(state, dtype=float).reshape(1, -1)
        probs = self._policy_probs(state_vector)[0]
        if self.deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(self.action_dim, p=probs))
        value = float(self._value_estimate(state_vector)[0])
        return {
            "action": action,
            "log_prob": float(np.log(probs[action] + 1e-12)),
            "value": value,
        }

    def set_deterministic(self, deterministic: bool) -> None:
        self.deterministic = bool(deterministic)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            policy_weights=self.policy_weights,
            policy_bias=self.policy_bias,
            value_weights=self.value_weights,
            value_bias=self.value_bias,
            deterministic=int(self.deterministic),
            hyperparameters=np.array([asdict(self.hyperparameters)], dtype=object),
        )
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        deterministic: bool = True,
        hyperparameters: PPOHyperparameters | None = None,
    ) -> "SharedLinearPPOPolicy":
        data = np.load(Path(path), allow_pickle=True)
        policy = cls(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            deterministic=deterministic,
            hyperparameters=(
                hyperparameters
                or PPOHyperparameters(**dict(data["hyperparameters"][0]))
                if "hyperparameters" in data
                else None
            ),
        )
        policy.policy_weights = np.array(data["policy_weights"], dtype=float)
        policy.policy_bias = np.array(data["policy_bias"], dtype=float)
        policy.value_weights = np.array(data["value_weights"], dtype=float)
        policy.value_bias = np.array(data["value_bias"], dtype=float)
        return policy

    def update(self, transitions: pd.DataFrame) -> dict[str, float]:
        if transitions.empty:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "wealth_delta_mean": 0.0,
                "inventory_penalty_mean": 0.0,
                "num_transitions": 0.0,
            }

        ordered_transitions = transitions.copy()
        sort_columns = [column for column in ("agent_id", "decision_index", "time_ns") if column in ordered_transitions.columns]
        if sort_columns:
            ordered_transitions = ordered_transitions.sort_values(sort_columns).reset_index(drop=True)
        batch = transitions_to_numpy(ordered_transitions)
        rewards = batch["rewards"] * float(self.hyperparameters.reward_scale)
        next_values = self._value_estimate(batch["next_states"])
        advantages, targets = _compute_gae(
            rewards,
            batch["old_values"],
            next_values,
            batch["dones"],
            batch["agent_ids"],
            gamma=self.hyperparameters.gamma,
            gae_lambda=self.hyperparameters.gae_lambda,
        )
        if self.hyperparameters.normalize_advantages and len(advantages) > 1:
            advantage_std = float(advantages.std(ddof=0))
            if advantage_std > 1e-8:
                advantages = (advantages - advantages.mean()) / advantage_std

        minibatch_size = min(int(self.hyperparameters.minibatch_size), len(batch["actions"]))
        indices = np.arange(len(batch["actions"]))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(int(self.hyperparameters.update_epochs)):
            self.update_random_state.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                mb = indices[start : start + minibatch_size]
                states = batch["states"][mb]
                actions = batch["actions"][mb]
                old_log_probs = batch["old_log_probs"][mb]
                old_values = batch["old_values"][mb]
                returns = targets[mb]
                batch_advantages = advantages[mb]

                probs = self._policy_probs(states)
                chosen_probs = probs[np.arange(len(mb)), actions]
                log_probs = np.log(chosen_probs + 1e-12)
                ratio = np.exp(log_probs - old_log_probs)
                clipped_ratio = np.clip(
                    ratio,
                    1.0 - self.hyperparameters.clip_epsilon,
                    1.0 + self.hyperparameters.clip_epsilon,
                )
                surrogate_1 = ratio * batch_advantages
                surrogate_2 = clipped_ratio * batch_advantages
                unclipped_active = surrogate_1 <= surrogate_2
                ratio_matches_clip = np.isclose(ratio, clipped_ratio)
                dloss_dlogp = np.where(
                    unclipped_active | ratio_matches_clip,
                    -batch_advantages * ratio,
                    0.0,
                )

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(mb)), actions] = 1.0
                grad_logits = dloss_dlogp[:, None] * (one_hot - probs)
                sample_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1, keepdims=True)
                grad_logits += (
                    self.hyperparameters.entropy_coefficient
                    * probs
                    * (np.log(probs + 1e-12) + sample_entropy)
                )
                grad_policy_weights = states.T @ grad_logits / len(mb)
                grad_policy_bias = grad_logits.mean(axis=0)

                values = self._value_estimate(states)
                value_errors = values - returns
                value_grad_errors = np.clip(
                    value_errors,
                    -self.hyperparameters.value_huber_delta,
                    self.hyperparameters.value_huber_delta,
                )
                grad_value_weights = states.T @ value_grad_errors / len(mb)
                grad_value_bias = np.array(value_grad_errors.mean(), dtype=float)

                grad_policy_weights = _clip_gradient(
                    grad_policy_weights,
                    self.hyperparameters.gradient_clip_norm,
                )
                grad_policy_bias = _clip_gradient(
                    grad_policy_bias,
                    self.hyperparameters.gradient_clip_norm,
                )
                grad_value_weights = _clip_gradient(
                    grad_value_weights,
                    self.hyperparameters.gradient_clip_norm,
                )
                grad_value_bias = _clip_gradient(
                    grad_value_bias,
                    self.hyperparameters.gradient_clip_norm,
                )

                self.policy_weights = self.policy_weight_optimizer.step(
                    self.policy_weights,
                    grad_policy_weights,
                    self.hyperparameters.actor_learning_rate,
                )
                self.policy_bias = self.policy_bias_optimizer.step(
                    self.policy_bias,
                    grad_policy_bias,
                    self.hyperparameters.actor_learning_rate,
                )
                self.value_weights = self.value_weight_optimizer.step(
                    self.value_weights,
                    grad_value_weights,
                    self.hyperparameters.critic_learning_rate,
                )
                self.value_bias = self.value_bias_optimizer.step(
                    self.value_bias,
                    grad_value_bias,
                    self.hyperparameters.critic_learning_rate,
                )

                policy_losses.append(float(-np.mean(np.minimum(surrogate_1, surrogate_2))))
                value_losses.append(
                    float(
                        np.mean(
                            _huber_loss(
                                value_errors,
                                self.hyperparameters.value_huber_delta,
                            )
                        )
                    )
                )
                entropies.append(float(-np.mean(np.sum(probs * np.log(probs + 1e-12), axis=1))))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "reward_mean": float(np.mean(batch["rewards"])) if len(batch["rewards"]) else 0.0,
            "reward_std": float(np.std(batch["rewards"], ddof=0)) if len(batch["rewards"]) else 0.0,
            "wealth_delta_mean": float(np.mean(batch["wealth_delta"])) if len(batch["wealth_delta"]) else 0.0,
            "inventory_penalty_mean": (
                float(np.mean(batch["inventory_penalty"])) if len(batch["inventory_penalty"]) else 0.0
            ),
            "flat_hold_penalty_mean": (
                float(np.mean(batch["flat_hold_penalty"])) if len(batch["flat_hold_penalty"]) else 0.0
            ),
            "num_transitions": float(len(batch["actions"])),
        }

    def _policy_logits(self, states: np.ndarray) -> np.ndarray:
        return states @ self.policy_weights + self.policy_bias

    def _policy_probs(self, states: np.ndarray) -> np.ndarray:
        return _softmax(self._policy_logits(states))

    def _value_estimate(self, states: np.ndarray) -> np.ndarray:
        return states @ self.value_weights + self.value_bias


@dataclass
class SharedPolicyBundle:
    """Role-aware shared policies for mixed taker/quoter runs."""

    taker_policy: SharedLinearPPOPolicy | None = None
    quoter_policy: SharedLinearPPOPolicy | None = None

    def policy_for_role(self, role: str) -> SharedLinearPPOPolicy:
        normalized_role = str(role).strip().lower()
        if normalized_role == "taker" and self.taker_policy is not None:
            return self.taker_policy
        if normalized_role == "quoter" and self.quoter_policy is not None:
            return self.quoter_policy
        raise ValueError(f"checkpoint does not contain a policy for rl role {normalized_role!r}.")

    def set_deterministic(self, deterministic: bool) -> None:
        if self.taker_policy is not None:
            self.taker_policy.set_deterministic(deterministic)
        if self.quoter_policy is not None:
            self.quoter_policy.set_deterministic(deterministic)


def _is_policy_bundle(policy: object | None) -> bool:
    return isinstance(policy, SharedPolicyBundle)


def _serialize_policy(prefix: str, policy: SharedLinearPPOPolicy | None) -> dict[str, object]:
    payload: dict[str, object] = {f"{prefix}_present": np.array(int(policy is not None), dtype=int)}
    if policy is None:
        return payload
    payload.update(
        {
            f"{prefix}_state_dim": np.array(policy.state_dim, dtype=int),
            f"{prefix}_action_dim": np.array(policy.action_dim, dtype=int),
            f"{prefix}_policy_weights": policy.policy_weights,
            f"{prefix}_policy_bias": policy.policy_bias,
            f"{prefix}_value_weights": policy.value_weights,
            f"{prefix}_value_bias": policy.value_bias,
            f"{prefix}_deterministic": np.array(int(policy.deterministic), dtype=int),
            f"{prefix}_hyperparameters": np.array([asdict(policy.hyperparameters)], dtype=object),
        }
    )
    return payload


def _deserialize_policy(
    data: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    deterministic: bool,
    hyperparameters: PPOHyperparameters | None = None,
) -> SharedLinearPPOPolicy | None:
    if int(data[f"{prefix}_present"]) == 0:
        return None
    policy = SharedLinearPPOPolicy(
        state_dim=int(data[f"{prefix}_state_dim"]),
        action_dim=int(data[f"{prefix}_action_dim"]),
        deterministic=deterministic,
        hyperparameters=(
            hyperparameters
            or PPOHyperparameters(**dict(data[f"{prefix}_hyperparameters"][0]))
        ),
    )
    policy.policy_weights = np.array(data[f"{prefix}_policy_weights"], dtype=float)
    policy.policy_bias = np.array(data[f"{prefix}_policy_bias"], dtype=float)
    policy.value_weights = np.array(data[f"{prefix}_value_weights"], dtype=float)
    policy.value_bias = np.array(data[f"{prefix}_value_bias"], dtype=float)
    return policy


def save_policy_artifact(policy: object, path: str | Path) -> Path:
    """Persist either a single shared policy or a role-aware policy bundle."""

    if not _is_policy_bundle(policy):
        return policy.save(path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = policy
    np.savez(
        target,
        artifact_type=np.array(["shared_policy_bundle"], dtype=object),
        **_serialize_policy("taker", bundle.taker_policy),
        **_serialize_policy("quoter", bundle.quoter_policy),
    )
    return target


def load_policy_artifact(
    path: str | Path,
    *,
    deterministic: bool = True,
    hyperparameters: PPOHyperparameters | None = None,
) -> object:
    """Load either a legacy single-policy checkpoint or a mixed-role bundle."""

    checkpoint = Path(path)
    data = np.load(checkpoint, allow_pickle=True)
    artifact_type = (
        str(data["artifact_type"][0])
        if "artifact_type" in data
        else ""
    )
    if artifact_type != "shared_policy_bundle":
        data.close()
        return SharedLinearPPOPolicy.load(
            checkpoint,
            deterministic=deterministic,
            hyperparameters=hyperparameters,
        )
    try:
        return SharedPolicyBundle(
            taker_policy=_deserialize_policy(
                data,
                "taker",
                deterministic=deterministic,
                hyperparameters=hyperparameters,
            ),
            quoter_policy=_deserialize_policy(
                data,
                "quoter",
                deterministic=deterministic,
                hyperparameters=hyperparameters,
            ),
        )
    finally:
        data.close()


def transitions_to_numpy(transitions: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert long-form transition logs into dense numpy arrays."""

    state_columns = sorted(column for column in transitions.columns if column.startswith("state_"))
    next_state_columns = sorted(column for column in transitions.columns if column.startswith("next_state_"))
    return {
        "agent_ids": transitions.get("agent_id", pd.Series(np.arange(len(transitions)), index=transitions.index)).to_numpy(dtype=int),
        "states": transitions[state_columns].to_numpy(dtype=float),
        "actions": transitions["action"].to_numpy(dtype=int),
        "rewards": transitions["reward"].to_numpy(dtype=float),
        "wealth_delta": transitions.get("wealth_delta", pd.Series(0.0, index=transitions.index)).to_numpy(dtype=float),
        "inventory_penalty": transitions.get(
            "inventory_penalty",
            pd.Series(0.0, index=transitions.index),
        ).to_numpy(dtype=float),
        "flat_hold_penalty": transitions.get(
            "flat_hold_penalty",
            pd.Series(0.0, index=transitions.index),
        ).to_numpy(dtype=float),
        "next_states": transitions[next_state_columns].to_numpy(dtype=float),
        "dones": transitions["done"].astype(float).to_numpy(dtype=float),
        "old_log_probs": transitions["log_prob"].fillna(0.0).to_numpy(dtype=float),
        "old_values": transitions["value_estimate"].fillna(0.0).to_numpy(dtype=float),
    }


def _role_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame.get("agent_role", pd.Series("taker", index=frame.index)),
        index=frame.index,
        dtype="string",
    ).fillna("taker")


def _build_policy_for_config(
    config: SimulationConfig,
    *,
    policy: object | None,
    hyperparameters: PPOHyperparameters | None,
) -> object:
    if policy is not None:
        return policy
    state_dim = int(config.return_window) + 3
    role_counts = config.rl_role_counts()
    if role_counts["quoter"] <= 0:
        return SharedLinearPPOPolicy(
            state_dim=state_dim,
            seed=int(config.seed),
            hyperparameters=hyperparameters,
        )
    return SharedPolicyBundle(
        taker_policy=(
            SharedLinearPPOPolicy(
                state_dim=state_dim,
                action_dim=3,
                seed=int(config.seed),
                hyperparameters=hyperparameters,
            )
            if role_counts["taker"] > 0
            else None
        ),
        quoter_policy=(
            SharedLinearPPOPolicy(
                state_dim=state_dim,
                action_dim=4,
                seed=int(config.seed) + 10_000,
                hyperparameters=hyperparameters,
            )
            if role_counts["quoter"] > 0
            else None
        ),
    )


def _combine_role_update_metrics(metrics_by_role: dict[str, dict[str, float]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    total_weight = sum(float(metrics.get("num_transitions", 0.0)) for metrics in metrics_by_role.values())
    all_keys = {key for metrics in metrics_by_role.values() for key in metrics}
    for key in all_keys:
        if key == "num_transitions":
            aggregate[key] = float(total_weight)
            continue
        weighted_sum = 0.0
        weighted = False
        fallback_values: list[float] = []
        for metrics in metrics_by_role.values():
            if key not in metrics:
                continue
            value = float(metrics[key])
            weight = float(metrics.get("num_transitions", 0.0))
            if total_weight > 0.0 and weight > 0.0:
                weighted_sum += value * weight
                weighted = True
            fallback_values.append(value)
        if weighted and total_weight > 0.0:
            aggregate[key] = float(weighted_sum / total_weight)
        elif fallback_values:
            aggregate[key] = float(np.mean(fallback_values))
    for role, metrics in metrics_by_role.items():
        for key, value in metrics.items():
            aggregate[f"{role}_{key}"] = float(value)
    return aggregate


def _update_policy_artifact(policy: object, transition_frame: pd.DataFrame) -> dict[str, float]:
    if not _is_policy_bundle(policy):
        return policy.update(transition_frame)
    role_series = _role_series(transition_frame)
    metrics_by_role: dict[str, dict[str, float]] = {}
    if policy.taker_policy is not None:
        metrics_by_role["taker"] = policy.taker_policy.update(
            transition_frame.loc[role_series == "taker"].copy()
        )
    if policy.quoter_policy is not None:
        metrics_by_role["quoter"] = policy.quoter_policy.update(
            transition_frame.loc[role_series == "quoter"].copy()
        )
    return _combine_role_update_metrics(metrics_by_role)


def summarize_episode(
    *,
    episode_index: int,
    phi: float,
    seed: int,
    rl_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
) -> dict[str, float]:
    """Summarize one training or evaluation episode."""

    if rl_frame.empty:
        return {
            "episode": float(episode_index),
            "phi": float(phi),
            "seed": float(seed),
            "total_training_reward": 0.0,
            "average_reward_per_rl_agent": 0.0,
            "average_inventory": 0.0,
            "average_abs_inventory": 0.0,
            "average_abs_ending_inventory": 0.0,
            "max_abs_inventory": 0.0,
            "buy_fraction": 0.0,
            "hold_fraction": 0.0,
            "sell_fraction": 0.0,
            "quote_hold_fraction": 0.0,
            "quote_bid_fraction": 0.0,
            "quote_ask_fraction": 0.0,
            "quote_both_fraction": 0.0,
            "collapse_state": "none",
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "wealth_delta_mean": 0.0,
            "inventory_penalty_mean": 0.0,
            "flat_hold_penalty_mean": 0.0,
            "blocked_buy_action_count": 0.0,
            "blocked_sell_action_count": 0.0,
            "blocked_action_count": 0.0,
            "inventory_at_cap_fraction": 0.0,
            "inventory_near_cap_fraction": 0.0,
            "num_rl_taker_decisions": 0.0,
            "num_rl_quoter_decisions": 0.0,
            "num_rl_decisions": 0.0,
        }

    role_series = _role_series(rl_frame)
    taker_frame = rl_frame.loc[role_series == "taker"].copy()
    quoter_frame = rl_frame.loc[role_series == "quoter"].copy()
    inventory_series = pd.to_numeric(rl_frame["inventory"], errors="coerce").fillna(0.0)
    ending_inventory = rl_frame.sort_values("time_ns").groupby("agent_id").tail(1)["inventory"].abs()
    average_reward_per_agent = rl_frame.groupby("agent_id")["reward"].mean().mean()
    taker_action_series = taker_frame["action"] if not taker_frame.empty else pd.Series(dtype=float)
    quoter_action_series = quoter_frame["action"] if not quoter_frame.empty else pd.Series(dtype=float)
    buy_fraction = float((taker_action_series == 2).mean()) if not taker_frame.empty else 0.0
    hold_fraction = float((taker_action_series == 1).mean()) if not taker_frame.empty else 0.0
    sell_fraction = float((taker_action_series == 0).mean()) if not taker_frame.empty else 0.0
    quote_hold_fraction = float((quoter_action_series == 0).mean()) if not quoter_frame.empty else 0.0
    quote_bid_fraction = float((quoter_action_series == 1).mean()) if not quoter_frame.empty else 0.0
    quote_ask_fraction = float((quoter_action_series == 2).mean()) if not quoter_frame.empty else 0.0
    quote_both_fraction = float((quoter_action_series == 3).mean()) if not quoter_frame.empty else 0.0
    transition_reward = pd.to_numeric(transition_frame["reward"], errors="coerce").fillna(0.0) if not transition_frame.empty else pd.Series(dtype=float)
    wealth_delta = pd.to_numeric(transition_frame["wealth_delta"], errors="coerce").fillna(0.0) if "wealth_delta" in transition_frame.columns else pd.Series(dtype=float)
    inventory_penalty = pd.to_numeric(transition_frame["inventory_penalty"], errors="coerce").fillna(0.0) if "inventory_penalty" in transition_frame.columns else pd.Series(dtype=float)
    flat_hold_penalty = pd.to_numeric(transition_frame["flat_hold_penalty"], errors="coerce").fillna(0.0) if "flat_hold_penalty" in transition_frame.columns else pd.Series(dtype=float)
    blocked_buy = pd.to_numeric(
        rl_frame.get("blocked_buy_due_to_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    blocked_sell = pd.to_numeric(
        rl_frame.get("blocked_sell_due_to_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    inventory_at_cap = pd.to_numeric(
        rl_frame.get("inventory_at_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    inventory_near_cap = pd.to_numeric(
        rl_frame.get("inventory_near_cap", pd.Series(False, index=rl_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return {
        "episode": float(episode_index),
        "phi": float(phi),
        "seed": float(seed),
        "total_training_reward": float(transition_reward.sum()) if not transition_frame.empty else 0.0,
        "average_reward_per_rl_agent": float(average_reward_per_agent),
        "average_inventory": float(inventory_series.mean()),
        "average_abs_inventory": float(inventory_series.abs().mean()),
        "average_abs_ending_inventory": float(ending_inventory.mean()),
        "max_abs_inventory": float(inventory_series.abs().max()),
        "buy_fraction": buy_fraction,
        "hold_fraction": hold_fraction,
        "sell_fraction": sell_fraction,
        "quote_hold_fraction": quote_hold_fraction,
        "quote_bid_fraction": quote_bid_fraction,
        "quote_ask_fraction": quote_ask_fraction,
        "quote_both_fraction": quote_both_fraction,
        "collapse_state": (
            _action_collapse_state(
                buy_fraction,
                hold_fraction,
                sell_fraction,
            )
            if not taker_frame.empty
            else "none"
        ),
        "reward_mean": float(transition_reward.mean()) if not transition_reward.empty else 0.0,
        "reward_std": float(transition_reward.std(ddof=0)) if not transition_reward.empty else 0.0,
        "wealth_delta_mean": float(wealth_delta.mean()) if not wealth_delta.empty else 0.0,
        "inventory_penalty_mean": float(inventory_penalty.mean()) if not inventory_penalty.empty else 0.0,
        "flat_hold_penalty_mean": float(flat_hold_penalty.mean()) if not flat_hold_penalty.empty else 0.0,
        "blocked_buy_action_count": float(blocked_buy.sum()),
        "blocked_sell_action_count": float(blocked_sell.sum()),
        "blocked_action_count": float(blocked_buy.sum() + blocked_sell.sum()),
        "inventory_at_cap_fraction": float(inventory_at_cap.mean()) if not inventory_at_cap.empty else 0.0,
        "inventory_near_cap_fraction": float(inventory_near_cap.mean()) if not inventory_near_cap.empty else 0.0,
        "num_rl_taker_decisions": float(len(taker_frame)),
        "num_rl_quoter_decisions": float(len(quoter_frame)),
        "num_rl_decisions": float(len(rl_frame)),
    }


def run_policy_episode(
    config: SimulationConfig,
    *,
    shared_policy: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one market session and return market, decision, and transition frames."""

    role_counts = config.rl_role_counts()
    if (
        shared_policy is not None
        and role_counts["taker"] > 0
        and role_counts["quoter"] > 0
        and not _is_policy_bundle(shared_policy)
    ):
        raise ValueError(
            "mixed rl liquidity mode requires a role-aware checkpoint with separate taker and quoter policies."
        )
    if (
        shared_policy is not None
        and role_counts["quoter"] > 0
        and role_counts["taker"] == 0
        and not _is_policy_bundle(shared_policy)
        and hasattr(shared_policy, "action_dim")
        and int(getattr(shared_policy, "action_dim")) < 4
    ):
        raise ValueError(
            "quoter-only evaluation requires a checkpoint with a 4-action quoter policy."
        )

    if _is_policy_bundle(shared_policy):
        policy_factory: Callable[..., RLPolicy] | None = (
            lambda agent_id, rng, role: shared_policy.policy_for_role(role)
        )
    elif shared_policy is not None:
        policy_factory = lambda agent_id, rng, role=None: shared_policy
    else:
        policy_factory = None
    simulator = MarketSimulator(
        config,
        rl_policy_factory=policy_factory,
    )
    market_frame = simulator.run()
    rl_frame = simulator.extract_rl_frame()
    transition_frame = simulator.extract_rl_transition_frame()
    return market_frame, rl_frame, transition_frame


def train_shared_policy(
    *,
    phi: float,
    episodes: int,
    start_seed: int = 7,
    training_seeds: Sequence[int] | None = None,
    config_factory: Callable[..., SimulationConfig] = build_abides_rmsc04_small_v1_config,
    config_overrides: dict[str, object] | None = None,
    policy: object | None = None,
    hyperparameters: PPOHyperparameters | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    evaluation_modes: Sequence[str] | None = None,
    evaluation_interval: int = 0,
    episode_callback: EpisodeCallback | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: CheckpointCallback | None = None,
) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    """Train one shared policy artifact from pooled asynchronous RL experience."""

    overrides = dict(config_overrides or {})
    sample_seed = int(training_seeds[0]) if training_seeds else int(start_seed)
    sample_config = config_factory(phi=phi, seed=sample_seed, **overrides)
    shared_policy = _build_policy_for_config(
        sample_config,
        policy=policy,
        hyperparameters=hyperparameters,
    )

    training_rows: list[dict[str, float]] = []
    evaluation_rows: list[dict[str, float]] = []
    episode_seeds = [int(seed) for seed in training_seeds] if training_seeds else [
        int(start_seed + episode_index) for episode_index in range(int(episodes))
    ]
    configured_evaluation_modes = [str(mode).strip().lower() for mode in (evaluation_modes or ("greedy",))]
    for episode_index, seed in enumerate(episode_seeds):
        config = config_factory(phi=phi, seed=seed, **overrides)
        market_frame, rl_frame, transition_frame = run_policy_episode(config, shared_policy=shared_policy)
        if episode_callback is not None:
            episode_callback("train", episode_index, seed, market_frame, rl_frame, transition_frame)
        update_metrics = _update_policy_artifact(shared_policy, transition_frame)
        summary = summarize_episode(
            episode_index=episode_index,
            phi=phi,
            seed=seed,
            rl_frame=rl_frame,
            transition_frame=transition_frame,
        )
        summary.update(update_metrics)
        training_rows.append(summary)

        if checkpoint_callback is not None and checkpoint_interval > 0 and (episode_index + 1) % checkpoint_interval == 0:
            checkpoint_callback(episode_index, seed, shared_policy)

        if evaluation_seeds and evaluation_interval > 0 and (episode_index + 1) % evaluation_interval == 0:
            for evaluation_mode in configured_evaluation_modes:
                _, evaluation_summary = evaluate_policy(
                    policy=shared_policy,
                    phi=phi,
                    seeds=evaluation_seeds,
                    deterministic=(evaluation_mode == "greedy"),
                    config_factory=config_factory,
                    config_overrides=overrides,
                    episode_callback=episode_callback,
                )
                evaluation_summary = dict(evaluation_summary)
                evaluation_summary["episode"] = float(episode_index)
                evaluation_summary["evaluation_mode"] = evaluation_mode
                evaluation_rows.append(evaluation_summary)

    return (
        shared_policy,
        pd.DataFrame(training_rows),
        pd.DataFrame(evaluation_rows),
    )


def evaluate_policy(
    *,
    policy: object | None,
    phi: float,
    seeds: Sequence[int],
    deterministic: bool = True,
    config_factory: Callable[..., SimulationConfig] = build_abides_rmsc04_small_v1_config,
    config_overrides: dict[str, object] | None = None,
    episode_callback: EpisodeCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate one shared policy across a fixed list of seeds."""

    overrides = dict(config_overrides or {})
    summaries: list[dict[str, float]] = []
    diagnostics_rows: list[dict[str, float | str]] = []
    deterministic_policy = policy
    original_deterministic: bool | None = None
    if _is_policy_bundle(deterministic_policy):
        deterministic_policy.set_deterministic(bool(deterministic))
    elif hasattr(deterministic_policy, "set_deterministic"):
        original_deterministic = bool(getattr(deterministic_policy, "deterministic", False))
        getattr(deterministic_policy, "set_deterministic")(bool(deterministic))
    try:
        for episode_index, seed in enumerate(seeds):
            config = config_factory(phi=phi, seed=seed, **overrides)
            market_frame, rl_frame, transition_frame = run_policy_episode(config, shared_policy=deterministic_policy)
            if episode_callback is not None:
                episode_callback("evaluation", episode_index, seed, market_frame, rl_frame, transition_frame)
            summaries.append(
                summarize_episode(
                    episode_index=episode_index,
                    phi=phi,
                    seed=seed,
                    rl_frame=rl_frame,
                    transition_frame=transition_frame,
                )
            )
            diagnostics, _ = compute_policy_evaluation_diagnostics(
                market_frame,
                rl_frame,
                transition_frame,
                policy=deterministic_policy,
            )
            diagnostics_rows.append(diagnostics)
        frame = pd.DataFrame(summaries)
        diagnostics_frame = pd.DataFrame(diagnostics_rows)
        aggregate = {
            "evaluation_total_reward_mean": float(frame["total_training_reward"].mean()) if not frame.empty else 0.0,
            "evaluation_average_reward_per_rl_agent_mean": float(frame["average_reward_per_rl_agent"].mean()) if not frame.empty else 0.0,
            "evaluation_average_abs_ending_inventory_mean": float(frame["average_abs_ending_inventory"].mean()) if not frame.empty else 0.0,
            "evaluation_buy_fraction_mean": float(frame["buy_fraction"].mean()) if not frame.empty else 0.0,
            "evaluation_hold_fraction_mean": float(frame["hold_fraction"].mean()) if not frame.empty else 0.0,
            "evaluation_sell_fraction_mean": float(frame["sell_fraction"].mean()) if not frame.empty else 0.0,
            "evaluation_quote_hold_fraction_mean": float(frame["quote_hold_fraction"].mean()) if not frame.empty and "quote_hold_fraction" in frame.columns else 0.0,
            "evaluation_quote_bid_fraction_mean": float(frame["quote_bid_fraction"].mean()) if not frame.empty and "quote_bid_fraction" in frame.columns else 0.0,
            "evaluation_quote_ask_fraction_mean": float(frame["quote_ask_fraction"].mean()) if not frame.empty and "quote_ask_fraction" in frame.columns else 0.0,
            "evaluation_quote_both_fraction_mean": float(frame["quote_both_fraction"].mean()) if not frame.empty and "quote_both_fraction" in frame.columns else 0.0,
        }
        if not diagnostics_frame.empty:
            aggregate.update(
                {
                    "evaluation_inventory_min_global": float(diagnostics_frame["inventory_overall_min"].min()),
                    "evaluation_inventory_max_global": float(diagnostics_frame["inventory_overall_max"].max()),
                    "evaluation_max_abs_inventory_reached_max": float(diagnostics_frame["max_abs_inventory_reached"].max()),
                    "evaluation_chosen_sell_count_total": float(diagnostics_frame["rl_sell_count"].sum()),
                    "evaluation_chosen_hold_count_total": float(diagnostics_frame["rl_hold_count"].sum()),
                    "evaluation_chosen_buy_count_total": float(diagnostics_frame["rl_buy_count"].sum()),
                    "evaluation_executed_sell_count_total": float(diagnostics_frame["executed_sell_action_count"].sum()),
                    "evaluation_executed_buy_count_total": float(diagnostics_frame["executed_buy_action_count"].sum()),
                    "evaluation_executed_sell_volume_total": float(diagnostics_frame["rl_executed_sell_volume"].sum()),
                    "evaluation_executed_buy_volume_total": float(diagnostics_frame["rl_executed_buy_volume"].sum()),
                    "evaluation_submitted_sell_count_total": float(diagnostics_frame["submitted_sell_action_count"].sum()),
                    "evaluation_submitted_buy_count_total": float(diagnostics_frame["submitted_buy_action_count"].sum()),
                    "evaluation_blocked_sell_action_count_total": float(diagnostics_frame["blocked_sell_action_count"].sum()),
                    "evaluation_blocked_buy_action_count_total": float(diagnostics_frame["blocked_buy_action_count"].sum()),
                    "evaluation_inventory_at_cap_fraction_mean": float(diagnostics_frame["inventory_at_cap_fraction"].mean()),
                    "evaluation_inventory_near_cap_fraction_mean": float(diagnostics_frame["inventory_near_cap_fraction"].mean()),
                }
            )
        return frame, aggregate
    finally:
        if _is_policy_bundle(deterministic_policy):
            deterministic_policy.set_deterministic(False)
        elif hasattr(deterministic_policy, "set_deterministic"):
            getattr(deterministic_policy, "set_deterministic")(
                bool(original_deterministic) if original_deterministic is not None else False
            )
