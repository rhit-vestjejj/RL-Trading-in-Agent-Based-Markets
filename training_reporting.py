"""Reporting helpers for longer shared-policy PPO training runs."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _load_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "rl_training_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    if "matplotlib.pyplot" not in sys.modules:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def build_combined_progress_frame(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return one combined long-form progress frame."""

    frames: list[pd.DataFrame] = []
    if not training_frame.empty:
        train = training_frame.copy()
        train["phase"] = "train"
        train["evaluation_mode"] = "train"
        frames.append(train)
    if not evaluation_frame.empty:
        evaluation = evaluation_frame.copy()
        evaluation["phase"] = "evaluation"
        frames.append(evaluation)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _window_mean(series: pd.Series, window: int = 5) -> float:
    if series.empty:
        return 0.0
    return float(series.tail(min(window, len(series))).mean())


def build_training_progress_report(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
) -> str:
    """Return a compact markdown report for one training run."""

    lines = ["# PPO Training Progress", ""]
    if training_frame.empty:
        lines.append("No training episodes were recorded.")
        return "\n".join(lines)

    reward_start = float(training_frame["total_training_reward"].head(min(5, len(training_frame))).mean())
    reward_end = _window_mean(training_frame["total_training_reward"])
    entropy_end = _window_mean(training_frame["entropy"])
    inventory_end = _window_mean(training_frame["average_abs_ending_inventory"])
    max_inventory_end = _window_mean(training_frame["max_abs_inventory"])
    lines.extend(
        [
            "## Training",
            f"- Episodes: {len(training_frame)}",
            f"- Mean reward first window: {reward_start:.4f}",
            f"- Mean reward last window: {reward_end:.4f}",
            f"- Mean abs ending inventory last window: {inventory_end:.4f}",
            f"- Mean max abs inventory last window: {max_inventory_end:.4f}",
            f"- Mean entropy last window: {entropy_end:.4f}",
        ]
    )

    if evaluation_frame.empty:
        lines.extend(["", "No periodic evaluation rows were recorded."])
        return "\n".join(lines)

    for mode in ("greedy", "stochastic"):
        mode_frame = evaluation_frame.loc[evaluation_frame["evaluation_mode"] == mode].sort_values("episode")
        if mode_frame.empty:
            continue
        reward_first = float(mode_frame["evaluation_total_reward_mean"].head(min(3, len(mode_frame))).mean())
        reward_last = _window_mean(mode_frame["evaluation_total_reward_mean"], window=3)
        hold_last = _window_mean(mode_frame["evaluation_hold_fraction_mean"], window=3)
        buy_last = _window_mean(mode_frame["evaluation_buy_fraction_mean"], window=3)
        sell_last = _window_mean(mode_frame["evaluation_sell_fraction_mean"], window=3)
        inventory_last = _window_mean(mode_frame["evaluation_average_abs_ending_inventory_mean"], window=3)
        inventory_range = (
            float(mode_frame["evaluation_inventory_min_global"].min())
            if "evaluation_inventory_min_global" in mode_frame.columns
            else float("nan"),
            float(mode_frame["evaluation_inventory_max_global"].max())
            if "evaluation_inventory_max_global" in mode_frame.columns
            else float("nan"),
        )
        lines.extend(
            [
                "",
                f"## {mode.capitalize()} Evaluation",
                f"- Checkpoints: {len(mode_frame)}",
                f"- Reward first window: {reward_first:.4f}",
                f"- Reward last window: {reward_last:.4f}",
                f"- Buy / hold / sell fractions last window: {buy_last:.4f} / {hold_last:.4f} / {sell_last:.4f}",
                f"- Mean abs ending inventory last window: {inventory_last:.4f}",
                f"- Inventory range across checkpoints: {inventory_range[0]:.4f} .. {inventory_range[1]:.4f}",
            ]
        )

    return "\n".join(lines)


def save_training_progress_plots(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save simple training/evaluation progress plots."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    plt = _load_matplotlib()
    saved_paths: list[Path] = []

    if not training_frame.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(training_frame["episode"], training_frame["total_training_reward"], linewidth=1.5)
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Reward")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            training_frame["episode"],
            training_frame["average_abs_ending_inventory"],
            linewidth=1.5,
            label="mean abs ending inv",
        )
        axes[1].plot(
            training_frame["episode"],
            training_frame["max_abs_inventory"],
            linewidth=1.2,
            label="max abs inv",
        )
        axes[1].set_ylabel("Inventory")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(training_frame["episode"], training_frame["entropy"], linewidth=1.5, label="entropy")
        axes[2].plot(training_frame["episode"], training_frame["value_loss"], linewidth=1.2, label="value loss")
        axes[2].set_ylabel("Training Stats")
        axes[2].set_xlabel("Episode")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        train_path = output_root / "training_progress.png"
        fig.savefig(train_path, dpi=150)
        plt.close(fig)
        saved_paths.append(train_path)

    if not evaluation_frame.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for mode, color in (("greedy", "tab:blue"), ("stochastic", "tab:orange")):
            mode_frame = evaluation_frame.loc[evaluation_frame["evaluation_mode"] == mode].sort_values("episode")
            if mode_frame.empty:
                continue
            axes[0].plot(
                mode_frame["episode"],
                mode_frame["evaluation_total_reward_mean"],
                linewidth=1.5,
                label=mode,
                color=color,
            )
            axes[1].plot(
                mode_frame["episode"],
                mode_frame["evaluation_hold_fraction_mean"],
                linewidth=1.5,
                label=mode,
                color=color,
            )
            axes[2].plot(
                mode_frame["episode"],
                mode_frame["evaluation_buy_fraction_mean"] - mode_frame["evaluation_sell_fraction_mean"],
                linewidth=1.5,
                label=mode,
                color=color,
            )
        axes[0].set_ylabel("Eval Reward")
        axes[0].set_title("Evaluation Reward")
        axes[1].set_ylabel("Hold Fraction")
        axes[1].set_title("Evaluation Hold Fraction")
        axes[2].set_ylabel("Buy-Sell Gap")
        axes[2].set_xlabel("Episode")
        axes[2].set_title("Evaluation Buy-Sell Fraction Gap")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper right")

        fig.tight_layout()
        eval_path = output_root / "evaluation_progress.png"
        fig.savefig(eval_path, dpi=150)
        plt.close(fig)
        saved_paths.append(eval_path)

    return saved_paths
