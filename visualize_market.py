"""Visualize a saved market log over time."""

from __future__ import annotations

import argparse

from visualization import (
    load_market_frame,
    save_market_plot,
    show_market_animation,
    show_market_plot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to a simulation log CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default="market_replay.png",
        help="Path for a saved plot image.",
    )
    parser.add_argument("--title", type=str, default="Market Replay")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Replay the market path in a Python window instead of a static plot.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=40,
        help="Animation frame interval in milliseconds.",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=0.0,
        help="Optional trailing time window. Use 0 to keep the axes fixed and just build the graph.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the static plot in a local window in addition to saving it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_market_frame(args.input)
    if args.animate:
        show_market_animation(
            frame,
            title=args.title,
            interval_ms=args.interval_ms,
            tail_seconds=args.tail_seconds if args.tail_seconds > 0 else None,
        )
    elif args.show_plot:
        show_market_plot(frame, title=args.title)
    path = save_market_plot(frame, args.output, title=args.title)
    print(f"Saved market plot to {path}")


if __name__ == "__main__":
    main()
