"""Rank sweep candidates and select a recommended phi=0 baseline."""

from __future__ import annotations

import argparse

from baseline_selection import (
    DiagnosticThresholds,
    RobustnessConfig,
    SelectionWeights,
    apply_threshold_overrides,
    apply_weight_overrides,
    assess_nearby_robustness,
    build_recommendation_report,
    load_selection_inputs,
    rank_candidate_settings,
    recommend_baseline,
    save_recommendation_report,
    save_selection_plots,
    selection_assumptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-input", type=str, default="")
    parser.add_argument("--summary-input", type=str, default="")
    parser.add_argument("--ranked-output", type=str, default="ranked_candidates.csv")
    parser.add_argument("--report-output", type=str, default="recommended_baseline.md")
    parser.add_argument("--plot-dir", type=str, default="")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Override a selection weight with name=value. Repeat as needed.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override a diagnostic threshold with name=value. Repeat as needed.",
    )
    parser.add_argument(
        "--neighbor-score-gap",
        type=float,
        default=8.0,
        help="Maximum score gap for a nearby candidate to count as similar.",
    )
    parser.add_argument(
        "--neighbor-stable-share",
        type=float,
        default=0.5,
        help="Minimum stable share for a nearby candidate to count as similar.",
    )
    parser.add_argument(
        "--neighbor-min-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of similar neighbors needed to call the recommendation robust.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.runs_input and not args.summary_input:
        raise SystemExit("At least one of --runs-input or --summary-input is required.")

    weights = apply_weight_overrides(SelectionWeights(), args.weight)
    thresholds = apply_threshold_overrides(DiagnosticThresholds(), args.threshold)

    _, summary, parameter_columns = load_selection_inputs(
        runs_input=args.runs_input or None,
        summary_input=args.summary_input or None,
    )
    ranked = rank_candidate_settings(
        summary,
        parameter_columns=parameter_columns,
        weights=weights,
        thresholds=thresholds,
    )
    recommended = recommend_baseline(ranked)
    robustness = assess_nearby_robustness(
        ranked,
        parameter_columns=parameter_columns,
        recommended=recommended,
        config=RobustnessConfig(
            score_gap=args.neighbor_score_gap,
            min_stable_share=args.neighbor_stable_share,
            min_similar_fraction=args.neighbor_min_fraction,
        ),
    )

    ranked.to_csv(args.ranked_output, index=False)
    report = build_recommendation_report(
        ranked,
        parameter_columns=parameter_columns,
        recommended=recommended,
        robustness=robustness,
    )
    save_recommendation_report(report, args.report_output)

    saved_plots = []
    if args.plot_dir:
        saved_plots = save_selection_plots(ranked, output_dir=args.plot_dir)

    print(f"Saved ranked candidates to {args.ranked_output}")
    print(f"Saved recommendation report to {args.report_output}")
    if saved_plots:
        print(f"Saved {len(saved_plots)} selection plots to {args.plot_dir}")
    print("Recommended baseline:")
    for column in parameter_columns:
        print(f"  {column}: {recommended[column]}")
    print(f"  selection_score: {recommended['selection_score']:.2f}")
    print(f"  stability_penalty: {recommended['stability_penalty']:.3f}")
    print(f"  robust_region: {robustness['label']}")
    print("Ranking assumptions:")
    for line in selection_assumptions(weights, thresholds):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
