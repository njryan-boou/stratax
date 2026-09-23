import argparse
import json
from pathlib import Path

import pandas as pd


RESULTS = Path("benchmarks/results/v2")
ANALYSIS = Path("benchmarks/analysis/v2")


def load_results():
    rows = []

    for platform_directory in sorted(RESULTS.iterdir()):
        if not platform_directory.is_dir():
            continue

        platform = platform_directory.name

        for path in sorted(platform_directory.glob("trial-*.json")):
            with path.open() as file:
                report = json.load(file)

            if report.get("benchmark_suite_version") != 2:
                continue

            trial = report["trial"]

            for result in report["results"]:
                rows.append({
                    "platform": platform,
                    "trial": trial,

                    "case": result["case"],
                    "rows": result["rows"],
                    "columns": result["columns"],
                    "elements": result["elements"],

                    "repeats": result["repeats"],

                    # Statistics within this individual trial.
                    "median_ns": result["median_ns"],
                    "mean_ns": result["mean_ns"],
                    "stdev_ns": result["stdev_ns"],
                    "mad_ns": result["mad_ns"],
                    "cv_percent": result["cv_percent"],

                    "min_ns": result["min_ns"],
                    "max_ns": result["max_ns"],
                })

    if not rows:
        raise RuntimeError(
            f"No V2 benchmark results found in {RESULTS}"
        )

    return pd.DataFrame(rows)


def aggregate_trials(df):
    """
    Combine independent trials for each
    platform / benchmark / size combination.
    """

    summary = (
        df.groupby(
            [
                "platform",
                "case",
                "rows",
                "columns",
                "elements",
            ],
            as_index=False,
        )
        .agg(
            trials=("trial", "count"),

            # Distribution of the trial medians.
            median_ns=("median_ns", "median"),
            mean_trial_median_ns=("median_ns", "mean"),
            trial_stdev_ns=("median_ns", "std"),
            min_trial_median_ns=("median_ns", "min"),
            max_trial_median_ns=("median_ns", "max"),

            # Typical within-trial noise.
            median_sample_stdev_ns=("stdev_ns", "median"),
            median_sample_mad_ns=("mad_ns", "median"),
            median_sample_cv_percent=("cv_percent", "median"),
            max_sample_cv_percent=("cv_percent", "max"),
        )
    )

    # A single trial has no across-trial sample standard deviation.
    summary["trial_stdev_ns"] = (
        summary["trial_stdev_ns"].fillna(0.0)
    )

    summary["trial_cv_percent"] = (
        summary["trial_stdev_ns"]
        / summary["mean_trial_median_ns"]
        * 100
    )

    # Convenient human-readable units.
    summary["median_us"] = (
        summary["median_ns"] / 1_000
    )

    summary["median_ms"] = (
        summary["median_ns"] / 1_000_000
    )

    return summary


def add_baseline_comparison(summary, baseline):
    """
    Compare every platform against one chosen baseline.

    speedup_vs_baseline > 1:
        current platform is faster

    speedup_vs_baseline < 1:
        current platform is slower
    """

    baseline_rows = summary[
        summary["platform"] == baseline
    ][
        [
            "case",
            "rows",
            "columns",
            "elements",
            "median_ns",
        ]
    ].rename(
        columns={
            "median_ns": "baseline_median_ns"
        }
    )

    if baseline_rows.empty:
        available = ", ".join(
            sorted(summary["platform"].unique())
        )

        raise ValueError(
            f"Unknown baseline '{baseline}'. "
            f"Available platforms: {available}"
        )

    compared = summary.merge(
        baseline_rows,
        on=[
            "case",
            "rows",
            "columns",
            "elements",
        ],
        how="left",
    )

    compared["speedup_vs_baseline"] = (
        compared["baseline_median_ns"]
        / compared["median_ns"]
    )

    compared["percent_faster_vs_baseline"] = (
        (
            compared["baseline_median_ns"]
            - compared["median_ns"]
        )
        / compared["baseline_median_ns"]
        * 100
    )

    return compared


def display_table(summary, case=None):
    table = summary

    if case:
        table = table[
            table["case"] == case
        ]

        if table.empty:
            raise ValueError(
                f"No benchmark case named '{case}'"
            )

    columns = [
        "platform",
        "case",
        "elements",
        "trials",
        "median_us",
        "trial_stdev_ns",
        "trial_cv_percent",
        "median_sample_cv_percent",
    ]

    print(
        table[columns]
        .sort_values(
            ["case", "elements", "median_us"]
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Stratax V2 multi-trial "
            "benchmark results."
        )
    )

    parser.add_argument(
        "--case",
        help="only display one benchmark case",
    )

    parser.add_argument(
        "--baseline",
        help=(
            "platform directory name to use "
            "as the performance baseline"
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write analysis CSV files",
    )

    args = parser.parse_args()

    raw = load_results()
    summary = aggregate_trials(raw)

    if args.baseline:
        summary = add_baseline_comparison(
            summary,
            args.baseline,
        )

    display_table(
        summary,
        case=args.case,
    )

    if not args.no_save:
        ANALYSIS.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw.to_csv(
            ANALYSIS / "trials.csv",
            index=False,
        )

        summary.to_csv(
            ANALYSIS / "summary.csv",
            index=False,
        )

        print()
        print(
            f"Saved analysis to {ANALYSIS}"
        )


if __name__ == "__main__":
    main()