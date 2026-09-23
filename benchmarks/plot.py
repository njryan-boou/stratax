import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


RESULTS = Path("benchmarks/results/v2")
PLOTS = Path("benchmarks/plots/v2")


def load_platforms():
    platforms = {}

    for directory in sorted(RESULTS.iterdir()):
        if not directory.is_dir():
            continue

        reports = []

        for path in sorted(directory.glob("trial-*.json")):
            with path.open() as file:
                report = json.load(file)

            if report.get("benchmark_suite_version") != 2:
                continue

            reports.append(report)

        if reports:
            platforms[directory.name] = reports

    return platforms


def platform_label(name):
    return name.replace("-", " ").title()


def benchmark_cases(platforms):
    cases = set()

    for reports in platforms.values():
        for report in reports:
            for result in report["results"]:
                cases.add(result["case"])

    return sorted(cases)


def results_for_case(report, case):
    return [
        result
        for result in report["results"]
        if result["case"] == case
    ]


def aggregate_case(reports, case):
    groups = {}

    for report in reports:
        for result in results_for_case(report, case):
            key = (
                result["rows"],
                result["columns"],
                result["elements"],
            )

            groups.setdefault(key, []).append(
                result["median_ns"]
            )

    aggregated = []

    for (rows, columns, elements), medians in groups.items():
        median_of_medians = statistics.median(medians)

        if len(medians) > 1:
            trial_stdev = statistics.stdev(medians)
        else:
            trial_stdev = 0.0

        aggregated.append({
            "rows": rows,
            "columns": columns,
            "elements": elements,
            "trial_count": len(medians),
            "median_ns": median_of_medians,
            "trial_stdev_ns": trial_stdev,
        })

    return sorted(
        aggregated,
        key=lambda result: result["elements"],
    )


def plot_case(case, platforms):
    plt.figure(figsize=(10, 6))

    element_counts = set()

    for platform_name, reports in platforms.items():
        results = aggregate_case(reports, case)

        if not results:
            continue

        x = [
            result["elements"]
            for result in results
        ]

        # Convert ns to microseconds.
        y = [
            result["median_ns"] / 1_000
            for result in results
        ]

        yerr = [
            result["trial_stdev_ns"] / 1_000
            for result in results
        ]

        element_counts.update(x)

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            label=platform_label(platform_name),
        )

    plt.xscale("log", base=2)
    plt.yscale("log")

    counts = sorted(element_counts)

    plt.xticks(
        counts,
        [f"{value:,}" for value in counts],
    )

    plt.xlabel("Elements")
    plt.ylabel("Median time (µs)")
    plt.title(case)

    plt.grid(
        True,
        which="both",
        alpha=0.25,
    )

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    plt.tight_layout()

    PLOTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        PLOTS / f"{case}.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main():
    platforms = load_platforms()

    if not platforms:
        raise RuntimeError(
            f"No V2 benchmark trials found in {RESULTS}"
        )

    for case in benchmark_cases(platforms):
        plot_case(case, platforms)


if __name__ == "__main__":
    main()