import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path("benchmarks/results")
PLOTS = Path("benchmarks/plots")


def load_reports():
    reports = []

    for path in RESULTS.glob("*.json"):
        with path.open() as file:
            reports.append(json.load(file))

    return reports


def platform_label(report):
    metadata = report["metadata"]

    cpu = metadata["cpu_model"]

    platform_name = metadata["platform"]
    if platform_name.startswith("Linux"):
        os_name = "Linux"
    elif platform_name.startswith("Windows"):
        os_name = "Windows"
    elif platform_name.startswith("macOS"):
        os_name = "macOS"
    else:
        os_name = platform_name

    compiler = metadata["compiler"].splitlines()[0]

    return f"{cpu} / {os_name} / {compiler}"


def benchmark_cases(reports):
    cases = set()

    for report in reports:
        for result in report["results"]:
            cases.add(result["case"])

    return sorted(cases)


def results_for_case(report, case):
    results = [
        result
        for result in report["results"]
        if result["case"] == case
    ]

    return sorted(results, key=lambda result: result["elements"])


def plot_case(case, reports):
    plt.figure()

    for report in reports:
        results = results_for_case(report, case)

        if not results:
            continue

        x = [result["elements"] for result in results]
        y = [result["median_ns"] for result in results]

        plt.plot(
            x,
            y,
            marker="o",
            label=platform_label(report),
        )

    plt.xlabel("Elements")
    plt.ylabel("Median time (ns)")
    plt.title(case)
    plt.legend()
    plt.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / f"{case}.png")
    plt.close()


def main():
    reports = load_reports()

    for case in benchmark_cases(reports):
        plot_case(case, reports)


if __name__ == "__main__":
    main()