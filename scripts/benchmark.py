#!/usr/bin/env python3
"""Build and run dependency-free C++ performance baselines with GCC or Clang."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SUITE_VERSION = 2
DEFAULT_REPEATS = 50
DEFAULT_WARMUPS = 10


def positive_integer(value):
    result = int(value)

    if result <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        )

    return result


def command_output(command):
    return subprocess.check_output(
        command,
        cwd=ROOT,
        text=True,
    ).strip()


def cpu_model():
    cpuinfo = Path("/proc/cpuinfo")

    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()

    if sys.platform == "darwin":
        return command_output([
            "sysctl",
            "-n",
            "machdep.cpu.brand_string",
        ])

    return platform.processor() or "unknown"


def source_digest():
    digest = hashlib.sha256()

    paths = sorted(
        path
        for path in (ROOT / "include").rglob("*")
        if path.suffix in {".hpp", ".h"}
    )

    paths.extend([
        ROOT / "benchmarks/baseline.cpp",
        Path(__file__).resolve(),
    ])

    for path in paths:
        digest.update(
            path.relative_to(ROOT)
            .as_posix()
            .encode()
        )

        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def source_tree_dirty():
    output = command_output([
        "git",
        "status",
        "--porcelain",
        "--",
        ".",
        ":(exclude)benchmarks/results/**",
        ":(exclude)benchmarks/plots/**",
    ])

    return bool(output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=positive_integer,
        default=[64, 256, 1024],
        help=(
            "square matrix extents "
            "(default: 64 256 1024)"
        ),
    )

    parser.add_argument(
        "--columns",
        type=positive_integer,
        help=(
            "use this column count for every "
            "requested row count"
        ),
    )

    parser.add_argument(
    "--repeats",
    type=positive_integer,
    default=DEFAULT_REPEATS,
    help=f"timed samples per benchmark case (default: {DEFAULT_REPEATS})",
)

    parser.add_argument(
    "--warmups",
    type=positive_integer,
    default=DEFAULT_WARMUPS,
    help=f"warmup executions per benchmark case (default: {DEFAULT_WARMUPS})",
)

    parser.add_argument(
        "--trial",
        type=positive_integer,
        default=1,
        help="trial number recorded in the report",
    )

    parser.add_argument(
        "--compiler",
        default=os.environ.get("CXX", "c++"),
        help=(
            "GCC/Clang compiler command "
            "(default: CXX or c++)"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: stdout)",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        help="also save individual samples as CSV",
    )

    args = parser.parse_args()

    compiler = shlex.split(args.compiler)

    if not compiler:
        parser.error(
            "--compiler must not be empty"
        )

    flags = [
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
    ]

    sources = source_digest()

    revision = command_output([
        "git",
        "rev-parse",
        "HEAD",
    ])

    samples = []

    with tempfile.TemporaryDirectory(
        prefix="stratax-benchmark-"
    ) as directory:
        binary = Path(directory) / (
            "baseline.exe"
            if os.name == "nt"
            else "baseline"
        )

        subprocess.run(
            [
                *compiler,
                *flags,
                "-I",
                str(ROOT / "include"),
                str(ROOT / "benchmarks/baseline.cpp"),
                "-o",
                str(binary),
            ],
            cwd=ROOT,
            check=True,
        )

        for rows in dict.fromkeys(args.sizes):
            columns = args.columns or rows

            result = command_output([
                str(binary),
                str(rows),
                str(columns),
                str(args.warmups),
                str(args.repeats),
            ])

            for sample in csv.DictReader(
                io.StringIO(result)
            ):
                samples.append({
                    key: (
                        value
                        if key == "case"
                        else float(value)
                        if key == "checksum"
                        else int(value)
                    )
                    for key, value in sample.items()
                })

    if source_digest() != sources:
        raise RuntimeError(
            "Sources changed during the benchmark run; "
            "rerun on a stable checkout"
        )

    groups = {}

    for sample in samples:
        key = (
            sample["case"],
            sample["rows"],
            sample["columns"],
        )

        groups.setdefault(key, []).append(sample)

    results = []

    for (name, rows, columns), group in groups.items():
        if len(group) != args.repeats:
            raise RuntimeError(
                f"expected {args.repeats} samples for "
                f"{name} {rows}x{columns}, "
                f"got {len(group)}"
            )

        sample_ids = sorted(
            sample["sample"]
            for sample in group
        )

        expected_ids = list(
            range(args.repeats)
        )

        if sample_ids != expected_ids:
            raise RuntimeError(
                f"invalid sample sequence: "
                f"{name}, {rows}x{columns}"
            )

        times = [
            sample["elapsed_ns"]
            for sample in group
        ]

        checksums = {
            sample["checksum"]
            for sample in group
        }

        if len(checksums) != 1:
            raise RuntimeError(
                f"unstable checksum: "
                f"{name}, {rows}x{columns}"
            )

        median = statistics.median(times)
        mean = statistics.mean(times)

        if len(times) > 1:
            stdev = statistics.stdev(times)
        else:
            stdev = 0.0

        mad = statistics.median(
            abs(value - median)
            for value in times
        )

        cv = (
            (stdev / mean) * 100
            if mean
            else 0.0
        )

        results.append({
            "case": name,
            "rows": rows,
            "columns": columns,
            "elements": rows * columns,
            "repeats": len(group),
            "checksum": group[0]["checksum"],

            "median_ns": median,
            "mean_ns": mean,
            "stdev_ns": stdev,
            "mad_ns": mad,
            "cv_percent": cv,

            "min_ns": min(times),
            "max_ns": max(times),
        })

    report = {
        "schema_version": 1,
        "benchmark_suite_version": (
            BENCHMARK_SUITE_VERSION
        ),
        "trial": args.trial,

        "metadata": {
            "recorded_at_utc": (
                datetime.now(timezone.utc)
                .isoformat()
            ),

            "git_revision": revision,

            "git_dirty": source_tree_dirty(),

            "source_sha256": sources,

            "platform": platform.platform(),
            "machine": platform.machine(),

            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),

            "compiler": command_output([
                *compiler,
                "--version",
            ]),

            "compiler_command": compiler,
            "compiler_flags": flags,

            "python_version": (
                platform.python_version()
            ),

            "benchmark_backend": "custom",

            "clock": (
                "C++ std::chrono::steady_clock"
            ),

            "warmups": args.warmups,

            "validation": (
                "One full untimed correctness "
                "validation per case; "
                "per-sample checksums outside timing"
            ),

            "timing_scope": (
                "Operation and result allocation; "
                "excludes input setup, checksum, "
                "result destruction, and output"
            ),
        },

        "results": results,
    }

    output = (
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(output)

        print(
            f"Saved {len(results)} validated cases "
            f"to {args.output}",
            file=sys.stderr,
        )

    else:
        print(
            output,
            end="",
        )

    if args.csv:
        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.csv.open(
            "w",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(samples[0]),
            )

            writer.writeheader()
            writer.writerows(samples)


if __name__ == "__main__":
    main()