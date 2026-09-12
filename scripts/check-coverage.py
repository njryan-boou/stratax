#!/usr/bin/env python3
"""Build and run C++ tests with GCC coverage, then report header line/branch coverage."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++", help="GCC C++ compiler")
    parser.add_argument("--gcov", default="gcov", help="gcov matching the compiler version")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum parallel compilations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "coverage")
    parser.add_argument("--cmake-arg", action="append", default=[], help="Additional CMake option (use --cmake-arg=-D...)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = []
    for command in ([args.compiler, "--version"], [args.gcov, "--version"],
                    ["cmake", "--version"], [sys.executable, "-m", "gcovr", "--version"]):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        versions.append(" ".join(command) + "\n" + result.stdout)
    (output / "toolchain.txt").write_text("\n".join(versions), encoding="utf-8")

    # A fresh build prevents old counters or a different compiler from inflating coverage.
    with tempfile.TemporaryDirectory(prefix="stratax-coverage-") as directory:
        build = Path(directory)
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build),
                        f"-DCMAKE_CXX_COMPILER={args.compiler}", "-DCMAKE_BUILD_TYPE=Debug",
                        "-DSTRATAX_BUILD_TESTS=ON", "-DSTRATAX_BUILD_PYTHON_BINDINGS=OFF",
                        "-DCMAKE_CXX_FLAGS=-O0 -g --coverage", "-DCMAKE_EXE_LINKER_FLAGS=--coverage",
                        *args.cmake_arg], check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel", str(args.jobs)], check=True)
        # GoogleTest discovery executes the binary during the build; count only test execution.
        for counter in build.rglob("*.gcda"):
            counter.unlink()
        tests = subprocess.run(["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"])
        subprocess.run([sys.executable, "-m", "gcovr", "--root", str(ROOT), "--filter", "include/",
                        "--gcov-executable", args.gcov, "--print-summary",
                        "--html-details", str(output / "index.html"),
                        "--xml", str(output / "coverage.xml"),
                        "--json", str(output / "coverage.json"), str(build)], cwd=ROOT, check=True)
        tests.check_returncode()
    print(f"Coverage reports: {output}")


if __name__ == "__main__":
    main()
