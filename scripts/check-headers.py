#!/usr/bin/env python3
"""Compile each public header as the only include in its own C++20 translation unit."""

import argparse
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="C++ compiler executable; otherwise use CMake's platform default")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum parallel compilations")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    headers = sorted(path for path in (ROOT / "include").rglob("*") if path.suffix in {".h", ".hpp"})
    if not headers:
        parser.error("No public headers found")

    with tempfile.TemporaryDirectory(prefix="stratax-headers-") as directory:
        work = Path(directory)
        project = [
            "cmake_minimum_required(VERSION 3.20)",
            "project(StrataxHeaderChecks LANGUAGES CXX)",
            "set(CMAKE_CXX_STANDARD 20)",
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            "set(CMAKE_CXX_EXTENSIONS OFF)",
            f"include_directories([=[{(ROOT / 'include').as_posix()}]=])",
        ]
        for index, header in enumerate(headers):
            source = f"header_{index}.cpp"
            relative = header.relative_to(ROOT / "include").as_posix()
            (work / source).write_text(f"#include <{relative}>\n", encoding="utf-8")
            project.append(f"add_library(header_{index} OBJECT {source})")
        (work / "CMakeLists.txt").write_text("\n".join(project) + "\n", encoding="utf-8")

        configure = ["cmake", "-S", str(work), "-B", str(work / "build")]
        if args.compiler:
            configure.append(f"-DCMAKE_CXX_COMPILER={args.compiler}")
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", str(work / "build"), "--config", "Release",
                        "--parallel", str(args.jobs)], check=True)
    print(f"{len(headers)} public headers compile independently")


if __name__ == "__main__":
    main()
