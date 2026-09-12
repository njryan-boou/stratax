#!/usr/bin/env python3
"""Compile/run documentation examples against this checkout's headers and extension."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FENCES = re.compile(r"^```(cpp|python)\s*\n(.*?)^```\s*$", re.M | re.S)
SOURCE_IMPORT = f"""
import sys
from importlib.machinery import PathFinder

class SourcePackageFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'stratax':
            return PathFinder.find_spec(fullname, [{str(ROOT / 'python')!r}])
        if fullname.startswith('stratax.'):
            spec = PathFinder.find_spec(fullname, path)
            if spec is None:
                raise ModuleNotFoundError(fullname + ' is missing; rebuild the source extension')
            return spec
        return None

sys.meta_path.insert(0, SourcePackageFinder())
"""


def run(command, label, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        print(f"FAIL: {label}", file=sys.stderr)
        print(result.stdout + result.stderr, file=sys.stderr)
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", action="store_true", help="Check C++ examples")
    parser.add_argument("--python", action="store_true", help="Check Python examples")
    parser.add_argument("--compiler", default="c++", help="C++20 compiler executable")
    args = parser.parse_args()
    if not args.cpp and not args.python:
        args.cpp = args.python = True

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.python:
        run([sys.executable, "-c", SOURCE_IMPORT + "\nimport stratax; import stratax._core"],
            "source extension import (build _core first)", cwd=ROOT, env=env)

    counts = {"cpp": 0, "python": 0}
    with tempfile.TemporaryDirectory(prefix="stratax-doc-examples-") as directory:
        work = Path(directory)

        def check(language, code, label):
            if not getattr(args, language):
                return
            counts[language] += 1
            if language == "cpp":
                source = work / "example.cpp"
                binary = work / ("example.exe" if os.name == "nt" else "example")
                source.write_text(code, encoding="utf-8")
                run([args.compiler, "-std=c++20", "-pedantic-errors",
                     "-Werror=narrowing", "-I", str(ROOT / "include"),
                     str(source), "-o", str(binary)], label + " (compile)", cwd=work)
                run([str(binary)], label + " (run)", cwd=work)
            else:
                run([sys.executable, "-c", SOURCE_IMPORT + "\n" + code], label, cwd=work, env=env)

        pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
        for path in pages:
            if "output" in path.relative_to(ROOT).parts:
                continue
            text = path.read_text(encoding="utf-8")
            for match in FENCES.finditer(text):
                line = text[:match.start()].count("\n") + 1
                check(match[1], match[2], f"{path.relative_to(ROOT)}:{line}")

        for language, extension in [("cpp", "cpp"), ("python", "py")]:
            if not getattr(args, language):
                continue
            for path in sorted((ROOT / "examples" / language).glob(f"*.{extension}")):
                if language == "cpp":
                    check(language, path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
                else:
                    counts[language] += 1
                    run([sys.executable, "-c", SOURCE_IMPORT +
                         "\nimport runpy; runpy.run_path(sys.argv[1], run_name='__main__')",
                         str(path)], str(path.relative_to(ROOT)), cwd=ROOT, env=env)

    for language, count in counts.items():
        if getattr(args, language):
            print(f"{language}: {count} examples passed")


if __name__ == "__main__":
    main()
