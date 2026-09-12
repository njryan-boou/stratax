#!/usr/bin/env python3
"""Check installed package provenance, then run Python tests outside the checkout."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = """
from importlib.metadata import distribution
from pathlib import Path
import stratax
import stratax._core

dist = distribution('stratax')
files = {Path(dist.locate_file(item)).resolve() for item in (dist.files or [])}
source = Path(SOURCE).resolve() / 'python'
for module in (stratax, stratax._core):
    path = Path(module.__file__).resolve()
    if path.is_relative_to(source) or path not in files:
        raise RuntimeError(f'{module.__name__} is not from the installed distribution: {path}')
    print(f'{module.__name__}: {path}')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true", help="Check provenance without running pytest")
    args = parser.parse_args()
    python = [sys.executable, "-I", "-B"]
    with tempfile.TemporaryDirectory(prefix="stratax-installed-tests-") as directory:
        subprocess.run([*python, "-c", "SOURCE = " + repr(str(ROOT)) + "\n" + PROVENANCE],
                       cwd=directory, check=True)
        if not args.imports_only:
            subprocess.run([*python, "-m", "pytest", str(ROOT / "tests" / "python"),
                            "--installed", "--import-mode=importlib", "-p", "no:cacheprovider"],
                           cwd=directory, check=True)


if __name__ == "__main__":
    main()
