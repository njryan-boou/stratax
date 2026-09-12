#!/usr/bin/env python3
"""Install a source archive in a fresh environment and run the archive's own tests."""

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import venv


def unpack(archive, destination):
    """Extract ordinary sdist files without allowing links or paths outside destination."""
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                raise ValueError(f"Unsafe archive path: {member.name}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes extraction directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("The source archive must contain one project directory")
    return roots[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="The .tar.gz source distribution to test")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum parallel compilations")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    archive = args.archive.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="stratax-sdist-") as directory:
        work = Path(directory)
        source = work / "source"
        source.mkdir()
        project = unpack(archive, source)
        for required in ("pyproject.toml", "include", "bindings", "python/stratax/__init__.py",
                         "tests/python/conftest.py", "scripts/check-installed-package.py",
                         "scripts/check-doc-examples.py", "README.md", "docs", "examples/python"):
            if not (project / required).exists():
                raise ValueError(f"Source archive is missing {required}")

        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
        # Installing the archive forces a new build without any checkout build tree.
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-cache-dir",
                        str(archive), "pytest>=8"], cwd=work, env=env, check=True)
        subprocess.run([str(python), "-I", str(project / "scripts/check-installed-package.py")],
                       cwd=work, check=True)
        subprocess.run([str(python), "-I", str(project / "scripts/check-doc-examples.py"),
                        "--python", "--installed"], cwd=work, check=True)
    print(f"Source distribution builds, installs, and passes its tests: {archive.name}")


if __name__ == "__main__":
    main()
