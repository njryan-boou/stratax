#!/usr/bin/env python3
"""Check source release versions and an optional v-prefixed publication tag.

Requires packaging and, on Python 3.10, tomli. CMake uses the numeric release
base; Python metadata, runtime, and documentation keep the full PEP 440 version.
"""

import argparse
from pathlib import Path
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]


def one_value(root, relative, pattern):
    matches = re.findall(pattern, (root / relative).read_text(encoding="utf-8"), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one version declaration in {relative}; found {len(matches)}")
    return matches[0]


def check_version(root, tag=None):
    """Return the canonical version only when all release declarations agree."""
    with (root / "pyproject.toml").open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    parsed = Version(version)
    if str(parsed) != version:
        raise ValueError(f"pyproject.toml must use canonical version {str(parsed)!r}")
    if len(parsed.release) != 3 or parsed.epoch or parsed.local:
        raise ValueError("Release versions must use major.minor.patch without an epoch or local suffix")

    declarations = {
        "bindings/module.cpp": one_value(root, "bindings/module.cpp", r'^constexpr const char\* PyVersion = "([^"\n]+)";$'),
        "docs/Doxyfile": one_value(root, "docs/Doxyfile", r'^PROJECT_NUMBER[ \t]*=[ \t]*(.*?)[ \t]*$'),
        "docs/mainpage.md": one_value(root, "docs/mainpage.md", r'^\*\*Stratax v([^*\s]+)\*\*'),
    }
    for path, actual in declarations.items():
        if path == "docs/Doxyfile" and actual == f'"{version}"':
            actual = version
        if actual != version:
            raise ValueError(f"{path} declares {actual!r}; expected {version!r}")

    cmake = one_value(root, "CMakeLists.txt", r'^\s*VERSION\s+([^\s)]+)\s*$')
    base = ".".join(map(str, parsed.release))
    if cmake != base:
        raise ValueError(f"CMakeLists.txt declares {cmake!r}; expected numeric base {base!r}")
    changelog = (root / "docs/guides/CHANGELOG.md").read_text(encoding="utf-8")
    entries = re.findall(r'^## \[([^]\n]+)\](?: - [^\n]+)?\s*$', changelog, re.MULTILINE)
    released = [entry for entry in entries if entry != "Unreleased"]
    if len(released) != len(set(released)):
        raise ValueError("CHANGELOG.md contains duplicate release entries")
    if not released or released[0] != version:
        raise ValueError(f"The newest CHANGELOG.md release entry must be [{version}]")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match source version; expected 'v{version}'")
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require an exact v-prefixed release tag")
    parser.add_argument("--github-output", type=Path, help="Append version to this GitHub Actions output file")
    args = parser.parse_args()
    try:
        version = check_version(ROOT, args.tag)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as output:
                output.write(f"version={version}\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Release version check failed: {error}\n")
    print(f"Verified source release {version}" + (f" and tag {args.tag}" if args.tag is not None else ""))


if __name__ == "__main__":
    main()
