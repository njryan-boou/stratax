#!/usr/bin/env python3
"""Verify release archive versions, package contents, and typing files.

Requires ``packaging`` (also installed by build and twine). Pass one or more
wheel/sdist paths or glob patterns; patterns are expanded here on every platform.
This checks archive contents without importing or extracting native code. Release
workflows must also run twine check and test each installed platform wheel.
"""

import argparse
from email import policy
from email.parser import BytesParser
import glob
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


NAME = "stratax"
PACKAGE_FILES = ("__init__.py", "_core.pyi", "py.typed")
EXTENSION = re.compile(r"stratax/_core(?:\.[^/]+)?\.(?:so|pyd)\Z")


def safe_path(name):
    """Reject ambiguous or traversing member names before inspecting an archive."""
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or str(path) != name.rstrip("/")):
        raise ValueError(f"Unsafe or noncanonical archive path: {name!r}")
    return path


def metadata_version(data, expected, location):
    """Require one distribution name and one canonical expected version."""
    message = BytesParser(policy=policy.default).parsebytes(data)
    if message.defects:
        raise ValueError(f"Malformed metadata in {location}: {message.defects}")
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if len(names) != 1 or canonicalize_name(str(names[0])) != NAME:
        raise ValueError(f"Expected one Name: {NAME} field in {location}")
    if len(versions) != 1:
        raise ValueError(f"Expected one Version field in {location}")
    value = str(versions[0])
    if Version(value) != expected or value != str(expected):
        raise ValueError(f"Version in {location} is {value!r}; expected {str(expected)!r}")


def require_files(files, required):
    missing = sorted(set(required) - set(files))
    if missing:
        raise ValueError(f"Archive is missing required files: {', '.join(missing)}")
    empty = sorted(name for name in required if not name.endswith("/py.typed") and not files[name])
    if empty:
        raise ValueError(f"Required archive files are empty: {', '.join(empty)}")


def zip_files(archive):
    """Read regular ZIP entries, rejecting duplicate names and symbolic links."""
    files = {}
    seen = set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            safe_path(member.filename)
            if member.filename in seen:
                raise ValueError(f"Duplicate archive member: {member.filename}")
            seen.add(member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsupported archive link: {member.filename}")
            if not member.is_dir():
                files[member.filename] = source.read(member)
    return files


def tar_files(archive):
    """Read regular tar entries without extraction or following archive links."""
    files = {}
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            safe_path(member.name)
            if member.name in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            if member.isfile():
                with source.extractfile(member) as incoming:
                    files[member.name] = incoming.read()
            elif not member.isdir():
                raise ValueError(f"Unsupported archive member: {member.name}")
    return files


def check_artifact(archive, expected):
    """Validate a wheel or source archive against the requested release version."""
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError(f"Artifact is not a file: {archive}")
    if archive.name.endswith(".whl"):
        name, version, _build, _tags = parse_wheel_filename(archive.name)
        files = zip_files(archive)
        dist_info = f"{NAME}-{expected}.dist-info"
        metadata = f"{dist_info}/METADATA"
        found_metadata = [name for name in files if name.endswith(".dist-info/METADATA")]
        if found_metadata != [metadata]:
            raise ValueError(f"Expected exactly {metadata}; found {found_metadata}")
        require_files(files, [*(f"stratax/{name}" for name in PACKAGE_FILES),
                              metadata, f"{dist_info}/WHEEL", f"{dist_info}/RECORD"])
        extensions = [name for name in files if EXTENSION.fullmatch(name)]
        if len(extensions) != 1 or not files[extensions[0]]:
            raise ValueError(f"Expected one nonempty stratax._core native extension; found {extensions}")
    elif archive.name.endswith((".tar.gz", ".zip")):
        name, version = parse_sdist_filename(archive.name)
        files = tar_files(archive) if archive.name.endswith(".tar.gz") else zip_files(archive)
        root = archive.name[:-7] if archive.name.endswith(".tar.gz") else archive.name[:-4]
        if any(PurePosixPath(member).parts[0] != root for member in files):
            raise ValueError(f"Source archive files must be under {root}/")
        metadata = f"{root}/PKG-INFO"
        require_files(files, [metadata, f"{root}/pyproject.toml",
                              *(f"{root}/python/stratax/{name}" for name in PACKAGE_FILES)])
    else:
        raise ValueError(f"Unsupported release artifact: {archive.name}")
    if name != NAME or version != expected:
        raise ValueError(f"Artifact filename identifies {name} {version}; expected {NAME} {expected}")
    metadata_version(files[metadata], expected, f"{archive.name}:{metadata}")
    print(f"Verified {archive.name}: {NAME} {expected}, package metadata and typing files")


def expand_artifacts(patterns):
    """Resolve every input, rejecting unmatched patterns instead of skipping them."""
    artifacts = []
    for pattern in patterns:
        matches = [pattern] if Path(pattern).is_file() else sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"No artifacts match {pattern!r}")
        for match in matches:
            artifact = Path(match).resolve()
            if not artifact.is_file():
                raise ValueError(f"Artifact is not a file: {artifact}")
            if artifact not in artifacts:
                artifacts.append(artifact)
    if not artifacts:
        raise ValueError("At least one release artifact is required")
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Expected canonical PEP 440 release version")
    parser.add_argument("artifacts", nargs="+", help="Wheel/sdist paths or glob patterns")
    args = parser.parse_args()
    try:
        expected = Version(args.version)
        if str(expected) != args.version:
            raise ValueError(f"Use canonical version {str(expected)!r}")
        for artifact in expand_artifacts(args.artifacts):
            check_artifact(artifact, expected)
    except (ValueError, InvalidVersion, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Release artifact check failed: {error}\n")


if __name__ == "__main__":
    main()
