"""Release checks reject inconsistent or incomplete distributable archives."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARTIFACTS = load_script("check-release-artifacts.py")
INSTALLED = load_script("check-installed-package.py")
VERSION = Version("0.4.0rc1")
METADATA = b"Metadata-Version: 2.1\nName: stratax\nVersion: 0.4.0rc1\n\nRelease candidate.\n"
DIST_INFO = "stratax-0.4.0rc1.dist-info"
SDIST_ROOT = "stratax-0.4.0rc1"


def wheel_files(extension="_core.cpython-313-x86_64-linux-gnu.so"):
    return {
        f"{DIST_INFO}/METADATA": METADATA,
        f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp313-cp313-linux_x86_64\n",
        f"{DIST_INFO}/RECORD": b"stratax/py.typed,,0\n",
        "stratax/__init__.py": b"from ._core import __version__\n",
        "stratax/_core.pyi": b"__version__: str\n",
        "stratax/py.typed": b"",
        f"stratax/{extension}": b"native-extension-placeholder",
    }


def sdist_files():
    return {
        f"{SDIST_ROOT}/PKG-INFO": METADATA,
        f"{SDIST_ROOT}/pyproject.toml": b'[project]\nname = "stratax"\nversion = "0.4.0rc1"\n',
        f"{SDIST_ROOT}/python/stratax/__init__.py": b"from ._core import __version__\n",
        f"{SDIST_ROOT}/python/stratax/_core.pyi": b"__version__: str\n",
        f"{SDIST_ROOT}/python/stratax/py.typed": b"",
    }


class ArchiveChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)

    def write_zip(self, files, filename="stratax-0.4.0rc1-cp313-cp313-linux_x86_64.whl"):
        archive = self.work / filename
        with zipfile.ZipFile(archive, "w") as output:
            for name, data in files.items():
                output.writestr(name, data)
        return archive

    def write_tar(self, files):
        archive = self.work / "stratax-0.4.0rc1.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        return archive

    def verify(self, archive):
        with redirect_stdout(io.StringIO()):
            ARTIFACTS.check_artifact(archive, VERSION)

    def test_valid_native_wheels_include_linux_windows_and_macos(self):
        for extension, tag in (("_core.cpython-313-x86_64-linux-gnu.so", "linux_x86_64"),
                               ("_core.cp313-win_amd64.pyd", "win_amd64"),
                               ("_core.cpython-313-darwin.so", "macosx_11_0_arm64")):
            with self.subTest(extension=extension):
                self.verify(self.write_zip(wheel_files(extension), f"stratax-0.4.0rc1-cp313-cp313-{tag}.whl"))

    def test_valid_tar_and_zip_source_archives(self):
        self.verify(self.write_tar(sdist_files()))
        self.verify(self.write_zip(sdist_files(), "stratax-0.4.0rc1.zip"))

    def test_missing_wheel_typing_or_extension_fails(self):
        for name in ("stratax/_core.pyi", "stratax/py.typed", "stratax/__init__.py",
                     "stratax/_core.cpython-313-x86_64-linux-gnu.so"):
            with self.subTest(name=name):
                files = wheel_files()
                del files[name]
                with self.assertRaises(ValueError):
                    self.verify(self.write_zip(files))

    def test_empty_native_extension_or_stub_fails(self):
        for name in ("stratax/_core.pyi", "stratax/_core.cpython-313-x86_64-linux-gnu.so"):
            with self.subTest(name=name):
                files = wheel_files()
                files[name] = b""
                with self.assertRaises(ValueError):
                    self.verify(self.write_zip(files))

    def test_multiple_native_extensions_fail(self):
        files = wheel_files()
        files["stratax/_core.cp313-win_amd64.pyd"] = b"other platform"
        with self.assertRaises(ValueError):
            self.verify(self.write_zip(files))

    def test_sdist_needs_root_metadata_and_typing_sources(self):
        for name in (f"{SDIST_ROOT}/PKG-INFO", f"{SDIST_ROOT}/python/stratax/py.typed"):
            with self.subTest(name=name):
                files = sdist_files()
                del files[name]
                with self.assertRaises(ValueError):
                    self.verify(self.write_tar(files))

    def test_filename_version_or_project_mismatch_fails(self):
        for filename in ("stratax-0.3.1-cp313-cp313-linux_x86_64.whl",
                         "elsewhere-0.4.0rc1-cp313-cp313-linux_x86_64.whl"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.verify(self.write_zip(wheel_files(), filename))

    def test_missing_duplicate_or_inconsistent_metadata_fields_fail(self):
        metadata_cases = (
            METADATA.replace(b"0.4.0rc1", b"0.3.1"),
            METADATA.replace(b"0.4.0rc1", b"not-a-version"),
            METADATA.replace(b"Name: stratax", b"Name: another-project"),
            METADATA.replace(b"Version: 0.4.0rc1\n", b""),
            METADATA.replace(b"Name: stratax\n", b""),
            METADATA.replace(b"Version: 0.4.0rc1", b"Version: 0.4.0rc1\nVersion: 0.3.1"),
            METADATA.replace(b"Name: stratax", b"Name: stratax\nName: elsewhere"),
        )
        for metadata in metadata_cases:
            with self.subTest(metadata=metadata):
                files = wheel_files()
                files[f"{DIST_INFO}/METADATA"] = metadata
                with self.assertRaises(ValueError):
                    self.verify(self.write_zip(files))

    def test_multiple_distribution_metadata_directories_fail(self):
        files = wheel_files()
        files["another-1.0.dist-info/METADATA"] = METADATA
        with self.assertRaises(ValueError):
            self.verify(self.write_zip(files))

    def test_duplicate_zip_members_fail(self):
        archive = self.write_zip(wheel_files())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "a") as output:
                output.writestr("stratax/py.typed", b"")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.verify(archive)

    def test_archive_paths_cannot_escape_root(self):
        for member in ("../unexpected", "/absolute", "stratax\\unexpected", "stratax/./unexpected"):
            with self.subTest(member=member):
                files = wheel_files()
                files[member] = b"unexpected"
                with self.assertRaisesRegex(ValueError, "archive path"):
                    self.verify(self.write_zip(files))

    def test_tar_links_fail(self):
        archive = self.work / "stratax-0.4.0rc1.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            member = tarfile.TarInfo(f"{SDIST_ROOT}/PKG-INFO")
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside/PKG-INFO"
            output.addfile(member)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.verify(archive)

    def test_source_archive_cannot_mix_root_directories(self):
        files = sdist_files()
        files["another-root/README.md"] = b"unexpected"
        with self.assertRaisesRegex(ValueError, "must be under"):
            self.verify(self.write_tar(files))

    def test_patterns_expand_and_duplicate_inputs_are_deduplicated(self):
        wheel = self.write_zip(wheel_files())
        sdist = self.write_tar(sdist_files())
        self.assertEqual(set(ARTIFACTS.expand_artifacts([str(self.work / "*"), str(wheel)])),
                         {wheel, sdist})

    def test_unmatched_patterns_empty_inputs_and_directories_fail(self):
        for patterns in ([], [str(self.work / "*.whl")], [str(self.work)]):
            with self.subTest(patterns=patterns), self.assertRaises(ValueError):
                ARTIFACTS.expand_artifacts(patterns)

    def test_cli_fails_closed_for_invalid_versions_and_missing_inputs(self):
        archive = self.write_zip(wheel_files())
        for args in (("--version", "invalid", str(archive)),
                     ("--version", "v0.4.0rc1", str(archive)),
                     ("--version", "0.4.0rc1"),
                     ("--version", "0.4.0rc1", str(self.work / "absent*.whl"))):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/check-release-artifacts.py"),
                                         *args], text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Verified", result.stdout)


class InstalledChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.package = self.work / "site-packages" / "stratax"
        self.package.mkdir(parents=True)
        self.paths = []
        for name in ("__init__.py", "_core.so", "_core.pyi", "py.typed"):
            path = self.package / name
            path.write_text("", encoding="utf-8")
            self.paths.append(path)
        self.stratax = ModuleType("stratax")
        self.stratax.__file__ = str(self.package / "__init__.py")
        self.stratax.__version__ = str(VERSION)
        self.core = ModuleType("stratax._core")
        self.core.__file__ = str(self.package / "_core.so")
        self.core.__version__ = str(VERSION)
        self.stratax._core = self.core
        self.distribution = SimpleNamespace(version=str(VERSION), files=self.paths,
                                            locate_file=lambda value: value)

    def verify(self):
        with patch.dict(sys.modules, {"stratax": self.stratax, "stratax._core": self.core}), \
                patch("importlib.metadata.distribution", return_value=self.distribution), \
                redirect_stdout(io.StringIO()):
            exec(INSTALLED.PROVENANCE, {"SOURCE": str(self.work / "checkout")})

    def test_matching_installed_distribution_passes(self):
        self.verify()

    def test_each_module_version_must_match_distribution(self):
        for module in (self.stratax, self.core):
            with self.subTest(module=module.__name__):
                module.__version__ = "0.3.1"
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    self.verify()
                module.__version__ = str(VERSION)

    def test_typing_files_must_be_recorded_and_physically_installed(self):
        for name in ("_core.pyi", "py.typed"):
            with self.subTest(name=name):
                path = self.package / name
                self.distribution.files = [entry for entry in self.paths if entry != path]
                with self.assertRaisesRegex(RuntimeError, "typing file"):
                    self.verify()
                self.distribution.files = self.paths
                path.unlink()
                with self.assertRaisesRegex(RuntimeError, "typing file"):
                    self.verify()
                path.touch()

    def test_unrecorded_module_still_fails_provenance(self):
        self.distribution.files = [path for path in self.paths if path.name != "_core.so"]
        with self.assertRaisesRegex(RuntimeError, "not from the installed distribution"):
            self.verify()

    def test_checkout_module_still_fails_provenance(self):
        self.stratax.__file__ = str(self.work / "checkout/python/stratax/__init__.py")
        self.distribution.files = [*self.paths, Path(self.stratax.__file__)]
        with self.assertRaisesRegex(RuntimeError, "not from the installed distribution"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
