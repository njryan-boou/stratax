"""Publication requires matching source versions, release notes, and tag."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_release_version", ROOT / "scripts/check-release-version.py")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseVersionChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write_project()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def write_project(self, version="0.4.0rc1", base="0.4.0"):
        self.write("pyproject.toml", f'[project]\nname = "stratax"\nversion = "{version}"\n')
        self.write("bindings/module.cpp", f'constexpr const char* PyVersion = "{version}";\n')
        self.write("docs/Doxyfile", f'PROJECT_NUMBER = "{version}"\n')
        self.write("docs/mainpage.md", f"# Stratax\n\n**Stratax v{version}**\n")
        self.write("CMakeLists.txt", f"project(stratax\n    VERSION {base}\n    LANGUAGES CXX\n)\n")
        self.write("docs/guides/CHANGELOG.md",
                   f"# Changelog\n\n## [Unreleased]\n\n## [{version}] - 2026-09-14\n\n"
                   "Release notes.\n\n## [0.3.1] - 2026-08-01\n\nEarlier notes.\n")

    def test_release_candidate_uses_full_version_with_numeric_cmake_base(self):
        self.assertEqual(RELEASE.check_version(self.root), "0.4.0rc1")
        self.assertEqual(RELEASE.check_version(self.root, "v0.4.0rc1"), "0.4.0rc1")

    def test_stable_release_and_updated_numeric_base_are_supported(self):
        for version, base in (("0.4.0", "0.4.0"), ("2.1.3rc2", "2.1.3")):
            with self.subTest(version=version):
                self.write_project(version, base)
                self.assertEqual(RELEASE.check_version(self.root, f"v{version}"), version)

    def test_cmake_cannot_contain_the_prerelease_suffix_or_an_old_base(self):
        for base in ("0.4.0rc1", "0.3.1"):
            with self.subTest(base=base):
                self.write_project(base=base)
                with self.assertRaisesRegex(ValueError, "numeric base"):
                    RELEASE.check_version(self.root)

    def test_each_runtime_or_documentation_version_must_agree(self):
        for path in ("bindings/module.cpp", "docs/Doxyfile", "docs/mainpage.md"):
            with self.subTest(path=path):
                self.write_project()
                original = (self.root / path).read_text(encoding="utf-8")
                self.write(path, original.replace("0.4.0rc1", "0.3.1"))
                with self.assertRaisesRegex(ValueError, "declares"):
                    RELEASE.check_version(self.root)

    def test_invalid_noncanonical_or_unsupported_version_forms_fail(self):
        for version in ("not-a-version", "v0.4.0rc1", "0.04.0rc1", "0.4", "0.4.0.1",
                        "1!0.4.0rc1", "0.4.0rc1+local", " 0.4.0rc1 "):
            with self.subTest(version=version):
                self.write_project(version)
                with self.assertRaises(ValueError):
                    RELEASE.check_version(self.root)

    def test_pyproject_version_is_required_and_must_be_text(self):
        for content in ('[project]\nname = "stratax"\n',
                        '[project]\nname = "stratax"\nversion = 4\n'):
            with self.subTest(content=content):
                self.write("pyproject.toml", content)
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    RELEASE.check_version(self.root)

    def test_duplicate_pyproject_version_declarations_fail(self):
        self.write("pyproject.toml", '[project]\nversion = "0.4.0rc1"\nversion = "0.4.0rc1"\n')
        with self.assertRaises(ValueError):
            RELEASE.check_version(self.root)

    def test_absent_or_duplicate_single_version_declarations_fail(self):
        declarations = {
            "bindings/module.cpp": 'constexpr const char* PyVersion = "0.4.0rc1";\n',
            "docs/Doxyfile": 'PROJECT_NUMBER = "0.4.0rc1"\n',
            "docs/mainpage.md": '**Stratax v0.4.0rc1**\n',
            "CMakeLists.txt": '    VERSION 0.4.0\n',
        }
        for path, declaration in declarations.items():
            for content in ("", declaration * 2):
                with self.subTest(path=path, content=content):
                    self.write_project()
                    self.write(path, content)
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        RELEASE.check_version(self.root)

    def test_doxygen_accepts_balanced_quotes_or_no_quotes(self):
        for declaration in ('PROJECT_NUMBER = "0.4.0rc1"\n', 'PROJECT_NUMBER = 0.4.0rc1\n'):
            with self.subTest(declaration=declaration):
                self.write("docs/Doxyfile", declaration)
                self.assertEqual(RELEASE.check_version(self.root), "0.4.0rc1")

    def test_doxygen_rejects_unbalanced_quotes(self):
        for declaration in ('PROJECT_NUMBER = "0.4.0rc1\n', 'PROJECT_NUMBER = 0.4.0rc1"\n'):
            with self.subTest(declaration=declaration):
                self.write("docs/Doxyfile", declaration)
                with self.assertRaises(ValueError):
                    RELEASE.check_version(self.root)

    def test_malformed_runtime_or_documentation_declarations_fail(self):
        malformed = {
            "bindings/module.cpp": 'constexpr const char* PyVersion = "0.4.0rc1;\n',
            "docs/mainpage.md": '**Stratax v0.4.0rc1*\n',
            "CMakeLists.txt": '    VERSION "0.4.0"\n',
        }
        for path, declaration in malformed.items():
            with self.subTest(path=path):
                self.write_project()
                self.write(path, declaration)
                with self.assertRaises(ValueError):
                    RELEASE.check_version(self.root)

    def test_latest_release_entry_must_match_even_if_expected_version_appears_later(self):
        self.write("docs/guides/CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n"
                   "## [0.4.0rc2] - 2026-09-15\n\n## [0.4.0rc1] - 2026-09-14\n")
        with self.assertRaisesRegex(ValueError, "newest"):
            RELEASE.check_version(self.root)

    def test_unreleased_alone_is_not_release_notes(self):
        for changelog in ("# Changelog\n", "# Changelog\n\n## [Unreleased]\n"):
            with self.subTest(changelog=changelog):
                self.write("docs/guides/CHANGELOG.md", changelog)
                with self.assertRaisesRegex(ValueError, "newest"):
                    RELEASE.check_version(self.root)

    def test_changelog_release_date_is_optional(self):
        self.write("docs/guides/CHANGELOG.md", "# Changelog\n\n## [0.4.0rc1]\n\nRelease notes.\n")
        self.assertEqual(RELEASE.check_version(self.root), "0.4.0rc1")

    def test_duplicate_current_release_notes_fail(self):
        self.write("docs/guides/CHANGELOG.md", "# Changelog\n\n## [0.4.0rc1] - 2026-09-14\n\n"
                   "## [0.3.1] - 2026-08-01\n\n## [0.4.0rc1] - 2026-09-13\n")
        with self.assertRaises(ValueError):
            RELEASE.check_version(self.root)

    def test_publication_tag_must_be_exact_including_prerelease_suffix(self):
        for tag in ("v0.4.0", "0.4.0rc1", "v0.4.0rc2", "refs/tags/v0.4.0rc1", "", "v0.4.0rc1 "):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Tag"):
                RELEASE.check_version(self.root, tag)

    def test_missing_source_file_fails_instead_of_skipping_its_check(self):
        (self.root / "bindings/module.cpp").unlink()
        with self.assertRaises(OSError):
            RELEASE.check_version(self.root)


if __name__ == "__main__":
    unittest.main()
