@page verification Verification

# Verification and release artifacts

CI runs the C++ test suite with GCC, Clang, MSVC, and Apple Clang. Clang also
checks address, undefined behavior, and leak errors. Each compiler separately
compiles every public header in its own C++20 translation unit, which catches
accidental dependence on another header's includes.

```sh
python scripts/check-headers.py
python scripts/check-headers.py --compiler clang++
```

The documentation example checker compiles and runs C++ examples, and runs
Python examples against either the source extension or an installed package.
The installed mode uses isolated Python processes and checks where the package
and native extension came from. CI runs the C++ examples on Linux and the Python
examples across its Python version matrix. The documentation workflow also runs
both sets of examples and fails if Doxygen emits a warning.

```sh
python scripts/check-doc-examples.py --cpp
python scripts/check-doc-examples.py --python
python scripts/check-doc-examples.py --python --installed
doxygen docs/Doxyfile
```

Build the source extension before selecting the default Python mode. Install
the wheel into the active environment before selecting `--installed`.

The coverage job uses a fresh GCC debug build and gcovr 8.6 to publish annotated
HTML, Cobertura XML, JSON, and tool versions in the `cpp-coverage` workflow artifact. Reports
include public headers and record line and branch coverage without a percentage
gate. Download the complete artifact and open `index.html` to inspect uncovered
branches.

```sh
python -m pip install gcovr==8.6
python scripts/check-coverage.py --output-dir build/coverage
```

Use `--compiler` and `--gcov` together when selecting a versioned GCC toolchain.
Additional CMake options can be supplied as `--cmake-arg=-DNAME=VALUE`, including
a local GoogleTest source directory for offline builds. The command produces
reports even if a test fails, then exits unsuccessfully. Build or reporting
failures also fail the job. Counts cover the C++ test suite and the template
instantiations it compiles; they do not measure Python bindings or prove that
all supported type combinations behave correctly. Compiler-generated exception
and cleanup branches can affect the denominator, so compare results from the
same compiler and reporter versions. The options come from the
[gcovr command reference](https://www.gcovr.com/en/stable/manpage.html).

Release jobs use cibuildwheel's test command on Linux, Windows, and macOS. The
repaired wheel is installed into a test environment before upload. The checker
requires both `stratax` and `stratax._core` to be files recorded in the installed
distribution, then runs the Python suite from a temporary directory using
`--installed` and pytest's importlib import mode. This also rejects editable
installs that resolve to the checkout. This follows cibuildwheel's
[wheel testing contract](https://cibuildwheel.pypa.io/en/latest/options/#test-command).

```sh
python -I scripts/check-installed-package.py
python -m build --sdist
python scripts/check-sdist.py dist/stratax-0.3.1.tar.gz
```

The source archive check creates a fresh virtual environment, installs the
archive to force a wheel build, and runs the tests and Python documentation
examples contained in that archive.
It uses neither the checkout's build tree nor its extension. It needs access
to the configured Python package index for build and test dependencies. CI and
release jobs both run it; source and wheel artifacts are uploaded only after
their release checks pass. PyPI publication depends on all platform wheel jobs
and the source archive job succeeding. These workflows validate the artifacts
and architectures they actually build; local execution does not establish that
the full hosted platform matrix has passed.
