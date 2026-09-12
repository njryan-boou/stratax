@page troubleshooting Troubleshooting

# Troubleshooting

Common setup, build, and import issues while working on Stratax.

## Python Import Cannot Find `_core`

The Python package depends on the compiled pybind11 extension module. Install
the package in editable mode from the repository root:

```powershell
python -m pip install -e .
```

Python examples import the installed package from the active environment. Run
them from the repository root with that environment's interpreter:

```powershell
python examples/python/vector.py
```

If a source test reports a missing or stale extension, rebuild with
`STRATAX_BUILD_PYTHON_BINDINGS=ON` using the intended interpreter; a plain CMake
build leaves bindings disabled. See @ref getting_started. To test an installed
wheel, use `python -m pytest tests/python --installed`.

Editable-install import hooks can take precedence over PYTHONPATH and select an
older extension. Source pytest and scripts/check-doc-examples.py explicitly pin
imports to this checkout. Reinstall after rebuilding if direct examples still
use an outdated installed package.

## Editable Install Fails

Check that Python and build tooling are available:

```powershell
python --version
cmake --version
python -m pip --version
```

Then upgrade build tooling inside your virtual environment:

```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade scikit-build-core pybind11
python -m pip install -e .
```

## CMake Cannot Find a Compiler

On Windows, install Visual Studio 2022 with the "Desktop Development with C++"
workload. Open a fresh terminal after installation and verify CMake can
configure the project:

```powershell
cmake -S . -B build
cmake --build build
```

If multiple compilers are installed, configure from a Visual Studio Developer
PowerShell or pass a CMake generator explicitly.

## C++ Examples Cannot Find `stratax.h`

Stratax is header-first for C++. Add the repository `include` directory to your
compiler or target.

```powershell
g++ -std=c++20 -I include examples/cpp/vector.cpp -o vector_example
```

For CMake targets:

```cmake
target_include_directories(my_app PRIVATE path/to/stratax/include)
target_compile_features(my_app PRIVATE cxx_std_20)
```

## Tests Do Not Run

Build C++ tests before running CTest:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run Python tests from the repository root after installing development
dependencies:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/python
```

## Documentation Looks Stale

The documentation site is generated from Markdown and headers by Doxygen.
GitHub Pages is deployed by `.github/workflows/docs.yml` on pushes to `master`.
Pull requests that touch docs or headers also run `.github/workflows/docs-check.yml`
to check local Markdown links and verify Doxygen generation.

For a local refresh:

```powershell
doxygen docs/Doxyfile
```

Edit Markdown files and headers, not generated files under `docs/output/html`.

For changes to examples or API descriptions, also run
`python scripts/check-doc-examples.py` after rebuilding the source extension.
