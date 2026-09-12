@page getting_started Getting Started

# Getting Started

Run commands from the repository root unless stated otherwise. C++ consumers
need a C++20 compiler and the include directory. Building repository tests needs
CMake 3.20 or newer; Python bindings also need Python 3.10 or newer and its
extension-development headers. GCC, Clang, Apple Clang, and MSVC builds are
configured in CI; Visual Studio is a Windows option, not a universal dependency.

## Checkout and Python setup

```sh
git clone https://github.com/njryan-boou/stratax.git
cd stratax
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS,
`.venv\Scripts\Activate.ps1` in PowerShell, or
`.venv\Scripts\activate.bat` in Command Prompt.

```sh
python -m pip install -e ".[dev]"
python -m pytest tests/python
python examples/python/vector.py
```

pip installs the scikit-build-core/pybind11 build requirements in an isolated
build environment. The dev extra supplies pytest. The extension build copies
_core into python/stratax, which source tests and the documentation checker use.
Direct Python examples import the package installed in the active environment.
Rebuild after C++ changes; do not assume an existing binary follows edited headers.

## C++ tests

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The configuration uses an installed GoogleTest when available, otherwise fetches
it. `STRATAX_BUILD_TESTS=OFF` disables that test dependency. On single-config
GCC/Clang generators, set `-DCMAKE_BUILD_TYPE=Release` at configure time when
Release is desired; --config selects the build on multi-config generators.

## Build a source extension explicitly

Inside the active virtual environment:

```sh
python -m pip install "pybind11>=2.12"
cmake -S . -B build-python -DSTRATAX_BUILD_PYTHON_BINDINGS=ON -DSTRATAX_BUILD_TESTS=OFF
cmake --build build-python --config Release --target _core
python -m pytest tests/python
```

If multiple interpreters exist, pass `-DPython_EXECUTABLE=/absolute/path/to/python`
when configuring. Python bindings are OFF in a plain CMake configuration.
Use a separate build directory for different interpreters or generators.
To test a separately installed current wheel, use pytest's `--installed` option.

## C++ example and documentation

With GCC or Clang:

```sh
c++ -std=c++20 -I include examples/cpp/vector.cpp -o vector_example
./vector_example
```

Install Doxygen and Graphviz to build the full HTML reference, then run:

```sh
doxygen docs/Doxyfile
```

Open docs/output/html/index.html. The `docs` CMake target is available when
Doxygen was found during configuration. See @ref user_guide, @ref examples,
@ref python_api, and @ref troubleshooting.
