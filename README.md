# Stratax

Stratax is a modern scientific computing library built around modern C++20
containers, shape metadata, and element-wise array operations. The core library
is header-first, with Python bindings powered by pybind11.

The project is still young: vectors, matrices, tensors, shape/stride metadata,
indexing, slicing, reshaping, conversions, creation helpers, printing,
comparison, and arithmetic are active. Linear algebra, calculus, random, and
statistics modules are currently reserved API areas.

## Features

- C++20 `Vector`, `Matrix`, and `Tensor` containers
- Contiguous `Buffer` storage with shape and stride metadata
- Bounds-checked `at(...)` and multidimensional `operator(...)` access
- Negative indexing and reverse slicing in both the C++ and Python APIs
- Element-wise arithmetic and comparison operators
- Reshape, flatten, slicing, and container conversion helpers
- Tensor creation helpers such as `zeros`, `ones`, `full`, and `identity`
- Axis reductions with optional `keepdims` and negative-axis support in the Python API
- Stream printing for vectors, matrices, tensors, shapes, and strides
- Python bindings and Python API wrappers for `Shape`, `Vector`, `Matrix`, and `Tensor`
- Python free functions for conversions and tensor creation helpers
- Doxygen API documentation

## Quick Start

### C++

```cpp
#include <stratax.h>
#include <iostream>

using namespace stratax::container;

int main()
{
    Vector<double> a{1.0, 2.0, 3.0};
    Vector<double> b{4.0, 5.0, 6.0};

    auto c = a + b;

    std::cout << c << '\n'; // [5, 7, 9]
}
```

### Python

```python
from stratax import Vector

v = Vector([1.0, 2.0, 3.0, 4.0])

print(v[::-1].tolist())

# [4.0, 3.0, 2.0, 1.0]
```

Python bindings expose `double`-based `Shape`, `Vector`, `Matrix`, and `Tensor` wrappers plus free-function conversions (`to_vector`, `to_matrix`, `to_tensor`) and creation helpers (`zeros`, `ones`, `full`, `identity`). The API is intentionally Python-like: use negative indexes, slice steps (including reverse slices), and `tolist()` for quick inspection.

Axis reductions are available through Python reduction helpers (`sum`, `prod`, `max`, `min`, `argmax`, `argmin`, `mean`, `var`, `std`), including `keepdims=True` for shape-preserving reductions and negative-axis indexing (for example `axis=-1` for the last dimension).

## Installation

### Python

```powershell
pip install stratax
```

### C++

Stratax is header-first for C++. Include the repository `include/` directory in
your build and include the umbrella header:

```cpp
#include <stratax.h>
```

For a small CMake target, wire the include directory and C++ standard like this:

```cmake
target_include_directories(my_app PRIVATE path/to/stratax/include)
target_compile_features(my_app PRIVATE cxx_std_20)
```

## Build & Development

### Requirements

- C++20 compiler
- CMake 3.20 or newer
- Python 3.10 or newer for bindings
- pybind11 2.12 or newer
- scikit-build-core 0.10 or newer
- pytest 8 or newer for Python tests
- Doxygen 1.9 or newer for API docs

### Build From Source

Configure and build with CMake:

```powershell
cmake -S . -B build
cmake --build build
```

For editable Python installation, use pip from an environment with the build
dependencies available:

```powershell
python -m pip install -e .
```

### Run Tests

The C++ tests live under `tests/cpp/` and are registered with CTest through
GoogleTest:

```powershell
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The Python tests live under `tests/python/` and use pytest:

```powershell
python -m pytest tests/python
```

### Documentation

 - [Documentation Site](https://njryan-boou.github.io/stratax)
 - [Local Documentation Index](docs/README.md)
 - [User Guide](docs/guides/user_guide.md)
 - [Python API Reference](docs/guides/python_api.md)
 - [Examples Guide](docs/guides/examples.md)
 - [Troubleshooting](docs/guides/troubleshooting.md)
 - [Releasing](docs/guides/releasing.md)

## Repository Layout

- [include/stratax/](include/stratax/) Public C++ headers
- [bindings/](bindings/) pybind11 binding sources
- [python/stratax/](python/stratax/) Python package
- [tests/cpp/](tests/cpp/) C++ tests
- [tests/python/](tests/python/) Python tests
- [examples/cpp/](examples/cpp/) C++ examples
- [examples/python/](examples/python/) Python examples
- [docs/](docs/) Documentation and API reference

## Project Status

Stratax is still evolving. The current container, indexing, slicing, reshape,
conversion, creation, printing, arithmetic, comparison, and reduction APIs are
usable, but naming and edge-case behavior may still change before a stable 1.0
release. Linear algebra, calculus, random, statistics, broadcasting, and file
I/O are reserved or planned areas.

### API Status

| Feature | C++ | Python |
|---------|-----|--------|
| `Shape`, `Vector`, `Matrix`, `Tensor` | Available | Available |
| Negative indexing | Available | Available |
| Slicing | Copy-based | Copy-based |
| Arithmetic and comparison | Available | Available |
| Bitwise operators | Available for integral containers | Not yet exposed |
| Creation helpers | Available | Available |
| Conversions | Available | Available |
| Reshape and flatten | Available | Available |
| Reductions | Available | Available, with `keepdims` |
| Broadcasting | Planned | Planned |
| Linear algebra | Planned | Planned |

### Implemented

- Core storage and metadata: `Buffer`, `Shape`, `Strides`, `Slice`
- Containers: `Vector`, `Matrix`, `Tensor`
- Operations: arithmetic, comparison, indexing, negative indexing, reshape, slicing
- Container helpers: creation and conversions
- I/O: stream printing
- Python bindings and Python API: `Shape`, `Vector`, `Matrix`, `Tensor`, conversion free functions, creation free functions, negative indexing, reverse slicing

### Roadmap

- Broadcasting
- Logical operations
- Linear algebra algorithms
- Calculus helpers
- Random sampling and distributions
- Statistics routines
- CSV and binary I/O

## Contributing

Contributions, bug reports, and feature requests are welcome through GitHub Issues and Pull Requests.
See [Contributing](docs/guides/contributing.md) for development notes and
[Releasing](docs/guides/releasing.md) for release checklist guidance.

## License

Stratax is licensed under the MIT License.
