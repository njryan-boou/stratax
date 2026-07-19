# Stratax

Stratax is a modern scientific computing library built around modern C++20
containers, shape metadata, and element-wise array operations. The core library
is header-first, with Python bindings powered by pybind11.

The project is still young: vectors, matrices, tensors, shape/stride metadata,
indexing, slicing, reshaping, conversions, creation helpers, printing,
comparison, and arithmetic are active. Linear algebra, calculus, random, and
statistics modules are currently reserved API areas.

## Links

- Homepage: [GitHub Repository](https://github.com/njryan-boou/stratax)
- PyPI: [stratax on PyPI](https://pypi.org/project/stratax/)
- Issues: [GitHub Issues](https://github.com/njryan-boou/stratax/issues)
- Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Getting Started: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- Developer Docs: [docs/dev/README.md](docs/dev/README.md)
- Roadmap: [docs/roadmap.md](docs/roadmap.md)

## Current Features

- C++20 `Vector`, `Matrix`, and `Tensor` containers
- Contiguous `Buffer` storage with shape and stride metadata
- Bounds-checked `at(...)` and multidimensional `operator(...)` access
- Negative indexing and reverse slicing in both the C++ and Python APIs
- Element-wise arithmetic and comparison operators
- Reshape, flatten, slicing, and container conversion helpers
- Tensor creation helpers such as `zeros`, `ones`, `full`, and `identity`
- Stream printing for vectors, matrices, tensors, shapes, and strides
- Python bindings and Python API wrappers for `Shape`, `Vector`, `Matrix`, and `Tensor`
- Python free functions for conversions and tensor creation helpers
- Doxygen API documentation

## Requirements

- C++20 compiler
- CMake 3.20 or newer
- Python 3.10 or newer for bindings
- pybind11 2.12 or newer
- scikit-build-core 0.10 or newer
- pytest 8 or newer for Python tests
- Doxygen 1.9 or newer for API docs

## Quick C++ Example

```cpp
#include <stratax.h>
#include <iostream>

int main()
{
    stratax::container::Vector<double> a{1.0, 2.0, 3.0};
    stratax::container::Vector<double> b{4.0, 5.0, 6.0};

    auto c = a + b;

    std::cout << c << '\n'; // [5, 7, 9]
}
```

## Quick Python Example

```python
from stratax import Matrix, Shape, Tensor, Vector
from stratax import identity, to_matrix, to_tensor, to_vector

# 1) Build a few basic containers.
shape = Shape([2, 3])
scores = Vector([10.0, 20.0, 30.0, 40.0, 50.0])
table = Matrix([[1.0, 2.0], [3.0, 4.0]])
grid = Tensor([2, 2], 1.0)

# 2) Python-like indexing and slicing.
last_score = scores[-1]          # 50.0
every_other_reversed = scores[::-2]
flipped_table = table[::-1, ::-1]

# 3) Update values.
scores[-1] = 99.0
grid[1, 1] = 9.0

# 4) Convert between container shapes.
scores_as_tensor = to_tensor(scores)
grid_as_vector = to_vector(grid)
matrix_from_tensor = to_matrix(Tensor([2, 2], 5.0))

# 5) Use creation helpers.
eye = identity(3)

# 6) Print plain Python data.
print("shape elements:", shape.elements)
print("last score:", last_score)
print("scores:", scores.tolist())
print("reverse step slice:", every_other_reversed.tolist())
print("table flipped:", flipped_table.tolist())
print("grid:", grid.tolist())
print("scores -> tensor:", scores_as_tensor.tolist())
print("grid -> vector:", grid_as_vector.tolist())
print("tensor -> matrix:", matrix_from_tensor.tolist())
print("identity(3):", eye.tolist())
```

Python bindings currently expose `double`-based `Shape`, `Vector`, `Matrix`,
and `Tensor` wrappers plus free-function conversions (`to_vector`,
`to_matrix`, `to_tensor`) and creation helpers (`zeros`, `ones`, `full`,
`identity`). The API is intentionally Python-like: use negative indexes,
slice steps (including reverse slices), and `tolist()` for quick inspection.

## Build From Source

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

## Run Tests

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

## Generate Documentation

Generate API documentation with Doxygen:

```powershell
doxygen Doxyfile
```

The generated HTML entry point is:

```text
docs/cpp/html/index.html
```

Open directly: [docs/cpp/html/index.html](docs/cpp/html/index.html)

If you configure through CMake and Doxygen is available, you can also run:

```powershell
cmake --build build --target docs
```

## Repository Layout

- [include/stratax/](include/stratax/) Public C++ headers
- [bindings/](bindings/) pybind11 binding sources
- [python/stratax/](python/stratax/) Python package
- [tests/cpp/](tests/cpp/) C++ tests
- [tests/python/](tests/python/) Python tests
- [examples/cpp/](examples/cpp/) C++ examples
- [examples/python/](examples/python/) Python examples
- [docs/cpp/](docs/cpp/) C++ API documentation
- [docs/dev/](docs/dev/) Developer documentation and design notes
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) Setup and build guide

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

## License

Stratax is licensed under the MIT License.

## Contributing

Contributions, bug reports, and feature requests are welcome through GitHub Issues and Pull Requests.

## Project Status

Implemented:

- Core storage and metadata: `Buffer`, `Shape`, `Strides`, `Slice`
- Containers: `Vector`, `Matrix`, `Tensor`
- Operations: arithmetic, comparison, indexing, negative indexing, reshape, slicing
- Container helpers: creation and conversions
- I/O: stream printing
- Python bindings and Python API: `Shape`, `Vector`, `Matrix`, `Tensor`, conversion free functions, creation free functions, negative indexing, reverse slicing

## Roadmap

- Reductions
- Broadcasting
- Logical operations
- Linear algebra algorithms
- Calculus helpers
- Random sampling and distributions
- Statistics routines
- CSV and binary I/O
