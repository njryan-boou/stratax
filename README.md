# Stratax

Stratax is a modern scientific computing library built around modern C++20
containers, shape metadata, and element-wise array operations. The core library
is header-first, with Python bindings powered by pybind11.

The project is still young: vectors, matrices, tensors, shape/stride metadata,
indexing, slicing, reshaping, conversions, creation helpers, printing,
comparison, and arithmetic are active. Linear algebra, calculus, random, and
statistics modules are currently reserved API areas.

## Current Features

- C++20 `Vector`, `Matrix`, and `Tensor` containers
- Contiguous `Buffer` storage with shape and stride metadata
- Bounds-checked `at(...)` and multidimensional `operator(...)` access
- Element-wise arithmetic and comparison operators
- Reshape, flatten, slicing, and container conversion helpers
- Tensor creation helpers such as `zeros`, `ones`, `full`, and `identity`
- Stream printing for vectors, matrices, tensors, shapes, and strides
- Python bindings for `Shape`, `Vector`, `Matrix`, and `Tensor`
- Doxygen API documentation

## Requirements

- C++20 compiler
- CMake 3.20 or newer
- Python 3.10 or newer for bindings
- pybind11 2.12 or newer
- scikit-build-core 0.10 or newer
- pytest 8 or newer for Python tests
- Doxygen 1.17 or newer for API docs

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

shape = Shape([2, 2])
vector = Vector([1.0, 2.0, 3.0])
matrix = Matrix([[1.0, 2.0], [3.0, 4.0]])
tensor = Tensor([2, 2], 1.0)

tensor[1, 1] = 9.0

print(shape.elements)
print(vector.tolist())
print(matrix.tolist())
print(tensor.tolist())
```

Python bindings currently expose `double`-based `Shape`, `Vector`, `Matrix`,
and `Tensor` wrappers. The C++ API remains the primary interface while the
binding surface grows.

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

If you configure through CMake and Doxygen is available, you can also run:

```powershell
cmake --build build --target docs
```

## Repository Layout

```text
include/stratax/      Public C++ headers
bindings/             pybind11 binding sources
python/stratax/       Python package
tests/cpp/            C++ tests
tests/python/         Python tests
examples/cpp/         C++ examples
examples/python/      Python examples
docs/cpp/             C++ API documentation
docs/python/          Python documentation
```

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
- Operations: arithmetic, comparison, indexing, reshape, slicing
- Container helpers: creation and conversions
- I/O: stream printing
- Python bindings: `Shape`, `Vector`, `Matrix`, `Tensor`

## Roadmap

- Broadcasting
- Logical operations
- Linear algebra algorithms
- Calculus helpers
- Random sampling and distributions
- Statistics routines
- CSV and binary I/O
