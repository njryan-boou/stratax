@page examples Examples

# Examples

Short examples for trying the current Stratax APIs from C++ and Python.

## C++

The C++ examples live under `examples/cpp`.

| File | Focus |
| ------ | ------- |
| `examples/cpp/vector.cpp` | Vector construction, indexing, arithmetic, and slicing |
| `examples/cpp/matrix.cpp` | Matrix construction, element access, and operations |
| `examples/cpp/tensor.cpp` | Tensor construction, shape-aware access, and slicing |
| `examples/cpp/shape.cpp` | Shape metadata basics |

Stratax is header-first. Point your target at the repository `include`
directory and compile with C++20 enabled:

```cmake
target_include_directories(my_app PRIVATE path/to/stratax/include)
target_compile_features(my_app PRIVATE cxx_std_20)
```

## Python

The Python examples live under `examples/python`.

| File | Focus |
| ------ | ------- |
| `examples/python/vector.py` | Vector construction, indexing, arithmetic, and slicing |
| `examples/python/matrix.py` | Matrix construction, element access, and operations |
| `examples/python/tensor.py` | Tensor construction, shape-aware access, and slicing |
| `examples/python/shape.py` | Shape metadata basics |

After installing the package, run examples from the repository root:

```powershell
python examples/python/vector.py
```

## Notes

- Slicing currently returns copy-based results.
- Python bindings currently expose `double` container specializations.
- Examples track implemented behavior and should be updated when public APIs
  change.
