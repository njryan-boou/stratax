# Stratax {#mainpage}

**Stratax** is a modern C++ library for efficient multi-dimensional array
operations, with Python bindings through pybind11.

## Features

- **Multi-dimensional arrays**: Vector, Matrix, and Tensor classes for structured data
- **Flexible operations**: Arithmetic, comparison, indexing, and slicing operations
- **Safe validation**: Built-in shape, dimension, type, and index validation
- **Python integration**: Python API via the `_core` extension module
- **Efficient computation**: Row-major storage with shape and stride metadata

## Quick Links

- @ref user_guide "User Guide" - Learn the current C++ and Python APIs
- @ref python_api "Python API Reference" - Scan the Python wrapper API
- @ref getting_started "Getting Started" - Set up and verify your installation
- @ref architecture "Architecture" - Understand the library design
- @ref examples "Examples" - Try focused C++ and Python examples
- @ref troubleshooting "Troubleshooting" - Diagnose common setup and build issues
- @ref dev_index "Developer Docs" - Deep dive into core concepts and implementation

## Core Components

### Data Structures

- **Vector** - 1D array type for sequential data
- **Matrix** - 2D array type for tabular data
- **Tensor** - N-dimensional array type for general use
- **Shape** - Dimension information and rank tracking
- **Buffer** - Underlying contiguous storage management

### Operations

- **Arithmetic** - Element-wise addition, subtraction, multiplication, division
- **Comparison** - Equality and inequality testing
- **Indexing** - Multi-dimensional array access with bounds checking
- **Slicing** - Sub-array extraction with copy-based results
- **Algorithms** - Creation helpers, conversion helpers, reductions, and reshaping

### Validation

- Shape compatibility checking
- Type consistency enforcement
- Index boundary validation
- Dimension count verification

## Documentation Structure

| Section | Purpose |
| --------- | --------- |
| Guides | Getting started, setup instructions, examples, and project workflow |
| API Reference | C++ header documentation and usage examples |
| Developer Docs | Architecture, design decisions, and contributor guidelines |

## Links

- [GitHub Repository](https://github.com/njryan-boou/stratax)
- [Python Package](https://pypi.org/project/stratax)
- License: MIT

---

**Stratax v0.3.0** - Efficient, validated array operations for C++ and Python.
