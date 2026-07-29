@page getting_started Getting Started

# Getting Started

Welcome to Stratax!

This guide walks you through setting up a development environment, building the project, and verifying that everything is working correctly.

Developer-focused design and implementation notes are available in @ref dev_index "Developer Docs".

**Documentation:**

- @subpage user_guide
- @subpage python_api
- @subpage architecture
- @subpage contributing
- @subpage examples
- @subpage troubleshooting
- @subpage changelog
- @subpage roadmap
- @ref dev_index "Developer Docs"

---

## Prerequisites

Install the following software before cloning the repository.

## Required

* Git
* Python 3.10 or newer
* CMake 3.20 or newer
* Visual Studio 2022 (Desktop Development with C++)
* Visual Studio Code (recommended)

Verify your installation:

```powershell
git --version
python --version
cmake --version
```

---

## Clone the Repository

```powershell
git clone https://github.com/njryan-boou/stratax.git
cd stratax
```

---

## Create a Virtual Environment

```powershell
python -m venv .venv
```

---

## Activate the Virtual Environment

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
.venv\Scripts\activate.bat
```

---

## Install Stratax

Install the project in editable mode:

```powershell
python -m pip install -e .
```

This command will:

* Install build dependencies
* Configure CMake
* Build the C++ extension
* Install the Python package into the virtual environment

For a quick Python-only setup from a fresh checkout:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/python/vector.py
```

---

## Build the Project

```powershell
cmake -S . -B build
cmake --build build
```

---

## Running Tests

## Python

```powershell
python -m pytest tests/python
```

## C++

```powershell
ctest --test-dir build --output-on-failure
```

---

## Try the Examples

Example programs are available in `examples/cpp` and `examples/python`.

Run a Python example from the repository root after installing the package:

```powershell
python examples/python/vector.py
```

For C++ examples, include the repository `include/` directory and compile with
C++20 support.

With GCC or Clang-like compilers:

```powershell
g++ -std=c++20 -I include examples/cpp/vector.cpp -o vector_example
```
