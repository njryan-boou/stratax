# Getting Started

Welcome to Stratax!

This guide walks you through setting up a development environment, building the project, and verifying that everything is working correctly.

---

# Prerequisites

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

# Clone the Repository

```powershell
git clone https://github.com/njryan-boou/stratax.git
cd stratax
```

---

# Create a Virtual Environment

```powershell
python -m venv .venv
```

---

# Activate the Virtual Environment

### PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
.venv\Scripts\activate.bat
```

---

# Install Stratax

Install the project in editable mode:

```powershell
python -m pip install -e .
```

This command will:

* Install build dependencies
* Configure CMake
* Build the C++ extension
* Install the Python package into the virtual environment

---

# Build the Project

```powershell
cmake -S . -B build
cmake --build build
```

---

# Running Tests

## Python

```powershell
python -m pytest tests/python
```

## C++

```powershell
ctest --test-dir build --output-on-failure
```

---
