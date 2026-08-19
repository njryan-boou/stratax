#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

source "$SCRIPT_DIR/pause-on-enter.sh"

echo "================================"
echo "       PYTHON BUILD + TEST"
echo "================================"

# Check virtual environment
if [[ ! -x "$PYTHON" ]]; then
    echo "✗ Virtual environment not found"
    echo "  Expected: $ROOT/.venv"
    exit 1
fi

# Check pytest
if ! "$PYTHON" -c "import pytest" 2>/dev/null; then
    echo "✗ pytest is not installed"
    echo "  Run: $PYTHON -m pip install pytest"
    exit 1
fi

# Build the Python extension first unless the caller has already done so.
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "==> Configuring Stratax"
    cmake \
        -S "$ROOT" \
        -B "$ROOT/build" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DSTRATAX_BUILD_PYTHON_BINDINGS=ON \
        -DPython_EXECUTABLE="$PYTHON"

    echo "==> Building Stratax"
    cmake --build "$ROOT/build" --parallel
    echo "✓ Build passed"
else
    echo "==> Skipping build (SKIP_BUILD=1)"
fi

echo
echo "==> Running Python tests"

cd "$ROOT"

# Avoid installed editable-import hooks taking precedence over the workspace
# package. With site initialization disabled, both paths are explicit and the
# freshly built extension copied into python/stratax is always tested.
SITE_PACKAGES="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYTHONPATH="$ROOT/python:$SITE_PACKAGES" "$PYTHON" -S -m pytest

echo
echo "================================"
echo "✓ PYTHON TESTS PASSED"
echo "================================"
pause_for_enter
