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

# Build Stratax first unless the caller has already done so.
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    SUPPRESS_PAUSE=1 "$SCRIPT_DIR/build.sh"
else
    echo "==> Skipping build (SKIP_BUILD=1)"
fi

echo
echo "==> Running Python tests"

cd "$ROOT"
"$PYTHON" -m pytest

echo
echo "================================"
echo "✓ PYTHON TESTS PASSED"
echo "================================"
pause_for_enter