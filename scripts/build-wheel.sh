#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "          BUILD WHEEL"
echo "================================"

# Check virtual environment
if [[ ! -x "$PYTHON" ]]; then
    echo "✗ Virtual environment not found"
    echo "  Expected: $ROOT/.venv"
    exit 1
fi

# Check Python build package
if ! "$PYTHON" -c "import build" 2>/dev/null; then
    echo "✗ Python 'build' package is not installed"
    echo "  Run: $PYTHON -m pip install build"
    exit 1
fi

echo
echo "==> Removing old distributions..."
rm -rf "$ROOT/dist"

echo
echo "==> Building wheel..."
"$PYTHON" -m build --wheel

echo
echo "==> Built:"
ls -lh "$ROOT/dist/"*.whl

echo
echo "================================"
echo "✓ WHEEL BUILD PASSED"
echo "================================"
pause_for_enter