#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "        BUILDING DOCS"
echo "================================"

# Check Doxygen
if ! command -v doxygen &>/dev/null; then
    echo "✗ Doxygen is not installed"
    exit 1
fi

echo
echo "==> Running Doxygen..."
doxygen docs/Doxyfile

echo
echo "✓ Documentation built successfully"

echo
echo "==> Opening documentation..."
xdg-open "$ROOT/docs/output/html/index.html"

echo
echo "================================"
echo "✓ DOCS COMPLETE"
echo "================================"
pause_for_enter