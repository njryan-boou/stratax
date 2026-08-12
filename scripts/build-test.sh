#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

echo "================================"
echo "        BUILD + TEST"
echo "================================"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    SUPPRESS_PAUSE=1 "$SCRIPT_DIR/build.sh"
else
    echo "==> Skipping build (SKIP_BUILD=1)"
fi

echo
echo "==> Running C++ tests"

ctest \
    --test-dir "$ROOT/build" \
    --output-on-failure \
    --parallel

echo
echo "================================"
echo "✓ BUILD + TEST PASSED"
echo "================================"
pause_for_enter
