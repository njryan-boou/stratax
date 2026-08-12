#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

echo "================================"
echo "      CLEAN BUILD + TEST"
echo "================================"

echo "==> Removing old build"
rm -rf "$ROOT/build"

SUPPRESS_PAUSE=1 "$SCRIPT_DIR/build-test.sh"

echo
echo "================================"
echo "✓ CLEAN BUILD + TEST PASSED"
echo "================================"
pause_for_enter
