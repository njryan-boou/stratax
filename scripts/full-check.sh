#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

echo "================================"
echo "          FULL CHECK"
echo "================================"

# Build + C++ tests
SUPPRESS_PAUSE=1 "$SCRIPT_DIR/build-test.sh"

# Python tests
echo
SUPPRESS_PAUSE=1 "$SCRIPT_DIR/python-test.sh"

echo
echo "================================"
echo "✓ ALL CHECKS PASSED"
echo "================================"
pause_for_enter