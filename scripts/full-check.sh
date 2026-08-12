#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

echo "================================"
echo "          FULL CHECK"
echo "================================"

# Build once up front.
SUPPRESS_PAUSE=1 "$SCRIPT_DIR/build.sh"

echo
echo "==> Running C++ and Python tests in parallel"

SUPPRESS_PAUSE=1 SKIP_BUILD=1 "$SCRIPT_DIR/build-test.sh" &
cpp_pid=$!

SUPPRESS_PAUSE=1 SKIP_BUILD=1 "$SCRIPT_DIR/python-test.sh" &
py_pid=$!

cpp_status=0
py_status=0

wait "$cpp_pid" || cpp_status=$?
wait "$py_pid" || py_status=$?

if [[ $cpp_status -ne 0 || $py_status -ne 0 ]]; then
	echo
	echo "✗ FULL CHECK FAILED"
	echo "  C++ test status: $cpp_status"
	echo "  Python test status: $py_status"
	exit 1
fi

echo
echo "================================"
echo "✓ ALL CHECKS PASSED"
echo "================================"
pause_for_enter