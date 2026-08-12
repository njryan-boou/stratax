#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "           GIT DIFF"
echo "================================"

echo
git diff --stat

echo
echo "================================"
echo "✓ DIFF COMPLETE"
echo "================================"
pause_for_enter