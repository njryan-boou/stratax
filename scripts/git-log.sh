#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "            GIT LOG"
echo "================================"

echo

git log \
    --graph \
    --oneline \
    --decorate \
    --all \
    -20

echo
echo "================================"
echo "Showing latest 20 commits"
echo "================================"
pause_for_enter