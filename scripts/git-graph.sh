#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "           GIT GRAPH"
echo "================================"

echo

git log \
    --graph \
    --decorate \
    --oneline \
    --all \
    --color=always

echo
echo "================================"
echo "✓ GIT GRAPH COMPLETE"
echo "================================"
pause_for_enter