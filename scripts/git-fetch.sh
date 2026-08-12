#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "          GIT FETCH"
echo "================================"

echo
echo "==> Fetching remote changes..."
git fetch --all --prune

echo
echo "==> Repository status:"
git status -sb

echo
echo "================================"
echo "✓ GIT FETCH COMPLETE"
echo "================================"
pause_for_enter