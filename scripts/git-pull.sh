#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "           GIT PULL"
echo "================================"

echo
echo "==> Current branch:"
git branch --show-current

echo
echo "==> Pulling changes..."
git pull

echo
echo "================================"
echo "✓ GIT PULL COMPLETE"
echo "================================"
pause_for_enter