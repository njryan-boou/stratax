#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "================================"
echo "          GIT BRANCHES"
echo "================================"

echo
git branch -a

echo
echo "================================"
echo "Current branch: $(git branch --show-current)"
echo "================================"
pause_for_enter