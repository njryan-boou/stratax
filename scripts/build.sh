#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/pause-on-enter.sh"

cd "$ROOT"

echo "==> Configuring Stratax"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

echo "==> Building Stratax"
cmake --build build --parallel

echo "✓ Build passed"
pause_for_enter
