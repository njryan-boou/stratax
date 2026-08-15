#!/usr/bin/env bash

set -e

cd ~/Development/stratax

if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo
    echo "Virtual environment created."
fi

source .venv/bin/activate

echo
echo "Virtual environment activated."
echo "Python: $(python --version)"
echo
echo "Environment: $VIRTUAL_ENV"
echo

exec bash