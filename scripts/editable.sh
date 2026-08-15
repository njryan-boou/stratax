#!/usr/bin/env bash

set -e

cd ~/Development/stratax

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Error: .venv does not exist."
    echo
    read -rp "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate

echo "Installing Stratax in editable mode..."
echo

if python -m pip install -e .; then
    echo
    echo "Editable install completed successfully."
    sleep 2
else
    echo
    echo "Editable install FAILED."
    read -rp "Press Enter to close..."
    exit 1
fi