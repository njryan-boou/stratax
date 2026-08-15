#!/usr/bin/env bash

cd ~/Development/stratax

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Error: .venv does not exist."
    echo "Use the VENV button first."
    echo
    read -rp "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate

echo "Starting Python..."
echo

python
