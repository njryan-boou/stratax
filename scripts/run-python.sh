#!/usr/bin/env bash

cd ~/Development/stratax

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Error: .venv does not exist."
    echo "Create the virtual environment first."
    read -rp "Press Enter to close..."
    exit 1
fi

if [[ ! -f "main.py" ]]; then
    echo "Error: main.py does not exist."
    read -rp "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate

echo "Running main.py..."
echo

if python main.py; then
    echo
    echo "Program exited successfully."
    sleep 2
else
    status=$?
    echo
    echo "Program exited with code $status."
    read -rp "Press Enter to close..."
    exit "$status"
fi