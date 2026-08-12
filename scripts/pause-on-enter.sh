#!/usr/bin/env bash

pause_for_enter() {
    if [[ "${SUPPRESS_PAUSE:-0}" == "1" ]]; then
        return
    fi

    if [[ -t 0 ]]; then
        read -r -p "Press Enter to exit... " _
    else
        echo "Press Enter to exit..."
        read -r _
    fi
}

handle_script_failure() {
    local exit_code=$?

    echo
    echo "Script failed with exit code ${exit_code}."

    if [[ "${SUPPRESS_PAUSE:-0}" == "1" ]]; then
        exit "$exit_code"
    fi

    pause_for_enter
    exit "$exit_code"
}

trap 'handle_script_failure' ERR
