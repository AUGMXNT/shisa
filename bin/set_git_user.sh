#!/bin/bash

# Get the username from the first argument
GIT_USER="$1"

# Check if GIT_USER is empty
if [ -z "$GIT_USER" ]; then
    echo "No GIT_USER specified. Exiting."
    exit 1
fi

# Exec into the user's default shell with the GIT_USER environment variable set
exec env GIT_USER="$GIT_USER" "$SHELL" -l
