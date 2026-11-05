#!/bin/sh
# Starts the live demo from any folder.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m demo.server "$@"
