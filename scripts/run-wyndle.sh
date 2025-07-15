#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go to the project root (one level up from scripts/)
cd "$SCRIPT_DIR/.."
source .venv/bin/activate
uv run wyndle-server