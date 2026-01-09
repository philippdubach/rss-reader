#!/bin/bash
# RSS Reader Pipeline - Fetch, Score, Check HN, Open Dashboard
# Usage: ./run.sh [hours] [limit]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

HOURS="${1:-24}"
LIMIT="${2:-50}"

echo "=== RSS Reader Pipeline ==="
echo ""

# Activate venv and run commands
source .venv/bin/activate

echo "[1/2] Fetching feeds and scoring..."
python main.py refresh

echo ""
echo "[2/2] Generating dashboard (checking HN)..."
python main.py dashboard --hours "$HOURS" --limit "$LIMIT" --open

echo ""
echo "Done!"
