#!/usr/bin/env bash
# One-command demo setup: creates/reuses .venv, installs requirements, starts
# the server. Ships with data/store.db already seeded (see README's
# "Offline demo mode") -- no API key needed to see a populated system;
# uploading a new PDF still needs GROQ_API_KEY/GOOGLE_API_KEY in .env.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -q -r requirements.txt

echo
echo "Starting server at http://127.0.0.1:8000 (Ctrl+C to stop)..."
echo "data/store.db is already seeded -- open the URL above to browse it immediately."
echo
.venv/bin/python -m uvicorn apps.api.main:app --reload
