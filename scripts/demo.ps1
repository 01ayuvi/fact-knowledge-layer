# One-command demo setup: creates/reuses .venv, installs requirements,
# starts the server. Ships with data/store.db already seeded (see README's
# "Offline demo mode") -- no API key needed to see a populated system;
# uploading a new PDF still needs GROQ_API_KEY/GOOGLE_API_KEY in .env.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Installing dependencies..."
& ".venv\Scripts\pip.exe" install -q -r requirements.txt

Write-Host ""
Write-Host "Starting server at http://127.0.0.1:8000 (Ctrl+C to stop)..."
Write-Host "data/store.db is already seeded -- open the URL above to browse it immediately."
Write-Host ""
& ".venv\Scripts\python.exe" -m uvicorn apps.api.main:app --reload
