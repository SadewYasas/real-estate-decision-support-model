# Creates the Python virtual environment and installs pinned dependencies.
# Usage (PowerShell, from the project root):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#   .\.venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.14 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Record the full resolved environment (including sub-dependencies) for the appendix
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-lock.txt

Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"
