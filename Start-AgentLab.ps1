$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path -LiteralPath ".env")) {
    throw "Missing .env. Run 'Copy-Item .env.example .env' and add your OmniRoute key to LLM_API_KEY."
}

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment. Install Python 3.12 and try again."
    }
}

& $python -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency installation failed."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js and npm are required for the React interface. Install the current Node.js LTS release and reopen PowerShell."
}

Push-Location (Join-Path $projectRoot "frontend")
try {
    if (-not (Test-Path -LiteralPath "node_modules")) {
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "React dependencies could not be installed."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "uvicorn", "agent_orchestration.api:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $projectRoot

# Wait for API to come online
$apiReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            $apiReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $apiReady) {
    throw "FastAPI did not start at http://127.0.0.1:8000. Check whether port 8000 is already in use."
}

Write-Host "API ready: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "OmniRoute model: $($health.llm_model); key configured: $($health.llm_credential_configured)" -ForegroundColor Green
Write-Host "Starting React Agent Lab at http://127.0.0.1:5173" -ForegroundColor Cyan

Push-Location (Join-Path $projectRoot "frontend")
try {
    npm run dev -- --host 127.0.0.1
    if ($LASTEXITCODE -ne 0) {
        throw "The React development server exited with an error."
    }
}
finally {
    Pop-Location
}
