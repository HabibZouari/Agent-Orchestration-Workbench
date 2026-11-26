<#
.SYNOPSIS
    Starts the Agent Lab: backing services (if needed), FastAPI, the Celery worker
    (if needed), and the React development server.

.DESCRIPTION
    Reads TASK_BACKEND and VECTOR_STORE from .env. In the default local mode only
    FastAPI and Vite are started. In distributed mode, Redis Stack and/or PostgreSQL
    are started with Docker Compose, and a Celery worker opens in its own window.

.PARAMETER ApiPort
    Port for FastAPI (default 8000). Change it if another app already uses 8000.

.PARAMETER SkipInstall
    Skip `pip install` / `npm install`.
#>
[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Read-DotEnv([string]$Path) {
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$' -and $line -notmatch '^\s*#') {
            $values[$Matches[1].ToUpper()] = $Matches[2].Trim('"').Trim("'")
        }
    }
    return $values
}

function Test-PortInUse([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Path -LiteralPath ".env")) {
    throw "Missing .env. Run 'Copy-Item .env.example .env' and set LLM_API_KEY."
}
$config = Read-DotEnv ".env"
$taskBackend = if ($config["TASK_BACKEND"]) { $config["TASK_BACKEND"] } else { "local" }
$vectorStore = if ($config["VECTOR_STORE"]) { $config["VECTOR_STORE"] } else { "sqlite" }

# --- Python environment ---------------------------------------------------------
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Install Python 3.11+ and retry." }
    $SkipInstall = $false
}
if (-not $SkipInstall) {
    & $python -m pip install --quiet -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
}

# --- Frontend dependencies ------------------------------------------------------
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js and npm are required for the React interface. Install the Node.js LTS release."
}
$frontend = Join-Path $projectRoot "frontend"
if (-not (Test-Path -LiteralPath (Join-Path $frontend "node_modules"))) {
    Push-Location $frontend
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "React dependencies could not be installed." }
    }
    finally { Pop-Location }
}

# --- Backing services (distributed mode only) -----------------------------------
$services = @()
if ($taskBackend -eq "celery") { $services += "redis" }
if ($vectorStore -eq "pgvector") { $services += "postgres" }
if ($services.Count -gt 0) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is required for TASK_BACKEND=celery or VECTOR_STORE=pgvector."
    }
    Write-Host "Starting $($services -join ', ') with Docker Compose..." -ForegroundColor Cyan
    docker compose up -d --wait @services
    if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed. Is Docker Desktop running?" }
}

# --- API ------------------------------------------------------------------------
if (Test-PortInUse $ApiPort) {
    throw "Port $ApiPort is already in use. Stop that process or run: .\Start-AgentLab.ps1 -ApiPort 8010"
}
Write-Host "Starting FastAPI on port $ApiPort..." -ForegroundColor Cyan
$api = Start-Process -PassThru -FilePath $python -WorkingDirectory $projectRoot -ArgumentList @(
    "-m", "uvicorn", "agent_orchestration.api:app", "--host", "127.0.0.1", "--port", "$ApiPort"
)

$worker = $null
if ($taskBackend -eq "celery") {
    Write-Host "Starting Celery worker..." -ForegroundColor Cyan
    $worker = Start-Process -PassThru -FilePath $python -WorkingDirectory $projectRoot -ArgumentList @(
        "-m", "celery", "-A", "agent_orchestration.celery_app", "worker", "--pool=solo", "--loglevel=info"
    )
}

try {
    $health = $null
    for ($attempt = 0; $attempt -lt 60 -and -not $health; $attempt++) {
        try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 }
        catch { Start-Sleep -Seconds 1 }
    }
    if (-not $health) { throw "FastAPI did not start on port $ApiPort. Check the API window for errors." }

    Write-Host "API ready: http://127.0.0.1:$ApiPort/docs  (status: $($health.status))" -ForegroundColor Green
    Write-Host "Mode: tasks=$($health.task_backend), store=$($health.vector_store); model=$($health.llm_model); key configured: $($health.llm_credential_configured)" -ForegroundColor Green
    Write-Host "Starting React Agent Lab at http://127.0.0.1:5173 (Ctrl+C to stop everything)" -ForegroundColor Cyan

    $env:API_PORT = "$ApiPort"
    Push-Location $frontend
    try { npm run dev }
    finally { Pop-Location }
}
finally {
    foreach ($process in @($api, $worker)) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    }
    Write-Host "Stopped API$(if ($worker) { ' and worker' }). Docker services keep running; stop them with 'docker compose stop'." -ForegroundColor Yellow
}
