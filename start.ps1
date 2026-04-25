# Federal_KG start script
# Usage: .\start.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "===== Federal_KG Starting =====" -ForegroundColor Cyan

# 1. Neo4j
Write-Host "[1/4] Starting Neo4j (Docker)..." -ForegroundColor Yellow
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker failed. Make sure Docker Desktop is running." -ForegroundColor Red
    exit 1
}
Write-Host "      Neo4j ready -> http://localhost:7474" -ForegroundColor Green

# 2. Wait
Write-Host "[2/4] Waiting for Neo4j (5s)..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 3. Backend (new window)
Write-Host "[3/4] Starting backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; backend\.venv\Scripts\activate; python backend\run.py"

# 4. Frontend (new window)
Write-Host "[4/4] Starting frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"

Write-Host ""
Write-Host "===== All services started =====" -ForegroundColor Cyan
Write-Host "Backend : http://localhost:5001" -ForegroundColor White
Write-Host "Frontend: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "To share via ngrok:" -ForegroundColor Yellow
Write-Host "  ngrok http 3000" -ForegroundColor White
