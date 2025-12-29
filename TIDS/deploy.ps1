##############################################################################
# GUARDIAN-SHIELD Production Deployment Script (PowerShell)
# Deploys complete system: Edge Device + Backend + Dashboard
##############################################################################

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  GUARDIAN-SHIELD DEPLOYMENT - PRODUCTION MODE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Configuration
$EDGE_DEVICE_DIR = "edge_device"
$BACKEND_DIR = "backend"
$DASHBOARD_DIR = "dashboard"
$ML_TRAINING_DIR = "ml_training"

# Check if .env exists
if (-not (Test-Path .env)) {
    Write-Host "Error: .env file not found" -ForegroundColor Red
    Write-Host "Copy .env.example to .env and configure:"
    Write-Host "  Copy-Item .env.example .env"
    exit 1
}

Write-Host "[1/8] Environment Configuration Verified" -ForegroundColor Green

##############################################################################
# Step 2: Install Edge Device Dependencies
##############################################################################
Write-Host "`n[2/8] Installing Edge Device Dependencies..." -ForegroundColor Yellow

Push-Location $EDGE_DEVICE_DIR

if (-not (Test-Path venv)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

.\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Edge device dependencies installed" -ForegroundColor Green
Pop-Location

##############################################################################
# Step 3: Install Backend Dependencies
##############################################################################
Write-Host "`n[3/8] Installing Backend Dependencies..." -ForegroundColor Yellow

Push-Location $BACKEND_DIR

if (-not (Test-Path venv)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

.\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
Write-Host "Initializing database..."
python -c "from database.queries import init_db; init_db()"

Write-Host "Backend dependencies installed" -ForegroundColor Green
Pop-Location

##############################################################################
# Step 4: Verify ML Model Exists
##############################################################################
Write-Host "`n[4/8] Verifying ML Model..." -ForegroundColor Yellow

$MODEL_PATH = Join-Path $ML_TRAINING_DIR "models\impact_classifier.tflite"

if (Test-Path $MODEL_PATH) {
    $MODEL_SIZE = (Get-Item $MODEL_PATH).Length / 1KB
    Write-Host "✓ TFLite model found: $($MODEL_SIZE.ToString('0.0')) KB" -ForegroundColor Green
} else {
    Write-Host "✗ TFLite model not found!" -ForegroundColor Red
    Write-Host "Run training pipeline first:"
    Write-Host "  cd $ML_TRAINING_DIR"
    Write-Host "  python generate_dataset.py"
    Write-Host "  python train_production_model.py"
    exit 1
}

##############################################################################
# Step 5: Install Dashboard Dependencies
##############################################################################
Write-Host "`n[5/8] Installing Dashboard Dependencies..." -ForegroundColor Yellow

Push-Location $DASHBOARD_DIR

if (-not (Test-Path node_modules)) {
    Write-Host "Installing npm packages..."
    npm install
} else {
    Write-Host "Node modules already installed"
}

Write-Host "Dashboard dependencies installed" -ForegroundColor Green
Pop-Location

##############################################################################
# Step 6: Build Dashboard for Production
##############################################################################
Write-Host "`n[6/8] Building Dashboard for Production..." -ForegroundColor Yellow

Push-Location $DASHBOARD_DIR
npm run build
Write-Host "Dashboard built successfully" -ForegroundColor Green
Pop-Location

##############################################################################
# Step 7: Security Validation
##############################################################################
Write-Host "`n[7/8] Running Security Checks..." -ForegroundColor Yellow

# Load environment variables
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#][^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

# Validate encryption keys
if ([string]::IsNullOrEmpty($env:ENCRYPTION_KEY)) {
    Write-Host "WARNING: ENCRYPTION_KEY not set in .env" -ForegroundColor Yellow
}

if ([string]::IsNullOrEmpty($env:JWT_SECRET)) {
    Write-Host "WARNING: JWT_SECRET not set in .env" -ForegroundColor Yellow
}

Write-Host "Security checks complete" -ForegroundColor Green

##############################################################################
# Step 8: Create Logs Directory
##############################################################################
if (-not (Test-Path logs)) {
    New-Item -ItemType Directory -Path logs | Out-Null
}

##############################################################################
# Step 9: Start Services
##############################################################################
Write-Host "`n[8/8] Starting Services..." -ForegroundColor Yellow

# Get backend port from env or use default
$BACKEND_PORT = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { "8000" }

# Start backend
Write-Host "Starting backend server..."
Push-Location $BACKEND_DIR
.\venv\Scripts\Activate.ps1

$backendJob = Start-Job -ScriptBlock {
    param($port, $dir)
    Set-Location $dir
    & .\venv\Scripts\Activate.ps1
    uvicorn app:app --host 0.0.0.0 --port $port
} -ArgumentList $BACKEND_PORT, (Resolve-Path $BACKEND_DIR)

Write-Host "Backend started (Job ID: $($backendJob.Id))" -ForegroundColor Green
Pop-Location

# Wait for backend to be ready
Write-Host "Waiting for backend to start..."
Start-Sleep -Seconds 5

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$BACKEND_PORT/api/v1/dashboard/summary" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "✓ Backend is responding" -ForegroundColor Green
} catch {
    Write-Host "⚠ Backend may not be fully started yet" -ForegroundColor Yellow
}

# Start edge device (simulation mode)
Write-Host "Starting edge device (simulation mode)..."
Push-Location $EDGE_DEVICE_DIR
.\venv\Scripts\Activate.ps1

$edgeJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    & .\venv\Scripts\Activate.ps1
    python main.py --simulate
} -ArgumentList (Resolve-Path $EDGE_DEVICE_DIR)

Write-Host "Edge device started (Job ID: $($edgeJob.Id))" -ForegroundColor Green
Pop-Location

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services Running:"
Write-Host "  Backend:      http://localhost:$BACKEND_PORT (Job ID: $($backendJob.Id))"
Write-Host "  Edge Device:  Running in simulation mode (Job ID: $($edgeJob.Id))"
Write-Host "  Dashboard:    Ready to serve from $DASHBOARD_DIR\build"
Write-Host ""
Write-Host "To stop services:"
Write-Host "  Stop-Job -Id $($backendJob.Id),$($edgeJob.Id)"
Write-Host "  Remove-Job -Id $($backendJob.Id),$($edgeJob.Id)"
Write-Host ""
Write-Host "To serve dashboard:"
Write-Host "  cd $DASHBOARD_DIR"
Write-Host "  npx serve -s build -p 3000"
Write-Host ""
Write-Host "To view logs:"
Write-Host "  Receive-Job -Id $($backendJob.Id)"
Write-Host "  Receive-Job -Id $($edgeJob.Id)"
Write-Host ""
Write-Host "========================================================================"
