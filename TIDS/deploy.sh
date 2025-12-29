#!/bin/bash

##############################################################################
# GUARDIAN-SHIELD Production Deployment Script
# Deploys complete system: Edge Device + Backend + Dashboard
##############################################################################

set -e  # Exit on error

echo "========================================================================"
echo "  GUARDIAN-SHIELD DEPLOYMENT - PRODUCTION MODE"
echo "========================================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EDGE_DEVICE_DIR="edge_device"
BACKEND_DIR="backend"
DASHBOARD_DIR="dashboard"
ML_TRAINING_DIR="ml_training"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Copy .env.example to .env and configure:"
    echo "  cp .env.example .env"
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '#' | xargs)

echo -e "${GREEN}[1/8] Environment Configuration Loaded${NC}"

##############################################################################
# Step 2: Install Edge Device Dependencies
##############################################################################
echo -e "${YELLOW}\n[2/8] Installing Edge Device Dependencies...${NC}"

cd $EDGE_DEVICE_DIR

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}Edge device dependencies installed${NC}"
cd ..

##############################################################################
# Step 3: Install Backend Dependencies
##############################################################################
echo -e "${YELLOW}\n[3/8] Installing Backend Dependencies...${NC}"

cd $BACKEND_DIR

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python -c "from database.queries import init_db; init_db()"

echo -e "${GREEN}Backend dependencies installed${NC}"
cd ..

##############################################################################
# Step 4: Verify ML Model Exists
##############################################################################
echo -e "${YELLOW}\n[4/8] Verifying ML Model...${NC}"

MODEL_PATH="$ML_TRAINING_DIR/models/impact_classifier.tflite"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo -e "${GREEN}✓ TFLite model found: $MODEL_SIZE${NC}"
else
    echo -e "${RED}✗ TFLite model not found!${NC}"
    echo "Run training pipeline first:"
    echo "  cd $ML_TRAINING_DIR"
    echo "  python generate_dataset.py"
    echo "  python train_production_model.py"
    exit 1
fi

##############################################################################
# Step 5: Install Dashboard Dependencies
##############################################################################
echo -e "${YELLOW}\n[5/8] Installing Dashboard Dependencies...${NC}"

cd $DASHBOARD_DIR

if [ ! -d "node_modules" ]; then
    echo "Installing npm packages..."
    npm install
else
    echo "Node modules already installed"
fi

echo -e "${GREEN}Dashboard dependencies installed${NC}"
cd ..

##############################################################################
# Step 6: Build Dashboard for Production
##############################################################################
echo -e "${YELLOW}\n[6/8] Building Dashboard for Production...${NC}"

cd $DASHBOARD_DIR
npm run build
echo -e "${GREEN}Dashboard built successfully${NC}"
cd ..

##############################################################################
# Step 7: Security Validation
##############################################################################
echo -e "${YELLOW}\n[7/8] Running Security Checks...${NC}"

# Check for .env in git
if git ls-files --error-unmatch .env 2>/dev/null; then
    echo -e "${RED}WARNING: .env file is tracked by git!${NC}"
    echo "Run: git rm --cached .env"
fi

# Validate encryption keys
if [ -z "$ENCRYPTION_KEY" ]; then
    echo -e "${RED}WARNING: ENCRYPTION_KEY not set in .env${NC}"
fi

if [ -z "$JWT_SECRET" ]; then
    echo -e "${RED}WARNING: JWT_SECRET not set in .env${NC}"
fi

echo -e "${GREEN}Security checks complete${NC}"

##############################################################################
# Step 8: Start Services
##############################################################################
echo -e "${YELLOW}\n[8/8] Starting Services...${NC}"

# Start backend
echo "Starting backend server..."
cd $BACKEND_DIR
source venv/bin/activate
nohup uvicorn app:app --host 0.0.0.0 --port ${BACKEND_PORT:-8000} > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"
cd ..

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:${BACKEND_PORT:-8000}/api/v1/dashboard/summary > /dev/null; then
    echo -e "${GREEN}✓ Backend is responding${NC}"
else
    echo -e "${YELLOW}⚠ Backend may not be fully started yet${NC}"
fi

# Start edge device (simulation mode for testing)
echo "Starting edge device (simulation mode)..."
cd $EDGE_DEVICE_DIR
source venv/bin/activate
nohup python main.py --simulate > ../logs/edge_device.log 2>&1 &
EDGE_PID=$!
echo "Edge device started (PID: $EDGE_PID)"
cd ..

# Dashboard is served via build directory (use nginx or serve)
echo "Dashboard built in $DASHBOARD_DIR/build"
echo "Serve with: npx serve -s $DASHBOARD_DIR/build -p 3000"

echo ""
echo "========================================================================"
echo -e "${GREEN}  DEPLOYMENT COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "Services Running:"
echo "  Backend:      http://localhost:${BACKEND_PORT:-8000} (PID: $BACKEND_PID)"
echo "  Edge Device:  Running in simulation mode (PID: $EDGE_PID)"
echo "  Dashboard:    Ready to serve from $DASHBOARD_DIR/build"
echo ""
echo "Logs:"
echo "  Backend:      logs/backend.log"
echo "  Edge Device:  logs/edge_device.log"
echo ""
echo "To stop services:"
echo "  kill $BACKEND_PID $EDGE_PID"
echo ""
echo "To serve dashboard:"
echo "  cd $DASHBOARD_DIR && npx serve -s build -p 3000"
echo ""
echo "========================================================================"
