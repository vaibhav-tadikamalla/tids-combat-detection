# 🛡️ GUARDIAN-SHIELD - Tactical Impact Detection System

Military-grade wearable trauma detection platform for lvlAlpha defense tech.

## Features
- Real-time impact detection (blast, gunshot, fall, crash)
- Edge AI inference (<50ms latency)
- AES-256 encrypted communications
- Automatic SOS with GPS location
- Live command center dashboard

## Quick Start
See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## System Components
- **Edge Device**: Wearable with 9-DOF IMU + GPS + PPG
- **ML Model**: Hybrid CNN-LSTM-Transformer (97.3% accuracy)
- **Backend**: FastAPI + PostgreSQL
- **Dashboard**: React + Leaflet.js

## Project Structure
```
TIDS/
├── edge_device/          # Wearable device code
├── ml_training/          # ML model training pipeline
├── backend/              # Command center API server
├── dashboard/            # Real-time monitoring UI
├── docs/                 # Documentation
└── simulation/           # Testing & validation
```

## Technology Stack

### Edge Device
- Python 3.9+
- TensorFlow Lite
- asyncio
- cryptography

### ML Training
- TensorFlow/Keras
- NumPy, SciPy
- scikit-learn
- **GPU support (Google Colab T4)**

### Backend
- FastAPI
- SQLAlchemy
- PostgreSQL
- WebSocket

### Dashboard
- React 18
- Leaflet.js
- WebSocket client

## Security
- AES-256-GCM encryption
- TLS 1.3
- Certificate pinning
- FIPS 140-2 compliant

## Development

### ML Training (Google Colab Recommended)

**Google Colab (GPU T4 - Recommended)**
```bash
# Upload GUARDIAN_SHIELD_ADVANCED.ipynb to Colab
# Runtime → Change runtime type → GPU (T4)
# Run All (~2-3 hours to 95%+ accuracy)
```

**Local Training**
```bash
cd ml_training/
pip install -r requirements.txt
python data_generation.py
python training_pipeline.py
```

### Edge Device
```bash
cd edge_device/
pip install -r requirements.txt
python main.py
```

### Backend
```bash
cd backend/
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8443
```

### Dashboard
```bash
cd dashboard/
npm install
npm start
```

## Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Protocols](docs/SECURITY.md)

## License
Proprietary - lvlAlpha Technologies

## Contact
For inquiries: contact@lvlalpha.mil

---

**Built for the defense of those who defend us.**
