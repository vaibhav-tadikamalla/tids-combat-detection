# Deployment Guide

## Edge Device Setup

### 1. Flash Firmware
```bash
cd edge_device/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Flash to device
esptool.py --chip esp32 write_flash -z 0x1000 firmware.bin
```

### 2. Configure Device
Edit `config.yaml`:

```yaml
device_id: GS-001-ALPHA
soldier_id: SOLDIER-12345
server_url: https://command.lvlalpha.mil:8443
encryption_key: <32-byte hex key>
```

### 3. Calibration
```bash
python calibrate_sensors.py
```

## Backend Deployment

### Docker Compose
```bash
cd backend/
docker-compose up -d
```

### Kubernetes (Production)
```bash
kubectl apply -f k8s/
kubectl rollout status deployment/guardian-backend
```

## Dashboard Deployment
```bash
cd dashboard/
npm install
npm run build
# Deploy to CDN or serve via nginx
```

## Monitoring
- Prometheus metrics: http://backend:9090/metrics
- Grafana dashboard: Import monitoring/grafana-dashboard.json
