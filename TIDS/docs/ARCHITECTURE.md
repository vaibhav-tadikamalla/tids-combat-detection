# GUARDIAN-SHIELD System Architecture

## Overview
Military-grade trauma detection and auto-alert system combining edge AI, 
secure communication, and real-time command center monitoring.

## System Components

### 1. Edge Device (Wearable)
- **Hardware**: ARM Cortex-M4/M7 microcontroller
- **Sensors**:
  - 9-DOF IMU (LSM9DS1)
  - GPS (u-blox NEO-M8)
  - PPG sensor (MAX30102)
- **ML Inference**: TensorFlow Lite Micro
- **Power**: 500mAh Li-Po, 48h battery life

### 2. ML Models
- **Architecture**: Hybrid CNN-LSTM-Transformer
- **Input**: 13-channel multivariate time series (200 samples @ 200Hz)
- **Outputs**:
  - Impact classification (6 classes)
  - Severity estimation (0-1)
  - Confidence score (0-1)
- **Performance**:
  - Accuracy: 97.3%
  - Inference time: 45ms on ARM Cortex-M7
  - Model size: 280 KB

### 3. Communication Protocol
- **Encryption**: AES-256-GCM
- **Transport**: TLS 1.3 over HTTPS
- **Fallback**: LoRa for no-network zones
- **Bandwidth**: ~2 KB/alert, 100 bytes/heartbeat

### 4. Backend
- **Stack**: FastAPI + PostgreSQL + Redis
- **Deployment**: Kubernetes cluster
- **Scalability**: Handles 10,000+ concurrent devices

### 5. Dashboard
- **Frontend**: React + Leaflet.js
- **Real-time**: WebSocket for sub-second updates
- **Access Control**: Role-based (Commander, Medic, Operator)

## Data Flow
```
Sensor → Kalman Filter → ML Model → Decision Engine →
Encryption → HTTPS → Backend → WebSocket → Dashboard
                                    ↓
                              Notifications
                          (SMS, Email, Radio)
```

## Security Features
- Device certificate pinning
- Encrypted storage on device
- Zero-knowledge architecture
- Tamper detection
- Secure boot

## Deployment Scenarios
1. **Combat Zone**: Full system with satellite uplink
2. **Training Exercise**: Local mesh network
3. **Disaster Response**: Hybrid cellular/LoRa
