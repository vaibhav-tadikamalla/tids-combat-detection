from fastapi import FastAPI, WebSocket, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import asyncio
from typing import List, Dict
import json
from datetime import datetime
import logging

from database.models import Alert, Telemetry, Device
from database.queries import get_db, init_db
from services.threat_analyzer import ThreatAnalyzer
from services.notification import NotificationService
from services.geo_fence import GeoFenceMonitor

# Import API routers
from api import alerts, telemetry, auth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Guardian-Shield Command Center", version="2.0")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and services"""
    logger.info("Initializing Guardian-Shield backend...")
    init_db()
    logger.info("Database initialized successfully")

# Include API routers
app.include_router(alerts.router)
app.include_router(telemetry.router)
app.include_router(auth.router)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://command-dashboard.lvlalpha.mil"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Services
threat_analyzer = ThreatAnalyzer()
notifier = NotificationService()
geo_monitor = GeoFenceMonitor()

# WebSocket connections
active_connections: Dict[str, WebSocket] = {}


def verify_device(device_id: str = Header(...)):
    """Verify device authentication"""
    # In production: JWT validation, certificate pinning
    if not device_id.startswith("GS-"):
        raise HTTPException(status_code=401, detail="Invalid device")
    return device_id


@app.post("/api/v1/alerts")
async def receive_alert(
    payload: dict,
    device_id: str = Depends(verify_device),
    db: Session = Depends(get_db)
):
    """Receive and process impact alerts"""
    
    # Decrypt payload (simplified)
    alert_data = payload  # Would decrypt here
    
    # Store in database
    alert = Alert(
        device_id=device_id,
        alert_type=alert_data['impact_type'],
        severity=alert_data['severity'],
        confidence=alert_data['confidence'],
        latitude=alert_data['location']['lat'],
        longitude=alert_data['location']['lon'],
        heart_rate=alert_data['vitals']['heart_rate'],
        spo2=alert_data['vitals']['spo2'],
        timestamp=datetime.fromtimestamp(alert_data['timestamp'])
    )
    
    db.add(alert)
    db.commit()
    
    # Threat analysis
    threat_level = await threat_analyzer.analyze(alert_data)
    
    # Geo-fence check
    zone_info = await geo_monitor.check_zone(
        alert_data['location']['lat'],
        alert_data['location']['lon']
    )
    
    # Send notifications
    if alert_data['severity'] > 0.7:
        await notifier.send_critical_alert(alert_data, threat_level, zone_info)
    
    # Broadcast to connected dashboards
    await broadcast_to_dashboards({
        'type': 'ALERT',
        'data': alert_data,
        'threat_level': threat_level,
        'zone': zone_info
    })
    
    return {"status": "received", "alert_id": alert.id}


@app.post("/api/v1/emergency")
async def receive_emergency(
    payload: dict,
    device_id: str = Depends(verify_device),
    db: Session = Depends(get_db)
):
    """Handle incapacitation emergencies"""
    
    # High-priority processing
    emergency_data = payload
    
    # Store with CRITICAL flag
    alert = Alert(
        device_id=device_id,
        alert_type="INCAPACITATION",
        severity=1.0,
        confidence=1.0,
        latitude=emergency_data['last_known_location']['latitude'],
        longitude=emergency_data['last_known_location']['longitude'],
        priority="CRITICAL",
        timestamp=datetime.fromtimestamp(emergency_data['timestamp'])
    )
    
    db.add(alert)
    db.commit()
    
    # Immediate escalation
    await notifier.send_emergency_escalation(emergency_data)
    
    # Activate rescue protocols
    await activate_rescue_protocol(emergency_data)
    
    return {"status": "emergency_received", "protocol_activated": True}


@app.post("/api/v1/telemetry")
async def receive_telemetry(
    payload: dict,
    device_id: str = Depends(verify_device),
    db: Session = Depends(get_db)
):
    """Receive routine telemetry"""
    
    telemetry = Telemetry(
        device_id=device_id,
        battery_level=payload['battery'],
        latitude=payload['location']['latitude'],
        longitude=payload['location']['longitude'],
        timestamp=datetime.fromtimestamp(payload['timestamp'])
    )
    
    db.add(telemetry)
    db.commit()
    
    return {"status": "ok"}


@app.post("/api/v1/heartbeat")
async def receive_heartbeat(payload: dict):
    """Maintain device connection status"""
    device_id = payload['device_id']
    
    # Update last_seen timestamp
    # Track connection health
    
    return {"status": "ok"}


@app.websocket("/ws/command")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time dashboard updates"""
    await websocket.accept()
    
    client_id = id(websocket)
    active_connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle commands from dashboard
            await process_command(json.loads(data))
    except:
        del active_connections[client_id]


async def broadcast_to_dashboards(message: dict):
    """Send updates to all connected dashboards"""
    disconnected = []
    
    for client_id, websocket in active_connections.items():
        try:
            await websocket.send_json(message)
        except:
            disconnected.append(client_id)
    
    # Cleanup
    for client_id in disconnected:
        del active_connections[client_id]


async def activate_rescue_protocol(emergency_data):
    """Initiate automated rescue sequence"""
    # In production:
    # - Alert nearest medical team
    # - Dispatch drone with supplies
    # - Notify chain of command
    # - Activate GPS beacon
    
    print(f"[RESCUE PROTOCOL] Activated for {emergency_data['device_id']}")


async def process_command(command: dict):
    """Process commands from command center"""
    # e.g., "Send status request to device X"
    # e.g., "Update geo-fence boundaries"
    pass


@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get overview stats for dashboard"""
    total_devices = db.query(Device).count()
    active_alerts = db.query(Alert).filter(Alert.acknowledged == False).count()
    
    recent_alerts = db.query(Alert).order_by(Alert.timestamp.desc()).limit(10).all()
    
    return {
        'total_devices': total_devices,
        'active_alerts': active_alerts,
        'recent_alerts': [alert.to_dict() for alert in recent_alerts]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8443, ssl_keyfile="./certs/key.pem", ssl_certfile="./certs/cert.pem")
