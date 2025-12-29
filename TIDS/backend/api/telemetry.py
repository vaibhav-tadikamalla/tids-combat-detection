"""
Telemetry API Module
Handles device telemetry and heartbeat data
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from database.queries import get_db
from database.models import Device, Telemetry
from pydantic import BaseModel
from datetime import datetime
import logging

router = APIRouter(prefix="/api/v1", tags=["telemetry"])


class HeartbeatRequest(BaseModel):
    device_id: str
    timestamp: float
    status: str
    battery: float
    location: dict = None


class TelemetryRequest(BaseModel):
    device_id: str
    battery_level: float
    latitude: float
    longitude: float
    timestamp: float


@router.post("/heartbeat")
async def receive_heartbeat(data: HeartbeatRequest, db: Session = Depends(get_db)):
    """
    Receive device heartbeat.
    
    Updates device status and last seen timestamp.
    """
    try:
        # Find or create device
        device = db.query(Device).filter(Device.device_id == data.device_id).first()
        
        if not device:
            device = Device(
                device_id=data.device_id,
                status=data.status,
                battery_level=data.battery,
                last_seen=datetime.fromtimestamp(data.timestamp)
            )
            db.add(device)
        else:
            device.status = data.status
            device.battery_level = data.battery
            device.last_seen = datetime.fromtimestamp(data.timestamp)
        
        if data.location:
            device.latitude = data.location.get('lat')
            device.longitude = data.location.get('lon')
        
        db.commit()
        
        return {"status": "ok", "message": "Heartbeat received"}
        
    except Exception as e:
        logging.error(f"Error processing heartbeat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry")
async def receive_telemetry(data: TelemetryRequest, db: Session = Depends(get_db)):
    """Receive device telemetry data"""
    try:
        telemetry = Telemetry(
            device_id=data.device_id,
            battery_level=data.battery_level,
            latitude=data.latitude,
            longitude=data.longitude,
            timestamp=datetime.fromtimestamp(data.timestamp)
        )
        db.add(telemetry)
        db.commit()
        
        return {"status": "ok"}
        
    except Exception as e:
        logging.error(f"Error saving telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices")
async def get_devices(db: Session = Depends(get_db)):
    """Get all registered devices"""
    devices = db.query(Device).all()
    
    return {
        "count": len(devices),
        "devices": [
            {
                "device_id": d.device_id,
                "soldier_id": d.soldier_id,
                "status": d.status,
                "battery": d.battery_level,
                "last_seen": d.last_seen.isoformat() if d.last_seen else None,
                "location": {
                    "lat": d.latitude,
                    "lon": d.longitude
                } if d.latitude else None
            }
            for d in devices
        ]
    }


@router.get("/dashboard/summary")
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get dashboard summary statistics"""
    from database.models import Alert
    
    total_devices = db.query(Device).count()
    active_devices = db.query(Device).filter(Device.status == 'active').count()
    
    # Recent alerts (last 24 hours)
    from datetime import timedelta
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    active_alerts = db.query(Alert).filter(
        Alert.timestamp >= recent_cutoff,
        Alert.acknowledged == False
    ).count()
    
    critical_alerts = db.query(Alert).filter(
        Alert.timestamp >= recent_cutoff,
        Alert.priority == 'CRITICAL'
    ).count()
    
    return {
        "total_devices": total_devices,
        "active_devices": active_devices,
        "active_alerts": active_alerts,
        "critical_alerts": critical_alerts,
        "timestamp": datetime.utcnow().isoformat()
    }
