"""
Alerts API Module
Handles alert reception and processing endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
from sqlalchemy.orm import Session
from database.queries import get_db
from database.models import Alert
from services.threat_analyzer import ThreatAnalyzer
from services.notification import NotificationService
from services.geo_fence import GeoFenceMonitor
from pydantic import BaseModel
from datetime import datetime
import logging

router = APIRouter(prefix="/api/v1", tags=["alerts"])

# Services
threat_analyzer = ThreatAnalyzer()
notification_service = NotificationService()
geo_fence = GeoFenceMonitor()


class AlertRequest(BaseModel):
    device_id: str
    impact_type: str
    severity: float
    confidence: float
    location: dict  # {"lat": float, "lon": float}
    vitals: dict
    timestamp: float


@router.post("/alerts")
async def receive_alert(
    alert_data: AlertRequest,
    device_id: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Receive and process impact alert from edge device.
    
    Args:
        alert_data: Alert information
        device_id: Device ID from header
        db: Database session
    
    Returns:
        Alert processing status
    """
    try:
        # Validate device
        if device_id != alert_data.device_id:
            raise HTTPException(status_code=403, detail="Device ID mismatch")
        
        # Analyze threat level
        threat_analysis = await threat_analyzer.analyze_alert(alert_data.dict())
        
        # Check geo-fence
        zone_info = await geo_fence.check_zone(
            alert_data.location['lat'],
            alert_data.location['lon']
        )
        
        # Create database record
        db_alert = Alert(
            device_id=alert_data.device_id,
            alert_type=alert_data.impact_type,
            severity=alert_data.severity,
            confidence=alert_data.confidence,
            latitude=alert_data.location['lat'],
            longitude=alert_data.location['lon'],
            heart_rate=alert_data.vitals.get('heart_rate'),
            spo2=alert_data.vitals.get('spo2'),
            timestamp=datetime.fromtimestamp(alert_data.timestamp),
            priority=threat_analysis['priority']
        )
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        
        # Send notifications for critical alerts
        if threat_analysis['threat_level'] > 0.7:
            await notification_service.send_critical_alert(
                alert_data.dict(),
                threat_analysis['threat_level'],
                zone_info
            )
        
        logging.info(f"Alert received from {device_id}: {alert_data.impact_type} "
                    f"(severity={alert_data.severity:.2f}, threat={threat_analysis['threat_level']:.2f})")
        
        return {
            "status": "received",
            "alert_id": db_alert.id,
            "threat_level": threat_analysis['threat_level'],
            "priority": threat_analysis['priority'],
            "zone": zone_info['zone'],
            "actions_taken": threat_analysis.get('recommended_actions', [])
        }
        
    except Exception as e:
        logging.error(f"Error processing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    limit: int = 50,
    device_id: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get recent alerts"""
    query = db.query(Alert)
    
    if device_id:
        query = query.filter(Alert.device_id == device_id)
    
    if acknowledged is not None:
        query = query.filter(Alert.acknowledged == acknowledged)
    
    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()
    
    return {
        "count": len(alerts),
        "alerts": [alert.to_dict() for alert in alerts]
    }


@router.patch("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    db.commit()
    
    return {"status": "acknowledged", "alert_id": alert_id}
