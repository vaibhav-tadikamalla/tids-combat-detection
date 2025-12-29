import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class ThreatAnalyzer:
    """AI-powered threat level assessment"""
    
    THREAT_WEIGHTS = {
        'blast': 1.0,
        'artillery': 1.0,
        'gunshot': 0.8,
        'vehicle_crash': 0.6,
        'fall': 0.4
    }
    
    def __init__(self):
        self.recent_incidents = []
        self.hotzone_map = {}
        
    async def analyze(self, alert_data: dict) -> str:
        """Multi-factor threat analysis"""
        
        # Factor 1: Impact type severity
        impact_severity = self.THREAT_WEIGHTS.get(alert_data['impact_type'], 0.5)
        
        # Factor 2: Injury severity
        injury_severity = alert_data['severity']
        
        # Factor 3: Vital signs deterioration
        vitals_score = self._assess_vitals(alert_data['vitals'])
        
        # Factor 4: Temporal clustering (multiple impacts nearby)
        cluster_score = self._check_temporal_clustering(
            alert_data['location'],
            alert_data['timestamp']
        )
        
        # Factor 5: Geographic threat level
        geo_threat = self._check_geographic_threat(alert_data['location'])
        
        # Weighted composite score
        threat_score = (
            impact_severity * 0.25 +
            injury_severity * 0.3 +
            vitals_score * 0.2 +
            cluster_score * 0.15 +
            geo_threat * 0.1
        )
        
        # Classify
        if threat_score > 0.8:
            threat_level = "CRITICAL"
        elif threat_score > 0.6:
            threat_level = "HIGH"
        elif threat_score > 0.4:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        # Update history
        self.recent_incidents.append({
            'location': alert_data['location'],
            'timestamp': alert_data['timestamp'],
            'type': alert_data['impact_type']
        })
        
        # Prune old incidents
        self._prune_old_incidents()
        
        return threat_level
    
    def _assess_vitals(self, vitals: dict) -> float:
        """Score vital signs criticality"""
        score = 0.0
        
        hr = vitals['heart_rate']
        spo2 = vitals['spo2']
        
        # Heart rate assessment
        if hr > 150 or hr < 40:
            score += 0.5
        elif hr > 120 or hr < 50:
            score += 0.3
        
        # SpO2 assessment
        if spo2 < 85:
            score += 0.5
        elif spo2 < 90:
            score += 0.3
        
        return min(score, 1.0)
    
    def _check_temporal_clustering(self, location: dict, timestamp: float) -> float:
        """Detect if multiple incidents in same area"""
        recent_threshold = timestamp - 300  # Last 5 minutes
        radius = 0.01  # ~1km
        
        nearby_count = 0
        for incident in self.recent_incidents:
            if incident['timestamp'] < recent_threshold:
                continue
            
            # Simple distance calc
            lat_diff = abs(incident['location']['lat'] - location['lat'])
            lon_diff = abs(incident['location']['lon'] - location['lon'])
            
            if lat_diff < radius and lon_diff < radius:
                nearby_count += 1
        
        # More incidents = higher threat
        return min(nearby_count * 0.25, 1.0)
    
    def _check_geographic_threat(self, location: dict) -> float:
        """Check if location is in known hostile zone"""
        # Would integrate with military intelligence databases
        # For now, placeholder
        return 0.5
    
    def _prune_old_incidents(self):
        """Remove incidents older than 1 hour"""
        cutoff = datetime.now().timestamp() - 3600
        self.recent_incidents = [
            inc for inc in self.recent_incidents
            if inc['timestamp'] > cutoff
        ]
