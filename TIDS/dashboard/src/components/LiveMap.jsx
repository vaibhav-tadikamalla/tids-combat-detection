import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Circle, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const IMPACT_ICONS = {
  blast: L.divIcon({ className: 'impact-icon blast', html: '💥' }),
  gunshot: L.divIcon({ className: 'impact-icon gunshot', html: '🔫' }),
  fall: L.divIcon({ className: 'impact-icon fall', html: '⬇️' }),
  vehicle_crash: L.divIcon({ className: 'impact-icon crash', html: '🚗' }),
};

const SEVERITY_COLORS = {
  CRITICAL: '#FF0000',
  HIGH: '#FF6600',
  MEDIUM: '#FFAA00',
  LOW: '#00FF00',
};

export function LiveMap({ alerts, soldiers, websocket }) {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const mapRef = useRef(null);

  useEffect(() => {
    if (!websocket) return;

    websocket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'ALERT') {
        // Flash animation on new alert
        flashLocation(message.data.location);
        
        // Play alert sound
        playAlertSound(message.threat_level);
      }
    };
  }, [websocket]);

  const flashLocation = (location) => {
    if (!mapRef.current) return;
    
    const map = mapRef.current;
    map.flyTo([location.lat, location.lon], 15, {
      duration: 2
    });
  };

  const playAlertSound = (threatLevel) => {
    const audio = new Audio(`/sounds/alert_${threatLevel.toLowerCase()}.mp3`);
    audio.play();
  };

  return (
    <MapContainer
      center={[19.0760, 72.8777]}
      zoom={13}
      style={{ height: '100vh', width: '100%' }}
      ref={mapRef}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap contributors'
      />

      {/* Active soldiers */}
      {soldiers.map((soldier) => (
        <Marker
          key={soldier.device_id}
          position={[soldier.latitude, soldier.longitude]}
          icon={L.divIcon({
            className: 'soldier-marker',
            html: `<div class="soldier-icon ${soldier.status}"></div>`
          })}
        >
          <Popup>
            <div>
              <h3>{soldier.name}</h3>
              <p>HR: {soldier.heart_rate} bpm</p>
              <p>SpO2: {soldier.spo2}%</p>
              <p>Battery: {soldier.battery}%</p>
            </div>
          </Popup>
        </Marker>
      ))}

      {/* Alert markers */}
      {alerts.map((alert) => (
        <React.Fragment key={alert.id}>
          <Marker
            position={[alert.latitude, alert.longitude]}
            icon={IMPACT_ICONS[alert.impact_type]}
            eventHandlers={{
              click: () => setSelectedAlert(alert)
            }}
          >
            <Popup>
              <div className="alert-popup">
                <h3>{alert.impact_type.toUpperCase()}</h3>
                <p>Severity: {(alert.severity * 100).toFixed(0)}%</p>
                <p>Time: {new Date(alert.timestamp * 1000).toLocaleTimeString()}</p>
                <button onClick={() => acknowledgeAlert(alert.id)}>
                  Acknowledge
                </button>
              </div>
            </Popup>
          </Marker>

          {/* Severity radius */}
          <Circle
            center={[alert.latitude, alert.longitude]}
            radius={alert.severity * 500}
            color={SEVERITY_COLORS[alert.threat_level]}
            fillOpacity={0.2}
          />
        </React.Fragment>
      ))}
    </MapContainer>
  );
}

function acknowledgeAlert(alertId) {
  fetch(`/api/v1/alerts/${alertId}/acknowledge`, { method: 'POST' });
}
