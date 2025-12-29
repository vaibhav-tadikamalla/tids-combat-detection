// Placeholder for AlertPanel component
import React from 'react';

export function AlertPanel({ alerts }) {
  return (
    <div className="alert-panel">
      <h2>Recent Alerts</h2>
      {alerts.map(alert => (
        <div key={alert.id} className="alert-item">
          <span>{alert.impact_type}</span>
          <span>{alert.severity}</span>
        </div>
      ))}
    </div>
  );
}
