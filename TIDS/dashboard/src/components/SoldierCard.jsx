// Placeholder for SoldierCard component
import React from 'react';

export function SoldierCard({ soldier }) {
  return (
    <div className="soldier-card">
      <h3>{soldier.device_id}</h3>
      <p>Battery: {soldier.battery}%</p>
      <p>Status: {soldier.status}</p>
    </div>
  );
}
