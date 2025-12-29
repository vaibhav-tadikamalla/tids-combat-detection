import React, { useEffect, useState } from 'react';
import { LiveMap } from './components/LiveMap';
import { AlertPanel } from './components/AlertPanel';
import { SoldierCard } from './components/SoldierCard';
import './App.css';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [soldiers, setSoldiers] = useState([]);
  const [websocket, setWebsocket] = useState(null);
  const [stats, setStats] = useState({
    total_devices: 0,
    active_alerts: 0,
    critical_count: 0
  });

  useEffect(() => {
    // Initialize WebSocket
    const ws = new WebSocket('wss://command.lvlalpha.mil:8443/ws/command');
    
    ws.onopen = () => {
      console.log('[DASHBOARD] Connected to Command Center');
      setWebsocket(ws);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
      console.error('[DASHBOARD] WebSocket error:', error);
    };

    // Fetch initial data
    fetchDashboardData();

    return () => ws.close();
  }, []);

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'ALERT':
        setAlerts(prev => [message.data, ...prev]);
        updateStats();
        break;
      
      case 'TELEMETRY_UPDATE':
        updateSoldierLocation(message.data);
        break;
      
      case 'DEVICE_STATUS':
        updateSoldierStatus(message.data);
        break;
    }
  };

  const fetchDashboardData = async () => {
    const response = await fetch('/api/v1/dashboard/summary');
    const data = await response.json();
    
    setAlerts(data.recent_alerts);
    setStats({
      total_devices: data.total_devices,
      active_alerts: data.active_alerts,
      critical_count: data.critical_count
    });
  };

  const updateSoldierLocation = (telemetry) => {
    setSoldiers(prev => {
      const index = prev.findIndex(s => s.device_id === telemetry.device_id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = { ...updated[index], ...telemetry };
        return updated;
      }
      return [...prev, telemetry];
    });
  };

  const updateStats = () => {
    setStats(prev => ({
      ...prev,
      active_alerts: prev.active_alerts + 1
    }));
  };

  return (
    <div className="command-center">
      <header className="dashboard-header">
        <h1>🛡️ GUARDIAN-SHIELD Command Center</h1>
        <div className="stats-bar">
          <div className="stat">
            <span className="stat-value">{stats.total_devices}</span>
            <span className="stat-label">Active Devices</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stats.active_alerts}</span>
            <span className="stat-label">Active Alerts</span>
          </div>
          <div className="stat critical">
            <span className="stat-value">{stats.critical_count}</span>
            <span className="stat-label">Critical</span>
          </div>
        </div>
      </header>

      <div className="dashboard-grid">
        <div className="map-container">
          <LiveMap alerts={alerts} soldiers={soldiers} websocket={websocket} />
        </div>

        <div className="side-panel">
          <AlertPanel alerts={alerts} />
          
          <div className="soldiers-list">
            <h2>Squad Status</h2>
            {soldiers.map(soldier => (
              <SoldierCard key={soldier.device_id} soldier={soldier} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
