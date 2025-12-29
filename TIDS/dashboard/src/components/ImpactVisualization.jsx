import React from 'react';

/**
 * Impact Visualization Component
 * Shows detailed impact event information
 */

const ImpactVisualization = ({ impact }) => {
  if (!impact) {
    return (
      <div style={styles.container}>
        <p style={styles.noData}>No impact data</p>
      </div>
    );
  }

  const getImpactIcon = (type) => {
    const icons = {
      blast: '💥',
      gunshot: '🔫',
      artillery: '🎯',
      vehicle_crash: '🚗',
      fall: '⬇️',
      normal: '✓'
    };
    return icons[type] || '⚠️';
  };

  const getSeverityColor = (severity) => {
    if (severity > 0.7) return '#ff0000';
    if (severity > 0.4) return '#ffaa00';
    return '#ffff00';
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.icon}>{getImpactIcon(impact.impact_type)}</span>
        <h3 style={styles.title}>{impact.impact_type.toUpperCase()}</h3>
      </div>

      <div style={styles.grid}>
        <div style={styles.metric}>
          <div style={styles.label}>Severity</div>
          <div 
            style={{
              ...styles.value,
              color: getSeverityColor(impact.severity)
            }}
          >
            {(impact.severity * 100).toFixed(0)}%
          </div>
        </div>

        <div style={styles.metric}>
          <div style={styles.label}>Confidence</div>
          <div style={styles.value}>
            {(impact.confidence * 100).toFixed(0)}%
          </div>
        </div>

        <div style={styles.metric}>
          <div style={styles.label}>Device</div>
          <div style={styles.value}>{impact.device_id}</div>
        </div>

        <div style={styles.metric}>
          <div style={styles.label}>Time</div>
          <div style={styles.value}>
            {new Date(impact.timestamp * 1000).toLocaleTimeString()}
          </div>
        </div>
      </div>

      {impact.vitals && (
        <div style={styles.vitals}>
          <h4 style={styles.vitalsTitle}>Vital Signs</h4>
          <div style={styles.vitalsGrid}>
            <div style={styles.vital}>
              <span>HR:</span>
              <strong>{impact.vitals.heart_rate} bpm</strong>
            </div>
            <div style={styles.vital}>
              <span>SpO2:</span>
              <strong>{impact.vitals.spo2}%</strong>
            </div>
            <div style={styles.vital}>
              <span>BR:</span>
              <strong>{impact.vitals.breathing_rate} /min</strong>
            </div>
            <div style={styles.vital}>
              <span>Temp:</span>
              <strong>{impact.vitals.skin_temp}°C</strong>
            </div>
          </div>
        </div>
      )}

      {impact.location && (
        <div style={styles.location}>
          <div style={styles.label}>Location</div>
          <div style={styles.coords}>
            {impact.location.lat.toFixed(6)}, {impact.location.lon.toFixed(6)}
          </div>
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    background: '#16213e',
    borderRadius: '8px',
    padding: '20px',
    border: '1px solid #00ff00',
  },
  noData: {
    textAlign: 'center',
    color: '#666',
    padding: '40px 0',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '20px',
    paddingBottom: '15px',
    borderBottom: '1px solid rgba(0, 255, 0, 0.3)',
  },
  icon: {
    fontSize: '32px',
    marginRight: '15px',
  },
  title: {
    fontSize: '24px',
    fontWeight: 'bold',
    color: '#00ff00',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '15px',
    marginBottom: '20px',
  },
  metric: {
    background: 'rgba(0, 255, 0, 0.1)',
    padding: '15px',
    borderRadius: '5px',
    border: '1px solid rgba(0, 255, 0, 0.2)',
  },
  label: {
    fontSize: '12px',
    color: '#888',
    marginBottom: '5px',
    textTransform: 'uppercase',
  },
  value: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#00ff00',
  },
  vitals: {
    marginTop: '20px',
    padding: '15px',
    background: 'rgba(0, 150, 255, 0.1)',
    borderRadius: '5px',
    border: '1px solid rgba(0, 150, 255, 0.3)',
  },
  vitalsTitle: {
    fontSize: '14px',
    color: '#0096ff',
    marginBottom: '10px',
  },
  vitalsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '10px',
  },
  vital: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '14px',
    color: '#fff',
  },
  location: {
    marginTop: '15px',
    paddingTop: '15px',
    borderTop: '1px solid rgba(0, 255, 0, 0.3)',
  },
  coords: {
    fontSize: '14px',
    color: '#00ff00',
    fontFamily: 'monospace',
    marginTop: '5px',
  },
};

export default ImpactVisualization;
