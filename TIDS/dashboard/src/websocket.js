/**
 * WebSocket Connection Manager
 * Handles real-time communication with backend
 */

class WebSocketManager {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.reconnectInterval = 3000;
    this.listeners = {
      alert: [],
      telemetry: [],
      heartbeat: [],
      connect: [],
      disconnect: [],
      error: []
    };
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected to backend');
        this.notifyListeners('connect', { status: 'connected' });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('[WebSocket] Received:', data);

          // Route message to appropriate listeners
          if (data.type) {
            this.notifyListeners(data.type, data);
          }
        } catch (error) {
          console.error('[WebSocket] Error parsing message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        this.notifyListeners('error', { error });
      };

      this.ws.onclose = () => {
        console.log('[WebSocket] Connection closed. Reconnecting...');
        this.notifyListeners('disconnect', { status: 'disconnected' });
        
        // Attempt reconnection
        setTimeout(() => this.connect(), this.reconnectInterval);
      };
    } catch (error) {
      console.error('[WebSocket] Connection failed:', error);
      setTimeout(() => this.connect(), this.reconnectInterval);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(type, data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, ...data }));
    } else {
      console.warn('[WebSocket] Cannot send - not connected');
    }
  }

  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  off(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
    }
  }

  notifyListeners(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`[WebSocket] Error in ${event} listener:`, error);
        }
      });
    }
  }
}

// Create singleton instance
const wsManager = new WebSocketManager(
  process.env.REACT_APP_WS_URL || 'ws://localhost:8443/ws/command'
);

export default wsManager;
