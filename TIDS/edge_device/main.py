import asyncio
import numpy as np
from sensors.sensor_fusion import SensorFusion
from models.model_loader import EdgeInferenceEngine
from security.secure_comm import SecureTransmitter
from collections import deque
import time
import json

class GuardianShield:
    """Main edge system running on wearable"""
    
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.sensor_fusion = SensorFusion(
            sample_rate=200,
            window_size=200
        )
        
        self.inference_engine = EdgeInferenceEngine(
            model_path='models/impact_classifier.tflite'
        )
        
        self.secure_tx = SecureTransmitter(
            server_url=self.config['server_url'],
            device_id=self.config['device_id'],
            encryption_key=self.config['encryption_key']
        )
        
        # State management
        self.alert_active = False
        self.last_impact_time = 0
        self.impact_history = deque(maxlen=10)
        
        # Power management
        self.power_mode = 'normal'  # normal, high_alert, low_power
        self.battery_level = 100
        
    async def run(self):
        """Main event loop"""
        print("[GUARDIAN-SHIELD] System initialized. Monitoring commenced.")
        
        tasks = [
            self.sensor_loop(),
            self.inference_loop(),
            self.heartbeat_loop(),
            self.power_management_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def sensor_loop(self):
        """Continuous sensor data acquisition"""
        while True:
            try:
                # Read all sensors
                sensor_data = await self.sensor_fusion.read_sensors()
                
                # Apply Kalman filtering
                filtered_data = self.sensor_fusion.apply_filters(sensor_data)
                
                # Buffer for inference
                self.sensor_fusion.buffer_data(filtered_data)
                
                # Adaptive sampling based on power mode
                if self.power_mode == 'normal':
                    await asyncio.sleep(0.005)  # 200 Hz
                elif self.power_mode == 'high_alert':
                    await asyncio.sleep(0.0025)  # 400 Hz
                else:
                    await asyncio.sleep(0.01)  # 100 Hz
                    
            except Exception as e:
                print(f"[SENSOR ERROR] {e}")
                await asyncio.sleep(0.1)
    
    async def inference_loop(self):
        """Real-time impact detection"""
        while True:
            try:
                # Get windowed data
                if not self.sensor_fusion.is_buffer_ready():
                    await asyncio.sleep(0.1)
                    continue
                
                window_data = self.sensor_fusion.get_inference_window()
                
                # Run inference
                start_time = time.time()
                prediction = self.inference_engine.predict(window_data)
                inference_time = (time.time() - start_time) * 1000
                
                # Parse results
                impact_type = prediction['impact_type']
                severity = prediction['severity']
                confidence = prediction['confidence']
                
                # Decision logic
                if confidence > 0.75 and impact_type != 'normal':
                    await self.handle_impact_detection(
                        impact_type=impact_type,
                        severity=severity,
                        confidence=confidence,
                        inference_time=inference_time
                    )
                
                # Log telemetry
                if np.random.random() < 0.1:  # Sample 10% for bandwidth efficiency
                    await self.send_telemetry(prediction, inference_time)
                
                await asyncio.sleep(0.05)  # Check every 50ms
                
            except Exception as e:
                print(f"[INFERENCE ERROR] {e}")
                await asyncio.sleep(0.1)
    
    async def handle_impact_detection(self, impact_type, severity, confidence, inference_time):
        """Trigger alert sequence"""
        current_time = time.time()
        
        # Debouncing - prevent duplicate alerts
        if current_time - self.last_impact_time < 5:
            return
        
        self.last_impact_time = current_time
        self.alert_active = True
        self.power_mode = 'high_alert'
        
        # Get current location
        gps_data = await self.sensor_fusion.get_gps()
        
        # Get vital signs
        vitals = await self.sensor_fusion.get_vitals()
        
        # Compile alert payload
        alert_payload = {
            'alert_id': f"IMPACT_{int(current_time)}",
            'timestamp': current_time,
            'device_id': self.config['device_id'],
            'soldier_id': self.config['soldier_id'],
            'impact_type': impact_type,
            'severity': float(severity),
            'confidence': float(confidence),
            'location': {
                'lat': gps_data['latitude'],
                'lon': gps_data['longitude'],
                'accuracy': gps_data['accuracy'],
                'altitude': gps_data['altitude']
            },
            'vitals': {
                'heart_rate': vitals['hr'],
                'spo2': vitals['spo2'],
                'breathing_rate': vitals['br'],
                'skin_temp': vitals['temp']
            },
            'inference_time_ms': inference_time,
            'priority': 'CRITICAL' if severity > 0.7 else 'HIGH'
        }
        
        # Send encrypted alert
        response = await self.secure_tx.send_alert(alert_payload)
        
        # Local notification
        self._trigger_local_alert(impact_type, severity)
        
        # Add to history
        self.impact_history.append(alert_payload)
        
        print(f"[ALERT] {impact_type.upper()} detected - Severity: {severity:.2f} - Alert sent")
        
        # Check for incapacitation
        await self.check_incapacitation()
    
    async def check_incapacitation(self):
        """Detect if soldier is unconscious/immobile"""
        await asyncio.sleep(10)  # Wait 10 seconds
        
        # Check for movement
        movement_detected = await self.sensor_fusion.detect_movement(threshold=0.5)
        
        # Check vitals
        vitals = await self.sensor_fusion.get_vitals()
        vitals_critical = (
            vitals['hr'] > 140 or vitals['hr'] < 40 or
            vitals['spo2'] < 85
        )
        
        if not movement_detected or vitals_critical:
            # Send incapacitation alert
            incap_alert = {
                'alert_type': 'INCAPACITATION',
                'device_id': self.config['device_id'],
                'timestamp': time.time(),
                'auto_escalate': True,
                'last_known_location': await self.sensor_fusion.get_gps()
            }
            
            await self.secure_tx.send_emergency(incap_alert)
            print("[CRITICAL] Incapacitation detected - Emergency protocols activated")
    
    async def send_telemetry(self, prediction, inference_time):
        """Send routine telemetry data"""
        telemetry = {
            'device_id': self.config['device_id'],
            'timestamp': time.time(),
            'battery': self.battery_level,
            'power_mode': self.power_mode,
            'inference_time': inference_time,
            'prediction_confidence': float(prediction['confidence']),
            'location': await self.sensor_fusion.get_gps()
        }
        
        await self.secure_tx.send_telemetry(telemetry)
    
    async def heartbeat_loop(self):
        """Maintain connection with command center"""
        while True:
            try:
                heartbeat = {
                    'device_id': self.config['device_id'],
                    'timestamp': time.time(),
                    'status': 'operational',
                    'battery': self.battery_level
                }
                
                await self.secure_tx.send_heartbeat(heartbeat)
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                print(f"[HEARTBEAT ERROR] {e}")
                await asyncio.sleep(5)
    
    async def power_management_loop(self):
        """Optimize power consumption"""
        while True:
            # Simulate battery drain
            if self.power_mode == 'normal':
                self.battery_level -= 0.01
            elif self.power_mode == 'high_alert':
                self.battery_level -= 0.03
            else:
                self.battery_level -= 0.005
            
            # Auto-adjust power mode
            if self.battery_level < 20 and self.power_mode != 'high_alert':
                self.power_mode = 'low_power'
            
            if self.battery_level < 10:
                await self.secure_tx.send_low_battery_alert(self.battery_level)
            
            await asyncio.sleep(60)
    
    def _trigger_local_alert(self, impact_type, severity):
        """Vibration/audio alert on device"""
        # In real implementation: GPIO control for haptic motor
        print(f"[LOCAL ALERT] Vibration pattern: {impact_type}")
    
    def _load_config(self, path):
        """Load device configuration"""
        # Placeholder - would load from secure storage
        return {
            'device_id': 'GS-001-ALPHA',
            'soldier_id': 'SOLDIER-12345',
            'server_url': 'https://command.lvlalpha.mil:8443',
            'encryption_key': 'AES256_KEY_HERE'
        }


if __name__ == "__main__":
    guardian = GuardianShield()
    asyncio.run(guardian.run())
