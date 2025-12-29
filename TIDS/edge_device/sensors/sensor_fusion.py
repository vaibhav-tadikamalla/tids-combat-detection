import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
import asyncio

class KalmanFilter:
    """Multi-dimensional Kalman filter for sensor fusion"""
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.x = np.zeros((dim_x, 1))  # State
        self.P = np.eye(dim_x) * 1000  # Covariance
        self.Q = np.eye(dim_x) * 0.01  # Process noise
        self.R = np.eye(dim_z) * 0.1   # Measurement noise
        self.F = np.eye(dim_x)          # State transition
        self.H = np.eye(dim_z, dim_x)  # Measurement function
        
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        y = z.reshape(-1, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        
        return self.x.flatten()


class SensorFusion:
    """Advanced multi-sensor data fusion system"""
    
    def __init__(self, sample_rate=200, window_size=200):
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Data buffers
        self.accel_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        self.mag_buffer = deque(maxlen=window_size)
        self.vitals_buffer = deque(maxlen=window_size)
        
        # Kalman filters for each sensor
        self.accel_kf = KalmanFilter(3, 3)
        self.gyro_kf = KalmanFilter(3, 3)
        
        # Digital filters
        self.create_filters()
        
        # Mock sensor interfaces (replace with actual hardware drivers)
        self.imu = None
        self.gps = None
        self.pulse_ox = None
        
    def create_filters(self):
        """Create Butterworth filters for noise reduction"""
        # Low-pass for accelerometer
        self.accel_filter = butter(4, 50, 'low', fs=self.sample_rate, output='sos')
        
        # Band-pass for impact detection
        self.impact_filter = butter(4, [10, 100], 'band', fs=self.sample_rate, output='sos')
        
    async def read_sensors(self):
        """Read all sensors - async I/O"""
        # Simulate sensor reads (replace with actual I2C/SPI communication)
        await asyncio.sleep(0.001)
        
        accel = np.random.randn(3)  # Would be: self.imu.read_accel()
        gyro = np.random.randn(3)   # Would be: self.imu.read_gyro()
        mag = np.random.randn(3)    # Would be: self.imu.read_mag()
        
        # Vitals from PPG sensor
        hr = 75 + np.random.randn()
        spo2 = 98 + np.random.randn() * 0.5
        
        return {
            'accel': accel,
            'gyro': gyro,
            'mag': mag,
            'vitals': np.array([hr, spo2, 16, 36.7])
        }
    
    def apply_filters(self, sensor_data):
        """Apply Kalman and digital filters"""
        # Kalman filtering
        accel_filtered = self.accel_kf.update(sensor_data['accel'])
        gyro_filtered = self.gyro_kf.update(sensor_data['gyro'])
        
        return {
            'accel': accel_filtered,
            'gyro': gyro_filtered,
            'mag': sensor_data['mag'],
            'vitals': sensor_data['vitals']
        }
    
    def buffer_data(self, filtered_data):
        """Add to sliding window buffer"""
        self.accel_buffer.append(filtered_data['accel'])
        self.gyro_buffer.append(filtered_data['gyro'])
        self.mag_buffer.append(filtered_data['mag'])
        self.vitals_buffer.append(filtered_data['vitals'])
    
    def is_buffer_ready(self):
        """Check if buffer is full"""
        return len(self.accel_buffer) == self.window_size
    
    def get_inference_window(self):
        """Prepare data for ML inference"""
        accel = np.array(self.accel_buffer)
        gyro = np.array(self.gyro_buffer)
        mag = np.array(self.mag_buffer)
        vitals = np.array(self.vitals_buffer)
        
        # Concatenate all features
        window = np.concatenate([accel, gyro, mag, vitals], axis=1)
        
        return window.reshape(1, self.window_size, 13)  # Add batch dimension
    
    async def get_gps(self):
        """Get current GPS coordinates"""
        # Simulate GPS read
        return {
            'latitude': 19.0760 + np.random.randn() * 0.001,
            'longitude': 72.8777 + np.random.randn() * 0.001,
            'accuracy': 5.0,
            'altitude': 10.0
        }
    
    async def get_vitals(self):
        """Get current vital signs"""
        if len(self.vitals_buffer) > 0:
            latest = self.vitals_buffer[-1]
            return {
                'hr': latest[0],
                'spo2': latest[1],
                'br': latest[2],
                'temp': latest[3]
            }
        return {'hr': 0, 'spo2': 0, 'br': 0, 'temp': 0}
    
    async def detect_movement(self, threshold=0.5, duration=5):
        """Detect if soldier is moving"""
        await asyncio.sleep(duration)
        
        if len(self.accel_buffer) < 50:
            return False
        
        recent_accel = np.array(list(self.accel_buffer)[-50:])
        movement_magnitude = np.std(recent_accel)
        
        return movement_magnitude > threshold
