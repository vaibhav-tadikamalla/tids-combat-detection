"""
IMU (Inertial Measurement Unit) Handler
Manages accelerometer, gyroscope, and magnetometer sensors
"""

import numpy as np
import time
from typing import Dict, Tuple
import logging

try:
    import board
    import busio
    import adafruit_lsm9ds1
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("IMU hardware libraries not available - using simulation mode")


class IMUHandler:
    """
    Handles IMU sensor data acquisition and processing.
    
    Sensors:
    - 3-axis accelerometer (±16g range)
    - 3-axis gyroscope (±2000°/s range)
    - 3-axis magnetometer
    """
    
    def __init__(self, simulate=False):
        self.simulate = simulate or not HARDWARE_AVAILABLE
        self.sensor = None
        
        if not self.simulate:
            try:
                # Initialize I2C bus
                i2c = busio.I2C(board.SCL, board.SDA)
                self.sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)
                
                # Configure sensor ranges
                self.sensor.accel_range = adafruit_lsm9ds1.ACCELRANGE_16G
                self.sensor.gyro_scale = adafruit_lsm9ds1.GYROSCALE_2000DPS
                
                logging.info("IMU sensor initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize IMU sensor: {e}")
                self.simulate = True
        
        # Calibration offsets
        self.accel_offset = np.zeros(3)
        self.gyro_offset = np.zeros(3)
        self.mag_offset = np.zeros(3)
        
        # Simulation state
        self.sim_time = 0
        self.last_read_time = time.time()
    
    def calibrate(self, samples=100):
        """
        Calibrate IMU by averaging readings while stationary.
        
        Args:
            samples: Number of samples to average for calibration
        """
        logging.info(f"Calibrating IMU with {samples} samples...")
        
        accel_sum = np.zeros(3)
        gyro_sum = np.zeros(3)
        
        for _ in range(samples):
            accel, gyro, _ = self.read_raw()
            accel_sum += accel
            gyro_sum += gyro
            time.sleep(0.01)
        
        # Gyroscope should read zero when stationary
        self.gyro_offset = gyro_sum / samples
        
        # Accelerometer should read [0, 0, 9.81] (gravity) when stationary
        avg_accel = accel_sum / samples
        self.accel_offset = avg_accel - np.array([0, 0, 9.81])
        
        logging.info("IMU calibration complete")
        logging.info(f"  Accel offset: {self.accel_offset}")
        logging.info(f"  Gyro offset: {self.gyro_offset}")
    
    def read_raw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read raw sensor data.
        
        Returns:
            Tuple of (acceleration, angular_velocity, magnetic_field)
        """
        if self.simulate:
            return self._simulate_reading()
        
        try:
            # Read accelerometer (m/s²)
            accel = np.array(self.sensor.acceleration)
            
            # Read gyroscope (rad/s)
            gyro = np.array(self.sensor.gyro)
            
            # Read magnetometer (gauss)
            mag = np.array(self.sensor.magnetic)
            
            return accel, gyro, mag
            
        except Exception as e:
            logging.error(f"Error reading IMU: {e}")
            return self._simulate_reading()
    
    def read(self) -> Dict[str, np.ndarray]:
        """
        Read calibrated sensor data.
        
        Returns:
            Dictionary with calibrated sensor readings
        """
        accel_raw, gyro_raw, mag_raw = self.read_raw()
        
        # Apply calibration
        accel = accel_raw - self.accel_offset
        gyro = gyro_raw - self.gyro_offset
        mag = mag_raw - self.mag_offset
        
        return {
            'acceleration': accel,
            'angular_velocity': gyro,
            'magnetic_field': mag,
            'timestamp': time.time()
        }
    
    def get_magnitude(self, vector: np.ndarray) -> float:
        """Calculate magnitude of a vector"""
        return np.linalg.norm(vector)
    
    def detect_impact(self, accel: np.ndarray, threshold=50.0) -> bool:
        """
        Detect sudden impact based on acceleration magnitude.
        
        Args:
            accel: Acceleration vector (m/s²)
            threshold: Impact threshold (m/s²)
        
        Returns:
            True if impact detected
        """
        magnitude = self.get_magnitude(accel)
        return magnitude > threshold
    
    def _simulate_reading(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate IMU readings for testing.
        
        Returns realistic sensor noise and occasional impacts.
        """
        current_time = time.time()
        dt = current_time - self.last_read_time
        self.last_read_time = current_time
        self.sim_time += dt
        
        # Base readings (stationary device)
        accel = np.array([0.0, 0.0, 9.81])  # Gravity
        gyro = np.array([0.0, 0.0, 0.0])
        mag = np.array([0.3, 0.1, 0.4])  # Earth's magnetic field
        
        # Add sensor noise
        accel += np.random.normal(0, 0.1, 3)
        gyro += np.random.normal(0, 0.01, 3)
        mag += np.random.normal(0, 0.02, 3)
        
        # Simulate occasional movement
        if np.random.random() < 0.01:  # 1% chance
            accel += np.random.normal(0, 5, 3)
            gyro += np.random.normal(0, 0.5, 3)
        
        # Simulate very rare impact event
        if np.random.random() < 0.001:  # 0.1% chance
            impact_magnitude = np.random.uniform(30, 100)
            impact_direction = np.random.randn(3)
            impact_direction /= np.linalg.norm(impact_direction)
            accel += impact_direction * impact_magnitude
        
        return accel, gyro, mag
    
    def get_orientation(self, accel: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
        """
        Calculate device orientation from accelerometer and magnetometer.
        
        Args:
            accel: Acceleration vector
            mag: Magnetic field vector
        
        Returns:
            Dictionary with roll, pitch, yaw angles (degrees)
        """
        # Normalize vectors
        accel_norm = accel / np.linalg.norm(accel)
        mag_norm = mag / np.linalg.norm(mag)
        
        # Calculate roll and pitch from accelerometer
        roll = np.arctan2(accel_norm[1], accel_norm[2])
        pitch = np.arctan2(-accel_norm[0], 
                          np.sqrt(accel_norm[1]**2 + accel_norm[2]**2))
        
        # Calculate yaw from magnetometer (compensated for roll/pitch)
        mag_x = mag_norm[0] * np.cos(pitch) + mag_norm[2] * np.sin(pitch)
        mag_y = (mag_norm[0] * np.sin(roll) * np.sin(pitch) + 
                mag_norm[1] * np.cos(roll) - 
                mag_norm[2] * np.sin(roll) * np.cos(pitch))
        yaw = np.arctan2(-mag_y, mag_x)
        
        return {
            'roll': np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw)
        }
    
    def close(self):
        """Release sensor resources"""
        if self.sensor:
            # LSM9DS1 doesn't require explicit cleanup
            pass
        logging.info("IMU handler closed")


if __name__ == "__main__":
    # Test IMU handler
    logging.basicConfig(level=logging.INFO)
    
    imu = IMUHandler(simulate=True)
    
    print("Testing IMU handler (simulation mode)...")
    print("Reading 10 samples:\n")
    
    for i in range(10):
        data = imu.read()
        accel = data['acceleration']
        gyro = data['angular_velocity']
        
        print(f"Sample {i+1}:")
        print(f"  Accel: [{accel[0]:6.2f}, {accel[1]:6.2f}, {accel[2]:6.2f}] m/s²")
        print(f"  Gyro:  [{gyro[0]:6.2f}, {gyro[1]:6.2f}, {gyro[2]:6.2f}] rad/s")
        print(f"  Magnitude: {imu.get_magnitude(accel):.2f} m/s²")
        
        if imu.detect_impact(accel):
            print("  ⚠ IMPACT DETECTED!")
        
        print()
        time.sleep(0.2)
    
    imu.close()
