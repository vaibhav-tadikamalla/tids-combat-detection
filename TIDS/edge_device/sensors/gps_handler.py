"""
GPS Handler Module
Manages GPS sensor for location tracking
"""

import time
import numpy as np
from typing import Optional, Tuple, Dict
import logging

try:
    import serial
    import adafruit_gps
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    logging.warning("GPS hardware libraries not available - using simulation mode")


class GPSHandler:
    """
    Handles GPS sensor data acquisition.
    
    Features:
    - Real-time location tracking
    - Speed and heading calculation
    - Satellite count monitoring
    - Fix quality assessment
    """
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, simulate=False):
        self.simulate = simulate or not HARDWARE_AVAILABLE
        self.gps = None
        
        if not self.simulate:
            try:
                uart = serial.Serial(port, baudrate=baudrate, timeout=10)
                self.gps = adafruit_gps.GPS(uart, debug=False)
                
                # Initialize GPS
                self.gps.send_command(b'PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0')
                self.gps.send_command(b'PMTK220,1000')  # 1Hz update rate
                
                logging.info("GPS sensor initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize GPS: {e}")
                self.simulate = True
        
        # Simulation state
        self.sim_lat = 19.0760  # Mumbai coordinates
        self.sim_lon = 72.8777
        self.sim_time = 0
        self.last_position = None
        self.last_time = time.time()
    
    def update(self):
        """Update GPS data (call regularly)"""
        if not self.simulate and self.gps:
            try:
                self.gps.update()
            except Exception as e:
                logging.error(f"GPS update error: {e}")
    
    def has_fix(self) -> bool:
        """Check if GPS has valid fix"""
        if self.simulate:
            return True
        return self.gps.has_fix if self.gps else False
    
    def get_location(self) -> Optional[Tuple[float, float]]:
        """
        Get current GPS location.
        
        Returns:
            Tuple of (latitude, longitude) or None if no fix
        """
        if self.simulate:
            return self._simulate_location()
        
        if not self.gps or not self.gps.has_fix:
            return None
        
        try:
            lat = self.gps.latitude
            lon = self.gps.longitude
            
            if lat is not None and lon is not None:
                self.last_position = (lat, lon)
                return (lat, lon)
            
            return None
        except Exception as e:
            logging.error(f"Error reading GPS location: {e}")
            return None
    
    def get_full_data(self) -> Dict:
        """
        Get complete GPS data.
        
        Returns:
            Dictionary with all GPS information
        """
        location = self.get_location()
        
        if self.simulate:
            return {
                'latitude': location[0] if location else None,
                'longitude': location[1] if location else None,
                'altitude': 10.0,
                'speed': 0.5,  # m/s
                'heading': 45.0,  # degrees
                'satellites': 8,
                'fix_quality': 3,
                'has_fix': True,
                'timestamp': time.time()
            }
        
        if not self.gps:
            return {'has_fix': False}
        
        return {
            'latitude': self.gps.latitude,
            'longitude': self.gps.longitude,
            'altitude': self.gps.altitude_m,
            'speed': self.gps.speed_knots * 0.514444 if self.gps.speed_knots else 0,  # Convert to m/s
            'heading': self.gps.track_angle_deg,
            'satellites': self.gps.satellites,
            'fix_quality': self.gps.fix_quality,
            'has_fix': self.gps.has_fix,
            'timestamp': time.time()
        }
    
    def calculate_distance(self, coord1: Tuple[float, float], 
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula.
        
        Args:
            coord1: (lat1, lon1)
            coord2: (lat2, lon2)
        
        Returns:
            Distance in meters
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi/2)**2 + 
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def calculate_bearing(self, coord1: Tuple[float, float], 
                         coord2: Tuple[float, float]) -> float:
        """
        Calculate bearing from coord1 to coord2.
        
        Args:
            coord1: Starting coordinate (lat1, lon1)
            coord2: Destination coordinate (lat2, lon2)
        
        Returns:
            Bearing in degrees (0-360)
        """
        lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
        lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])
        
        delta_lon = lon2 - lon1
        
        x = np.sin(delta_lon) * np.cos(lat2)
        y = (np.cos(lat1) * np.sin(lat2) - 
             np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon))
        
        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def _simulate_location(self) -> Tuple[float, float]:
        """
        Simulate GPS location with realistic movement.
        
        Returns:
            Tuple of (latitude, longitude)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.sim_time += dt
        
        # Simulate slow walking (1 m/s)
        speed_mps = 1.0
        
        # Random walk pattern
        heading = (self.sim_time * 10) % 360  # Slow circular pattern
        
        # Convert speed to lat/lon change
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        distance_m = speed_mps * dt
        
        delta_lat = distance_m * np.cos(np.radians(heading)) / 111000
        delta_lon = distance_m * np.sin(np.radians(heading)) / (111000 * np.cos(np.radians(self.sim_lat)))
        
        self.sim_lat += delta_lat
        self.sim_lon += delta_lon
        
        # Add GPS noise (typical accuracy ±5m)
        noise_lat = np.random.normal(0, 5 / 111000)
        noise_lon = np.random.normal(0, 5 / (111000 * np.cos(np.radians(self.sim_lat))))
        
        return (self.sim_lat + noise_lat, self.sim_lon + noise_lon)
    
    def close(self):
        """Release GPS resources"""
        if self.gps:
            # Close serial connection if needed
            pass
        logging.info("GPS handler closed")


if __name__ == "__main__":
    # Test GPS handler
    logging.basicConfig(level=logging.INFO)
    
    gps = GPSHandler(simulate=True)
    
    print("Testing GPS handler (simulation mode)...")
    print("Reading 10 samples:\n")
    
    start_location = gps.get_location()
    
    for i in range(10):
        gps.update()
        data = gps.get_full_data()
        
        print(f"Sample {i+1}:")
        print(f"  Location: {data['latitude']:.6f}, {data['longitude']:.6f}")
        print(f"  Altitude: {data['altitude']:.1f} m")
        print(f"  Speed: {data['speed']:.2f} m/s")
        print(f"  Heading: {data['heading']:.1f}°")
        print(f"  Satellites: {data['satellites']}")
        print(f"  Fix: {data['has_fix']}")
        
        current_location = (data['latitude'], data['longitude'])
        if start_location and i > 0:
            distance = gps.calculate_distance(start_location, current_location)
            bearing = gps.calculate_bearing(start_location, current_location)
            print(f"  Distance from start: {distance:.2f} m")
            print(f"  Bearing from start: {bearing:.1f}°")
        
        print()
        time.sleep(1)
    
    gps.close()
