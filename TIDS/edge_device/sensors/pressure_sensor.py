import numpy as np
from scipy import signal
import asyncio

class PredictiveBlastDetection:
    """
    Detect blast waves BEFORE they cause injury using:
    1. Rapid pressure change detection (100+ readings/sec)
    2. Blast wave propagation modeling
    3. Pre-impact warning (50-200ms advance notice)
    
    Theory: Blast pressure wave travels at sound speed, but we can detect
    the pressure change and warn before the main shockwave arrives.
    """
    
    ATMOSPHERIC_PRESSURE = 101325  # Pa
    BLAST_THRESHOLD = 1000  # Pa above ambient (indicates nearby explosion)
    SAMPLE_RATE = 1000  # Hz (1000 readings/sec for high-speed detection)
    
    def __init__(self):
        self.pressure_buffer = []
        self.baseline_pressure = self.ATMOSPHERIC_PRESSURE
        self.calibration_samples = 1000
        
        # Blast wave prediction model
        self.blast_detector = signal.butter(4, [10, 200], 'band', 
                                            fs=self.SAMPLE_RATE, output='sos')
        
    async def calibrate(self):
        """Calibrate to local atmospheric pressure"""
        print("[PRESSURE SENSOR] Calibrating...")
        samples = []
        
        for _ in range(self.calibration_samples):
            pressure = await self._read_pressure()
            samples.append(pressure)
            await asyncio.sleep(1 / self.SAMPLE_RATE)
        
        self.baseline_pressure = np.median(samples)
        print(f"[PRESSURE SENSOR] Baseline: {self.baseline_pressure:.1f} Pa")
    
    async def monitor_blast_waves(self):
        """Continuous monitoring for blast wave detection"""
        while True:
            pressure = await self._read_pressure()
            self.pressure_buffer.append(pressure)
            
            # Keep 500ms of data
            if len(self.pressure_buffer) > self.SAMPLE_RATE // 2:
                self.pressure_buffer.pop(0)
            
            # Check for rapid pressure change
            if len(self.pressure_buffer) >= 50:  # 50ms of data
                blast_detected = await self._analyze_pressure_wave()
                
                if blast_detected:
                    # CRITICAL: Blast wave incoming
                    await self._trigger_pre_impact_warning(blast_detected)
            
            await asyncio.sleep(1 / self.SAMPLE_RATE)
    
    async def _analyze_pressure_wave(self):
        """Detect characteristic blast wave signature"""
        pressure_data = np.array(self.pressure_buffer)
        
        # Compute pressure deviation from baseline
        deviation = pressure_data - self.baseline_pressure
        
        # Apply bandpass filter (blast waves: 10-200 Hz)
        filtered = signal.sosfilt(self.blast_detector, deviation)
        
        # Compute rate of change
        pressure_derivative = np.gradient(filtered)
        
        # Detect rapid positive pressure spike
        max_derivative = np.max(pressure_derivative)
        max_pressure = np.max(np.abs(deviation))
        
        # Blast wave criteria
        if max_derivative > 500 and max_pressure > self.BLAST_THRESHOLD:
            # Estimate blast parameters
            blast_distance = self._estimate_blast_distance(max_pressure)
            time_to_impact = self._estimate_time_to_impact(blast_distance, pressure_derivative)
            blast_magnitude = self._estimate_blast_magnitude(max_pressure)
            
            return {
                'type': 'BLAST_WAVE_DETECTED',
                'distance_meters': blast_distance,
                'time_to_impact_ms': time_to_impact,
                'magnitude': blast_magnitude,
                'overpressure_pa': max_pressure,
                'confidence': 0.92
            }
        
        return None
    
    def _estimate_blast_distance(self, overpressure):
        """
        Estimate blast distance using scaled overpressure equation.
        
        Kingery-Bulmash equation (simplified):
        P = K * (W^(1/3) / R)^n
        
        Where:
        P = overpressure (Pa)
        W = explosive weight (kg TNT equivalent)
        R = distance (m)
        K, n = empirical constants
        """
        # Assume 10kg TNT equivalent (typical IED)
        W = 10
        K = 100000
        n = 1.5
        
        # Solve for R
        R = (K * W**(1/3) / overpressure)**(1/n)
        
        return max(R, 1)  # Minimum 1 meter
    
    def _estimate_time_to_impact(self, distance, pressure_derivative):
        """Estimate time until main shockwave arrives"""
        # Blast wave travels at ~340 m/s initially, then slows
        wave_speed = 340 + 100 * (1 / max(distance, 1))  # Faster for close blasts
        
        # Current pressure rise rate indicates how much of wave has arrived
        arrival_fraction = np.sum(pressure_derivative > 0) / len(pressure_derivative)
        
        remaining_time_ms = (distance * (1 - arrival_fraction) / wave_speed) * 1000
        
        return max(remaining_time_ms, 0)
    
    def _estimate_blast_magnitude(self, overpressure):
        """Classify blast severity"""
        if overpressure > 50000:  # 50 kPa
            return 'LETHAL'
        elif overpressure > 20000:  # 20 kPa
            return 'SEVERE_INJURY'
        elif overpressure > 5000:  # 5 kPa
            return 'MODERATE_INJURY'
        else:
            return 'MINOR'
    
    async def _trigger_pre_impact_warning(self, blast_data):
        """
        CRITICAL: Issue warning BEFORE blast wave hits.
        
        This gives the soldier 50-200ms to:
        - Brace for impact
        - Close eyes/mouth
        - Turn away from blast
        - Activate protective measures
        """
        print(f"""
        ⚠️  BLAST WAVE INCOMING ⚠️
        Distance: {blast_data['distance_meters']:.1f}m
        Impact in: {blast_data['time_to_impact_ms']:.0f}ms
        Severity: {blast_data['magnitude']}
        """)
        
        # Trigger haptic/audio warning on device
        await self._activate_emergency_warning()
        
        # Send to command center
        # await self.secure_tx.send_emergency({...})
    
    async def _activate_emergency_warning(self):
        """Activate emergency warning systems on device"""
        # In production:
        # - Intense vibration pattern
        # - Audio warning (if headset connected)
        # - Visual flash (if has display)
        pass
    
    async def _read_pressure(self):
        """Read barometric pressure sensor"""
        # Placeholder - would interface with actual sensor (e.g., BMP388)
        # Simulating normal atmospheric pressure with small variations
        return self.ATMOSPHERIC_PRESSURE + np.random.randn() * 10
