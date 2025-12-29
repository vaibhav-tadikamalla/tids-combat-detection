import numpy as np
from scipy import signal
from scipy.optimize import least_squares
from collections import deque
import asyncio
import time

class AcousticSniperDetection:
    """
    Detect gunshots and triangulate shooter location using:
    1. Muzzle blast detection
    2. Supersonic shockwave detection
    3. Multi-device triangulation via mesh network
    """
    
    SOUND_SPEED = 343  # m/s at 20°C
    MUZZLE_BLAST_FREQ = (100, 2000)  # Hz
    SHOCKWAVE_FREQ = (2000, 8000)  # Hz
    
    def __init__(self, sample_rate=44100, device_positions=None):
        self.sample_rate = sample_rate
        self.device_positions = device_positions or {}  # {device_id: (x, y, z)}
        
        # Audio buffer (2 seconds)
        self.buffer_size = sample_rate * 2
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Bandpass filters
        self.muzzle_filter = signal.butter(4, self.MUZZLE_BLAST_FREQ, 'band', 
                                           fs=sample_rate, output='sos')
        self.shockwave_filter = signal.butter(4, self.SHOCKWAVE_FREQ, 'band',
                                              fs=sample_rate, output='sos')
        
        # Detection state
        self.last_detection_time = 0
        self.detection_events = []
        
    async def process_audio_stream(self, audio_chunk):
        """Real-time audio processing"""
        self.audio_buffer.extend(audio_chunk)
        
        if len(self.audio_buffer) < self.sample_rate * 0.1:  # Need 100ms minimum
            return None
        
        # Convert to numpy array
        audio_data = np.array(list(self.audio_buffer)[-int(self.sample_rate * 0.5):])
        
        # Detect muzzle blast
        muzzle_detected, muzzle_time = self._detect_impulse(
            audio_data, self.muzzle_filter, threshold=0.3
        )
        
        # Detect shockwave
        shockwave_detected, shockwave_time = self._detect_impulse(
            audio_data, self.shockwave_filter, threshold=0.5
        )
        
        # Gunshot = both muzzle blast and shockwave
        if muzzle_detected and shockwave_detected:
            detection = await self._analyze_gunshot(
                muzzle_time, shockwave_time, audio_data
            )
            return detection
        
        return None
    
    def _detect_impulse(self, audio_data, filter_sos, threshold):
        """Detect sudden impulse in filtered audio"""
        filtered = signal.sosfilt(filter_sos, audio_data)
        
        # Compute energy envelope
        envelope = np.abs(signal.hilbert(filtered))
        
        # Detect peaks
        peaks, properties = signal.find_peaks(
            envelope, 
            height=threshold * np.max(envelope),
            distance=self.sample_rate * 0.01  # 10ms minimum separation
        )
        
        if len(peaks) > 0:
            # Return time of first peak
            detection_time = time.time() - (len(audio_data) - peaks[0]) / self.sample_rate
            return True, detection_time
        
        return False, None
    
    async def _analyze_gunshot(self, muzzle_time, shockwave_time, audio_data):
        """Analyze gunshot characteristics"""
        
        # Time difference between shockwave and muzzle blast
        time_diff = abs(muzzle_time - shockwave_time)
        
        # Estimate distance (shockwave arrives first for supersonic bullets)
        if shockwave_time < muzzle_time:
            # Supersonic bullet
            bullet_speed = 900  # m/s (typical rifle)
            distance = (muzzle_time - shockwave_time) * self.SOUND_SPEED * bullet_speed / (bullet_speed - self.SOUND_SPEED)
        else:
            # Subsonic or distant shot
            distance = time_diff * self.SOUND_SPEED
        
        # Analyze frequency to identify weapon type
        weapon_type = self._identify_weapon(audio_data)
        
        # Direction estimation (requires multiple microphones or mesh network)
        direction = await self._estimate_direction(muzzle_time, shockwave_time)
        
        detection = {
            'type': 'GUNSHOT',
            'timestamp': time.time(),
            'distance': distance,
            'direction': direction,
            'weapon_type': weapon_type,
            'confidence': 0.85,
            'requires_triangulation': True
        }
        
        # Broadcast to mesh network for triangulation
        await self._broadcast_for_triangulation(detection)
        
        return detection
    
    def _identify_weapon(self, audio_data):
        """Identify weapon type from acoustic signature"""
        # Compute power spectral density
        f, psd = signal.welch(audio_data, self.sample_rate, nperseg=1024)
        
        # Peak frequency analysis
        peak_freq = f[np.argmax(psd)]
        
        # Weapon classification (simplified)
        if peak_freq < 500:
            return 'Heavy caliber (sniper rifle or machine gun)'
        elif peak_freq < 1000:
            return 'Assault rifle'
        elif peak_freq < 2000:
            return 'Pistol or SMG'
        else:
            return 'Unknown small arms'
    
    async def _estimate_direction(self, muzzle_time, shockwave_time):
        """Estimate direction (requires microphone array or mesh network)"""
        # Placeholder - would use TDOA (Time Difference of Arrival)
        # In production: use multiple microphones on device or mesh network data
        return {
            'azimuth': None,  # degrees from north
            'elevation': None,  # degrees from horizontal
            'method': 'pending_triangulation'
        }
    
    async def _broadcast_for_triangulation(self, detection):
        """Send detection to nearby devices for triangulation"""
        # Would use mesh network to share detection times
        # Multiple devices can then triangulate shooter position
        pass
    
    async def triangulate_shooter(self, detection_events):
        """
        Triangulate shooter position from multiple device detections.
        
        Args:
            detection_events: List of {device_id, timestamp, position}
        
        Returns:
            Shooter 3D coordinates (x, y, z)
        """
        if len(detection_events) < 3:
            return None  # Need at least 3 devices
        
        # Extract positions and times
        positions = np.array([
            self.device_positions[evt['device_id']] 
            for evt in detection_events
        ])
        
        times = np.array([evt['timestamp'] for evt in detection_events])
        
        # TDOA (Time Difference of Arrival) triangulation
        def residuals(shooter_pos):
            distances = np.linalg.norm(positions - shooter_pos, axis=1)
            time_diffs = distances / self.SOUND_SPEED
            return time_diffs - (times - times[0])
        
        # Initial guess (centroid)
        x0 = np.mean(positions, axis=0)
        
        # Solve
        result = least_squares(residuals, x0)
        
        if result.success:
            return {
                'shooter_position': result.x.tolist(),
                'accuracy': np.linalg.norm(result.fun),
                'num_devices': len(detection_events),
                'method': 'TDOA_triangulation'
            }
        
        return None


# Integration with main system
class EnhancedSensorFusion:
    """Extended sensor fusion with acoustic detection"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acoustic_detector = AcousticSniperDetection()
        
    async def read_sensors(self):
        """Override to include acoustic sensor"""
        base_data = await super().read_sensors()
        
        # Read microphone
        audio_chunk = await self._read_microphone()
        
        # Process for gunshot detection
        gunshot_detection = await self.acoustic_detector.process_audio_stream(audio_chunk)
        
        if gunshot_detection:
            # Immediate high-priority alert
            await self._trigger_gunshot_alert(gunshot_detection)
        
        base_data['acoustic'] = gunshot_detection
        return base_data
    
    async def _read_microphone(self):
        """Read audio from microphone (placeholder)"""
        # In production: interface with actual microphone hardware
        # For now, return simulated data
        return np.random.randn(4410)  # 100ms at 44.1kHz
    
    async def _trigger_gunshot_alert(self, detection):
        """Send immediate gunshot alert"""
        alert = {
            'alert_type': 'ACTIVE_SHOOTER',
            'priority': 'CRITICAL',
            'detection': detection,
            'timestamp': time.time(),
            'requires_immediate_action': True
        }
        # Would integrate with existing alert system
        print(f"[GUNSHOT DETECTED] Distance: {detection['distance']:.1f}m, Weapon: {detection['weapon_type']}")
