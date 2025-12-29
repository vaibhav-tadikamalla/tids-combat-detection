import numpy as np
from scipy import signal
from scipy.fft import fft
import h5py

class MilitaryScenarioGenerator:
    """Generate realistic combat scenario data"""
    
    IMPACT_PROFILES = {
        'blast': {
            'peak_acceleration': (80, 250),  # g-force
            'duration': (0.1, 0.5),  # seconds
            'frequency_range': (50, 200),  # Hz
            'shockwave_pattern': 'exponential_decay'
        },
        'gunshot': {
            'peak_acceleration': (30, 80),
            'duration': (0.05, 0.15),
            'frequency_range': (100, 500),
            'shockwave_pattern': 'sharp_spike'
        },
        'vehicle_crash': {
            'peak_acceleration': (20, 60),
            'duration': (0.5, 2.0),
            'frequency_range': (5, 50),
            'shockwave_pattern': 'sustained_impact'
        },
        'fall': {
            'peak_acceleration': (10, 40),
            'duration': (0.2, 0.8),
            'frequency_range': (2, 20),
            'shockwave_pattern': 'sudden_stop'
        },
        'artillery': {
            'peak_acceleration': (100, 300),
            'duration': (0.2, 0.7),
            'frequency_range': (30, 150),
            'shockwave_pattern': 'double_pulse'
        },
        'normal': {
            'peak_acceleration': (0.5, 3),
            'duration': None,
            'frequency_range': (0.1, 5),
            'shockwave_pattern': 'random_walk'
        }
    }
    
    def __init__(self, sample_rate=200, sequence_length=200):
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        
    def generate_blast_signature(self, params):
        """Generate physics-based blast wave signature"""
        duration = np.random.uniform(*params['duration'])
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        peak_accel = np.random.uniform(*params['peak_acceleration'])
        
        # Friedlander waveform (standard blast wave equation)
        pressure = peak_accel * (1 - t/duration) * np.exp(-t * 5)
        
        # Add high-frequency shock components
        shock_freq = np.random.uniform(*params['frequency_range'])
        shock = np.sin(2 * np.pi * shock_freq * t) * np.exp(-t * 10)
        
        signature = pressure + 0.3 * shock
        return self._pad_sequence(signature)
    
    def generate_gunshot_signature(self, params):
        """Generate gunshot recoil/impact signature"""
        duration = np.random.uniform(*params['duration'])
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        peak_accel = np.random.uniform(*params['peak_acceleration'])
        
        # Sharp impulse with rapid decay
        impulse = peak_accel * np.exp(-t * 50) * (1 + 0.5 * np.sin(2 * np.pi * 200 * t))
        
        return self._pad_sequence(impulse)
    
    def generate_fall_signature(self, params):
        """Generate fall impact signature"""
        # Pre-impact (freefall)
        freefall_duration = np.random.uniform(0.1, 0.5)
        freefall_samples = int(freefall_duration * self.sample_rate)
        freefall = np.random.normal(-9.81, 0.5, freefall_samples)
        
        # Impact
        impact_duration = np.random.uniform(*params['duration'])
        impact_samples = int(impact_duration * self.sample_rate)
        t = np.linspace(0, impact_duration, impact_samples)
        
        peak_accel = np.random.uniform(*params['peak_acceleration'])
        impact = peak_accel * np.exp(-t * 8) * (1 - np.cos(2 * np.pi * 10 * t))
        
        # Post-impact settling
        settling = np.random.normal(1, 0.3, 20)
        
        signature = np.concatenate([freefall, impact, settling])
        return self._pad_sequence(signature)
    
    def add_sensor_noise(self, signal_data, snr_db=25):
        """Add realistic sensor noise"""
        signal_power = np.mean(signal_data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal_data.shape)
        return signal_data + noise
    
    def generate_vital_response(self, impact_type, severity):
        """Simulate physiological response to trauma"""
        baseline_hr = np.random.uniform(60, 80)
        baseline_spo2 = np.random.uniform(95, 99)
        
        if impact_type in ['blast', 'artillery', 'gunshot']:
            # Acute stress response
            hr_spike = severity * np.random.uniform(40, 80)
            spo2_drop = severity * np.random.uniform(5, 15)
            
            hr_curve = baseline_hr + hr_spike * (1 - np.exp(-np.linspace(0, 5, self.sequence_length)))
            spo2_curve = baseline_spo2 - spo2_drop * (1 - np.exp(-np.linspace(0, 3, self.sequence_length)))
            
        elif impact_type == 'fall':
            hr_spike = severity * np.random.uniform(20, 40)
            hr_curve = baseline_hr + hr_spike * np.exp(-np.linspace(0, 2, self.sequence_length))
            spo2_curve = np.full(self.sequence_length, baseline_spo2)
            
        else:
            hr_curve = np.full(self.sequence_length, baseline_hr)
            spo2_curve = np.full(self.sequence_length, baseline_spo2)
        
        # Add breathing rate and skin temperature
        br = np.random.uniform(12, 18) + np.random.normal(0, 1, self.sequence_length)
        temp = np.full(self.sequence_length, 36.5 + np.random.uniform(-0.5, 0.5))
        
        return np.stack([hr_curve, spo2_curve, br, temp], axis=-1)
    
    def generate_sample(self, impact_type):
        """Generate complete multi-sensor sample"""
        params = self.IMPACT_PROFILES[impact_type]
        
        # Generate accelerometer signature (3-axis)
        if impact_type == 'blast':
            accel_z = self.generate_blast_signature(params)
        elif impact_type == 'gunshot':
            accel_z = self.generate_gunshot_signature(params)
        elif impact_type == 'fall':
            accel_z = self.generate_fall_signature(params)
        else:
            accel_z = np.random.normal(0, 1, self.sequence_length)
        
        # Add perpendicular axes
        accel_x = 0.3 * accel_z + np.random.normal(0, 0.5, self.sequence_length)
        accel_y = 0.2 * accel_z + np.random.normal(0, 0.5, self.sequence_length)
        
        # Generate gyroscope (rotational motion)
        gyro = np.random.normal(0, 10, (self.sequence_length, 3))
        if impact_type != 'normal':
            impact_idx = np.argmax(np.abs(accel_z))
            gyro[impact_idx:impact_idx+20] += np.random.uniform(50, 200, (min(20, self.sequence_length-impact_idx), 3))
        
        # Generate magnetometer
        mag = np.random.normal([30, 20, -40], 5, (self.sequence_length, 3))
        
        # Add sensor noise
        accel = self.add_sensor_noise(np.stack([accel_x, accel_y, accel_z], axis=-1))
        gyro = self.add_sensor_noise(gyro)
        mag = self.add_sensor_noise(mag)
        
        # Generate vitals
        severity = np.random.uniform(0.3, 1.0) if impact_type != 'normal' else 0.0
        vitals = self.generate_vital_response(impact_type, severity)
        
        # Concatenate all sensors
        sample = np.concatenate([accel, gyro, mag, vitals], axis=-1)
        
        return sample, severity
    
    def _pad_sequence(self, signal_data):
        """Pad or trim to fixed sequence length"""
        if len(signal_data) >= self.sequence_length:
            return signal_data[:self.sequence_length]
        else:
            padding = np.zeros(self.sequence_length - len(signal_data))
            return np.concatenate([signal_data, padding])
    
    def generate_dataset(self, samples_per_class=1000):
        """Generate complete training dataset"""
        impact_types = list(self.IMPACT_PROFILES.keys())
        X, y_type, y_severity = [], [], []
        
        for i, impact_type in enumerate(impact_types):
            print(f"Generating {samples_per_class} samples for {impact_type}...")
            for _ in range(samples_per_class):
                sample, severity = self.generate_sample(impact_type)
                X.append(sample)
                
                # One-hot encode type
                type_label = np.zeros(len(impact_types))
                type_label[i] = 1
                y_type.append(type_label)
                
                y_severity.append(severity)
        
        return np.array(X), np.array(y_type), np.array(y_severity)


# Generate dataset
if __name__ == "__main__":
    generator = MilitaryScenarioGenerator()
    X_train, y_type, y_severity = generator.generate_dataset(samples_per_class=2000)
    
    # Save to HDF5
    with h5py.File('combat_trauma_dataset.h5', 'w') as f:
        f.create_dataset('X', data=X_train)
        f.create_dataset('y_type', data=y_type)
        f.create_dataset('y_severity', data=y_severity)
    
    print(f"Dataset generated: {X_train.shape}")
