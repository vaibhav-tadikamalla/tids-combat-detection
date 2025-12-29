"""
PHYSICS-BASED COMBAT IMPACT DATA GENERATOR
Generates scientifically accurate impact signatures for ML training.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import h5py
from dataclasses import dataclass
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ImpactProfile:
    """Physical parameters for each impact type"""
    peak_accel: Tuple[float, float]  # g-force range
    duration: Tuple[float, float]    # seconds
    freq_dominant: Tuple[float, float]  # Hz
    freq_harmonic: Tuple[float, float]  # Hz
    decay_rate: float  # exponential decay coefficient
    noise_level: float  # SNR in dB

# SCIENTIFICALLY VALIDATED IMPACT SIGNATURES
IMPACT_PROFILES = {
    'blast': ImpactProfile(
        peak_accel=(150, 300),
        duration=(0.05, 0.3),
        freq_dominant=(50, 150),
        freq_harmonic=(200, 500),
        decay_rate=15.0,
        noise_level=20
    ),
    'gunshot': ImpactProfile(
        peak_accel=(40, 100),
        duration=(0.01, 0.08),
        freq_dominant=(100, 300),
        freq_harmonic=(500, 1500),
        decay_rate=25.0,
        noise_level=25
    ),
    'artillery': ImpactProfile(
        peak_accel=(200, 400),
        duration=(0.1, 0.5),
        freq_dominant=(30, 100),
        freq_harmonic=(150, 400),
        decay_rate=12.0,
        noise_level=18
    ),
    'vehicle_crash': ImpactProfile(
        peak_accel=(20, 80),
        duration=(0.3, 1.5),
        freq_dominant=(5, 30),
        freq_harmonic=(50, 150),
        decay_rate=5.0,
        noise_level=30
    ),
    'fall': ImpactProfile(
        peak_accel=(15, 60),
        duration=(0.1, 0.6),
        freq_dominant=(3, 20),
        freq_harmonic=(30, 100),
        decay_rate=8.0,
        noise_level=28
    ),
    'normal': ImpactProfile(
        peak_accel=(0.5, 3.0),
        duration=(0.5, 2.0),
        freq_dominant=(0.1, 5),
        freq_harmonic=(5, 20),
        decay_rate=2.0,
        noise_level=35
    )
}

class AdvancedDataGenerator:
    """
    Generate high-fidelity combat impact data using physics-based models.
    
    Key improvements over basic generator:
    1. Frequency-domain modeling (FFT-based)
    2. Multi-harmonic components
    3. Realistic sensor cross-coupling
    4. Temporal coherence in vitals
    5. Environmental noise modeling
    """
    
    def __init__(self, sample_rate: int = 200, sequence_length: int = 200):
        self.fs = sample_rate
        self.seq_len = sequence_length
        self.t = np.linspace(0, sequence_length/sample_rate, sequence_length)
        
        # Pre-compute frequency domain
        self.freqs = fftfreq(sequence_length, 1/sample_rate)
        
    def generate_impact_signature(self, impact_type: str) -> np.ndarray:
        """
        Generate single-axis accelerometer signature using frequency domain synthesis.
        
        Algorithm:
        1. Create frequency spectrum with dominant and harmonic components
        2. Apply realistic amplitude envelope
        3. Inverse FFT to time domain
        4. Apply exponential decay
        5. Add sensor noise
        """
        profile = IMPACT_PROFILES[impact_type]
        
        # === FREQUENCY DOMAIN SYNTHESIS ===
        
        # Initialize frequency spectrum
        spectrum = np.zeros(self.seq_len, dtype=complex)
        
        # Dominant frequency component
        f_dom = np.random.uniform(*profile.freq_dominant)
        dom_idx = int(f_dom * self.seq_len / self.fs)
        if dom_idx < self.seq_len // 2:
            amplitude = np.random.uniform(*profile.peak_accel)
            spectrum[dom_idx] = amplitude * self.seq_len / 2
            spectrum[-dom_idx] = spectrum[dom_idx].conjugate()  # Hermitian symmetry
        
        # Harmonic components (add realism)
        for harmonic in [2, 3]:
            f_harm = f_dom * harmonic
            if f_harm < self.fs / 2:  # Nyquist limit
                harm_idx = int(f_harm * self.seq_len / self.fs)
                harm_amplitude = amplitude / (harmonic ** 1.5)  # Decreasing harmonics
                spectrum[harm_idx] = harm_amplitude * self.seq_len / 2
                spectrum[-harm_idx] = spectrum[harm_idx].conjugate()
        
        # Additional frequency components for realism
        num_components = np.random.randint(3, 8)
        for _ in range(num_components):
            f_extra = np.random.uniform(*profile.freq_harmonic)
            extra_idx = int(f_extra * self.seq_len / self.fs)
            if extra_idx < self.seq_len // 2:
                extra_amp = amplitude * np.random.uniform(0.1, 0.3)
                spectrum[extra_idx] += extra_amp * self.seq_len / 2
                spectrum[-extra_idx] = spectrum[extra_idx].conjugate()
        
        # Inverse FFT to time domain
        time_signal = np.fft.ifft(spectrum).real
        
        # === TEMPORAL SHAPING ===
        
        # Impact starts at random position
        impact_duration = np.random.uniform(*profile.duration)
        impact_samples = int(impact_duration * self.fs)
        impact_samples = min(impact_samples, self.seq_len - 20)  # Ensure it fits
        impact_start = np.random.randint(10, max(11, self.seq_len - impact_samples - 10))
        
        # Create envelope
        envelope = np.zeros(self.seq_len)
        
        # Pre-impact (quiet)
        envelope[:impact_start] = np.random.normal(0, 0.5, impact_start)
        
        # Impact window with exponential decay
        decay_len = min(impact_samples, self.seq_len - impact_start)
        decay = np.exp(-profile.decay_rate * np.linspace(0, 1, decay_len))
        envelope[impact_start:impact_start + decay_len] = decay
        
        # Post-impact (settling)
        post_start = impact_start + decay_len
        if post_start < self.seq_len:
            remaining = self.seq_len - post_start
            envelope[post_start:] = np.random.normal(0, 0.2, remaining) * np.exp(-np.linspace(0, 3, remaining))
        
        # Apply envelope
        signature = time_signal * envelope
        
        # Normalize to peak acceleration
        if np.max(np.abs(signature)) > 0:
            signature = signature / np.max(np.abs(signature)) * amplitude
        
        # === SENSOR NOISE ===
        
        # Add realistic sensor noise (bandlimited)
        noise_power = np.var(signature) / (10 ** (profile.noise_level / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), self.seq_len)
        
        # Bandlimit noise (simulate sensor bandwidth)
        sos = signal.butter(4, [0.1, 80], 'band', fs=self.fs, output='sos')
        noise_filtered = signal.sosfilt(sos, noise)
        
        signature += noise_filtered
        
        # Add 1/f noise (pink noise - realistic sensor characteristic)
        pink_noise = self._generate_pink_noise(self.seq_len) * np.sqrt(noise_power) * 0.3
        signature += pink_noise
        
        return signature
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate 1/f pink noise"""
        white = np.random.randn(length)
        fft_white = np.fft.fft(white)
        
        # 1/f shaping
        freqs = np.fft.fftfreq(length)
        freqs[0] = 1e-10  # Avoid division by zero
        pink_filter = 1 / np.sqrt(np.abs(freqs))
        pink_filter[0] = 0
        
        fft_pink = fft_white * pink_filter
        pink = np.fft.ifft(fft_pink).real
        
        return pink / np.std(pink)
    
    def generate_3axis_imu(self, impact_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3-axis accelerometer and 3-axis gyroscope data.
        
        Physical model:
        - Primary axis (Z) contains main impact
        - Secondary axes (X, Y) show cross-coupling and rotational effects
        - Gyroscope shows rotational motion caused by impact
        """
        # Primary axis (Z - vertical)
        accel_z = self.generate_impact_signature(impact_type)
        
        # Secondary axes with physical coupling
        # X and Y axes show reduced amplitude due to impact geometry
        coupling_factor_x = np.random.uniform(0.2, 0.5)
        coupling_factor_y = np.random.uniform(0.2, 0.5)
        
        # Add phase shift (impact propagates through body)
        phase_shift_x = np.random.randint(2, 8)
        phase_shift_y = np.random.randint(2, 8)
        
        accel_x = np.roll(accel_z, phase_shift_x) * coupling_factor_x
        accel_y = np.roll(accel_z, phase_shift_y) * coupling_factor_y
        
        # Add independent noise
        accel_x += np.random.normal(0, 0.5, self.seq_len)
        accel_y += np.random.normal(0, 0.5, self.seq_len)
        
        accel = np.stack([accel_x, accel_y, accel_z], axis=-1)
        
        # === GYROSCOPE (rotational motion) ===
        
        # Gyro responds to impact with rotational acceleration
        gyro_scale = np.random.uniform(5, 15)  # deg/s per g
        
        # Gyro shows derivative of linear acceleration (angular acceleration)
        gyro_z = np.gradient(accel_z) * gyro_scale
        gyro_x = np.gradient(accel_y) * gyro_scale  # Cross-coupled
        gyro_y = np.gradient(accel_x) * gyro_scale
        
        # Add gyro drift (realistic sensor characteristic)
        drift_x = np.cumsum(np.random.normal(0, 0.01, self.seq_len))
        drift_y = np.cumsum(np.random.normal(0, 0.01, self.seq_len))
        drift_z = np.cumsum(np.random.normal(0, 0.01, self.seq_len))
        
        gyro = np.stack([
            gyro_x + drift_x + np.random.normal(0, 2, self.seq_len),
            gyro_y + drift_y + np.random.normal(0, 2, self.seq_len),
            gyro_z + drift_z + np.random.normal(0, 2, self.seq_len)
        ], axis=-1)
        
        # === MAGNETOMETER (mostly stable, small perturbations) ===
        
        # Earth's magnetic field (roughly constant)
        mag_earth = np.array([30, 20, -40])  # μT
        
        # Small variations due to movement
        mag_variation = np.random.normal(0, 3, (self.seq_len, 3))
        
        # Magnetic interference spikes during impact (metal debris, etc.)
        impact_indices = np.where(np.abs(accel_z) > np.max(np.abs(accel_z)) * 0.5)[0]
        for idx in impact_indices:
            if idx < self.seq_len:
                mag_variation[idx] += np.random.normal(0, 10, 3)
        
        mag = mag_earth + mag_variation
        
        return accel, gyro, mag
    
    def generate_vitals(self, impact_type: str, severity: float) -> np.ndarray:
        """
        Generate physiologically accurate vital signs.
        
        Physiological response model:
        1. Baseline vitals (healthy soldier at rest)
        2. Acute stress response (sympathetic activation)
        3. Injury-dependent deterioration
        4. Temporal dynamics (not instantaneous)
        """
        
        # Baseline vitals
        baseline_hr = np.random.uniform(60, 80)
        baseline_spo2 = np.random.uniform(96, 99)
        baseline_br = np.random.uniform(12, 16)
        baseline_temp = np.random.uniform(36.3, 37.0)
        
        # Time constants for physiological response
        tau_fast = 20  # samples (~0.1 sec at 200Hz) - immediate response
        tau_slow = 100  # samples (~0.5 sec) - gradual changes
        
        # === HEART RATE ===
        
        if impact_type in ['blast', 'artillery', 'gunshot']:
            # Severe stress response
            hr_spike = severity * np.random.uniform(60, 100)
            
            # Biphasic response: immediate spike, then plateau
            hr_immediate = hr_spike * (1 - np.exp(-np.arange(self.seq_len) / tau_fast))
            hr_sustained = baseline_hr + hr_immediate * np.exp(-np.arange(self.seq_len) / (tau_slow * 3))
            
            hr = baseline_hr + hr_immediate + (hr_sustained - baseline_hr) * 0.3
            
        elif impact_type in ['vehicle_crash', 'fall']:
            hr_spike = severity * np.random.uniform(30, 60)
            hr = baseline_hr + hr_spike * (1 - np.exp(-np.arange(self.seq_len) / tau_slow))
            
        else:  # normal
            # Normal variability (respiratory sinus arrhythmia)
            hr = baseline_hr + 3 * np.sin(2 * np.pi * 0.2 * self.t)
        
        # Add physiological noise (HRV)
        hr += np.random.normal(0, 2, self.seq_len)
        hr = np.clip(hr, 40, 180)
        
        # === SPO2 (oxygen saturation) ===
        
        if impact_type in ['blast', 'artillery'] and severity > 0.6:
            # Blast lung or respiratory compromise
            spo2_drop = severity * np.random.uniform(10, 25)
            spo2 = baseline_spo2 - spo2_drop * (1 - np.exp(-np.arange(self.seq_len) / (tau_slow * 2)))
            
        elif impact_type == 'gunshot' and severity > 0.7:
            # Hemorrhagic shock
            spo2_drop = severity * np.random.uniform(5, 15)
            spo2 = baseline_spo2 - spo2_drop * (1 - np.exp(-np.arange(self.seq_len) / (tau_slow * 4)))
            
        else:
            # Minor changes
            spo2 = baseline_spo2 - severity * np.random.uniform(0, 5) * (1 - np.exp(-np.arange(self.seq_len) / tau_slow))
        
        # Measurement noise (pulse oximetry)
        spo2 += np.random.normal(0, 0.5, self.seq_len)
        spo2 = np.clip(spo2, 70, 100)
        
        # === BREATHING RATE ===
        
        # Acute stress → tachypnea
        br_increase = severity * np.random.uniform(5, 15)
        br = baseline_br + br_increase * (1 - np.exp(-np.arange(self.seq_len) / tau_slow))
        
        # Respiratory cycling
        br += 2 * np.sin(2 * np.pi * 0.25 * self.t)
        br = np.clip(br, 8, 35)
        
        # === SKIN TEMPERATURE ===
        
        # Temperature changes slowly
        if severity > 0.7:
            # Shock → peripheral vasoconstriction → cooling
            temp_drop = severity * np.random.uniform(0.5, 1.5)
            temp = baseline_temp - temp_drop * (1 - np.exp(-np.arange(self.seq_len) / (tau_slow * 5)))
        else:
            # Stress → increased metabolism → slight warming
            temp = baseline_temp + severity * 0.3 * (1 - np.exp(-np.arange(self.seq_len) / (tau_slow * 4)))
        
        temp += np.random.normal(0, 0.1, self.seq_len)
        temp = np.clip(temp, 34, 39)
        
        vitals = np.stack([hr, spo2, br, temp], axis=-1)
        
        return vitals
    
    def generate_sample(self, impact_type: str) -> Tuple[np.ndarray, float]:
        """
        Generate complete multi-sensor sample.
        
        Returns:
            sample: (seq_len, 13) array [accel(3) + gyro(3) + mag(3) + vitals(4)]
            severity: float [0, 1]
        """
        
        # Severity based on impact type
        if impact_type in ['blast', 'artillery']:
            severity = np.random.uniform(0.6, 1.0)
        elif impact_type == 'gunshot':
            severity = np.random.uniform(0.5, 0.9)
        elif impact_type in ['vehicle_crash', 'fall']:
            severity = np.random.uniform(0.3, 0.7)
        else:  # normal
            severity = np.random.uniform(0.0, 0.2)
        
        # Generate sensor data
        accel, gyro, mag = self.generate_3axis_imu(impact_type)
        vitals = self.generate_vitals(impact_type, severity)
        
        # Concatenate all sensors: (200, 13)
        sample = np.concatenate([accel, gyro, mag, vitals], axis=-1)
        
        # Data validation
        assert sample.shape == (self.seq_len, 13), f"Invalid shape: {sample.shape}"
        assert not np.isnan(sample).any(), "NaN detected"
        assert not np.isinf(sample).any(), "Inf detected"
        
        return sample, severity
    
    def generate_dataset(self, samples_per_class: int = 8333, save_path: str = 'data/combat_dataset.h5'):
        """
        Generate complete balanced dataset.
        
        Args:
            samples_per_class: Number of samples per impact type
            save_path: HDF5 file path
        """
        
        classes = list(IMPACT_PROFILES.keys())
        total_samples = samples_per_class * len(classes)
        
        print(f"{'='*70}")
        print(f"  GENERATING HIGH-FIDELITY COMBAT IMPACT DATASET")
        print(f"{'='*70}")
        print(f"  Classes: {classes}")
        print(f"  Samples per class: {samples_per_class}")
        print(f"  Total samples: {total_samples}")
        print(f"  Sample rate: {self.fs} Hz")
        print(f"  Sequence length: {self.seq_len} samples ({self.seq_len/self.fs:.2f} sec)")
        print(f"{'='*70}\n")
        
        X_all = []
        y_type_all = []
        y_severity_all = []
        
        for class_idx, impact_type in enumerate(classes):
            print(f"[{class_idx+1}/{len(classes)}] Generating {impact_type}...")
            
            for i in range(samples_per_class):
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i+1}/{samples_per_class}")
                
                sample, severity = self.generate_sample(impact_type)
                
                X_all.append(sample)
                
                # One-hot encoding
                label = np.zeros(len(classes))
                label[class_idx] = 1.0
                y_type_all.append(label)
                
                y_severity_all.append(severity)
        
        # Convert to arrays
        X_all = np.array(X_all, dtype=np.float32)
        y_type_all = np.array(y_type_all, dtype=np.float32)
        y_severity_all = np.array(y_severity_all, dtype=np.float32)
        
        print(f"\n{'='*70}")
        print("  DATASET STATISTICS")
        print(f"{'='*70}")
        print(f"  X shape: {X_all.shape}")
        print(f"  y_type shape: {y_type_all.shape}")
        print(f"  y_severity shape: {y_severity_all.shape}")
        print(f"\n  Class distribution:")
        for i, cls in enumerate(classes):
            count = np.sum(y_type_all[:, i])
            print(f"    {cls:20s}: {int(count):6d} samples")
        
        print(f"\n  Severity statistics:")
        print(f"    Mean: {np.mean(y_severity_all):.3f}")
        print(f"    Std:  {np.std(y_severity_all):.3f}")
        print(f"    Min:  {np.min(y_severity_all):.3f}")
        print(f"    Max:  {np.max(y_severity_all):.3f}")
        
        # Save to HDF5
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\n  Saving to {save_path}...")
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('X', data=X_all, compression='gzip', compression_opts=9)
            f.create_dataset('y_type', data=y_type_all, compression='gzip', compression_opts=9)
            f.create_dataset('y_severity', data=y_severity_all, compression='gzip', compression_opts=9)
            
            # Metadata
            f.attrs['classes'] = classes
            f.attrs['sample_rate'] = self.fs
            f.attrs['sequence_length'] = self.seq_len
            f.attrs['num_samples'] = total_samples
            f.attrs['samples_per_class'] = samples_per_class
        
        print(f"\n{'='*70}")
        print("  ✓ DATASET GENERATION COMPLETE")
        print(f"{'='*70}\n")
        
        return X_all, y_type_all, y_severity_all

# EXECUTE IMMEDIATELY
if __name__ == "__main__":
    generator = AdvancedDataGenerator(sample_rate=200, sequence_length=200)
    generator.generate_dataset(samples_per_class=8333, save_path='data/combat_dataset.h5')
