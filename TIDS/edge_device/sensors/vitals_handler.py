"""
Vitals Monitoring Handler
Manages physiological sensor data (heart rate, SpO2, temperature, etc.)
"""

import time
import numpy as np
from typing import Dict
from collections import deque
import logging


class VitalsHandler:
    """
    Handles vital signs monitoring.
    
    Sensors:
    - Heart rate (PPG - photoplethysmography)
    - SpO2 (blood oxygen saturation)
    - Skin temperature
    - Breathing rate (derived from heart rate variability)
    """
    
    def __init__(self, simulate=True):
        """
        Initialize vitals handler.
        
        Args:
            simulate: Use simulated data (hardware not yet implemented)
        """
        self.simulate = simulate
        
        # Baseline vital signs (healthy resting soldier)
        self.baseline_hr = 70  # bpm
        self.baseline_spo2 = 98  # %
        self.baseline_temp = 36.8  # °C
        self.baseline_br = 16  # breaths per minute
        
        # Current state
        self.stress_level = 0.0  # 0-1
        self.activity_level = 0.0  # 0-1
        
        # History buffers for signal processing
        self.hr_history = deque(maxlen=60)  # 1 minute at 1Hz
        self.spo2_history = deque(maxlen=60)
        
        # Simulation time
        self.sim_time = 0
        self.last_read = time.time()
        
        logging.info("Vitals handler initialized (simulation mode)")
    
    def read(self) -> Dict[str, float]:
        """
        Read current vital signs.
        
        Returns:
            Dictionary with vital sign measurements
        """
        if self.simulate:
            return self._simulate_vitals()
        
        # Hardware implementation would go here
        return self._simulate_vitals()
    
    def _simulate_vitals(self) -> Dict[str, float]:
        """
        Simulate realistic vital signs.
        
        Models:
        - Normal variation
        - Stress response (increased HR, decreased HRV)
        - Physical activity (increased HR, BR)
        - Shock response (decreased BP, increased HR)
        """
        current_time = time.time()
        dt = current_time - self.last_read
        self.last_read = current_time
        self.sim_time += dt
        
        # Base vital signs
        heart_rate = self.baseline_hr
        spo2 = self.baseline_spo2
        temperature = self.baseline_temp
        breathing_rate = self.baseline_br
        
        # Activity effect (running, combat stress)
        activity_factor = 1.0 + self.activity_level * 0.8
        heart_rate *= activity_factor
        breathing_rate *= activity_factor
        temperature += self.activity_level * 1.5
        
        # Stress effect
        stress_factor = 1.0 + self.stress_level * 0.3
        heart_rate *= stress_factor
        
        # Circadian rhythm (slight variations)
        circadian = 0.05 * np.sin(2 * np.pi * self.sim_time / 86400)
        heart_rate *= (1 + circadian)
        
        # Respiratory sinus arrhythmia (HR varies with breathing)
        breathing_phase = 2 * np.pi * self.sim_time * breathing_rate / 60
        hr_breathing_var = 3 * np.sin(breathing_phase)
        heart_rate += hr_breathing_var
        
        # Random physiological noise
        heart_rate += np.random.normal(0, 2)
        spo2 += np.random.normal(0, 0.5)
        temperature += np.random.normal(0, 0.1)
        breathing_rate += np.random.normal(0, 1)
        
        # Clamp to realistic ranges
        heart_rate = np.clip(heart_rate, 40, 200)
        spo2 = np.clip(spo2, 85, 100)
        temperature = np.clip(temperature, 35.0, 41.0)
        breathing_rate = np.clip(breathing_rate, 8, 40)
        
        # Update history
        self.hr_history.append(heart_rate)
        self.spo2_history.append(spo2)
        
        # Calculate derived metrics
        hrv = self._calculate_hrv()
        blood_pressure = self._estimate_blood_pressure(heart_rate)
        
        return {
            'heart_rate': round(heart_rate, 1),
            'spo2': round(spo2, 1),
            'skin_temp': round(temperature, 2),
            'breathing_rate': round(breathing_rate, 1),
            'hrv': round(hrv, 2),
            'blood_pressure_systolic': round(blood_pressure[0], 1),
            'blood_pressure_diastolic': round(blood_pressure[1], 1),
            'timestamp': current_time
        }
    
    def _calculate_hrv(self) -> float:
        """
        Calculate Heart Rate Variability (RMSSD metric).
        
        HRV is important indicator of stress and autonomic nervous system state.
        Higher HRV = better recovery, lower stress
        Lower HRV = fatigue, stress, potential health issues
        
        Returns:
            HRV in milliseconds
        """
        if len(self.hr_history) < 10:
            return 50.0  # Default value
        
        # Convert HR to RR intervals (time between beats)
        hr_array = np.array(list(self.hr_history)[-30:])  # Last 30 samples
        rr_intervals = 60000 / hr_array  # milliseconds
        
        # Calculate successive differences
        diff_rr = np.diff(rr_intervals)
        
        # RMSSD (Root Mean Square of Successive Differences)
        rmssd = np.sqrt(np.mean(diff_rr**2))
        
        # Adjust for stress level (stress reduces HRV)
        rmssd *= (1 - 0.5 * self.stress_level)
        
        return max(10.0, min(rmssd, 200.0))
    
    def _estimate_blood_pressure(self, heart_rate: float) -> tuple:
        """
        Estimate blood pressure from heart rate.
        
        Note: This is a rough estimation. Real BP monitoring requires
        dedicated sensor (e.g., cuff or continuous monitoring device).
        
        Args:
            heart_rate: Current heart rate in bpm
        
        Returns:
            Tuple of (systolic, diastolic) pressure in mmHg
        """
        # Baseline BP for healthy adult
        systolic_base = 120
        diastolic_base = 80
        
        # HR-BP relationship (increased HR often increases BP)
        hr_effect = (heart_rate - 70) * 0.3
        
        systolic = systolic_base + hr_effect + np.random.normal(0, 5)
        diastolic = diastolic_base + hr_effect * 0.5 + np.random.normal(0, 3)
        
        # Activity increases systolic more than diastolic
        systolic += self.activity_level * 30
        diastolic += self.activity_level * 10
        
        # Stress increases both
        systolic += self.stress_level * 20
        diastolic += self.stress_level * 10
        
        # Clamp to realistic ranges
        systolic = np.clip(systolic, 80, 200)
        diastolic = np.clip(diastolic, 50, 130)
        
        return (systolic, diastolic)
    
    def set_activity_level(self, level: float):
        """
        Set activity level (0=rest, 1=intense activity).
        
        Args:
            level: Activity level 0.0-1.0
        """
        self.activity_level = np.clip(level, 0, 1)
    
    def set_stress_level(self, level: float):
        """
        Set stress level (0=calm, 1=extreme stress).
        
        Args:
            level: Stress level 0.0-1.0
        """
        self.stress_level = np.clip(level, 0, 1)
    
    def assess_vitals(self, vitals: Dict) -> Dict[str, str]:
        """
        Assess vital signs and provide clinical interpretation.
        
        Args:
            vitals: Dictionary of vital signs
        
        Returns:
            Dictionary with assessment for each vital
        """
        assessment = {}
        
        # Heart rate assessment
        hr = vitals['heart_rate']
        if hr < 50:
            assessment['heart_rate'] = 'BRADYCARDIA - Abnormally low'
        elif hr < 60:
            assessment['heart_rate'] = 'LOW - Below normal range'
        elif hr <= 100:
            assessment['heart_rate'] = 'NORMAL'
        elif hr <= 120:
            assessment['heart_rate'] = 'ELEVATED - Expected during activity'
        elif hr <= 150:
            assessment['heart_rate'] = 'HIGH - Monitor for stress/shock'
        else:
            assessment['heart_rate'] = 'CRITICAL - Tachycardia, immediate attention'
        
        # SpO2 assessment
        spo2 = vitals['spo2']
        if spo2 >= 95:
            assessment['spo2'] = 'NORMAL'
        elif spo2 >= 90:
            assessment['spo2'] = 'LOW - Monitor closely'
        elif spo2 >= 85:
            assessment['spo2'] = 'CRITICAL - Hypoxemia, oxygen needed'
        else:
            assessment['spo2'] = 'SEVERE - Life-threatening hypoxemia'
        
        # Temperature assessment
        temp = vitals['skin_temp']
        if temp < 35.0:
            assessment['temperature'] = 'HYPOTHERMIA - Dangerous'
        elif temp < 36.0:
            assessment['temperature'] = 'LOW - Monitor for hypothermia'
        elif temp <= 37.5:
            assessment['temperature'] = 'NORMAL'
        elif temp <= 38.5:
            assessment['temperature'] = 'ELEVATED - Possible fever/exertion'
        elif temp <= 40.0:
            assessment['temperature'] = 'FEVER - High temperature'
        else:
            assessment['temperature'] = 'CRITICAL - Hyperthermia, cooling needed'
        
        # Breathing rate assessment
        br = vitals['breathing_rate']
        if br < 10:
            assessment['breathing_rate'] = 'LOW - Bradypnea'
        elif br <= 20:
            assessment['breathing_rate'] = 'NORMAL'
        elif br <= 25:
            assessment['breathing_rate'] = 'ELEVATED'
        else:
            assessment['breathing_rate'] = 'HIGH - Tachypnea, check for distress'
        
        return assessment
    
    def detect_shock(self, vitals: Dict) -> bool:
        """
        Detect signs of shock (hemorrhagic or other).
        
        Shock indicators:
        - Tachycardia (high HR)
        - Low BP
        - Low SpO2
        - Rapid breathing
        
        Args:
            vitals: Current vital signs
        
        Returns:
            True if shock suspected
        """
        shock_score = 0
        
        if vitals['heart_rate'] > 120:
            shock_score += 2
        elif vitals['heart_rate'] > 100:
            shock_score += 1
        
        if vitals.get('blood_pressure_systolic', 120) < 90:
            shock_score += 3
        
        if vitals['spo2'] < 90:
            shock_score += 2
        
        if vitals['breathing_rate'] > 25:
            shock_score += 1
        
        return shock_score >= 4
    
    def close(self):
        """Release resources"""
        logging.info("Vitals handler closed")


if __name__ == "__main__":
    # Test vitals handler
    logging.basicConfig(level=logging.INFO)
    
    vitals = VitalsHandler(simulate=True)
    
    print("Testing Vitals Handler (simulation mode)")
    print("="*60)
    
    # Test normal state
    print("\n1. RESTING STATE:")
    vitals.set_activity_level(0.0)
    vitals.set_stress_level(0.0)
    
    for i in range(5):
        data = vitals.read()
        print(f"\n  Sample {i+1}:")
        print(f"    HR: {data['heart_rate']} bpm")
        print(f"    SpO2: {data['spo2']}%")
        print(f"    Temp: {data['skin_temp']}°C")
        print(f"    BR: {data['breathing_rate']} /min")
        print(f"    HRV: {data['hrv']} ms")
        print(f"    BP: {data['blood_pressure_systolic']}/{data['blood_pressure_diastolic']} mmHg")
        
        assessment = vitals.assess_vitals(data)
        print(f"    Assessment: HR={assessment['heart_rate']}, SpO2={assessment['spo2']}")
        
        if vitals.detect_shock(data):
            print("    ⚠️  SHOCK DETECTED!")
        
        time.sleep(0.5)
    
    # Test high activity
    print("\n\n2. HIGH ACTIVITY (running):")
    vitals.set_activity_level(0.8)
    vitals.set_stress_level(0.3)
    
    for i in range(3):
        data = vitals.read()
        print(f"\n  Sample {i+1}:")
        print(f"    HR: {data['heart_rate']} bpm")
        print(f"    SpO2: {data['spo2']}%")
        print(f"    BR: {data['breathing_rate']} /min")
        time.sleep(0.5)
    
    vitals.close()
