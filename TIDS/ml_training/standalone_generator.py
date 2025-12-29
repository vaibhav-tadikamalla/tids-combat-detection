"""
Standalone Dataset Generator - NO external dependencies required
Generates JSON dataset that can be used immediately
"""

import json
import random
import math
import os

def generate_sensor_data(impact_type, severity):
    """Generate 200 timesteps of 13-feature sensor data"""
    data = []
    
    # Base parameters vary by impact type
    params = {
        'blast': {'accel': 150, 'gyro': 300, 'hr': 180, 'acoustic': 0.9},
        'gunshot': {'accel': 80, 'gyro': 150, 'hr': 160, 'acoustic': 0.95},
        'artillery': {'accel': 200, 'gyro': 350, 'hr': 190, 'acoustic': 0.85},
        'vehicle_crash': {'accel': 120, 'gyro': 200, 'hr': 170, 'acoustic': 0.3},
        'fall': {'accel': 60, 'gyro': 100, 'hr': 140, 'acoustic': 0.1},
        'normal': {'accel': 1, 'gyro': 5, 'hr': 80, 'acoustic': 0.05}
    }
    
    p = params[impact_type]
    
    for t in range(200):  # 200 timesteps (1 second at 200Hz)
        # Impact spike in first 20 timesteps, then decay
        spike = math.exp(-t / 20.0) if t < 50 else 0.1
        
        # Accelerometer (m/s²) - 3 axes
        accel_magnitude = p['accel'] * severity * spike + random.gauss(0, 2)
        accel_x = accel_magnitude * math.sin(t * 0.1) + random.gauss(0, 1)
        accel_y = accel_magnitude * math.cos(t * 0.1) + random.gauss(0, 1)
        accel_z = accel_magnitude * 0.5 + random.gauss(9.81, 0.5)
        
        # Gyroscope (deg/s) - 3 axes
        gyro_magnitude = p['gyro'] * severity * spike + random.gauss(0, 5)
        gyro_x = gyro_magnitude * math.sin(t * 0.15) + random.gauss(0, 2)
        gyro_y = gyro_magnitude * math.cos(t * 0.15) + random.gauss(0, 2)
        gyro_z = gyro_magnitude * 0.3 + random.gauss(0, 2)
        
        # Magnetometer (µT) - 3 axes
        mag_x = 25 + random.gauss(0, 2)
        mag_y = 30 + random.gauss(0, 2)
        mag_z = 45 + random.gauss(0, 2)
        
        # Heart rate (bpm) - spikes then elevated
        hr_base = p['hr'] if t < 100 else (p['hr'] + 80) / 2
        hr = hr_base + random.gauss(0, 5)
        
        # Temperature (°C) - slight increase
        temp = 36.5 + severity * 0.5 + random.gauss(0, 0.2)
        
        # Pressure (kPa) - spike for blasts
        pressure_spike = p['accel'] * 0.5 * spike if 'blast' in impact_type or 'artillery' in impact_type else 0
        pressure = 101.3 + pressure_spike + random.gauss(0, 0.5)
        
        # Acoustic (normalized) - loud for gunshots/blasts
        acoustic = p['acoustic'] * spike + random.gauss(0, 0.05)
        acoustic = max(0, min(1, acoustic))
        
        data.append([
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z,
            mag_x, mag_y, mag_z,
            hr, temp, pressure, acoustic
        ])
    
    return data

def generate_dataset(samples_per_class=1000):
    """Generate complete dataset"""
    classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
    dataset = []
    
    print("Generating dataset...")
    total = samples_per_class * len(classes)
    
    for class_idx, impact_type in enumerate(classes):
        print(f"  {impact_type}: ", end='', flush=True)
        for i in range(samples_per_class):
            if i % 200 == 0:
                print('.', end='', flush=True)
            
            severity = random.uniform(0.3, 1.0) if impact_type != 'normal' else random.uniform(0, 0.2)
            sensor_data = generate_sensor_data(impact_type, severity)
            
            dataset.append({
                'data': sensor_data,
                'impact_type': impact_type,
                'class_id': class_idx,
                'severity': severity
            })
        print(f" {samples_per_class} samples")
    
    # Shuffle
    random.shuffle(dataset)
    print(f"\nTotal: {len(dataset)} samples")
    return dataset

if __name__ == '__main__':
    # Generate dataset
    dataset = generate_dataset(samples_per_class=1000)
    
    # Save as JSON
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'combat_trauma_dataset.json')
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f)
    
    print(f"✓ Dataset saved ({os.path.getsize(output_file) / 1024 / 1024:.1f} MB)")
    print(f"✓ Ready for training!")
