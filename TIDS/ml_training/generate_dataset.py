"""
ML Dataset Generation Script
Generates high-quality training dataset for production model
"""

import numpy as np
import h5py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_generation import MilitaryScenarioGenerator
from tqdm import tqdm

def generate_production_dataset():
    """Generate production-grade dataset with validation"""
    
    print("="*60)
    print("GUARDIAN-SHIELD DATASET GENERATION")
    print("="*60)
    
    # Initialize generator
    generator = MilitaryScenarioGenerator(sample_rate=200, sequence_length=200)
    
    # Configuration
    samples_per_class = 5000
    classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
    
    print(f"\nGenerating {samples_per_class} samples per class...")
    print(f"Total samples: {len(classes) * samples_per_class}")
    
    X_all = []
    y_type_all = []
    y_severity_all = []
    
    for class_idx, impact_type in enumerate(classes):
        print(f"\n[{class_idx+1}/{len(classes)}] Generating {impact_type} samples...")
        
        for i in tqdm(range(samples_per_class)):
            # Generate sample
            sample, severity = generator.generate_sample(impact_type)
            
            # Validate sample
            assert sample.shape == (200, 13), f"Invalid sample shape: {sample.shape}"
            assert not np.isnan(sample).any(), "Sample contains NaN"
            assert not np.isinf(sample).any(), "Sample contains Inf"
            
            X_all.append(sample)
            
            # One-hot encode
            type_label = np.zeros(len(classes))
            type_label[class_idx] = 1
            y_type_all.append(type_label)
            
            y_severity_all.append(severity)
    
    X_all = np.array(X_all)
    y_type_all = np.array(y_type_all)
    y_severity_all = np.array(y_severity_all)
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"X shape: {X_all.shape}")
    print(f"y_type shape: {y_type_all.shape}")
    print(f"y_severity shape: {y_severity_all.shape}")
    print(f"\nClass distribution:")
    for i, class_name in enumerate(classes):
        count = np.sum(y_type_all[:, i])
        print(f"  {class_name}: {int(count)} samples")
    
    print(f"\nSeverity statistics:")
    print(f"  Mean: {np.mean(y_severity_all):.3f}")
    print(f"  Std: {np.std(y_severity_all):.3f}")
    print(f"  Min: {np.min(y_severity_all):.3f}")
    print(f"  Max: {np.max(y_severity_all):.3f}")
    
    # Data quality checks
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    checks = {
        "No NaN values": not np.isnan(X_all).any(),
        "No Inf values": not np.isinf(X_all).any(),
        "Correct shape": X_all.shape == (30000, 200, 13),
        "Balanced classes": np.std([np.sum(y_type_all[:, i]) for i in range(len(classes))]) == 0,
        "Severity in range": np.all((y_severity_all >= 0) & (y_severity_all <= 1))
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        raise ValueError("Dataset validation failed!")
    
    # Save dataset
    os.makedirs('data', exist_ok=True)
    filepath = 'data/combat_trauma_dataset.h5'
    
    print(f"\nSaving dataset to {filepath}...")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('X', data=X_all, compression='gzip')
        f.create_dataset('y_type', data=y_type_all, compression='gzip')
        f.create_dataset('y_severity', data=y_severity_all, compression='gzip')
        f.attrs['classes'] = classes
        f.attrs['sample_rate'] = 200
        f.attrs['sequence_length'] = 200
    
    print("✓ Dataset saved successfully!")
    print("\n" + "="*60)
    
    return filepath

if __name__ == "__main__":
    generate_production_dataset()
