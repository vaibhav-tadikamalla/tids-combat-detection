"""
Integration Test Suite
End-to-end testing of GUARDIAN-SHIELD system
"""

import pytest
import asyncio
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'edge_device'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

from ml_training.data_generation import MilitaryScenarioGenerator

@pytest.mark.asyncio
async def test_scenario_generation():
    """Test military scenario generation"""
    generator = MilitaryScenarioGenerator(sample_rate=200, sequence_length=200)
    
    # Test all impact types
    impact_types = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
    
    for impact_type in impact_types:
        sample, severity = generator.generate_sample(impact_type)
        
        assert sample.shape == (200, 13), f"Invalid shape for {impact_type}"
        assert not np.isnan(sample).any(), f"NaN in {impact_type}"
        assert not np.isinf(sample).any(), f"Inf in {impact_type}"
        assert 0 <= severity <= 1, f"Invalid severity for {impact_type}"
    
    print("✓ Scenario generation test PASSED")

@pytest.mark.asyncio
async def test_sensor_data_simulation():
    """Test sensor data simulation"""
    try:
        from edge_device.sensors.imu_handler import IMUHandler
        from edge_device.sensors.gps_handler import GPSHandler
        from edge_device.sensors.vitals_handler import VitalsHandler
        
        # Test IMU
        imu = IMUHandler(simulate=True)
        imu_data = imu.read()
        assert 'acceleration' in imu_data
        assert len(imu_data['acceleration']) == 3
        
        # Test GPS
        gps = GPSHandler(simulate=True)
        location = gps.get_location()
        assert location is not None
        assert len(location) == 2
        
        # Test Vitals
        vitals = VitalsHandler(simulate=True)
        vital_data = vitals.read()
        assert 'heart_rate' in vital_data
        assert 40 <= vital_data['heart_rate'] <= 200
        assert 85 <= vital_data['spo2'] <= 100
        
        print("✓ Sensor simulation test PASSED")
        
    except ImportError as e:
        pytest.skip(f"Sensor modules not available: {e}")

def test_data_quality():
    """Test generated data quality"""
    generator = MilitaryScenarioGenerator()
    
    # Generate multiple samples and check consistency
    samples = []
    for _ in range(100):
        sample, _ = generator.generate_sample('blast')
        samples.append(sample)
    
    samples = np.array(samples)
    
    # Check statistical properties
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    # Data should have variation
    assert np.all(std > 0), "Data lacks variation"
    
    # No extreme outliers
    assert np.all(np.abs(mean) < 100), "Extreme values detected"
    
    print("✓ Data quality test PASSED")

@pytest.mark.asyncio
async def test_blast_physics():
    """Test blast simulation physics"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
        from simulation.blast_simulator import BlastSimulator
        
        simulator = BlastSimulator()
        
        # Test grenade scenario
        result = simulator.simulate_scenario('grenade')
        
        assert result['peak_pressure'] > 0
        assert result['impulse'] > 0
        assert result['distance'] > 0
        assert 'injury_assessment' in result
        
        # Physics validation
        assert result['t_arrival'] > 0  # Shock wave takes time
        assert result['scaled_distance'] > 0
        
        print("✓ Blast physics test PASSED")
        
    except ImportError:
        pytest.skip("Blast simulator not available")

def test_model_file_exists():
    """Verify model file location"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_training', 'models')
    
    # Check if models directory exists
    if not os.path.exists(model_path):
        pytest.skip("Model not yet trained - run training pipeline first")
    
    print("✓ Model path exists")

@pytest.mark.asyncio
async def test_medical_ai_assessment():
    """Test medical AI triage system"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from backend.services.medical_ai import PredictiveMedicalAI
        
        ai = PredictiveMedicalAI()
        
        # Test casualty assessment
        alert_data = {
            'impact_type': 'blast',
            'severity': 0.8,
            'location': (19.0760, 72.8777),
            'vitals': {
                'heart_rate': 130,
                'spo2': 92,
                'breathing_rate': 28,
                'skin_temp': 37.5
            }
        }
        
        assessment = await ai.assess_casualty(alert_data)
        
        assert 'triage_category' in assessment
        assert 'survival_probability' in assessment
        assert 'recommended_treatments' in assessment
        assert assessment['triage_category'] in ['IMMEDIATE', 'DELAYED', 'MINIMAL', 'EXPECTANT']
        
        print("✓ Medical AI test PASSED")
        
    except ImportError:
        pytest.skip("Backend modules not available")

if __name__ == "__main__":
    print("="*60)
    print("GUARDIAN-SHIELD INTEGRATION TEST SUITE")
    print("="*60)
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
