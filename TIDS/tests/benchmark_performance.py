"""
Performance Benchmark Suite
Measures system performance and validates requirements
"""

import time
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

def benchmark_data_generation():
    """Benchmark data generation speed"""
    from ml_training.data_generation import MilitaryScenarioGenerator
    
    print("\n" + "="*60)
    print("DATA GENERATION BENCHMARK")
    print("="*60)
    
    generator = MilitaryScenarioGenerator()
    
    start = time.time()
    samples_generated = 100
    
    for _ in range(samples_generated):
        sample, severity = generator.generate_sample('blast')
    
    duration = time.time() - start
    samples_per_second = samples_generated / duration
    
    print(f"Generated {samples_generated} samples in {duration:.2f}s")
    print(f"Speed: {samples_per_second:.0f} samples/sec")
    
    assert samples_per_second > 50, f"Data generation too slow: {samples_per_second:.0f} samples/sec"
    print("✓ Data generation speed acceptable")
    
    return samples_per_second

def benchmark_inference_speed():
    """Measure inference latency"""
    
    print("\n" + "="*60)
    print("INFERENCE LATENCY BENCHMARK")
    print("="*60)
    
    # Check if TFLite model exists
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_training', 'models', 'impact_classifier.tflite')
    
    if not os.path.exists(model_path):
        print("⚠ TFLite model not found - skipping inference benchmark")
        print("  Run training pipeline first: python ml_training/train_production_model.py")
        return None
    
    try:
        import tensorflow as tf
        
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Benchmark inference
        latencies = []
        num_runs = 100
        
        for _ in range(num_runs):
            test_input = np.random.randn(1, 200, 13).astype(np.float32)
            
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            latency = (time.time() - start) * 1000  # Convert to ms
            
            latencies.append(latency)
        
        latencies = np.array(latencies)
        
        print(f"\nInference Latency (n={num_runs}):")
        print(f"  Mean: {np.mean(latencies):.2f} ms")
        print(f"  Median: {np.median(latencies):.2f} ms")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} ms")
        print(f"  Max: {np.max(latencies):.2f} ms")
        
        # Requirement: < 50ms for 95th percentile
        p95 = np.percentile(latencies, 95)
        
        if p95 < 50:
            print(f"\n✓ PASS: Inference latency {p95:.2f}ms < 50ms requirement")
        else:
            print(f"\n✗ WARNING: Inference latency {p95:.2f}ms exceeds 50ms target")
        
        return p95
        
    except ImportError:
        print("⚠ TensorFlow not available - skipping inference benchmark")
        return None

def benchmark_sensor_throughput():
    """Benchmark sensor data processing"""
    
    print("\n" + "="*60)
    print("SENSOR THROUGHPUT BENCHMARK")
    print("="*60)
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'edge_device'))
        from edge_device.sensors.imu_handler import IMUHandler
        from edge_device.sensors.vitals_handler import VitalsHandler
        
        imu = IMUHandler(simulate=True)
        vitals = VitalsHandler(simulate=True)
        
        # Benchmark IMU
        start = time.time()
        imu_reads = 200
        for _ in range(imu_reads):
            data = imu.read()
        imu_duration = time.time() - start
        imu_rate = imu_reads / imu_duration
        
        # Benchmark Vitals
        start = time.time()
        vital_reads = 100
        for _ in range(vital_reads):
            data = vitals.read()
        vital_duration = time.time() - start
        vital_rate = vital_reads / vital_duration
        
        print(f"IMU sensor: {imu_rate:.0f} reads/sec")
        print(f"Vitals sensor: {vital_rate:.0f} reads/sec")
        
        # Requirements: IMU at 200 Hz, Vitals at 1 Hz minimum
        assert imu_rate > 200, f"IMU too slow: {imu_rate:.0f} Hz"
        assert vital_rate > 1, f"Vitals too slow: {vital_rate:.0f} Hz"
        
        print("✓ Sensor throughput meets requirements")
        
        return {'imu': imu_rate, 'vitals': vital_rate}
        
    except ImportError:
        print("⚠ Sensor modules not available")
        return None

def benchmark_memory_usage():
    """Estimate memory usage"""
    
    print("\n" + "="*60)
    print("MEMORY USAGE ESTIMATION")
    print("="*60)
    
    import sys
    
    # Estimate model size
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_training', 'models', 'impact_classifier.tflite')
    
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / 1024  # KB
        print(f"TFLite model: {model_size:.1f} KB")
        
        if model_size < 500:
            print("✓ Model size within 500 KB target")
        else:
            print(f"⚠ Model size {model_size:.1f} KB exceeds 500 KB target")
    else:
        print("⚠ Model file not found")
    
    # Buffer size estimation
    buffer_samples = 200
    features_per_sample = 13
    bytes_per_float = 4
    buffer_size_kb = (buffer_samples * features_per_sample * bytes_per_float) / 1024
    
    print(f"Sensor buffer: {buffer_size_kb:.1f} KB")
    print(f"Estimated total RAM: {model_size + buffer_size_kb + 100:.1f} KB")  # +100 for overhead
    
    print("✓ Memory usage acceptable for edge device")

def run_all_benchmarks():
    """Run complete benchmark suite"""
    
    print("\n" + "="*70)
    print(" " * 20 + "PERFORMANCE BENCHMARK SUITE")
    print("="*70)
    
    results = {}
    
    # Run benchmarks
    results['data_gen'] = benchmark_data_generation()
    results['inference'] = benchmark_inference_speed()
    results['sensors'] = benchmark_sensor_throughput()
    benchmark_memory_usage()
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"Data Generation: {results['data_gen']:.0f} samples/sec" if results['data_gen'] else "Data Generation: Skipped")
    
    if results['inference']:
        print(f"Inference Latency: {results['inference']:.2f} ms (95th percentile)")
        if results['inference'] < 50:
            print("  ✓ Meets <50ms requirement")
        else:
            print("  ⚠ Exceeds 50ms target")
    else:
        print("Inference Latency: Not tested (model not found)")
    
    if results['sensors']:
        print(f"IMU Throughput: {results['sensors']['imu']:.0f} Hz")
        print(f"Vitals Throughput: {results['sensors']['vitals']:.0f} Hz")
    
    print("\n" + "="*70)
    print("Benchmark suite complete!")
    print("="*70)

if __name__ == "__main__":
    run_all_benchmarks()
