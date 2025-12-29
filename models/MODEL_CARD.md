---
language: en
license: mit
tags:
- tensorflow
- combat-detection
- impact-classification
- military-ai
- sensor-fusion
- tflite
datasets:
- synthetic
metrics:
- accuracy
library_name: tensorflow
model-index:
- name: Guardian-Shield Combat Detector
  results:
  - task:
      type: impact-classification
      name: Combat Impact Classification
    metrics:
    - type: accuracy
      value: 0.9546
      name: Test Accuracy
    - type: f1
      value: 0.95
      name: F1 Score
---

# 🎯 Guardian-Shield Combat Impact Detector

**Developed by**: Vaibhav Tadikamalla  
**License**: MIT  
**Model Version**: 1.0.0  
**Release Date**: December 2025

## Model Description

Guardian-Shield is an AI-powered combat impact classification system that achieves **95.46% accuracy** in real-time detection of combat-related impacts from wearable sensor data. The model is optimized for edge deployment using TensorFlow Lite.

### Key Features

- ✅ **6 Impact Classes**: blast, gunshot, artillery, vehicle_crash, fall, normal
- ✅ **Severity Assessment**: 0-1 continuous injury severity score
- ✅ **High Accuracy**: 95.46% test accuracy
- ✅ **Real-time Processing**: <10ms inference on GPU T4
- ✅ **Edge Optimized**: TensorFlow Lite format (~7 MB)
- ✅ **Multi-sensor Fusion**: IMU + vital signs integration

---

## 🏗️ Model Architecture

**Type**: Deep Residual Network with Multi-Head Attention

```
Input: (200 timesteps, 13 channels)
├─ Initial Conv1D (64 filters, kernel=7)
├─ Residual Stage 1: 2 blocks (64 filters, dilations: 1, 2)
├─ MaxPooling1D (pool_size=2)
├─ Residual Stage 2: 3 blocks (128 filters, dilations: 1, 2, 4)
├─ MaxPooling1D (pool_size=2)
├─ Residual Stage 3: 3 blocks (256 filters, dilations: 1, 2, 4)
├─ Residual Stage 4: 2 blocks (512 filters, dilations: 1, 2)
├─ Multi-Head Attention (16 heads, key_dim=32)
├─ Squeeze-and-Excitation Block
├─ Dual Pooling (Global Average + Max)
├─ Dense Layers: 768 → 512 → 256
└─ Output Heads:
    ├─ Impact Type: Softmax (6 classes)
    └─ Severity: Sigmoid (0-1)
```

**Parameters**: ~8.4 Million  
**Framework**: TensorFlow 2.x / Keras  
**Optimization**: TensorFlow Lite with DEFAULT optimizations

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.46% |
| **Validation Accuracy** | ~95.8% |
| **Training Samples** | 60,000 (synthetic) |
| **Inference Time** | <10ms (GPU T4) |
| **Model Size** | 6.64 MB (TFLite) |
| **Input Shape** | (1, 200, 13) |
| **Outputs** | 2 (impact_type, severity) |

### Per-Class Performance

Training used balanced dataset (10,000 samples per class):
- Blast: High-energy explosions
- Gunshot: Ballistic impacts  
- Artillery: Heavy ordnance
- Vehicle Crash: Vehicular collisions
- Fall: Ground impacts
- Normal: Baseline activity

---

## 💾 Model Files

This repository contains:

1. **`impact_classifier.tflite`** (6.64 MB) - TensorFlow Lite model
2. **`norm.npz`** (604 bytes) - Normalization parameters (mean, std)

---

## 🚀 Usage

### Requirements

```bash
pip install tensorflow numpy
```

### Quick Start

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='impact_classifier.tflite')
interpreter.allocate_tensors()

# Load normalization parameters
norm = np.load('norm.npz')
mean, std = norm['mean'], norm['std']

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input (200 timesteps, 13 channels)
# Channels: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, 
#            mag_x, mag_y, mag_z, heart_rate, spo2, breathing, temp]
sensor_data = np.random.randn(1, 200, 13).astype(np.float32)

# Normalize
normalized_data = (sensor_data - mean) / std

# Run inference
interpreter.set_tensor(input_details[0]['index'], normalized_data)
interpreter.invoke()

# Get predictions
impact_type = interpreter.get_tensor(output_details[0]['index'])
severity = interpreter.get_tensor(output_details[1]['index'])

# Decode results
classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
predicted_class = classes[impact_type.argmax()]
severity_score = severity[0][0]

print(f"🎯 Impact: {predicted_class}")
print(f"⚠️  Severity: {severity_score:.2f}")
```

### Expected Input Format

**Shape**: `(batch_size, 200, 13)`  
**Data Type**: `float32`  
**Sampling Rate**: 100 Hz (2 seconds of data)

**Channel Order**:
1. Accelerometer X (m/s²)
2. Accelerometer Y (m/s²)
3. Accelerometer Z (m/s²)
4. Gyroscope X (°/s)
5. Gyroscope Y (°/s)
6. Gyroscope Z (°/s)
7. Magnetometer X (μT)
8. Magnetometer Y (μT)
9. Magnetometer Z (μT)
10. Heart Rate (bpm)
11. SpO2 (%)
12. Breathing Rate (breaths/min)
13. Body Temperature (°C)

**Important**: Always normalize using provided `norm.npz` parameters!

---

## 🎓 Training Details

### Dataset

- **Type**: Synthetic physics-based simulation
- **Size**: 60,000 samples
- **Distribution**: 10,000 per class (balanced)
- **Split**: 70% train, 15% validation, 15% test
- **Augmentation**: Time shifts, scaling, additive noise

### Data Generation

Multi-harmonic signal modeling with:
- Exponential decay (impact physics)
- Multiple frequency components
- Realistic physiological responses
- Cross-channel correlations

### Training Configuration

```python
Optimizer: Adam (lr=0.001, clipnorm=1.0)
Loss Functions:
  - Impact Type: Categorical Crossentropy (label_smoothing=0.1)
  - Severity: MSE
Loss Weights: [1.0, 0.2]
Batch Size: 32
Epochs: 200 (early stopping enabled)
Callbacks:
  - EarlyStopping (patience=30, monitor='val_impact_type_accuracy')
  - ReduceLROnPlateau (factor=0.5, patience=12)
  - ModelCheckpoint (save_best_only=True)
```

**Training Time**: ~2-3 hours on Google Colab GPU T4

---

## 🎯 Intended Use

### Primary Applications

✅ **Military Wearable Systems**: Real-time combat impact detection on soldier-worn devices  
✅ **Medical Triage AI**: Automated injury severity assessment  
✅ **Combat Analytics**: Post-mission impact analysis  
✅ **Research**: Combat biomechanics and injury patterns

### Out-of-Scope Use

❌ Civilian fall detection (different sensor characteristics)  
❌ Medical diagnosis (requires real combat data validation)  
❌ Legal/forensic evidence (synthetic training data)

---

## ⚠️ Limitations

1. **Synthetic Data Only**: Trained on physics-based simulations, not real combat data
2. **Sensor Calibration**: Requires properly calibrated IMU sensors
3. **Environmental Factors**: Performance may vary with extreme temperatures, vibrations
4. **Generalization**: Real-world combat scenarios may differ from training distribution
5. **Validation Needed**: Requires extensive field testing before operational deployment

---

## 🔒 Ethical Considerations

This model is designed for **defensive military applications** to:
- Improve soldier safety
- Enable faster medical response
- Reduce combat casualties

**Not intended for**:
- Offensive weapon systems
- Autonomous targeting
- Non-military surveillance

Users must comply with international humanitarian law and Geneva Conventions.

---

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{guardian_shield_2025,
  author = {Tadikamalla, Vaibhav},
  title = {Guardian-Shield: Deep Learning for Combat Impact Detection},
  year = {2025},
  url = {https://huggingface.co/vaibhav-tadikamalla/guardian-shield-combat-detector},
  version = {1.0.0},
  note = {TensorFlow Lite model achieving 95.46\% accuracy on 6-class impact classification}
}
```

---

## 📞 Contact & Support

- **GitHub**: [tids-combat-detection](https://github.com/vaibhav-tadikamalla/tids-combat-detection)
- **Issues**: Report bugs or request features via GitHub Issues
- **Email**: [Contact via GitHub profile]

---

## 📜 License

This model is released under the **MIT License**. See [LICENSE](https://github.com/vaibhav-tadikamalla/tids-combat-detection/blob/main/LICENSE) for details.

---

## 🙏 Acknowledgments

- **TensorFlow Team**: Deep learning framework
- **Google Colab**: Free GPU resources for training
- **Military Research Community**: Domain expertise and feedback

---

## 🔄 Version History

**v1.0.0** (December 2025)
- Initial release
- 95.46% test accuracy
- TensorFlow Lite optimization
- 6-class impact classification + severity prediction

---

**⭐ Star this model if you find it useful!**

**🔗 Related Resources**:
- [Full TIDS System](https://github.com/vaibhav-tadikamalla/tids-combat-detection)
- [Training Notebook](https://github.com/vaibhav-tadikamalla/tids-combat-detection/blob/main/GUARDIAN_SHIELD_FINAL_TRAINABLE.ipynb)
- [Documentation](https://github.com/vaibhav-tadikamalla/tids-combat-detection/blob/main/README.md)
