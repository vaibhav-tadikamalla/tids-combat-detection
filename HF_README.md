---
language: en
license: mit
tags:
- tensorflow
- combat-detection
- impact-classification
- military-ai
- sensor-fusion
datasets:
- synthetic
metrics:
- accuracy
model-index:
- name: Guardian-Shield Combat Detector
  results:
  - task:
      type: impact-classification
    metrics:
    - type: accuracy
      value: 0.9546
      name: Test Accuracy
---

# Guardian-Shield Combat Impact Detector

## Model Description

AI-powered combat impact classification system achieving **95.46% accuracy** on 6 impact classes:
- 💥 Blast
- 🔫 Gunshot  
- 🎯 Artillery
- 🚗 Vehicle Crash
- 🤕 Fall
- ✅ Normal

## Model Architecture

**ResNet-inspired Deep Learning Model**
- Dilated convolutions (rates: 1, 2, 4)
- Multi-head attention (16 heads)
- Squeeze-and-Excitation blocks
- Dual pooling (Average + Max)
- Parameters: ~8.4M
- Format: TensorFlow Lite (optimized for edge deployment)

## Intended Use

**Primary Use**: Real-time combat impact detection on wearable military devices

**Input**: 13-channel sensor data (200 timesteps)
- 3-axis accelerometer
- 3-axis gyroscope
- 3-axis magnetometer
- Heart rate, SpO2, breathing rate, temperature

**Output**: 
- Impact classification (6 classes)
- Severity score (0-1)

## Training Data

- **Size**: 60,000 synthetic samples
- **Method**: Physics-based signal modeling with physiological responses
- **Split**: 70/15/15 (train/val/test)
- **Augmentation**: Time shifting, scaling, noise injection

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.46% |
| **Validation Accuracy** | ~95.8% |
| **Inference Time** | <10ms (GPU T4) |
| **Model Size** | 6.64 MB (TFLite) |

## Usage Example

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter('impact_classifier.tflite')
interpreter.allocate_tensors()

# Load normalization parameters
norm = np.load('norm.npz')
mean, std = norm['mean'], norm['std']

# Prepare sensor data (shape: [1, 200, 13])
sensor_data = your_sensor_reading  # Your 13-channel sensor data
normalized = (sensor_data - mean) / std

# Run inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], normalized)
interpreter.invoke()

# Get predictions
impact_type = interpreter.get_tensor(output_details[0]['index'])
severity = interpreter.get_tensor(output_details[1]['index'])

# Decode results
classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
predicted_class = classes[impact_type.argmax()]
severity_score = severity[0][0]

print(f"Impact: {predicted_class}")
print(f"Severity: {severity_score:.2f}")
```

## Files in This Repository

- `impact_classifier.tflite` - Optimized TensorFlow Lite model (6.64 MB)
- `norm.npz` - Normalization parameters (mean and std)
- `README.md` - This file

## Deployment

**Supported Platforms**:
- ✅ Raspberry Pi (TensorFlow Lite)
- ✅ Android devices (TFLite Android SDK)
- ✅ Embedded systems (ARM Cortex-M7+)
- ✅ Edge TPU devices (Google Coral)

## Limitations

⚠️ **Important Considerations**:
- Trained on synthetic data only
- Requires calibrated sensors (IMU + vitals)
- Performance may vary with real-world sensor noise
- Needs field validation with actual combat scenarios
- Environmental factors (temperature, altitude) may affect accuracy

## Training Details

**Framework**: TensorFlow 2.x  
**Hardware**: Google Colab GPU (T4)  
**Training Time**: ~2-3 hours  
**Optimizer**: Adam (lr=0.001, clipnorm=1.0)  
**Loss**: Categorical Crossentropy (label smoothing=0.1) + MSE  
**Callbacks**: Early stopping, learning rate reduction, model checkpointing

## Ethical Considerations

This model is designed for **defensive military applications** to:
- Improve soldier safety and survivability
- Enable faster medical response
- Reduce combat casualties

**Intended Users**: Military medical personnel, combat medics, field commanders

## Citation

```bibtex
@software{guardian_shield_2025,
  author = {Tadikamalla, Vaibhav},
  title = {Guardian-Shield Combat Impact Detector},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/tadikamallavaibhav/guardian-shield-combat-detector}
}
```

## Related Links

- **GitHub Repository**: https://github.com/vaibhav-tadikamalla/tids-combat-detection
- **Training Notebook**: Available in GitHub repo (GUARDIAN_SHIELD_FINAL_TRAINABLE.ipynb)
- **Full TIDS System**: See GitHub for complete edge device implementation

## Contact

**Author**: Vaibhav Tadikamalla  
**Project**: TIDS - Tactical Impact Detection System

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

## License

MIT License - See LICENSE file in GitHub repository
