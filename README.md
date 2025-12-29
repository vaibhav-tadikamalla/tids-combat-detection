# 🎯 TIDS - Tactical Impact Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Project Overview

**TIDS (Tactical Impact Detection System)** is an AI-powered combat impact classification system designed for military applications. It uses deep learning to analyze sensor data from wearable devices and classify combat impacts in real-time with **95.46% accuracy**.

**🤗 Try it now**: [Hugging Face Model](https://huggingface.co/tadikamallavaibhav/GuardianShieldCombatDetector)

### Key Features
- ✅ **6 Impact Classes**: blast, gunshot, artillery, vehicle_crash, fall, normal
- ✅ **Severity Assessment**: 0-1 injury severity score
- ✅ **95.46% Test Accuracy**: Trained on 60,000 synthetic samples
- ✅ **Real-time Processing**: TensorFlow Lite optimized
- ✅ **Multi-sensor Fusion**: IMU (accelerometer, gyroscope, magnetometer) + vitals (HR, SpO2, breathing, temp)
- ✅ **API Access**: Use via Hugging Face Inference API (no download needed)

---

## 📊 Model Architecture

**Guardian-Shield Deep ResNet with Attention**

```
Input: (200 timesteps, 13 channels)
├─ Initial Conv1D (64 filters)
├─ Stage 1: 2x Residual Blocks (64 filters, dilations: 1, 2)
├─ Stage 2: 3x Residual Blocks (128 filters, dilations: 1, 2, 4)
├─ Stage 3: 3x Residual Blocks (256 filters, dilations: 1, 2, 4)
├─ Stage 4: 2x Residual Blocks (512 filters, dilations: 1, 2)
├─ Multi-Head Attention (16 heads)
├─ Squeeze-and-Excitation Block
├─ Dual Pooling (Avg + Max)
└─ Classification Head → [Impact Type (6 classes), Severity (0-1)]
```

**Parameters**: ~8.4M  
**Inference Time**: <10ms on GPU T4

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda
- (Optional) Google Colab account for training

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/military_project.git
cd military_project

# Install dependencies
pip install -r requirements.txt

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## 📁 Project Structure

```
military_project/
├── TIDS/                              # Main TIDS system
│   ├── ml_training/                   # ML training scripts
│   │   ├── advanced_data_generator.py
│   │   ├── autonomous_training.py
│   │   ├── production_model.py
│   │   └── models/                    # Trained models (download separately)
│   ├── sensors/                       # Sensor integration
│   ├── communication/                 # Alert system
│   └── README.md
├── GUARDIAN_SHIELD_FINAL_TRAINABLE.ipynb  # Google Colab training notebook
├── instant_colab_setup.py             # Auto-generates Colab notebook
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## 🎓 Training the Model

### Option 1: Google Colab (Recommended)

1. **Upload Notebook**
   - Open [Google Colab](https://colab.research.google.com/)
   - Upload `GUARDIAN_SHIELD_FINAL_TRAINABLE.ipynb`

2. **Configure Runtime**
   - Runtime → Change runtime type → **GPU (T4)**

3. **Run Training**
   - Runtime → Run all
   - Wait 2-3 hours
   - Download `impact_classifier.tflite` and `norm.npz`

### Option 2: Local Training

```bash
cd TIDS/ml_training
python autonomous_training.py
```

**Requirements**: 
- GPU with 8GB+ VRAM recommended
- ~12GB RAM
- 2-4 hours training time

---

## 🔥 Using the Trained Model

### 🎯 Hugging Face Inference API (Recommended)

Use the pre-trained model **without downloading** via Hugging Face Inference API:

**Model**: [tadikamallavaibhav/GuardianShieldCombatDetector](https://huggingface.co/tadikamallavaibhav/GuardianShieldCombatDetector)

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import InferenceClient
import numpy as np

# Initialize client
client = InferenceClient(model="tadikamallavaibhav/GuardianShieldCombatDetector")

# Prepare sensor data (200 timesteps, 13 channels)
sensor_data = np.random.randn(200, 13).astype(np.float32)

# Get predictions (model hosted on HF servers)
result = client.predict(sensor_data)

print(f"Impact: {result['impact_type']} | Severity: {result['severity']:.2f}")
```

**Benefits:**
- ✅ No model download needed (zero setup)
- ✅ Always uses latest model version
- ✅ Hugging Face handles infrastructure
- ✅ Fast inference with HF accelerators

### 📥 For Research/Offline Use

To access the model files for research or offline deployment, request access:

1. Visit [Model Page](https://huggingface.co/tadikamallavaibhav/GuardianShieldCombatDetector)
2. Click **"Request Access"**
3. Approval typically granted within 24 hours

Once approved:

```bash
huggingface-cli download tadikamallavaibhav/GuardianShieldCombatDetector
```

### Local Inference (After Download)

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter(model_path='impact_classifier.tflite')
interpreter.allocate_tensors()

# Load normalization params
norm = np.load('norm.npz')
mean, std = norm['mean'], norm['std']

# Prepare input (200 timesteps, 13 channels)
sensor_data = np.random.randn(1, 200, 13).astype(np.float32)
normalized_data = (sensor_data - mean) / std

# Run inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], normalized_data)
interpreter.invoke()

impact_type = interpreter.get_tensor(output_details[0]['index'])
severity = interpreter.get_tensor(output_details[1]['index'])

# Decode results
classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
predicted_class = classes[impact_type.argmax()]
severity_score = severity[0][0]

print(f"Impact: {predicted_class} | Severity: {severity_score:.2f}")
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.46% |
| **Training Samples** | 60,000 |
| **Validation Accuracy** | ~95.8% |
| **Inference Time** | <10ms (GPU T4) |
| **Model Size** | ~34 MB (TFLite) |

### Confusion Matrix
(To be added after evaluation)

---

## 🧪 Dataset

**Synthetic Physics-Based Data**
- **Size**: 60,000 samples (10,000 per class)
- **Sampling Rate**: 200 timesteps
- **Channels**: 13 (3-axis accel, 3-axis gyro, 3-axis mag, HR, SpO2, breathing, temp)
- **Generation**: Multi-harmonic signal modeling with realistic physiological responses

---

## 🚀 Deployment

### Hardware Options

1. **Raspberry Pi** (Recommended for prototyping)
   ```bash
   pip install tensorflow-lite
   python inference.py
   ```

2. **Android** (Mobile deployment)
   - Use TensorFlow Lite Android SDK
   - See: [TFLite Android Guide](https://www.tensorflow.org/lite/android)

3. **Embedded Systems** (Production)
   - ARM Cortex-M7+ with TFLite Micro
   - Edge TPU for accelerated inference

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Google Colab for free GPU resources
- Military research community for domain expertise

---

## 📧 Contact

**Project Maintainer**: Vaibhav Tadikamalla

**Questions?** Open an issue or reach out via email.

---

## 🔮 Future Work

- [ ] Real sensor data collection and validation
- [ ] Multi-modal fusion (audio + sensor)
- [ ] Edge optimization (quantization to INT8)
- [ ] Real-time alert system integration
- [ ] Field testing with military personnel
- [ ] Integration with command center dashboard

---

**⭐ Star this repo if you find it useful!**
