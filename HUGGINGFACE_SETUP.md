# 🤗 Hugging Face Model Upload Guide

## What is Hugging Face?

**Hugging Face** is like GitHub but specifically for machine learning models. It's the best place to share your AI model because:
- ✅ Free hosting (no size limits for models)
- ✅ Model versioning and tracking
- ✅ Automatic model cards (documentation)
- ✅ Easy downloads via API
- ✅ Community engagement and stars

---

## Step 1: Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up (free account)
3. Verify your email

---

## Step 2: Create Access Token

1. Click your profile (top right) → **Settings**
2. Click **Access Tokens** (left sidebar)
3. Click **New token**
4. Fill in:
   - Name: `tids-model-upload`
   - Role: **Write**
5. Click **Generate token**
6. **COPY THE TOKEN** - you won't see it again!

---

## Step 3: Install Hugging Face CLI

```powershell
pip install huggingface_hub
```

---

## Step 4: Login to Hugging Face

```powershell
huggingface-cli login
```

When prompted, paste your access token (from Step 2)

---

## Step 5: Create Model Repository

### Option A: Via Website (Easiest)

1. Go to https://huggingface.co/new
2. Fill in:
   - Owner: **Your username**
   - Model name: `guardian-shield-combat-detector`
   - License: **MIT**
   - Visibility: **Public**
3. Click **Create model**

### Option B: Via CLI

```powershell
huggingface-cli repo create guardian-shield-combat-detector --type model
```

---

## Step 6: Upload Your Model Files

**IMPORTANT**: First, download these files from your Colab training:
- `impact_classifier.tflite` (the trained model)
- `norm.npz` (normalization parameters)

Then upload them:

```powershell
cd 'C:\Users\Vaibhav Tadikamalla\Desktop\military_project'

# Upload model file
huggingface-cli upload YOUR_USERNAME/guardian-shield-combat-detector impact_classifier.tflite

# Upload normalization params
huggingface-cli upload YOUR_USERNAME/guardian-shield-combat-detector norm.npz
```

**Replace `YOUR_USERNAME`** with your actual Hugging Face username!

---

## Step 7: Create Model Card (README)

Create a file `MODEL_CARD.md`:

```markdown
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
- Blast
- Gunshot  
- Artillery
- Vehicle Crash
- Fall
- Normal

## Model Architecture

- ResNet-inspired with dilated convolutions
- Multi-head attention (16 heads)
- Squeeze-and-Excitation blocks
- Parameters: ~8.4M
- Format: TensorFlow Lite (optimized for edge deployment)

## Intended Use

**Primary Use**: Real-time combat impact detection on wearable military devices

**Input**: 13-channel sensor data (IMU + vitals) - 200 timesteps
- 3-axis accelerometer
- 3-axis gyroscope
- 3-axis magnetometer
- Heart rate, SpO2, breathing rate, temperature

**Output**: 
- Impact classification (6 classes)
- Severity score (0-1)

## Training Data

- 60,000 synthetic samples (physics-based modeling)
- Multi-harmonic signal generation with physiological responses
- Stratified train/val/test split (70/15/15)

## Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 95.46% |
| Inference Time | <10ms (GPU T4) |
| Model Size | ~34 MB |

## Usage

```python
import tensorflow as tf
import numpy as np

# Load model
interpreter = tf.lite.Interpreter('impact_classifier.tflite')
interpreter.allocate_tensors()

# Load normalization
norm = np.load('norm.npz')
mean, std = norm['mean'], norm['std']

# Prepare input (200 timesteps, 13 channels)
sensor_data = your_sensor_data  # shape: (1, 200, 13)
normalized = (sensor_data - mean) / std

# Inference
interpreter.set_tensor(0, normalized)
interpreter.invoke()

impact_type = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
severity = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])

classes = ['blast', 'gunshot', 'artillery', 'vehicle_crash', 'fall', 'normal']
print(f"Impact: {classes[impact_type.argmax()]}, Severity: {severity[0][0]:.2f}")
```

## Limitations

- Trained on synthetic data only
- Requires calibrated sensors
- Performance may vary with real-world sensor noise
- Needs validation with actual combat data

## Citation

```bibtex
@software{guardian_shield_2025,
  author = {Tadikamalla, Vaibhav},
  title = {Guardian-Shield Combat Impact Detector},
  year = {2025},
  url = {https://huggingface.co/YOUR_USERNAME/guardian-shield-combat-detector}
}
```

## Contact

For questions or collaboration: [Your email/contact]
```

Upload the model card:

```powershell
# Save the above content as MODEL_CARD.md, then:
huggingface-cli upload YOUR_USERNAME/guardian-shield-combat-detector MODEL_CARD.md --path-in-repo README.md
```

---

## ✅ Done! Your Model is Now Public

**Share your model**: `https://huggingface.co/YOUR_USERNAME/guardian-shield-combat-detector`

### What Others Can Do:

**Download via CLI:**
```bash
huggingface-cli download YOUR_USERNAME/guardian-shield-combat-detector
```

**Download in Python:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/guardian-shield-combat-detector",
    filename="impact_classifier.tflite"
)
norm_path = hf_hub_download(
    repo_id="YOUR_USERNAME/guardian-shield-combat-detector", 
    filename="norm.npz"
)
```

**Direct Download Links:**
- Model: `https://huggingface.co/YOUR_USERNAME/guardian-shield-combat-detector/resolve/main/impact_classifier.tflite`
- Norm: `https://huggingface.co/YOUR_USERNAME/guardian-shield-combat-detector/resolve/main/norm.npz`

---

## 🎯 Update Your README

Add these links to your GitHub README.md:

```markdown
## 📥 Download Pre-trained Model

- [Hugging Face](https://huggingface.co/YOUR_USERNAME/guardian-shield-combat-detector) 🤗
- [GitHub Releases](https://github.com/YOUR_USERNAME/tids-combat-detection/releases)
```

---

## 🌟 Pro Tips

1. **Add example outputs** - Upload prediction examples
2. **Enable Spaces** - Create an interactive demo
3. **Community engagement** - Respond to issues/discussions
4. **Model versioning** - Tag releases (v1.0, v1.1, etc.)
5. **Metrics tracking** - Upload training logs and charts
