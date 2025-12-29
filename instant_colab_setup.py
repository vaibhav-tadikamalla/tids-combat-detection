#!/usr/bin/env python3
"""
INSTANT GOOGLE COLAB SETUP
Creates notebook with all code embedded - just upload and run.
"""

import json
import webbrowser
import os

def create_colab_notebook():
    """Create complete Colab notebook"""
    
    print("="*70)
    print(" "*20 + "CREATING COLAB NOTEBOOK")
    print("="*70)
    
    # Read all Python code
    code_files = {
        'data_gen': 'TIDS/ml_training/advanced_data_generator.py',
        'model': 'TIDS/ml_training/production_model.py',
        'trainer': 'TIDS/ml_training/autonomous_training.py'
    }
    
    code = {}
    for key, filepath in code_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                code[key] = f.read()
            print(f"✓ Loaded {filepath}")
        else:
            print(f"✗ Missing {filepath}")
            return
    
    # Create notebook with embedded code
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎯 GUARDIAN-SHIELD AUTONOMOUS TRAINING\n",
                    "\n",
                    "**Target: 95% Accuracy Minimum**\n",
                    "\n",
                    "This notebook will:\n",
                    "1. Generate 50,000 physics-based combat impact samples\n",
                    "2. Train ResNet+BiGRU+Attention model\n",
                    "3. Auto-retry until 95% accuracy achieved (max 5 attempts)\n",
                    "4. Convert to TFLite for edge deployment\n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 🚀 EXECUTION INSTRUCTIONS\n",
                    "\n",
                    "### Google Colab:\n",
                    "**STEP 1:** `Runtime` → `Change runtime type` → Select `GPU` (T4)\n",
                    "**STEP 2:** `Runtime` → `Run all`\n",
                    "**STEP 3:** Wait 2-3 hours\n",
                    "\n",
                    "### Kaggle (Recommended for TPU):\n",
                    "**STEP 1:** Upload to Kaggle Notebooks\n",
                    "**STEP 2:** Accelerator → Select `TPU v5e-8`\n",
                    "**STEP 3:** Click `Run All`\n",
                    "**STEP 4:** Wait 1-2 hours (faster!)\n",
                    "\n",
                    "**STEP 4:** Download `impact_classifier.tflite` when finished\n",
                    "\n",
                    "---\n",
                    "\n",
                    "**DO NOT MODIFY ANY CODE BELOW**"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === ENVIRONMENT SETUP ===\n",
                    "print('🔧 Setting up environment...')\n",
                    "\n",
                    "import tensorflow as tf\n",
                    "print(f'TensorFlow version: {tf.__version__}')\n",
                    "print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\"))} device(s)')\n",
                    "\n",
                    "!pip install -q h5py scikit-learn scipy matplotlib\n",
                    "\n",
                    "import os\n",
                    "os.makedirs('data', exist_ok=True)\n",
                    "os.makedirs('models', exist_ok=True)\n",
                    "os.makedirs('logs', exist_ok=True)\n",
                    "\n",
                    "print('✓ Environment ready')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === DATA GENERATOR CODE ===\n",
                    "# Physics-based FFT synthesis with 6 impact classes\n",
                    "\n",
                    code['data_gen'].replace('if __name__ == "__main__":', 'if False:')
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === MODEL ARCHITECTURE CODE ===\n",
                    "# ResNet+BiGRU+Attention (target: >95% accuracy)\n",
                    "\n",
                    code['model'].replace('if __name__ == "__main__":', 'if False:')
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === AUTONOMOUS TRAINING CODE ===\n",
                    "# 5-attempt retry with progressive hyperparameters\n",
                    "\n",
                    # Fix imports for Colab
                    code['trainer'].replace(
                        "# Workaround for Python 3.13 compatibility\\nimport sys\\nimport signal\\n\\n# Disable KeyboardInterrupt during import (Python 3.13 TF compatibility issue)\\nold_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)\\n\\n# Use Keras 3 directly (better Python 3.13 support)\\nimport os\\nos.environ['KERAS_BACKEND'] = 'tensorflow'\\n\\ntry:\\n    import keras\\n    from production_model import build_production_model, count_parameters\\nfinally:\\n    # Restore handler\\n    signal.signal(signal.SIGINT, old_handler)",
                        "import tensorflow as tf\\nfrom tensorflow import keras"
                    ).replace(
                        "if __name__ == \"__main__\":",
                        "if False:"
                    )
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === STEP 1: GENERATE DATASET ===\n",
                    "print('\\n' + '='*70)\n",
                    "print(' '*20 + 'GENERATING 50K SAMPLES')\n",
                    "print('='*70 + '\\n')\n",
                    "\n",
                    "generator = AdvancedDataGenerator(sample_rate=200, sequence_length=200)\n",
                    "X, y_type, y_severity = generator.generate_dataset(\n",
                    "    samples_per_class=8333,\n",
                    "    save_path='data/combat_dataset.h5'\n",
                    ")\n",
                    "\n",
                    "print('\\n✓ Dataset generation complete')\n",
                    "print(f'  Shape: {X.shape}')\n",
                    "print(f'  Classes: {y_type.shape[1]}')\n",
                    "print(f'  File: data/combat_dataset.h5')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === STEP 2: AUTONOMOUS TRAINING ===\n",
                    "print('\\n' + '='*70)\n",
                    "print(' '*15 + 'STARTING AUTONOMOUS TRAINING')\n",
                    "print(' '*20 + 'Target: 95% Accuracy')\n",
                    "print('='*70 + '\\n')\n",
                    "\n",
                    "trainer = AutonomousTrainer(target_accuracy=0.95, max_attempts=5)\n",
                    "trainer.run()\n",
                    "\n",
                    "print('\\n' + '='*70)\n",
                    "print(' '*20 + 'TRAINING COMPLETE')\n",
                    "print('='*70)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === STEP 3: VERIFY RESULTS ===\n",
                    "import json\n",
                    "\n",
                    "with open('logs/training_summary.json', 'r') as f:\n",
                    "    summary = json.load(f)\n",
                    "\n",
                    "print('\\n' + '='*70)\n",
                    "print(' '*25 + 'FINAL RESULTS')\n",
                    "print('='*70)\n",
                    "print(f\"\\nBest Accuracy: {summary['best_accuracy']:.4f} ({summary['best_accuracy']*100:.2f}%)\")\n",
                    "print(f\"Target: {summary['target_accuracy']:.4f} ({summary['target_accuracy']*100:.2f}%)\")\n",
                    "print(f\"Total Attempts: {len(summary['attempts'])}\")\n",
                    "\n",
                    "if summary['best_accuracy'] >= summary['target_accuracy']:\n",
                    "    print('\\n🎉 SUCCESS! Target accuracy achieved!')\n",
                    "    print('✓ TFLite model ready for deployment')\n",
                    "else:\n",
                    "    print(f\"\\n⚠ Best accuracy: {summary['best_accuracy']*100:.2f}%\")\n",
                    "    print(f\"  (Target was {summary['target_accuracy']*100:.2f}%)\")\n",
                    "\n",
                    "print('\\n' + '='*70)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# === STEP 4: DOWNLOAD MODELS ===\n",
                    "from google.colab import files\n",
                    "\n",
                    "print('\\nDownloading trained models...')\n",
                    "\n",
                    "# TFLite model (for deployment)\n",
                    "if os.path.exists('models/impact_classifier.tflite'):\n",
                    "    files.download('models/impact_classifier.tflite')\n",
                    "    print('✓ Downloaded: impact_classifier.tflite')\n",
                    "else:\n",
                    "    print('✗ TFLite model not found')\n",
                    "\n",
                    "# Keras model (for fine-tuning)\n",
                    "if os.path.exists('models/best_model_overall.h5'):\n",
                    "    files.download('models/best_model_overall.h5')\n",
                    "    print('✓ Downloaded: best_model_overall.h5')\n",
                    "\n",
                    "# Training summary\n",
                    "if os.path.exists('logs/training_summary.json'):\n",
                    "    files.download('logs/training_summary.json')\n",
                    "    print('✓ Downloaded: training_summary.json')\n",
                    "\n",
                    "print('\\n✅ ALL FILES READY FOR DEPLOYMENT')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## ✅ DEPLOYMENT CHECKLIST\n",
                    "\n",
                    "After download, you should have:\n",
                    "\n",
                    "1. **`impact_classifier.tflite`** - Deploy this to Raspberry Pi\n",
                    "2. **`best_model_overall.h5`** - Keep for future fine-tuning\n",
                    "3. **`training_summary.json`** - Training metrics and history\n",
                    "\n",
                    "---\n",
                    "\n",
                    "### Next Steps:\n",
                    "\n",
                    "```bash\n",
                    "# On Raspberry Pi:\n",
                    "# 1. Install TensorFlow Lite runtime\n",
                    "pip3 install tflite-runtime\n",
                    "\n",
                    "# 2. Copy impact_classifier.tflite to deployment directory\n",
                    "cp impact_classifier.tflite /path/to/tids/\n",
                    "\n",
                    "# 3. Run inference\n",
                    "python3 impact_detector.py\n",
                    "```"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "private_outputs": True
            },
            "kaggle": {
                "accelerator": "tpu1vmV5-8",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    # Write notebook
    output_path = 'GUARDIAN_SHIELD_TRAINING.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Created: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    return output_path

def main():
    """Execute instant setup"""
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  INSTANT GOOGLE COLAB SETUP".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # Create notebook
    notebook_path = create_colab_notebook()
    
    if not notebook_path:
        print("\n✗ Failed to create notebook")
        return
    
    # Open Colab
    print(f"\n{'='*70}")
    print("  NEXT STEPS (AUTOMATED)")
    print(f"{'='*70}")
    print("\n1. Opening Google Colab...")
    
    try:
        webbrowser.open('https://colab.research.google.com')
        print("   ✓ Browser opened")
    except:
        print("   ✗ Could not open browser automatically")
        print("   → Manually open: https://colab.research.google.com")
    
    print(f"\n2. Upload notebook:")
    print(f"   → File: {notebook_path}")
    print(f"   → Click: 'Upload notebook' → Choose file")
    
    print(f"\n3. Configure GPU:")
    print(f"   → Runtime → Change runtime type → GPU (T4)")
    
    print(f"\n4. Execute training:")
    print(f"   → Runtime → Run all")
    
    print(f"\n5. Wait for completion:")
    print(f"   → Estimated time: 1-3 hours")
    print(f"   → Watch for '🎉 SUCCESS!' message")
    
    print(f"\n6. Download models:")
    print(f"   → impact_classifier.tflite (auto-download)")
    print(f"   → best_model_overall.h5 (auto-download)")
    
    print(f"\n{'='*70}")
    print("  ✓ SETUP COMPLETE")
    print(f"{'='*70}\n")
    
    print("📋 SUMMARY:")
    print(f"   Notebook: {notebook_path}")
    print(f"   Colab: https://colab.research.google.com")
    print(f"   Action: Upload notebook and click 'Run all'")
    print()

if __name__ == "__main__":
    main()
