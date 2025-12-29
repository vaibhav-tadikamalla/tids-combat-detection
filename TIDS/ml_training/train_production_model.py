"""
Production Model Training Script
Trains GUARDIAN-SHIELD ML model to >95% accuracy
"""

import tensorflow as tf
from models.hybrid_cnn_lstm import ImpactClassifier
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

def train_production_model():
    """Train and validate production model"""
    
    print("="*60)
    print("GUARDIAN-SHIELD MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    if not os.path.exists('data/combat_trauma_dataset.h5'):
        print("ERROR: Dataset not found. Run generate_dataset.py first!")
        sys.exit(1)
    
    with h5py.File('data/combat_trauma_dataset.h5', 'r') as f:
        X = f['X'][:]
        y_type = f['y_type'][:]
        y_severity = f['y_severity'][:]
        classes = list(f.attrs['classes'])
    
    print(f"Loaded {len(X)} samples")
    print(f"Classes: {classes}")
    
    # Split data
    X_train, X_temp, y_type_train, y_type_temp, y_sev_train, y_sev_temp = train_test_split(
        X, y_type, y_severity, test_size=0.3, random_state=42, stratify=y_type.argmax(axis=1)
    )
    
    X_val, X_test, y_type_val, y_type_test, y_sev_val, y_sev_test = train_test_split(
        X_temp, y_type_temp, y_sev_temp, test_size=0.5, random_state=42, stratify=y_type_temp.argmax(axis=1)
    )
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Build model
    print("\nBuilding model...")
    model = ImpactClassifier()
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_impact_type_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_impact_type_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        ),
        tf.keras.callbacks.CSVLogger('training_history.csv')
    ]
    
    # Training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = model.fit(
        X_train,
        {
            'impact_type': y_type_train,
            'severity': y_sev_train,
            'confidence': np.ones(len(y_sev_train))
        },
        validation_data=(
            X_val,
            {
                'impact_type': y_type_val,
                'severity': y_sev_val,
                'confidence': np.ones(len(y_sev_val))
            }
        ),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_results = model.evaluate(
        X_test,
        {
            'impact_type': y_type_test,
            'severity': y_sev_test,
            'confidence': np.ones(len(y_sev_test))
        },
        verbose=1
    )
    
    print("\nTest Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"  {name}: {value:.4f}")
    
    # Check accuracy threshold
    impact_accuracy_idx = [i for i, name in enumerate(model.metrics_names) if 'impact_type_accuracy' in name][0]
    impact_accuracy = test_results[impact_accuracy_idx]
    
    print(f"\n{'='*60}")
    print("ACCURACY VALIDATION")
    print("="*60)
    
    if impact_accuracy >= 0.95:
        print(f"✓ PASS: Impact classification accuracy {impact_accuracy:.2%} >= 95%")
    else:
        print(f"✗ FAIL: Impact classification accuracy {impact_accuracy:.2%} < 95%")
        print("⚠ Model needs retraining with more data or architecture changes")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('models/guardian_shield_model.h5')
    print("\n✓ Model saved to models/guardian_shield_model.h5")
    
    # Convert to TFLite
    convert_to_tflite(model)
    
    return model, history

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    if 'impact_type_accuracy' in history.history:
        axes[0, 0].plot(history.history['impact_type_accuracy'], label='Train')
        axes[0, 0].plot(history.history['val_impact_type_accuracy'], label='Val')
        axes[0, 0].set_title('Impact Classification Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Val')
    axes[0, 1].set_title('Total Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Severity MAE
    if 'severity_mae' in history.history:
        axes[1, 0].plot(history.history['severity_mae'], label='Train')
        axes[1, 0].plot(history.history['val_severity_mae'], label='Val')
        axes[1, 0].set_title('Severity Estimation MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ Training plots saved to training_history.png")

def convert_to_tflite(model):
    """Convert to TFLite for edge deployment"""
    print("\n" + "="*60)
    print("CONVERTING TO TFLITE")
    print("="*60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_path = 'models/impact_classifier.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\n✓ TFLite model saved to {tflite_path}")
    print(f"  Model size: {size_kb:.1f} KB")
    
    # Validate TFLite model
    validate_tflite_model(tflite_path)

def validate_tflite_model(tflite_path):
    """Test TFLite model inference"""
    print("\nValidating TFLite model...")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output tensors: {len(output_details)}")
    
    # Test inference
    test_input = np.random.randn(1, 200, 13).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    print("  ✓ TFLite inference successful")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    train_production_model()
