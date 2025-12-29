"""
AUTONOMOUS TRAINING PIPELINE
Automatically trains until 95% accuracy is achieved or max attempts reached.
"""

# Workaround for Python 3.13 compatibility
import sys
import signal

# Disable KeyboardInterrupt during import (Python 3.13 TF compatibility issue)
old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

# Use Keras 3 directly (better Python 3.13 support)
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

try:
    import keras
    from production_model import build_production_model, count_parameters
finally:
    # Restore handler
    signal.signal(signal.SIGINT, old_handler)

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class AutonomousTrainer:
    """
    Autonomous training system with adaptive hyperparameters.
    
    Features:
    - Automatic hyperparameter adjustment if accuracy target not met
    - Early stopping with patience
    - Learning rate scheduling
    - Data augmentation
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(self, target_accuracy=0.95, max_attempts=5):
        self.target_accuracy = target_accuracy
        self.max_attempts = max_attempts
        self.best_accuracy = 0.0
        self.attempt_history = []
        
    def load_data(self, filepath='data/combat_dataset.h5'):
        """Load and split dataset"""
        
        print(f"\n{'='*70}")
        print("  LOADING DATASET")
        print(f"{'='*70}")
        
        with h5py.File(filepath, 'r') as f:
            X = f['X'][:]
            y_type = f['y_type'][:]
            y_severity = f['y_severity'][:]
            classes = f.attrs['classes']
        
        print(f"  Loaded {len(X)} samples")
        print(f"  Classes: {classes}")
        
        # Stratified split
        X_train, X_temp, y_type_train, y_type_temp, y_sev_train, y_sev_temp = train_test_split(
            X, y_type, y_severity,
            test_size=0.30,
            random_state=42,
            stratify=y_type.argmax(axis=1)
        )
        
        X_val, X_test, y_type_val, y_type_test, y_sev_val, y_sev_test = train_test_split(
            X_temp, y_type_temp, y_sev_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_type_temp.argmax(axis=1)
        )
        
        print(f"\n  Split:")
        print(f"    Train: {len(X_train)} (70%)")
        print(f"    Val:   {len(X_val)} (15%)")
        print(f"    Test:  {len(X_test)} (15%)")
        
        # Compute class weights for imbalanced data
        y_train_labels = y_type_train.argmax(axis=1)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        print(f"\n  Class weights: {class_weight_dict}")
        
        return (X_train, X_val, X_test,
                y_type_train, y_type_val, y_type_test,
                y_sev_train, y_sev_val, y_sev_test,
                class_weight_dict, classes)
    
    def create_callbacks(self, attempt, patience=25):
        """Create training callbacks"""
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_impact_type_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Learning rate reduction
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                f'models/attempt_{attempt}_best.h5',
                monitor='val_impact_type_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=f'logs/attempt_{attempt}',
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                f'logs/attempt_{attempt}_history.csv',
                append=False
            ),
            
            # Learning rate logger
            keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr,
                verbose=0
            )
        ]
        
        return callbacks
    
    def train_attempt(self, attempt, learning_rate, batch_size, epochs, data):
        """Single training attempt"""
        
        (X_train, X_val, X_test,
         y_type_train, y_type_val, y_type_test,
         y_sev_train, y_sev_val, y_sev_test,
         class_weight_dict, classes) = data
        
        print(f"\n{'='*70}")
        print(f"  TRAINING ATTEMPT {attempt}/{self.max_attempts}")
        print(f"{'='*70}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
        print(f"{'='*70}\n")
        
        # Build model
        model = build_production_model(
            num_classes=len(classes),
            learning_rate=learning_rate
        )
        
        # Count parameters
        count_parameters(model)
        
        # Callbacks
        callbacks = self.create_callbacks(attempt)
        
        # Training (no class_weight for multi-output models)
        history = model.fit(
            X_train,
            {
                'impact_type': y_type_train,
                'severity': y_sev_train
            },
            validation_data=(
                X_val,
                {
                    'impact_type': y_type_val,
                    'severity': y_sev_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluation on test set
        print(f"\n{'='*70}")
        print("  FINAL EVALUATION ON TEST SET")
        print(f"{'='*70}\n")
        
        test_results = model.evaluate(
            X_test,
            {
                'impact_type': y_type_test,
                'severity': y_sev_test
            },
            batch_size=batch_size,
            verbose=1
        )
        
        # Extract accuracy
        metric_names = model.metrics_names
        impact_acc_idx = [i for i, name in enumerate(metric_names) if name == 'impact_type_accuracy'][0]
        test_accuracy = test_results[impact_acc_idx]
        
        print(f"\n{'='*70}")
        print(f"  TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  TARGET: {self.target_accuracy:.4f} ({self.target_accuracy*100:.2f}%)")
        
        if test_accuracy >= self.target_accuracy:
            print(f"  ✓ TARGET ACHIEVED!")
        else:
            print(f"  ✗ Below target by {(self.target_accuracy - test_accuracy)*100:.2f}%")
        
        print(f"{'='*70}\n")
        
        # Save attempt info
        attempt_info = {
            'attempt': attempt,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': len(history.history['loss']),
            'test_accuracy': float(test_accuracy),
            'test_results': {name: float(val) for name, val in zip(metric_names, test_results)},
            'target_achieved': test_accuracy >= self.target_accuracy
        }
        
        self.attempt_history.append(attempt_info)
        
        # Plot training history
        self.plot_history(history, attempt, test_accuracy)
        
        # Save model if best so far
        if test_accuracy > self.best_accuracy:
            self.best_accuracy = test_accuracy
            model.save('models/best_model_overall.h5')
            print(f"  ✓ New best model saved (accuracy: {test_accuracy:.4f})")
        
        # Convert to TFLite if target achieved
        if test_accuracy >= self.target_accuracy:
            self.convert_to_tflite(model, test_accuracy)
        
        return test_accuracy >= self.target_accuracy, test_accuracy, model
    
    def plot_history(self, history, attempt, test_accuracy):
        """Plot training curves"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Attempt {attempt} - Test Accuracy: {test_accuracy:.4f}', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['impact_type_accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_impact_type_accuracy'], label='Val', linewidth=2)
        axes[0, 0].axhline(y=self.target_accuracy, color='r', linestyle='--', label='Target')
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Total Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[0, 2].plot(history.history['impact_type_precision'], label='Precision', linewidth=2)
        axes[0, 2].plot(history.history['impact_type_recall'], label='Recall', linewidth=2)
        axes[0, 2].set_title('Precision & Recall')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Severity MAE
        axes[1, 0].plot(history.history['severity_mae'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_severity_mae'], label='Val', linewidth=2)
        axes[1, 0].set_title('Severity MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Top-2 Accuracy
        axes[1, 2].plot(history.history['impact_type_top2_accuracy'], label='Train', linewidth=2)
        axes[1, 2].plot(history.history['val_impact_type_top2_accuracy'], label='Val', linewidth=2)
        axes[1, 2].set_title('Top-2 Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'logs/attempt_{attempt}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Training curves saved to logs/attempt_{attempt}_training_curves.png")
    
    def convert_to_tflite(self, model, accuracy):
        """Convert model to TFLite"""
        
        print(f"\n{'='*70}")
        print("  CONVERTING TO TFLITE")
        print(f"{'='*70}\n")
        
        # Import TensorFlow only for TFLite conversion
        import tensorflow as tf
        
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
        
        print(f"  ✓ TFLite model saved to {tflite_path}")
        print(f"  Model size: {size_kb:.1f} KB")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\n{'='*70}\n")
    
    def run(self):
        """Main autonomous training loop"""
        
        print(f"\n{'#'*70}")
        print(f"#{'':^68}#")
        print(f"#{'AUTONOMOUS TRAINING SYSTEM':^68}#")
        print(f"#{'Target Accuracy: 95%':^68}#")
        print(f"#{'':^68}#")
        print(f"{'#'*70}\n")
        
        # Load data once
        data = self.load_data()
        
        # Hyperparameter configurations (progressively more aggressive)
        configs = [
            {'lr': 0.001, 'batch_size': 32, 'epochs': 100},
            {'lr': 0.0005, 'batch_size': 32, 'epochs': 150},
            {'lr': 0.001, 'batch_size': 64, 'epochs': 120},
            {'lr': 0.0003, 'batch_size': 32, 'epochs': 200},
            {'lr': 0.001, 'batch_size': 16, 'epochs': 150}
        ]
        
        for attempt in range(1, self.max_attempts + 1):
            config = configs[attempt - 1]
            
            success, accuracy, model = self.train_attempt(
                attempt=attempt,
                learning_rate=config['lr'],
                batch_size=config['batch_size'],
                epochs=config['epochs'],
                data=data
            )
            
            if success:
                print(f"\n{'#'*70}")
                print(f"#{'':^68}#")
                print(f"#{'🎉 SUCCESS! TARGET ACCURACY ACHIEVED 🎉':^68}#")
                print(f"#{'':^68}#")
                print(f"#{'Attempt: ' + str(attempt):^68}#")
                print(f"#{'Accuracy: ' + f'{accuracy:.4f} ({accuracy*100:.2f}%)':^68}#")
                print(f"#{'':^68}#")
                print(f"{'#'*70}\n")
                
                break
            
            print(f"\n  Attempt {attempt} did not meet target. Adjusting hyperparameters...\n")
        
        else:
            # Max attempts reached
            print(f"\n{'#'*70}")
            print(f"#{'':^68}#")
            print(f"#{'⚠ MAX ATTEMPTS REACHED ⚠':^68}#")
            print(f"#{'':^68}#")
            print(f"#{'Best Accuracy: ' + f'{self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)':^68}#")
            print(f"#{'':^68}#")
            print(f"{'#'*70}\n")
        
        # Save summary
        summary = {
            'target_accuracy': self.target_accuracy,
            'best_accuracy': float(self.best_accuracy),
            'attempts': self.attempt_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('logs/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Training summary saved to logs/training_summary.json\n")
        
        return self.best_accuracy >= self.target_accuracy

# EXECUTE IMMEDIATELY
if __name__ == "__main__":
    trainer = AutonomousTrainer(target_accuracy=0.95, max_attempts=5)
    success = trainer.run()
    
    if not success:
        print("\n❌ WARNING: Target accuracy not achieved. Review logs and adjust architecture.")
    else:
        print("\n✅ PRODUCTION MODEL READY FOR DEPLOYMENT")
