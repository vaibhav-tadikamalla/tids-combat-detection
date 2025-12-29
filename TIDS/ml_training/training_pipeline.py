import tensorflow as tf
from models.hybrid_cnn_lstm import build_model
from data_generation import MilitaryScenarioGenerator
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

class TrainingPipeline:
    def __init__(self, model_save_path='./models/'):
        self.model_save_path = model_save_path
        self.model = build_model()
        
    def load_data(self, filepath='combat_trauma_dataset.h5'):
        """Load generated dataset"""
        with h5py.File(filepath, 'r') as f:
            X = f['X'][:]
            y_type = f['y_type'][:]
            y_severity = f['y_severity'][:]
        
        return train_test_split(X, y_type, y_severity, test_size=0.2, random_state=42)
    
    def create_data_augmentation(self):
        """Combat-specific data augmentation"""
        def augment(x, y_type, y_severity):
            # Time warping
            if tf.random.uniform([]) > 0.5:
                x = tf.image.random_crop(tf.pad(x, [[10, 10], [0, 0]]), [200, 13])
            
            # Magnitude scaling (simulate different sensor sensitivities)
            scale = tf.random.uniform([], 0.9, 1.1)
            x = x * scale
            
            # Gaussian noise injection
            noise = tf.random.normal(tf.shape(x), 0, 0.01)
            x = x + noise
            
            return x, {'impact_type': y_type, 'severity': y_severity}
        
        return augment
    
    def train(self, epochs=100):
        """Train with advanced techniques"""
        X_train, X_val, y_type_train, y_type_val, y_sev_train, y_sev_val = self.load_data()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_impact_type_accuracy',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{self.model_save_path}/best_model.h5',
                save_best_only=True,
                monitor='val_impact_type_accuracy'
            ),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]
        
        # Class weights for imbalanced data
        class_weights = self._compute_class_weights(y_type_train)
        
        # Training
        history = self.model.fit(
            X_train,
            {
                'impact_type': y_type_train,
                'severity': y_sev_train,
                'confidence': np.ones(len(y_sev_train))  # Placeholder
            },
            validation_data=(
                X_val,
                {
                    'impact_type': y_type_val,
                    'severity': y_sev_val,
                    'confidence': np.ones(len(y_sev_val))
                }
            ),
            epochs=epochs,
            batch_size=32,
            class_weight={'impact_type': class_weights},
            callbacks=callbacks
        )
        
        return history
    
    def convert_to_tflite(self):
        """Optimize for edge deployment"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimization for military-grade edge devices
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Quantization for extreme low-power operation
        def representative_dataset():
            X_train, _, _, _, _, _ = self.load_data()
            for i in range(100):
                yield [X_train[i:i+1].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        # Save
        with open(f'{self.model_save_path}/impact_classifier.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    def _compute_class_weights(self, y):
        """Handle class imbalance"""
        class_counts = np.sum(y, axis=0)
        total = np.sum(class_counts)
        weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        return weights


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.train(epochs=100)
    pipeline.convert_to_tflite()
