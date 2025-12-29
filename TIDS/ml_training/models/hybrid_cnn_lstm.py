import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class SpatioTemporalAttention(layers.Layer):
    """Custom attention mechanism for multi-sensor fusion"""
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
        
    def call(self, features):
        score = tf.nn.tanh(self.W1(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        return tf.reduce_sum(context_vector, axis=1)


class ImpactClassifier(keras.Model):
    def __init__(self, num_classes=6, sequence_length=200):
        super().__init__()
        
        # Multi-sensor channels: Accel(3) + Gyro(3) + Mag(3) + Vitals(4) = 13
        self.input_channels = 13
        
        # 1D CNN for spatial feature extraction
        self.conv1 = layers.Conv1D(64, 7, padding='same', activation='relu')
        self.conv2 = layers.Conv1D(128, 5, padding='same', activation='relu')
        self.conv3 = layers.Conv1D(256, 3, padding='same', activation='relu')
        self.pool = layers.MaxPooling1D(2)
        self.dropout1 = layers.Dropout(0.3)
        
        # Bidirectional LSTM for temporal patterns
        self.lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        
        # Custom attention mechanism
        self.attention = SpatioTemporalAttention(128)
        
        # Transformer encoder block
        self.mha = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ff = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        
        # Classification heads
        self.impact_classifier = layers.Dense(num_classes, activation='softmax', name='impact_type')
        self.severity_estimator = layers.Dense(1, activation='sigmoid', name='severity')
        self.confidence_estimator = layers.Dense(1, activation='sigmoid', name='confidence')
        
    def call(self, inputs, training=False):
        # CNN feature extraction
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.dropout1(x, training=training)
        
        # LSTM temporal modeling
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Transformer self-attention
        attn_output = self.mha(x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        # Custom attention pooling
        x = self.attention(x)
        
        # Multi-task outputs
        impact_type = self.impact_classifier(x)
        severity = self.severity_estimator(x)
        confidence = self.confidence_estimator(x)
        
        return {
            'impact_type': impact_type,
            'severity': severity,
            'confidence': confidence
        }


def build_model():
    """Build and compile the model"""
    model = ImpactClassifier(num_classes=6)  # [Blast, Gunshot, Fall, Crash, None, Unknown]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'impact_type': 'categorical_crossentropy',
            'severity': 'binary_crossentropy',
            'confidence': 'mse'
        },
        loss_weights={
            'impact_type': 1.0,
            'severity': 0.5,
            'confidence': 0.3
        },
        metrics={
            'impact_type': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2)],
            'severity': ['mae'],
            'confidence': ['mae']
        }
    )
    
    return model
