"""
PRODUCTION-GRADE IMPACT CLASSIFICATION MODEL
Architecture: ResNet-inspired CNN + Bidirectional GRU + Multi-Head Attention
Target Accuracy: >95%
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers, Model
import numpy as np

class ResidualBlock(layers.Layer):
    """Residual block for deep feature extraction with projection shortcut"""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.add = layers.Add()
        self.projection = None
        
    def build(self, input_shape):
        # Add projection layer if dimensions don't match
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Residual connection with projection if needed
        shortcut = inputs
        if self.projection is not None:
            shortcut = self.projection(inputs)
        
        x = self.add([shortcut, x])
        x = self.activation(x)
        
        return x

class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention for temporal dependencies"""
    
    def __init__(self, embed_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.1
        )
        self.layernorm = layers.LayerNormalization()
        
    def call(self, inputs, training=False):
        attn_output = self.mha(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training
        )
        return self.layernorm(inputs + attn_output)

class ImpactClassificationModel(Model):
    """
    Advanced deep learning model for impact classification.
    
    Architecture:
    1. Input: (batch, 200, 13) - multivariate time series
    2. Feature extraction: Residual CNN blocks
    3. Temporal modeling: Bidirectional GRU
    4. Attention: Multi-head self-attention
    5. Classification: Dense layers with dropout
    
    Innovations:
    - Residual connections prevent vanishing gradients
    - Bidirectional GRU captures forward/backward temporal context
    - Attention mechanism focuses on critical time steps
    - Batch normalization stabilizes training
    - Label smoothing prevents overconfidence
    """
    
    def __init__(self, num_classes=6, **kwargs):
        super().__init__(**kwargs)
        
        # === FEATURE EXTRACTION (CNN) ===
        
        # Initial conv layer
        self.conv_input = layers.Conv1D(64, 7, padding='same')
        self.bn_input = layers.BatchNormalization()
        self.relu_input = layers.ReLU()
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64, kernel_size=5)
        self.pool1 = layers.MaxPooling1D(2, padding='same')
        
        self.res_block2 = ResidualBlock(128, kernel_size=5)
        self.pool2 = layers.MaxPooling1D(2, padding='same')
        
        self.res_block3 = ResidualBlock(256, kernel_size=3)
        
        # === TEMPORAL MODELING (RNN) ===
        
        self.bigru1 = layers.Bidirectional(
            layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )
        self.bigru2 = layers.Bidirectional(
            layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )
        
        # === ATTENTION MECHANISM ===
        
        self.attention = MultiHeadSelfAttention(embed_dim=128, num_heads=8)
        
        # Global pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.global_max_pool = layers.GlobalMaxPooling1D()
        
        # === CLASSIFICATION HEAD ===
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.bn_dense1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.4)
        
        self.dense2 = layers.Dense(128, activation='relu')
        self.bn_dense2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        
        # Multi-task outputs
        self.impact_classifier = layers.Dense(num_classes, activation='softmax', name='impact_type')
        self.severity_regressor = layers.Dense(1, activation='sigmoid', name='severity')
        
    def call(self, inputs, training=False):
        # Input shape: (batch, 200, 13)
        
        # CNN feature extraction
        x = self.conv_input(inputs)
        x = self.bn_input(x, training=training)
        x = self.relu_input(x)
        
        x = self.res_block1(x, training=training)
        x = self.pool1(x)
        
        x = self.res_block2(x, training=training)
        x = self.pool2(x)
        
        x = self.res_block3(x, training=training)
        
        # Temporal modeling
        x = self.bigru1(x, training=training)
        x = self.bigru2(x, training=training)
        
        # Attention
        x = self.attention(x, training=training)
        
        # Global pooling (both avg and max for richer representation)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = layers.concatenate([avg_pool, max_pool])
        
        # Dense layers
        x = self.dense1(x)
        x = self.bn_dense1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn_dense2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Outputs
        impact_type = self.impact_classifier(x)
        severity = self.severity_regressor(x)
        
        return {'impact_type': impact_type, 'severity': severity}

def build_production_model(num_classes=6, learning_rate=0.001):
    """
    Build and compile production model.
    
    Hyperparameters (OPTIMIZED):
    - Learning rate: 0.001 with ReduceLROnPlateau
    - Optimizer: Adam with AMSGrad (more stable)
    - Loss: Categorical crossentropy with label smoothing (0.1)
    - Metrics: Accuracy, Top-2 accuracy, Precision, Recall
    """
    
    model = ImpactClassificationModel(num_classes=num_classes)
    
    # Build model by calling it once
    model.build(input_shape=(None, 200, 13))
    
    # Optimizer with gradient clipping (prevents exploding gradients)
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
        amsgrad=True
    )
    
    # Loss functions with label smoothing
    impact_loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    severity_loss = keras.losses.MeanSquaredError()
    
    model.compile(
        optimizer=optimizer,
        loss={
            'impact_type': impact_loss,
            'severity': severity_loss
        },
        loss_weights={
            'impact_type': 1.0,
            'severity': 0.3
        },
        metrics={
            'impact_type': [
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ],
            'severity': [
                keras.metrics.MeanAbsoluteError(name='mae'),
                keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        }
    )
    
    return model

# Count parameters
def count_parameters(model):
    """Count trainable parameters"""
    trainable = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    
    print(f"\nModel Parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Non-trainable: {non_trainable:,}")
    print(f"  Total: {trainable + non_trainable:,}")
    
    return trainable

if __name__ == "__main__":
    model = build_production_model()
    model.summary()
    count_parameters(model)
