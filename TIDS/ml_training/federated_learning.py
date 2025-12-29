import tensorflow as tf
import numpy as np
from typing import List, Dict
import asyncio
import time

class OnDeviceLearning:
    """
    Continual learning on edge device:
    1. Adapt to new threat patterns in real-time
    2. Federated learning across squad
    3. Privacy-preserving (never send raw data)
    4. Low-power incremental updates
    """
    
    def __init__(self, base_model_path, learning_rate=0.0001):
        self.model = tf.lite.Interpreter(model_path=base_model_path)
        self.learning_rate = learning_rate
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 1000
        
        # Adaptation parameters
        self.adaptation_threshold = 0.6  # Adapt when confidence < 60%
        self.min_samples_for_update = 10
        
    async def observe_and_learn(self, sensor_data, ground_truth_label=None):
        """
        Observe new data and decide whether to learn from it.
        
        Args:
            sensor_data: Raw sensor window
            ground_truth_label: User confirmation of event type (optional)
        """
        # Run inference
        prediction = self._infer(sensor_data)
        
        # If low confidence or user provided correction
        if prediction['confidence'] < self.adaptation_threshold or ground_truth_label:
            # Add to replay buffer
            label = ground_truth_label if ground_truth_label else prediction['impact_type']
            
            self.replay_buffer.append({
                'data': sensor_data,
                'label': label,
                'timestamp': time.time()
            })
            
            # Trim buffer
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)
            
            # Trigger learning if enough samples
            if len(self.replay_buffer) >= self.min_samples_for_update:
                await self._incremental_update()
    
    async def _incremental_update(self):
        """Update model with recent experiences"""
        print("[ON-DEVICE LEARNING] Performing incremental update...")
        
        # Sample from replay buffer
        batch = np.random.choice(
            self.replay_buffer,
            size=min(32, len(self.replay_buffer)),
            replace=False
        )
        
        X = np.array([sample['data'] for sample in batch])
        y = np.array([sample['label'] for sample in batch])
        
        # Perform gradient update (simplified - actual implementation more complex)
        # In production: use TFLite converter with gradient support or
        # maintain separate trainable model
        
        # For now, log for later batch update
        await self._queue_for_federated_update(X, y)
    
    async def _queue_for_federated_update(self, X, y):
        """Queue samples for federated learning"""
        # Compute local gradients (privacy-preserving)
        local_gradients = self._compute_gradients(X, y)
        
        # Share gradients (NOT raw data) with squad via mesh network
        await self._share_gradients_via_mesh(local_gradients)
    
    def _compute_gradients(self, X, y):
        """Compute model gradients without sharing data"""
        # Placeholder - would compute actual gradients
        return {'layer1': np.random.randn(128, 64), 'layer2': np.random.randn(64, 6)}
    
    async def _share_gradients_via_mesh(self, gradients):
        """Share gradients with squad for federated aggregation"""
        # Use mesh network to share encrypted gradients
        # Other devices aggregate and update their models
        pass
    
    def _infer(self, sensor_data):
        """Run inference with current model"""
        # Use TFLite interpreter
        return {'impact_type': 'blast', 'confidence': 0.85}


class FederatedAggregator:
    """
    Aggregate model updates from multiple soldiers.
    
    Implements Federated Averaging (FedAvg) algorithm for privacy-preserving
    collaborative learning across the squad.
    """
    
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.gradient_cache = {}
        self.aggregation_round = 0
        
    async def collect_gradients(self, device_id, gradients, num_samples):
        """Collect gradients from device"""
        self.gradient_cache[device_id] = {
            'gradients': gradients,
            'num_samples': num_samples,
            'timestamp': time.time()
        }
        
        # If enough devices have contributed
        if len(self.gradient_cache) >= self.num_devices * 0.7:  # 70% participation
            await self._aggregate_and_update()
    
    async def _aggregate_and_update(self):
        """Federated averaging"""
        print(f"[FEDERATED] Aggregating round {self.aggregation_round}")
        
        # Weighted average of gradients
        total_samples = sum(info['num_samples'] for info in self.gradient_cache.values())
        
        aggregated_gradients = {}
        
        for device_id, info in self.gradient_cache.items():
            weight = info['num_samples'] / total_samples
            
            for layer_name, gradient in info['gradients'].items():
                if layer_name not in aggregated_gradients:
                    aggregated_gradients[layer_name] = gradient * weight
                else:
                    aggregated_gradients[layer_name] += gradient * weight
        
        # Broadcast aggregated update to all devices
        await self._broadcast_model_update(aggregated_gradients)
        
        # Clear cache
        self.gradient_cache = {}
        self.aggregation_round += 1
    
    async def _broadcast_model_update(self, gradients):
        """Send updated model to all devices"""
        # Devices apply these gradients to their local models
        pass
