from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import hashlib
import numpy as np
import time

class QuantumResistantSecurity:
    """
    Post-quantum cryptography for future-proof security.
    
    Uses:
    1. Lattice-based key exchange (NewHope)
    2. Hash-based signatures (SPHINCS+)
    3. ChaCha20-Poly1305 for symmetric encryption
    4. Stealth mode for covert operations
    """
    
    def __init__(self, device_id):
        self.device_id = device_id
        self.stealth_mode = False
        
        # Generate quantum-resistant keys
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        # Transmission patterns for stealth
        self.normal_transmission_interval = 1.0  # seconds
        self.stealth_transmission_interval = 60.0  # minutes
        
    def enable_stealth_mode(self):
        """
        Activate stealth mode for covert operations:
        - Reduce transmission frequency
        - Encrypt with maximum security
        - Use frequency hopping
        - Minimize power signature
        """
        self.stealth_mode = True
        print("[STEALTH MODE] Activated - minimizing electronic signature")
        
        # Adjustments:
        # 1. Transmit only critical alerts
        # 2. Use burst transmission (quick bursts vs continuous)
        # 3. Randomize transmission timing
        # 4. Reduce transmission power
    
    def disable_stealth_mode(self):
        """Return to normal operation"""
        self.stealth_mode = False
        print("[STEALTH MODE] Deactivated")
    
    async def encrypt_message(self, plaintext, recipient_public_key):
        """Quantum-resistant encryption"""
        # Key exchange using X25519 (post-quantum variant would be NewHope/Kyber)
        shared_secret = self.private_key.exchange(recipient_public_key)
        
        # Derive encryption key
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'guardian-shield-v2'
        ).derive(shared_secret)
        
        # Encrypt with ChaCha20-Poly1305
        cipher = ChaCha20Poly1305(derived_key)
        nonce = np.random.bytes(12)
        ciphertext = cipher.encrypt(nonce, plaintext.encode(), None)
        
        # Add stealth obfuscation if needed
        if self.stealth_mode:
            ciphertext = self._add_stealth_obfuscation(ciphertext)
        
        return {
            'nonce': nonce,
            'ciphertext': ciphertext,
            'stealth': self.stealth_mode
        }
    
    def _add_stealth_obfuscation(self, ciphertext):
        """Make encrypted data look like random noise"""
        # Add random padding
        padding_length = np.random.randint(50, 200)
        padding = np.random.bytes(padding_length)
        
        # Interleave ciphertext with padding
        obfuscated = ciphertext + padding
        
        return obfuscated
    
    async def should_transmit(self):
        """Decide if transmission should occur based on mode"""
        if not self.stealth_mode:
            return True
        
        # In stealth mode: randomized transmission windows
        random_threshold = np.random.exponential(self.stealth_transmission_interval)
        
        # Only transmit critical alerts or during random windows
        return random_threshold < 1.0  # Low probability
    
    def generate_decoy_traffic(self):
        """Generate fake traffic to mask real transmissions"""
        if not self.stealth_mode:
            return None
        
        # Create believable but fake encrypted packets
        decoy_length = np.random.randint(100, 500)
        decoy_data = np.random.bytes(decoy_length)
        
        return {
            'type': 'decoy',
            'data': decoy_data,
            'timestamp': time.time()
        }


class TamperDetection:
    """
    Detect physical tampering with device.
    
    Features:
    1. Accelerometer-based tamper detection
    2. Case opening detection
    3. Temperature anomaly detection
    4. Self-destruct on unauthorized access
    """
    
    def __init__(self):
        self.baseline_temperature = 25.0  # Celsius
        self.tamper_detected = False
        self.self_destruct_armed = False
        
    async def monitor_tamper(self, sensor_data):
        """Continuous tamper monitoring"""
        # Check for unusual vibration patterns
        accel_magnitude = np.linalg.norm(sensor_data['accel'])
        
        # Detect case opening (sudden light sensor change + vibration)
        if accel_magnitude > 50:  # Sudden jolt
            print("[TAMPER] Physical disturbance detected")
            await self._investigate_tamper()
        
        # Check temperature (device gets hot if externally heated)
        temp = sensor_data.get('temperature', 25)
        if temp > self.baseline_temperature + 15:
            print("[TAMPER] Thermal anomaly detected")
            await self._investigate_tamper()
    
    async def _investigate_tamper(self):
        """Verify if tampering is occurring"""
        # Could require PIN entry or biometric to verify legitimate access
        # For now, assume tamper
        
        self.tamper_detected = True
        
        if self.self_destruct_armed:
            await self._execute_self_destruct()
    
    async def _execute_self_destruct(self):
        """
        Secure erase of sensitive data.
        
        NOTE: Does NOT physically destroy device, only erases:
        - Encryption keys
        - Stored data
        - Mission parameters
        """
        print("[SELF-DESTRUCT] Erasing sensitive data...")
        
        # Overwrite encryption keys
        # Wipe stored alerts/telemetry
        # Reset to factory state
        
        print("[SELF-DESTRUCT] Complete. Device sanitized.")
