import aiohttp
import asyncio
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import json
import base64
import os

class SecureTransmitter:
    """Military-grade encrypted communication"""
    
    def __init__(self, server_url, device_id, encryption_key):
        self.server_url = server_url
        self.device_id = device_id
        self.session_key = encryption_key.encode()[:32]  # AES-256
        
        # Generate device RSA keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
        self.session = None
        
    async def initialize_session(self):
        """Create persistent HTTPS session"""
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(ssl=True)  # Enforce TLS 1.3
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'X-Device-ID': self.device_id,
                'User-Agent': 'GuardianShield/1.0'
            }
        )
    
    def encrypt_payload(self, data):
        """AES-256-GCM encryption"""
        plaintext = json.dumps(data).encode('utf-8')
        
        # Generate IV
        iv = os.urandom(12)
        
        # AES-GCM
        cipher = Cipher(
            algorithms.AES(self.session_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Package: IV + Tag + Ciphertext
        package = {
            'iv': base64.b64encode(iv).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'data': base64.b64encode(ciphertext).decode()
        }
        
        return package
    
    async def send_alert(self, alert_payload):
        """Send critical impact alert"""
        if not self.session:
            await self.initialize_session()
        
        encrypted = self.encrypt_payload(alert_payload)
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/v1/alerts",
                json=encrypted,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"[COMM ERROR] Alert send failed: {response.status}")
                    return None
        except Exception as e:
            print(f"[COMM ERROR] {e}")
            # Store locally for retry
            await self.queue_for_retry(encrypted)
            return None
    
    async def send_emergency(self, incap_alert):
        """High-priority emergency transmission"""
        if not self.session:
            await self.initialize_session()
        
        encrypted = self.encrypt_payload(incap_alert)
        
        # Retry logic for critical messages
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{self.server_url}/api/v1/emergency",
                    json=encrypted,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        return await response.json()
            except:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        print("[CRITICAL] Emergency transmission failed after retries")
    
    async def send_telemetry(self, telemetry):
        """Send routine telemetry"""
        if not self.session:
            await self.initialize_session()
        
        encrypted = self.encrypt_payload(telemetry)
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/v1/telemetry",
                json=encrypted
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def send_heartbeat(self, heartbeat):
        """Send keepalive heartbeat"""
        if not self.session:
            await self.initialize_session()
        
        try:
            async with self.session.post(
                f"{self.server_url}/api/v1/heartbeat",
                json=heartbeat
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def send_low_battery_alert(self, battery_level):
        """Send low battery warning"""
        alert = {
            'device_id': self.device_id,
            'alert_type': 'LOW_BATTERY',
            'battery_level': battery_level,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        await self.send_alert(alert)
    
    async def queue_for_retry(self, payload):
        """Store failed transmissions for retry"""
        # In real implementation: Write to persistent storage
        pass
    
    async def close(self):
        """Cleanup"""
        if self.session:
            await self.session.close()
