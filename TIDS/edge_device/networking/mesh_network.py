import asyncio
import json
import time
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from collections import defaultdict
import numpy as np

class MilitaryMeshNetwork:
    """
    LoRa/BLE mesh network for soldier-to-soldier communication.
    
    Features:
    1. Peer-to-peer encrypted messages
    2. Multi-hop routing
    3. Automatic topology discovery
    4. Collaborative threat detection
    5. Works without cellular/internet
    """
    
    def __init__(self, device_id, mesh_key):
        self.device_id = device_id
        self.cipher = ChaCha20Poly1305(mesh_key)
        
        # Mesh topology
        self.peers = {}  # {peer_id: {last_seen, position, hop_count, rssi}}
        self.routing_table = {}  # {destination: next_hop}
        
        # Collaborative data
        self.shared_threats = []  # Threats detected by any squad member
        self.squad_formation = {}  # Real-time squad positions
        
        # Message queue
        self.outgoing_queue = asyncio.Queue()
        self.incoming_queue = asyncio.Queue()
        
    async def start(self):
        """Initialize mesh network"""
        tasks = [
            self.discovery_loop(),
            self.message_sender(),
            self.message_receiver(),
            self.topology_update_loop()
        ]
        await asyncio.gather(*tasks)
    
    async def discovery_loop(self):
        """Discover nearby devices"""
        while True:
            # Broadcast beacon
            beacon = {
                'type': 'BEACON',
                'device_id': self.device_id,
                'timestamp': time.time(),
                'position': await self._get_current_position()
            }
            
            await self._broadcast(beacon)
            await asyncio.sleep(5)  # Beacon every 5 seconds
    
    async def message_receiver(self):
        """Process incoming mesh messages"""
        while True:
            # Simulate receiving message (in production: LoRa/BLE radio)
            message = await self._receive_radio()
            
            if message:
                await self._process_message(message)
            
            await asyncio.sleep(0.1)
    
    async def _process_message(self, encrypted_message):
        """Decrypt and handle incoming message"""
        try:
            # Decrypt
            nonce = encrypted_message['nonce'].encode()
            ciphertext = encrypted_message['data'].encode()
            plaintext = self.cipher.decrypt(nonce, ciphertext, None)
            
            message = json.loads(plaintext)
            
            # Handle different message types
            if message['type'] == 'BEACON':
                await self._handle_beacon(message)
            
            elif message['type'] == 'THREAT_ALERT':
                await self._handle_threat_alert(message)
            
            elif message['type'] == 'GUNSHOT_TRIANGULATION':
                await self._handle_gunshot_data(message)
            
            elif message['type'] == 'MEDICAL_EMERGENCY':
                await self._handle_medical_emergency(message)
            
            elif message['type'] == 'ROUTE_UPDATE':
                await self._update_routing_table(message)
            
            # Forward if not for us
            if message.get('destination') and message['destination'] != self.device_id:
                await self._forward_message(message)
        
        except Exception as e:
            print(f"[MESH] Failed to process message: {e}")
    
    async def _handle_beacon(self, beacon):
        """Update peer information"""
        peer_id = beacon['device_id']
        
        self.peers[peer_id] = {
            'last_seen': time.time(),
            'position': beacon['position'],
            'hop_count': beacon.get('hop_count', 1),
            'rssi': beacon.get('rssi', -50)  # Signal strength
        }
        
        # Update squad formation
        self.squad_formation[peer_id] = beacon['position']
    
    async def _handle_threat_alert(self, alert):
        """Process threat detected by squad member"""
        threat = {
            'source_device': alert['device_id'],
            'threat_type': alert['threat_type'],
            'location': alert['location'],
            'timestamp': alert['timestamp'],
            'severity': alert['severity']
        }
        
        self.shared_threats.append(threat)
        
        # Collaborative threat assessment
        threat_cluster = await self._analyze_threat_cluster(threat)
        
        if threat_cluster:
            print(f"[MESH] COORDINATED ATTACK DETECTED: {threat_cluster}")
            # Escalate to all squad members
            await self.broadcast_emergency(threat_cluster)
    
    async def _handle_gunshot_data(self, data):
        """Contribute to gunshot triangulation"""
        # Receive detection time from other devices
        detection_events = data['detection_events']
        
        # Add our own detection if we have one
        # Then triangulate
        from edge_device.sensors.acoustic_sensor import AcousticSniperDetection
        detector = AcousticSniperDetection(device_positions=self.squad_formation)
        
        shooter_location = await detector.triangulate_shooter(detection_events)
        
        if shooter_location:
            # Broadcast shooter position to squad
            await self.broadcast_threat({
                'type': 'SHOOTER_LOCATED',
                'position': shooter_location,
                'confidence': shooter_location['accuracy']
            })
    
    async def _handle_medical_emergency(self, emergency):
        """Coordinate medical response"""
        injured_position = emergency['location']
        
        # Find nearest squad member with medical training
        medic = await self._find_nearest_medic(injured_position)
        
        if medic:
            # Route medic to casualty
            await self.send_direct_message(medic, {
                'type': 'CASUALTY_LOCATION',
                'position': injured_position,
                'injury_type': emergency['injury_type'],
                'priority': emergency['priority']
            })
    
    async def broadcast_threat(self, threat_data):
        """Alert all squad members of threat"""
        message = {
            'type': 'THREAT_ALERT',
            'device_id': self.device_id,
            'threat_type': threat_data['type'],
            'location': threat_data.get('position'),
            'timestamp': time.time(),
            'severity': threat_data.get('confidence', 1.0)
        }
        
        await self._broadcast(message)
    
    async def broadcast_emergency(self, emergency_data):
        """Highest priority broadcast"""
        message = {
            'type': 'MEDICAL_EMERGENCY',
            'device_id': self.device_id,
            'location': await self._get_current_position(),
            'injury_type': emergency_data.get('injury_type', 'UNKNOWN'),
            'priority': 'CRITICAL',
            'timestamp': time.time()
        }
        
        await self._broadcast(message, priority=True)
    
    async def send_direct_message(self, destination_id, message):
        """Send message to specific device"""
        message['destination'] = destination_id
        message['source'] = self.device_id
        message['timestamp'] = time.time()
        
        # Find next hop
        next_hop = self.routing_table.get(destination_id, destination_id)
        
        await self._send_via_radio(next_hop, message)
    
    async def _broadcast(self, message, priority=False):
        """Broadcast to all nearby devices"""
        # Encrypt message
        plaintext = json.dumps(message).encode()
        nonce = np.random.bytes(12)
        ciphertext = self.cipher.encrypt(nonce, plaintext, None)
        
        encrypted_message = {
            'nonce': nonce.hex(),
            'data': ciphertext.hex()
        }
        
        # Send via radio
        await self._send_via_radio('broadcast', encrypted_message, priority)
    
    async def _send_via_radio(self, destination, message, priority=False):
        """Send message via LoRa/BLE radio"""
        # Placeholder - in production: interface with LoRa module
        # e.g., sx1276 driver
        if priority:
            # Use high-power transmission
            pass
        
        await self.outgoing_queue.put({
            'destination': destination,
            'message': message,
            'priority': priority
        })
    
    async def _receive_radio(self):
        """Receive message from radio"""
        # Placeholder - would read from LoRa/BLE module
        # Simulating random incoming messages
        if np.random.random() < 0.1:  # 10% chance per cycle
            return {
                'nonce': '0' * 24,
                'data': '0' * 32
            }
        return None
    
    async def message_sender(self):
        """Process outgoing message queue"""
        while True:
            msg = await self.outgoing_queue.get()
            # Actually transmit via hardware
            # await lora_module.send(msg)
            await asyncio.sleep(0.1)
    
    async def topology_update_loop(self):
        """Maintain mesh topology"""
        while True:
            # Remove stale peers
            current_time = time.time()
            stale_peers = [
                peer_id for peer_id, info in self.peers.items()
                if current_time - info['last_seen'] > 30  # 30 seconds timeout
            ]
            
            for peer_id in stale_peers:
                del self.peers[peer_id]
                if peer_id in self.squad_formation:
                    del self.squad_formation[peer_id]
            
            # Update routing table (simplified Dijkstra)
            await self._update_routes()
            
            await asyncio.sleep(10)
    
    async def _update_routes(self):
        """Compute optimal routes to all peers"""
        # Simplified routing based on hop count
        for peer_id, info in self.peers.items():
            if info['hop_count'] == 1:
                # Direct connection
                self.routing_table[peer_id] = peer_id
            # Multi-hop routing would be more complex
    
    async def _analyze_threat_cluster(self, new_threat):
        """Detect coordinated attacks from multiple threats"""
        recent_threats = [
            t for t in self.shared_threats
            if time.time() - t['timestamp'] < 60  # Last minute
        ]
        
        if len(recent_threats) >= 3:
            # Multiple threats in short time = coordinated attack
            positions = [t['location'] for t in recent_threats]
            
            # Check if threats form a pattern
            distances = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(
                        np.array(positions[i][:2]) - np.array(positions[j][:2])
                    )
                    distances.append(dist)
            
            if np.mean(distances) < 500:  # Within 500m
                return {
                    'type': 'COORDINATED_ATTACK',
                    'threat_count': len(recent_threats),
                    'area_radius_meters': np.max(distances),
                    'recommendation': 'IMMEDIATE_EVACUATION'
                }
        
        return None
    
    async def _find_nearest_medic(self, position):
        """Find closest squad member with medical training"""
        # Would check squad database for medic designation
        # For now, find closest peer
        if not self.peers:
            return None
        
        closest_peer = None
        min_distance = float('inf')
        
        for peer_id, info in self.peers.items():
            peer_pos = info['position']
            distance = np.linalg.norm(
                np.array(position[:2]) - np.array(peer_pos[:2])
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_peer = peer_id
        
        return closest_peer
    
    async def _get_current_position(self):
        """Get device's current GPS position"""
        # Would integrate with GPS module
        return [19.0760, 72.8777, 10.0]  # Placeholder
    
    async def _forward_message(self, message):
        """Forward message to next hop"""
        destination = message['destination']
        next_hop = self.routing_table.get(destination)
        
        if next_hop:
            message['hop_count'] = message.get('hop_count', 0) + 1
            await self._send_via_radio(next_hop, message)
