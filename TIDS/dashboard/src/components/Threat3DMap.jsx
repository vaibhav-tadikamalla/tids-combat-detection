import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

/**
 * 3D Battlefield Visualization
 * 
 * Features:
 * - Real-time 3D soldier positions
 * - Threat zones with pulsing animations
 * - Impact trajectory visualization
 * - Topographic terrain rendering
 * - Interactive camera controls
 */

const Threat3DMap = ({ soldiers, threats, impacts }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const soldiersRef = useRef({});
  const threatsRef = useRef({});
  
  const [selectedSoldier, setSelectedSoldier] = useState(null);

  useEffect(() => {
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e27);
    scene.fog = new THREE.Fog(0x0a0e27, 50, 200);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      60,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(50, 80, 50);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.1;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(100, 100, 50);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Create terrain
    createTerrain(scene);

    // Grid helper
    const gridHelper = new THREE.GridHelper(200, 40, 0x00ff00, 0x003300);
    gridHelper.position.y = 0.01;
    scene.add(gridHelper);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      controls.update();
      
      // Animate threats (pulsing effect)
      Object.values(threatsRef.current).forEach(threat => {
        const scale = 1 + 0.3 * Math.sin(Date.now() * 0.003);
        threat.scale.set(scale, scale, scale);
      });
      
      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  // Update soldiers
  useEffect(() => {
    if (!sceneRef.current) return;

    soldiers.forEach(soldier => {
      if (!soldiersRef.current[soldier.device_id]) {
        // Create new soldier
        const soldierMesh = createSoldier(soldier);
        sceneRef.current.add(soldierMesh);
        soldiersRef.current[soldier.device_id] = soldierMesh;
      } else {
        // Update existing soldier
        const mesh = soldiersRef.current[soldier.device_id];
        updateSoldierPosition(mesh, soldier);
        updateSoldierStatus(mesh, soldier);
      }
    });

    // Remove soldiers no longer in list
    Object.keys(soldiersRef.current).forEach(id => {
      if (!soldiers.find(s => s.device_id === id)) {
        sceneRef.current.remove(soldiersRef.current[id]);
        delete soldiersRef.current[id];
      }
    });
  }, [soldiers]);

  // Update threats
  useEffect(() => {
    if (!sceneRef.current) return;

    threats.forEach(threat => {
      if (!threatsRef.current[threat.id]) {
        // Create new threat visualization
        const threatMesh = createThreatVisualization(threat);
        sceneRef.current.add(threatMesh);
        threatsRef.current[threat.id] = threatMesh;
      }
    });

    // Remove old threats
    Object.keys(threatsRef.current).forEach(id => {
      if (!threats.find(t => t.id === id)) {
        sceneRef.current.remove(threatsRef.current[id]);
        delete threatsRef.current[id];
      }
    });
  }, [threats]);

  const createTerrain = (scene) => {
    // Create simple terrain with height variation
    const geometry = new THREE.PlaneGeometry(200, 200, 50, 50);
    const vertices = geometry.attributes.position.array;

    // Add height variation
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const y = vertices[i + 1];
      // Simple Perlin-like noise
      vertices[i + 2] = 
        Math.sin(x * 0.05) * 3 + 
        Math.cos(y * 0.07) * 2 + 
        Math.sin((x + y) * 0.03) * 1.5;
    }
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0x2a4a2a,
      roughness: 0.9,
      metalness: 0.1,
      flatShading: true
    });

    const terrain = new THREE.Mesh(geometry, material);
    terrain.rotation.x = -Math.PI / 2;
    terrain.receiveShadow = true;
    scene.add(terrain);
  };

  const createSoldier = (soldier) => {
    const group = new THREE.Group();

    // Body (cylinder)
    const bodyGeometry = new THREE.CylinderGeometry(0.3, 0.3, 1.8, 8);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: getSoldierColor(soldier.status),
      roughness: 0.7,
      metalness: 0.3
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.castShadow = true;
    body.position.y = 0.9;
    group.add(body);

    // Head (sphere)
    const headGeometry = new THREE.SphereGeometry(0.25, 16, 16);
    const head = new THREE.Mesh(headGeometry, bodyMaterial);
    head.castShadow = true;
    head.position.y = 2.1;
    group.add(head);

    // Status indicator (floating sphere above head)
    const indicatorGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    const indicatorMaterial = new THREE.MeshBasicMaterial({
      color: getVitalsColor(soldier.vitals),
      transparent: true,
      opacity: 0.8
    });
    const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
    indicator.position.y = 3;
    group.add(indicator);

    // Glow effect for critical soldiers
    if (soldier.status === 'critical') {
      const glowGeometry = new THREE.SphereGeometry(0.4, 16, 16);
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0xff0000,
        transparent: true,
        opacity: 0.3
      });
      const glow = new THREE.Mesh(glowGeometry, glowMaterial);
      glow.position.y = 0.9;
      group.add(glow);
    }

    // Position soldier
    const [lat, lon] = soldier.location;
    group.position.set(
      (lon - baseLocation.lon) * 111000,
      0,
      (lat - baseLocation.lat) * 111000
    );

    group.userData = { soldier };

    return group;
  };

  const createThreatVisualization = (threat) => {
    const group = new THREE.Group();

    // Threat zone (semi-transparent cylinder)
    const radius = threat.radius || 10;
    const geometry = new THREE.CylinderGeometry(radius, radius, 0.5, 32);
    const material = new THREE.MeshStandardMaterial({
      color: getThreatColor(threat.severity),
      transparent: true,
      opacity: 0.3,
      emissive: getThreatColor(threat.severity),
      emissiveIntensity: 0.5
    });
    const zone = new THREE.Mesh(geometry, material);
    zone.position.y = 0.25;
    group.add(zone);

    // Threat icon (pyramid)
    const iconGeometry = new THREE.ConeGeometry(1, 2, 4);
    const iconMaterial = new THREE.MeshStandardMaterial({
      color: getThreatColor(threat.severity),
      emissive: getThreatColor(threat.severity),
      emissiveIntensity: 0.8
    });
    const icon = new THREE.Mesh(iconGeometry, iconMaterial);
    icon.position.y = 2;
    icon.rotation.y = Math.PI / 4;
    group.add(icon);

    // Position threat
    const [lat, lon] = threat.location;
    group.position.set(
      (lon - baseLocation.lon) * 111000,
      0,
      (lat - baseLocation.lat) * 111000
    );

    return group;
  };

  const updateSoldierPosition = (mesh, soldier) => {
    const [lat, lon] = soldier.location;
    mesh.position.set(
      (lon - baseLocation.lon) * 111000,
      0,
      (lat - baseLocation.lat) * 111000
    );
  };

  const updateSoldierStatus = (mesh, soldier) => {
    // Update body color
    mesh.children[0].material.color.set(getSoldierColor(soldier.status));
    
    // Update indicator color
    if (mesh.children[2]) {
      mesh.children[2].material.color.set(getVitalsColor(soldier.vitals));
    }
  };

  const getSoldierColor = (status) => {
    switch (status) {
      case 'active': return 0x00ff00;
      case 'injured': return 0xffaa00;
      case 'critical': return 0xff0000;
      case 'offline': return 0x666666;
      default: return 0x00ff00;
    }
  };

  const getVitalsColor = (vitals) => {
    if (!vitals) return 0x00ff00;
    
    const hr = vitals.heart_rate;
    const spo2 = vitals.spo2;
    
    if (hr > 120 || hr < 50 || spo2 < 90) return 0xff0000;
    if (hr > 100 || hr < 60 || spo2 < 95) return 0xffaa00;
    return 0x00ff00;
  };

  const getThreatColor = (severity) => {
    if (severity > 0.7) return 0xff0000;
    if (severity > 0.4) return 0xffaa00;
    return 0xffff00;
  };

  // Base location for coordinate conversion
  const baseLocation = { lat: 28.6139, lon: 77.2090 }; // Example: New Delhi

  return (
    <div style={{ position: 'relative', width: '100%', height: '600px' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      
      {/* Controls overlay */}
      <div style={{
        position: 'absolute',
        top: 10,
        left: 10,
        background: 'rgba(0,0,0,0.7)',
        padding: '10px',
        borderRadius: '5px',
        color: 'white',
        fontSize: '12px'
      }}>
        <div><strong>Controls:</strong></div>
        <div>Left Mouse: Rotate</div>
        <div>Right Mouse: Pan</div>
        <div>Scroll: Zoom</div>
      </div>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        top: 10,
        right: 10,
        background: 'rgba(0,0,0,0.7)',
        padding: '10px',
        borderRadius: '5px',
        color: 'white',
        fontSize: '12px'
      }}>
        <div><strong>Status Legend:</strong></div>
        <div style={{ color: '#00ff00' }}>● Active</div>
        <div style={{ color: '#ffaa00' }}>● Injured</div>
        <div style={{ color: '#ff0000' }}>● Critical</div>
        <div style={{ color: '#666666' }}>● Offline</div>
      </div>

      {/* Selected soldier info */}
      {selectedSoldier && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          background: 'rgba(0,0,0,0.7)',
          padding: '15px',
          borderRadius: '5px',
          color: 'white',
          fontSize: '14px',
          minWidth: '200px'
        }}>
          <div><strong>Soldier: {selectedSoldier.device_id}</strong></div>
          <div>Status: {selectedSoldier.status}</div>
          <div>HR: {selectedSoldier.vitals?.heart_rate} bpm</div>
          <div>SpO2: {selectedSoldier.vitals?.spo2}%</div>
          <div>Location: [{selectedSoldier.location[0].toFixed(4)}, {selectedSoldier.location[1].toFixed(4)}]</div>
        </div>
      )}
    </div>
  );
};

export default Threat3DMap;
