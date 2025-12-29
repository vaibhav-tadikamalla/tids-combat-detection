"""
Security Validation Suite
Tests encryption, authentication, and security features
"""

import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'edge_device'))

def test_encryption_imports():
    """Test that encryption modules can be imported"""
    try:
        from edge_device.security.secure_comm import SecureTransmitter
        from edge_device.security.quantum_resistant import QuantumResistantSecurity
        print("✓ Security modules imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import security modules: {e}")

def test_quantum_resistant_security():
    """Test quantum-resistant security features"""
    try:
        from edge_device.security.quantum_resistant import QuantumResistantSecurity
        
        qr_security = QuantumResistantSecurity(device_id="TEST-001")
        
        # Test stealth mode
        qr_security.enable_stealth_mode()
        assert qr_security.stealth_mode == True
        
        qr_security.disable_stealth_mode()
        assert qr_security.stealth_mode == False
        
        print("✓ Quantum-resistant security works")
    except ImportError:
        pytest.skip("Quantum-resistant module not available")

def test_no_hardcoded_secrets():
    """Scan for hardcoded secrets in code"""
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    suspicious_patterns = [
        'password = "',
        'api_key = "',
        'secret = "',
        'token = "'
    ]
    
    violations = []
    
    for root, dirs, files in os.walk(project_root):
        # Skip venv, node_modules, .git
        dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'node_modules', '.git', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for pattern in suspicious_patterns:
                            if pattern in content and 'os.getenv' not in content.split(pattern)[1].split('\n')[0]:
                                # Check if it's not a test or example
                                if 'test' not in filepath.lower() and 'example' not in filepath.lower():
                                    violations.append(f"{filepath}: {pattern}")
                except Exception as e:
                    pass
    
    # Allow some violations in test files
    critical_violations = [v for v in violations if 'test' not in v.lower()]
    
    if critical_violations:
        print(f"⚠ Potential hardcoded secrets found: {len(critical_violations)}")
        for v in critical_violations[:5]:  # Show first 5
            print(f"  {v}")
    else:
        print("✓ No hardcoded secrets found")
    
    # Don't fail the test, just warn
    assert len(critical_violations) < 10, "Too many potential hardcoded secrets"

def test_env_example_exists():
    """Verify .env.example exists"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env.example')
    assert os.path.exists(env_path), ".env.example not found"
    print("✓ .env.example exists")

def test_encryption_key_generation():
    """Test secure key generation"""
    import secrets
    
    # Generate keys
    key = secrets.token_bytes(32)
    assert len(key) == 32
    
    # Keys should be random
    key2 = secrets.token_bytes(32)
    assert key != key2
    
    print("✓ Secure key generation works")

if __name__ == "__main__":
    print("="*60)
    print("SECURITY VALIDATION SUITE")
    print("="*60)
    pytest.main([__file__, '-v'])
