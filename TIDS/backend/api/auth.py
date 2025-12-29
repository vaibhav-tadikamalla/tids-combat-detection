"""
Authentication API Module
Handles device and user authentication
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os
import hashlib
import secrets

router = APIRouter(prefix="/api/v1", tags=["auth"])

# In production, this would use proper database and JWT tokens
DEVICE_KEYS = {}  # device_id -> api_key


class DeviceRegistration(BaseModel):
    device_id: str
    soldier_id: str
    unit: str


class AuthToken(BaseModel):
    device_id: str
    api_key: str


@router.post("/register")
async def register_device(data: DeviceRegistration):
    """
    Register a new device and generate API key.
    
    In production, this would:
    - Verify authorization
    - Store in secure database
    - Generate JWT token
    - Set up encryption keys
    """
    if data.device_id in DEVICE_KEYS:
        raise HTTPException(status_code=400, detail="Device already registered")
    
    # Generate secure API key
    api_key = secrets.token_urlsafe(32)
    DEVICE_KEYS[data.device_id] = hashlib.sha256(api_key.encode()).hexdigest()
    
    return {
        "device_id": data.device_id,
        "api_key": api_key,
        "status": "registered",
        "message": "Store this API key securely - it won't be shown again"
    }


@router.post("/authenticate")
async def authenticate_device(auth: AuthToken):
    """Authenticate a device"""
    if auth.device_id not in DEVICE_KEYS:
        raise HTTPException(status_code=401, detail="Device not registered")
    
    key_hash = hashlib.sha256(auth.api_key.encode()).hexdigest()
    
    if key_hash != DEVICE_KEYS[auth.device_id]:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "status": "authenticated",
        "device_id": auth.device_id,
        "message": "Authentication successful"
    }


@router.get("/verify")
async def verify_token(device_id: str, api_key: str):
    """Verify an API key"""
    if device_id not in DEVICE_KEYS:
        return {"valid": False}
    
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return {"valid": key_hash == DEVICE_KEYS[device_id]}
