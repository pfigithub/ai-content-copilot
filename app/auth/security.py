import time
from importlib import import_module
from datetime import datetime, timedelta, timezone
import jwt
from app.config import settings

PasswordHasher = import_module("argon2").PasswordHasher
VerifyMismatchError = import_module("argon2.exceptions").VerifyMismatchError

# Argon2 is the winner of the Password Hashing Competition (PHC)
# It is significantly more secure than bcrypt or SHA-256 for passwords.
_hasher = PasswordHasher()

def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    return _hasher.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its Argon2 hash."""
    try:
        _hasher.verify(hashed_password, plain_password)
        
        # Proactive security: Check if the hashing parameters need upgrading
        if _hasher.check_needs_rehash(hashed_password):
            # In a full flow, you'd trigger a background re-hash here
            pass
            
        return True
    except VerifyMismatchError:
        return False

def create_access_token(user_id: str, role: str, expires_delta_hours: int = 24) -> str:
    """Generate a JWT for authenticated sessions."""
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured. Cannot issue tokens.")
        
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=expires_delta_hours)
    
    payload = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": expire,
        "sub": user_id,  # Subject (User ID)
        "role": role     # 'admin' or 'user'
    }
    
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")

def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT."""
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")
        
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")