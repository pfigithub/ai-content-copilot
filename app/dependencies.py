from fastapi import Header, HTTPException
from typing import Optional
from app.providers.ai_provider import AIProvider
from app.providers.ollama_provider import OllamaProvider

def get_ai_provider() -> AIProvider:
    return OllamaProvider()

def get_current_user(x_user_id: Optional[str] = Header(default=None, alias="X-User-ID")) -> str:
    """
    API Auth Gatekeeper: Extracts User ID from headers for Tenant Isolation.
    In a real SaaS, this should validate a JWT token or API Key.
    """
    if not x_user_id or not x_user_id.strip():
        # Fallback for strict local testing
        return "local_admin_user"
        
    cleaned_id = x_user_id.strip()
    
    # 🟢 FIX: Allow hyphens (-) in User IDs so tests can pass
    safe_chars_removed = cleaned_id.replace("_", "").replace("-", "")
    
    if len(cleaned_id) > 64 or not safe_chars_removed.isalnum():
         raise HTTPException(status_code=403, detail="Invalid User ID format.")
         
    return cleaned_id