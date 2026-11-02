import httpx
import ipaddress
from urllib.parse import urlparse
from app.exceptions import URLFetchError
from app.services.html_service import extract_text_from_html

DEFAULT_TIMEOUT = 20.0

def is_safe_url(url: str) -> bool:
    """Prevent SSRF attacks by blocking local, private, and reserved IP spaces."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Block localhost and typical internal aliases
        if hostname.lower() in ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"]:
            return False
            
        # Attempt to parse as IP to block private subnets
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
        except ValueError:
            pass # It's a domain name, not a direct IP
            
        return True
    except Exception:
        return False

async def fetch_url_content(url: str) -> str:
    if not is_safe_url(url):
         raise URLFetchError() # Or a specific SSRF error

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT,
            max_redirects=3 # Limit redirects to prevent infinite loops
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Prevent downloading massive non-text files
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower() and "text/plain" not in content_type.lower():
                 raise URLFetchError()

    except httpx.HTTPError as exc:
        raise URLFetchError() from exc

    return extract_text_from_html(response.text)