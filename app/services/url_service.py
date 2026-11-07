import httpx
import socket
from urllib.parse import urlparse
from app.exceptions import URLFetchError
from app.services.html_service import extract_text_from_html

DEFAULT_TIMEOUT = 20.0
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB max response size

def is_safe_url(url: str) -> bool:
    """Prevent SSRF attacks by resolving DNS and blocking local, private, and reserved IP spaces."""
    import ipaddress
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
            
        hostname = parsed.hostname
        if not hostname:
            return False
            
        # Resolve hostname to IP to prevent DNS Rebinding attacks
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False # Unresolvable host
            
        # Check every IP resolved. If ANY point to internal networks, block the URL.
        for info in addr_info:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
                
        return True
    except Exception:
        return False

async def fetch_url_content(url: str) -> str:
    if not is_safe_url(url):
         raise URLFetchError()

    try:
        # We disable automatic redirects to force explicit URL validation and prevent SSRF bypass
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=DEFAULT_TIMEOUT,
        ) as client:
            
            # Using stream to strictly enforce payload size limit during download
            async with client.stream("GET", url) as response:
                if 300 <= response.status_code < 400:
                    raise URLFetchError()
                    
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower() and "text/plain" not in content_type.lower():
                     raise URLFetchError()

                content_chunks = []
                bytes_downloaded = 0
                
                async for chunk in response.aiter_bytes():
                    bytes_downloaded += len(chunk)
                    if bytes_downloaded > MAX_CONTENT_LENGTH:
                        raise URLFetchError() # Abort if stream exceeds limit
                    content_chunks.append(chunk)
                    
                raw_html = b"".join(content_chunks).decode("utf-8", errors="ignore")

    except Exception as exc:
        raise URLFetchError() from exc

    return extract_text_from_html(raw_html)