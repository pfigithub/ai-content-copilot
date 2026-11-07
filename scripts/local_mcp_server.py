"""
A standalone MCP Server providing local system tools.
This runs as a completely separate process from our API.
"""
from mcp.server.fastmcp import FastMCP
import datetime
import os
from pathlib import Path

mcp = FastMCP("LocalSystem")

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = (BASE_DIR / ".." / "workspace").resolve()
WORKSPACE_DIR.mkdir(exist_ok=True)

@mcp.tool()
def get_current_time() -> str:
    """Get the current system date and exact time. Use this when the user asks about today's date or time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List all files and folders in the designated secure workspace on the local system."""
    try:
        target_path = (WORKSPACE_DIR / path).resolve()
        if not target_path.is_relative_to(WORKSPACE_DIR):
            return "Security Error: Access denied. You can only read files inside the designated workspace."
            
        if not target_path.exists():
            return f"Path does not exist: {path}"
            
        files = os.listdir(target_path)
        return "\n".join(files) if files else "Directory is empty."
    except Exception as e:
        return f"Error reading directory: {e}"

if __name__ == "__main__":
    mcp.run(transport='stdio')