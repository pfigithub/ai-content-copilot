"""
A standalone MCP Server providing local system tools.
This runs as a completely separate process from our API.
"""

from mcp.server.fastmcp import FastMCP
import datetime
import os

# Create an MCP server named "LocalSystem"
mcp = FastMCP("LocalSystem")

@mcp.tool()
def get_current_time() -> str:
    """Get the current system date and exact time. Use this when the user asks about today's date or time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List all files and folders in a specific directory on the local system."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error reading directory: {e}"

if __name__ == "__main__":
    # The server communicates via standard input/output (stdio)
    mcp.run(transport='stdio')