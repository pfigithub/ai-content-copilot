import os

# Keep ordinary tests independent of a live PostgreSQL database.
os.environ["APP_ENV"] = "test"

# Set this before test modules import LangGraph.
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"