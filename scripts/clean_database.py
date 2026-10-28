"""
Maintenance script to wipe the Qdrant collection and remove all poisoned data.
"""

import asyncio
from qdrant_client import AsyncQdrantClient
from app.config import settings

async def main():
    print("1. Connecting to Qdrant Database...")
    # Initialize the async Qdrant client
    client = AsyncQdrantClient(url=settings.qdrant_url)
    
    collection_name = settings.qdrant_collection
    
    print(f"2. Attempting to drop the poisoned collection: '{collection_name}'...")
    try:
        # Dropping the collection permanently deletes all vectors and documents inside it
        await client.delete_collection(collection_name=collection_name)
        print(f"[SUCCESS] Collection '{collection_name}' has been completely wiped.")
        print("[INFO] The database is now sterile. The collection will be automatically recreated upon next document upload.")
    except Exception as e:
        print(f"[ERROR] Failed to wipe collection. It might already be empty or missing. Details: {e}")

if __name__ == "__main__":
    asyncio.run(main())