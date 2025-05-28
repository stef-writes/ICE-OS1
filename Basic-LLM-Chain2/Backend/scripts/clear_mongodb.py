import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

async def clear_mongodb():
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB URL from .env
    mongodb_url = os.getenv("MONGODB_URL")
    
    if not mongodb_url:
        print("ERROR: MONGODB_URL not found in .env file")
        return
    
    print("Connecting to MongoDB...")
    
    # Create client
    client = AsyncIOMotorClient(mongodb_url)
    db = client.albus_db
    
    # Count documents before deletion
    nodes_count = await db.nodes.count_documents({})
    chains_count = await db.chains.count_documents({})
    
    print(f"Found {nodes_count} nodes and {chains_count} chains in database")
    print("Preparing to delete all data...")
    
    # Ask for confirmation
    confirm = input("Are you sure you want to delete all data? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Delete all documents from collections
    nodes_result = await db.nodes.delete_many({})
    chains_result = await db.chains.delete_many({})
    
    print(f"✅ Deleted {nodes_result.deleted_count} nodes")
    print(f"✅ Deleted {chains_result.deleted_count} chains")
    
    # Verify collections are empty
    nodes_remaining = await db.nodes.count_documents({})
    chains_remaining = await db.chains.count_documents({})
    
    if nodes_remaining == 0 and chains_remaining == 0:
        print("All data successfully cleared from MongoDB!")
    else:
        print(f"WARNING: {nodes_remaining} nodes and {chains_remaining} chains still remain")

if __name__ == "__main__":
    asyncio.run(clear_mongodb()) 