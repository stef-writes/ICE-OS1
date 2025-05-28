import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime

async def debug_database():
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
    
    # Count documents
    nodes_count = await db.nodes.count_documents({})
    chains_count = await db.chains.count_documents({})
    
    print(f"\n=== DATABASE OVERVIEW ===")
    print(f"Total nodes: {nodes_count}")
    print(f"Total chains: {chains_count}")
    
    # List all nodes
    print(f"\n=== ALL NODES ===")
    if nodes_count > 0:
        async for node in db.nodes.find({}):
            print(f"Node ID: {node.get('node_id', 'N/A')}")
            print(f"  Name: {node.get('name', 'N/A')}")
            print(f"  Type: {node.get('node_type', 'N/A')}")
            print(f"  Output: {str(node.get('output', 'None'))[:50]}...")
            print(f"  Created: {node.get('created_at', 'N/A')}")
            print(f"  Updated: {node.get('updated_at', 'N/A')}")
            print("  ---")
    else:
        print("No nodes found in database")
    
    # List all chains
    print(f"\n=== ALL CHAINS ===")
    if chains_count > 0:
        async for chain in db.chains.find({}):
            print(f"Chain ID: {chain.get('_id', 'N/A')}")
            print(f"  Name: {chain.get('name', 'N/A')}")
            print(f"  Flow Nodes Count: {len(chain.get('flow_nodes', []))}")
            print(f"  Edges Count: {len(chain.get('edges', []))}")
            print(f"  Created: {chain.get('created_at', 'N/A')}")
            print(f"  Updated: {chain.get('updated_at', 'N/A')}")
            
            # Show flow nodes details
            flow_nodes = chain.get('flow_nodes', [])
            if flow_nodes:
                print("  Flow Nodes:")
                for fn in flow_nodes:
                    print(f"    - {fn.get('node_id', 'N/A')} ({fn.get('name', 'N/A')})")
            
            # Show edges
            edges = chain.get('edges', [])
            if edges:
                print("  Edges:")
                for edge in edges:
                    print(f"    - {edge.get('from_node', 'N/A')} -> {edge.get('to_node', 'N/A')}")
            print("  ---")
    else:
        print("No chains found in database")
    
    # Check for orphaned nodes (nodes in chains but not in nodes collection)
    print(f"\n=== CHECKING FOR ORPHANED NODES ===")
    all_node_ids = set()
    async for node in db.nodes.find({}, {"node_id": 1}):
        all_node_ids.add(node["node_id"])
    
    chain_node_ids = set()
    async for chain in db.chains.find({}):
        flow_nodes = chain.get('flow_nodes', [])
        for fn in flow_nodes:
            chain_node_ids.add(fn.get('node_id'))
    
    orphaned_in_chains = chain_node_ids - all_node_ids
    orphaned_in_nodes = all_node_ids - chain_node_ids
    
    if orphaned_in_chains:
        print(f"Nodes referenced in chains but missing from nodes collection:")
        for node_id in orphaned_in_chains:
            print(f"  - {node_id}")
    
    if orphaned_in_nodes:
        print(f"Nodes in nodes collection but not referenced in any chain:")
        for node_id in orphaned_in_nodes:
            print(f"  - {node_id}")
    
    if not orphaned_in_chains and not orphaned_in_nodes:
        print("No orphaned nodes found - all nodes are properly synchronized!")
    
    print(f"\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(debug_database()) 