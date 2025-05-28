from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, List, Dict, Any, Union
import os
from dotenv import load_dotenv
import re
from datetime import datetime
from bson import ObjectId

# Import Pydantic models from models.py
from models import NodeDocument, ChainDocument, EdgeDocument, LLMConfigDocument

# Import utility for extracting template variables
from utils import extract_template_variables

load_dotenv()

# MongoDB connection string
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# Create MongoDB client and database
client = AsyncIOMotorClient(MONGODB_URL)
db = client.albus_db  # Database name

# Collections
nodes_collection = db.nodes
chains_collection = db.chains

async def save_node(node_data: Union[NodeDocument, Dict[str, Any]]) -> str:
    """Save a node to the database or update if it already exists.
    Accepts either a NodeDocument instance or a dictionary."""
    if isinstance(node_data, dict):
        try:
            node_doc = NodeDocument(**node_data)
        except Exception as e:
            raise ValueError(f"Invalid node_data dictionary: {e}")
    elif isinstance(node_data, NodeDocument):
        node_doc = node_data
    else:
        raise TypeError("node_data must be a NodeDocument instance or a dictionary")

    node_id = node_doc.node_id
    existing_node_dict = await nodes_collection.find_one({"node_id": node_id})

    if existing_node_dict:
        # Update existing node
        doc_to_update_with = node_doc.model_dump(by_alias=True, exclude_none=True, exclude_unset=True) 
        update_payload = {k: v for k, v in doc_to_update_with.items() if k not in ["_id", "id", "created_at", "node_id"]}
        
        if not update_payload: # if only ids were passed, nothing to update
            return str(existing_node_dict["_id"])

        update_payload["updated_at"] = datetime.utcnow()
        await nodes_collection.update_one(
            {"node_id": node_id},
            {"$set": update_payload}
        )
        return str(existing_node_dict["_id"])
    else:
        # Insert new node
        doc_to_insert = node_doc.model_dump(by_alias=True, exclude_none=True)
        if "_id" in doc_to_insert:
            del doc_to_insert["_id"]
        if "id" in doc_to_insert: 
            del doc_to_insert["id"]

        doc_to_insert["created_at"] = datetime.utcnow()
        doc_to_insert["updated_at"] = doc_to_insert["created_at"]

        result = await nodes_collection.insert_one(doc_to_insert)
        return str(result.inserted_id)

async def get_node(node_id: str) -> Optional[NodeDocument]:
    """Retrieve a node by its ID and parse it into a NodeDocument."""
    node_dict = await nodes_collection.find_one({"node_id": node_id})
    if node_dict:
        return NodeDocument(**node_dict)
    return None

async def get_node_name(node_id: str) -> str:
    """Get a node's name by its ID. Returns node_id if name is not set."""
    node = await nodes_collection.find_one({"node_id": node_id}, {"name": 1})
    return node.get("name", node_id) if node else node_id

async def save_chain(chain_data: Union[ChainDocument, Dict[str, Any]]) -> str:
    """Save a chain to the database. Accepts ChainDocument or dict."""
    if isinstance(chain_data, dict):
        try:
            chain_doc = ChainDocument(**chain_data)
        except Exception as e:
            raise ValueError(f"Invalid chain_data dictionary: {e}")
    elif isinstance(chain_data, ChainDocument):
        chain_doc = chain_data
    else:
        raise TypeError("chain_data must be a ChainDocument instance or a dictionary")

    # Auto-population for nodes_info is no longer needed as flow_nodes contains all info.
    # Ensure flow_nodes has the data required for enhanced_edges if they are to be populated.

    # Auto-populate enhanced_edges if not fully provided and edges are present.
    # This now uses names from chain_doc.flow_nodes.
    if not chain_doc.enhanced_edges and chain_doc.edges and chain_doc.flow_nodes:
        # Create a quick lookup for node names from flow_nodes
        node_names_map = {fn.node_id: fn.name for fn in chain_doc.flow_nodes}
        
        chain_doc.enhanced_edges = [] # Initialize to ensure it's clean
        for edge_dict in chain_doc.edges:
            from_node_id = edge_dict.get("from_node")
            to_node_id = edge_dict.get("to_node")
            if from_node_id and to_node_id:
                from_node_name = node_names_map.get(from_node_id, from_node_id) # Default to ID if name not in map
                to_node_name = node_names_map.get(to_node_id, to_node_id)       # Default to ID if name not in map
                chain_doc.enhanced_edges.append(EdgeDocument(
                    from_node=from_node_id, from_node_name=from_node_name,
                    to_node=to_node_id, to_node_name=to_node_name
                ))
    
    chain_doc.created_at = datetime.utcnow()
    chain_doc.updated_at = chain_doc.created_at

    doc_to_insert = chain_doc.model_dump(by_alias=True, exclude_none=True)
    # Ensure MongoDB generates the _id by removing any client-side generated/serialized one
    if "_id" in doc_to_insert:
        del doc_to_insert["_id"]
    if "id" in doc_to_insert: # Just in case Pydantic adds 'id' if alias wasn't used somewhere
        del doc_to_insert["id"]
            
    result = await chains_collection.insert_one(doc_to_insert)
    return str(result.inserted_id)

async def get_chain(chain_id_str: str) -> Optional[ChainDocument]:
    """Retrieve a chain by its MongoDB ObjectId string and parse it."""
    if not ObjectId.is_valid(chain_id_str):
        # If the provided ID is not a valid ObjectId string, it cannot be found by _id.
        # Optionally, log a warning or raise an error if strict ObjectId format is always expected.
        # print(f"Warning: Invalid ObjectId string passed to get_chain: {chain_id_str}")
        return None 
        
    try:
        mongo_id = ObjectId(chain_id_str)
    except Exception as e:
        # print(f"Error converting string to ObjectId in get_chain: {chain_id_str}, Error: {e}")
        return None # Conversion failed

    chain_dict = await chains_collection.find_one({"_id": mongo_id})
    
    if chain_dict:
        try:
            return ChainDocument(**chain_dict)
        except Exception as e:
            # print(f"Error parsing chain_dict into ChainDocument: {chain_dict}, Error: {e}")
            return None # Parsing failed
    return None

async def get_all_chains() -> List[ChainDocument]:
    """Retrieve all chains and parse them into ChainDocument instances."""
    chains = []
    async for chain_dict in chains_collection.find():
        chains.append(ChainDocument(**chain_dict))
    return chains

async def get_all_nodes(name_query: Optional[str] = None, limit: int = 0) -> List[NodeDocument]:
    """Retrieve nodes, optionally filtering by name and limiting results."""
    query = {}
    if name_query:
        escaped_query = re.escape(name_query)
        query["name"] = {"$regex": escaped_query, "$options": "i"}
        
    nodes_cursor = nodes_collection.find(query)
    if limit > 0:
        nodes_cursor = nodes_cursor.limit(limit)
    
    nodes = []
    async for node_dict in nodes_cursor:
        nodes.append(NodeDocument(**node_dict))
    return nodes

async def update_node_output(node_id: str, output_data: Any) -> bool:
    """Update only the output field of a node and its updated_at timestamp."""
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {"output": output_data, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0

async def update_node_name(node_id: str, name: str) -> bool:
    """Update the name of a node by its ID."""
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {"name": name, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0

async def delete_node(node_id: str) -> bool:
    """Delete a node by its ID. Also consider implications for chains."""
    # TODO: Consider chain cleanup logic if a node is deleted.
    # E.g., remove node from 'nodes' lists in chains, remove relevant edges.
    result = await nodes_collection.delete_one({"node_id": node_id})
    # Add chain update logic here if necessary
    return result.deleted_count > 0

async def delete_chain(chain_id: str) -> bool:
    """Delete a chain by its MongoDB ObjectId."""
    if not ObjectId.is_valid(chain_id):
        return False # Or raise error
    result = await chains_collection.delete_one({"_id": ObjectId(chain_id)})
    return result.deleted_count > 0

async def get_node_with_inputs(node_id: str) -> Optional[NodeDocument]:
    """Get a node along with information about its input nodes (experimental feature)."""
    target_node = await get_node(node_id)
    if not target_node:
        return None

    # This is a simplified way to represent inputs. A more robust solution 
    # might involve a dedicated field in NodeDocument or specific query.
    input_nodes_data: List[Dict[str, Any]] = [] 

    # Find chains where this node is a target_node in an edge
    async for chain_doc in chains_collection.find({"enhanced_edges.to_node": node_id}):
        for edge in chain_doc.enhanced_edges:
            if edge.to_node == node_id:
                input_node_doc = await get_node(edge.from_node)
                if input_node_doc:
                    input_nodes_data.append({
                        "node_id": input_node_doc.node_id,
                        "name": input_node_doc.name,
                        "node_type": input_node_doc.node_type,
                        "output": input_node_doc.output # The output of the source node
                    })
    
    # We are not modifying the NodeDocument schema itself here, so we'll return
    # the node and one could pass input_nodes_data separately or attach it if using a more flexible dict.
    # For now, just printing for demonstration and returning the node. 
    # If you want to embed this in the returned NodeDocument, the model would need an `input_nodes` field.
    # print(f"Input nodes for {target_node.name}: {input_nodes_data}") 
    # As a temporary measure, let's add it to the dict representation for this specific function call.
    # This won't be part of the formal NodeDocument schema unless defined.
    target_node_dict = target_node.model_dump()
    target_node_dict["input_nodes_details"] = input_nodes_data
    # And then re-parse. This is a bit hacky. Better to add to model if always needed.
    try:
        return NodeDocument(**target_node_dict) # This will fail if input_nodes_details is not in NodeDocument
    except:
        # Fallback: return the original node document without the extra field if parsing fails.
        # For this to work properly, NodeDocument should have an Optional field for input_nodes_details.
        return target_node 

async def update_node_prompt(node_id: str, new_prompt: str) -> bool:
    """Update the prompt field of a node and dynamically update its input_keys based on the new prompt."""
    # new_prompt can be an empty string, which is a valid state.
    
    # Extract variables from the new prompt to update input_keys
    new_input_keys = extract_template_variables(new_prompt)
    
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {
            "prompt": new_prompt, 
            "input_keys": new_input_keys, # Dynamically set input_keys
            "updated_at": datetime.utcnow()
            }
        }
    )
    return result.modified_count > 0

async def update_node_llm_config(node_id: str, new_llm_config: LLMConfigDocument) -> bool:
    """Update only the llm_config field of a node and its updated_at timestamp."""
    # Convert Pydantic model to dict for MongoDB storage
    llm_config_dict = new_llm_config.model_dump()
    result = await nodes_collection.update_one(
        {"node_id": node_id},
        {"$set": {"llm_config": llm_config_dict, "updated_at": datetime.utcnow()}}
    )
    return result.modified_count > 0

async def update_chain(chain_id_str: str, chain_update_data: ChainDocument) -> bool:
    """Update an existing chain by its MongoDB ObjectId string."""
    if not ObjectId.is_valid(chain_id_str):
        # print(f"Warning: Invalid ObjectId string passed to update_chain: {chain_id_str}")
        return False

    try:
        mongo_id = ObjectId(chain_id_str)
    except Exception as e:
        # print(f"Error converting string to ObjectId in update_chain: {chain_id_str}, Error: {e}")
        return False

    # Prepare the update document
    # We want to update fields that can be changed, and set updated_at
    # Exclude _id and created_at from the update payload
    update_payload = chain_update_data.model_dump(exclude={"id", "_id", "created_at"}, exclude_none=True)
    
    # Explicitly set updated_at to current time
    update_payload["updated_at"] = datetime.utcnow()

    # Auto-population for nodes_info is removed from update_payload generation.
    # flow_nodes from chain_update_data will be part of update_payload directly.

    # Auto-populate enhanced_edges if edges or flow_nodes were part of the update and enhanced_edges were not.
    # This now uses names from chain_update_data.flow_nodes.
    if ("edges" in update_payload or "flow_nodes" in update_payload) and \
       ("enhanced_edges" not in update_payload or not update_payload.get("enhanced_edges")) and \
       chain_update_data.edges and chain_update_data.flow_nodes:
        
        node_names_map = {fn.node_id: fn.name for fn in chain_update_data.flow_nodes}
        update_payload["enhanced_edges"] = [] # Initialize/clear before repopulating

        for edge_dict in chain_update_data.edges: # Use edges from the input data
            from_node_id = edge_dict.get("from_node")
            to_node_id = edge_dict.get("to_node")
            if from_node_id and to_node_id:
                from_node_name = node_names_map.get(from_node_id, from_node_id)
                to_node_name = node_names_map.get(to_node_id, to_node_id)
                update_payload["enhanced_edges"].append(EdgeDocument(
                    from_node=from_node_id, from_node_name=from_node_name,
                    to_node=to_node_id, to_node_name=to_node_name
                ).model_dump()) # Ensure it's dict for $set
    
    if not update_payload:
        # print(f"No actual data to update for chain {chain_id_str} after exclusions.")
        return False # Or True if no change is considered a success

    result = await chains_collection.update_one(
        {"_id": mongo_id},
        {"$set": update_payload}
    )
    return result.modified_count > 0 