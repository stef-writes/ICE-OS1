from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional, Union

from models import (
    NodeDocument, ChainDocument, NodeNameUpdate, 
    NodePromptUpdate, NodeLLMConfigUpdate
) # Assuming these are accessible
from database import (
    save_node, get_node, save_chain, get_chain, 
    get_all_chains, get_all_nodes, delete_node, 
    update_node_name, update_node_output, update_chain,
    update_node_prompt, update_node_llm_config
) # Assuming these are accessible

router = APIRouter()

# --- New Database CRUD Operations using Pydantic Models ---

@router.post("/nodes/legacy_create", status_code=201, summary="Legacy: Create node with minimal data", deprecated=True)
async def create_node_legacy(node_id: str, output: Any):
    node_data_for_save = {
        "node_id": node_id, 
        "output": output,
        "name": f"Legacy Node {node_id}",
        "node_type": "unknown"
    }
    try:
        created_node = await save_node(node_data_for_save)
        return created_node
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legacy node creation: {str(e)}")

@router.post("/nodes/", response_model=NodeDocument, status_code=201, summary="Create a new node document")
async def create_node_document(node_doc: NodeDocument):
    try:
        node_data = node_doc.model_dump(by_alias=True, exclude={"id"})
        new_node_id_str = await save_node(node_data)
        if not new_node_id_str:
            raise HTTPException(status_code=500, detail="Failed to save node or retrieve ID.")
        created_node_doc = await get_node(node_doc.node_id)
        if not created_node_doc:
            raise HTTPException(status_code=500, detail=f"Node created with logical ID {node_doc.node_id} (DB ID {new_node_id_str}) but could not be retrieved.")
        return created_node_doc
    except Exception as e:
        print(f"Error creating node document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/{node_id}", response_model=NodeDocument, summary="Get a node by its ID")
async def read_node(node_id: str):
    node = await get_node(node_id)
    if node:
        return node
    raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

@router.put("/nodes/{node_id}/output", summary="Update a node's output field")
async def update_node_output_field(node_id: str, output_data: Dict[str, Any]):
    if "output" not in output_data:
        raise HTTPException(status_code=400, detail="Request body must contain 'output' field.")
    try:
        success = await update_node_output(node_id, output_data["output"])
        if success:
            updated_node_doc = await get_node(node_id)
            if updated_node_doc:
                return updated_node_doc
            else:
                raise HTTPException(status_code=404, detail=f"Node '{node_id}' updated but could not be retrieved.")
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found for update or no changes made.")
    except Exception as e:
        print(f"Error updating node output for {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/nodes/{node_id}", status_code=200, summary="Delete a node by its ID")
async def remove_node(node_id: str):
    deleted_count = await delete_node(node_id)
    if deleted_count > 0:
        return {"message": f"Node '{node_id}' deleted successfully."}
    raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found or already deleted.")

@router.get("/nodes/", response_model=List[NodeDocument], summary="List all nodes")
async def list_nodes(q: Optional[str] = None):
    nodes = await get_all_nodes()
    return nodes

@router.post("/chains/legacy_create", status_code=201, summary="Legacy: Create chain with dictionary", deprecated=True)
async def create_chain_legacy(chain_data: Dict[str, Any]):
    try:
        created_chain = await save_chain(chain_data)
        return created_chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legacy chain creation: {str(e)}")

@router.post("/chains/", response_model=ChainDocument, status_code=201, summary="Create a new chain document")
async def create_chain_document(chain_doc: Union[ChainDocument, Dict[str, Any]]):
    try:
        if isinstance(chain_doc, dict):
            chain_data = chain_doc
        else:
            chain_data = chain_doc.model_dump(by_alias=True, exclude={"id"})
        
        new_chain_id = await save_chain(chain_data)
        if not new_chain_id:
            raise HTTPException(status_code=500, detail="Failed to save chain or retrieve ID.")
        
        created_chain_doc = await get_chain(new_chain_id)
        if not created_chain_doc:
            raise HTTPException(status_code=500, detail=f"Chain created with ID {new_chain_id} but could not be retrieved.")
        return created_chain_doc
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid chain data: {str(ve)}")
    except Exception as e:
        print(f"Error creating chain document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chains/{chain_id}", response_model=ChainDocument, summary="Get a chain by its ID")
async def read_chain(chain_id: str):
    chain = await get_chain(chain_id)
    if chain:
        return chain
    raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")

@router.get("/chains/", response_model=List[ChainDocument], summary="List all chains")
async def list_chains():
    return await get_all_chains()

@router.put("/chains/{chain_id}", response_model=ChainDocument, summary="Update an existing chain")
async def update_chain_api(chain_id: str, chain_data: ChainDocument):
    if chain_data.id and str(chain_data.id) != chain_id:
        raise HTTPException(
            status_code=400, 
            detail=f"Chain ID in path ({chain_id}) does not match ID in body ({chain_data.id})"
        )
    
    success = await update_chain(chain_id, chain_data)
    if not success:
        existing_chain = await get_chain(chain_id)
        if not existing_chain:
            raise HTTPException(status_code=404, detail=f"Chain with ID '{chain_id}' not found.")
        else:
            raise HTTPException(status_code=304, detail=f"Chain with ID '{chain_id}' was not modified. The provided data might be identical or an update error occurred.")

    updated_chain_doc = await get_chain(chain_id)
    if not updated_chain_doc:
        raise HTTPException(status_code=500, detail=f"Chain with ID '{chain_id}' was updated but could not be retrieved.")
    return updated_chain_doc

@router.put("/nodes/{node_id}/name", summary="Update a node's name")
async def update_node_name_api(node_id: str, update_data: NodeNameUpdate):
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    if not update_data.name or not update_data.name.strip():
            raise HTTPException(status_code=400, detail="Node name cannot be empty.")

    if node_doc.name == update_data.name:
        return {"message": f"Node {node_id} name is already '{update_data.name}'. No change made.", "node": node_doc}

    success = await update_node_name(node_id, update_data.name)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update node name for ID '{node_id}'. An unexpected error occurred.")
    
    updated_node_doc = await get_node(node_id)
    return {"message": f"Node {node_id} name updated to {update_data.name}", "node": updated_node_doc}

@router.put("/nodes/{node_id}/prompt", response_model=NodeDocument, summary="Update a node's prompt field")
async def update_node_prompt_api(node_id: str, payload: NodePromptUpdate):
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    if node_doc.prompt == payload.prompt:
        return node_doc

    success = await update_node_prompt(node_id, payload.prompt)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update node prompt for '{node_id}'. Update was not applied.")
    
    updated_node_doc = await get_node(node_id)
    return updated_node_doc

@router.put("/nodes/{node_id}/llm_config", response_model=NodeDocument, summary="Update a node's LLM config field")
async def update_node_llm_config_api(node_id: str, payload: NodeLLMConfigUpdate):
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    if node_doc.llm_config and node_doc.llm_config.model_dump() == payload.llm_config.model_dump():
        return node_doc
    if not node_doc.llm_config and not payload.llm_config:
        return node_doc

    success = await update_node_llm_config(node_id, payload.llm_config)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update node LLM config for '{node_id}'. Update was not applied.")

    updated_node_doc = await get_node(node_id)
    return updated_node_doc 