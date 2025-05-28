from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional

from scriptchain import get_script_chain, ScriptChain # Assuming these are accessible
from database import get_chain, get_node # Assuming these are accessible
from llm import LLMConfig # Assuming this is accessible

router = APIRouter()

@router.post("/execute")
async def execute_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain with concurrent execution (default behavior)."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            # script_chain.storage.data[key] = value # Direct access removed
            script_chain.storage.store("initial_inputs", {key: value}) # Store under a generic node_id or handle differently
        print(f"Initial storage set with inputs: {initial_inputs}")

    try:
        results = await script_chain.execute_concurrent()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")

@router.post("/execute_concurrent")
async def execute_concurrent_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain with concurrent execution for better performance."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute_concurrent request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            # script_chain.storage.data[key] = value
            script_chain.storage.store("initial_inputs", {key: value}) 
        print(f"Initial storage set with inputs: {initial_inputs}")

    try:
        results = await script_chain.execute_concurrent()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during concurrent chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Concurrent chain execution failed: {str(e)}")

@router.post("/execute_sequential")
async def execute_sequential_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain sequentially (original behavior)."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute_sequential request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            # script_chain.storage.data[key] = value
            script_chain.storage.store("initial_inputs", {key: value})
        print(f"Initial storage set with inputs: {initial_inputs}")

    try:
        results = await script_chain.execute()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during sequential chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Sequential chain execution failed: {str(e)}")

@router.post("/execute_chain/{chain_id}")
async def execute_chain_by_id(chain_id: str, execution_mode: str = "sequential", initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database by its ID."""
    try:
        chain_doc = await get_chain(chain_id)
        if not chain_doc:
            raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")
        
        temp_script_chain = ScriptChain()
        
        for flow_node in chain_doc.flow_nodes:
            node_doc = await get_node(flow_node.node_id)
            if node_doc:
                llm_config = None
                if flow_node.llm_config:
                    llm_config = LLMConfig(
                        provider=flow_node.llm_config.provider,
                        model=flow_node.llm_config.model,
                        temperature=flow_node.llm_config.temperature,
                        max_tokens=flow_node.llm_config.max_tokens
                    )
                
                temp_script_chain.add_node(
                    flow_node.node_id,
                    flow_node.node_type,
                    node_doc.input_keys,
                    node_doc.output_keys,
                    llm_config
                )
        
        for edge in chain_doc.edges:
            temp_script_chain.add_edge(edge["from_node"], edge["to_node"])
        
        if initial_inputs:
            for key, value in initial_inputs.items():
                # temp_script_chain.storage.data[key] = value
                temp_script_chain.storage.store("initial_inputs", {key: value})
        
        if execution_mode == "concurrent":
            results = await temp_script_chain.execute_concurrent()
        else:
            results = await temp_script_chain.execute()
        
        if results and "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error executing chain {chain_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")

@router.post("/execute_chain_sequential/{chain_id}")
async def execute_chain_sequential_by_id(chain_id: str, initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database sequentially."""
    return await execute_chain_by_id(chain_id, "sequential", initial_inputs)

@router.post("/execute_chain_concurrent/{chain_id}")
async def execute_chain_concurrent_by_id(chain_id: str, initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database concurrently."""
    return await execute_chain_by_id(chain_id, "concurrent", initial_inputs) 