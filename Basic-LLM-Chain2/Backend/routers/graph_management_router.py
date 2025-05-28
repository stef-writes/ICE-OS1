from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional

from llm import LLMConfig, default_llm_config # Assuming these are accessible
from models import NodeInput, EdgeInput, GenerateTextNodeRequest, GenerateTextNodeResponse # Assuming these are accessible
from scriptchain import get_script_chain, Node # Assuming these are accessible
from templates import template_processor # Assuming template_processor is accessible
from database import save_node, update_node_output # Assuming these are accessible
import networkx as nx

router = APIRouter()

@router.post("/add_node", status_code=201)
async def add_node_api(node: NodeInput, session_id: str = "default"):
    """Adds a node to the script chain via API and saves it to the database."""
    
    if not node.node_id or not node.node_id.strip():
        raise HTTPException(status_code=400, detail="node_id cannot be empty or whitespace")
    
    internal_node_type = "text_generation"
    if node.node_type == "llmNode":
        pass
    elif node.node_type == "text_generation":
        pass
    else:
        raise HTTPException(status_code=400, detail=f"node_type '{node.node_type}' not supported. Only 'llmNode' or 'text_generation' is currently supported from the frontend.")
    
    script_chain = get_script_chain(session_id)
    llm_config_for_node = default_llm_config
    if node.llm_config:
        llm_config_for_node = LLMConfig(
            provider=node.llm_config.provider,
            model=node.llm_config.model,
            temperature=node.llm_config.temperature,
            max_tokens=node.llm_config.max_tokens
        )
    try:
        script_chain.add_node(
            node_id=node.node_id,
            node_type=internal_node_type,
            input_keys=node.input_keys,
            output_keys=node.output_keys,
            model_config=llm_config_for_node
        )
        print(f"Added node: {node.node_id} (type: {internal_node_type}) to in-memory chain for session {session_id}")
        
        try:
            node_db_data = {
                "node_id": node.node_id,
                "name": node.name if node.name else node.node_id,
                "node_type": internal_node_type,
                "input_keys": node.input_keys,
                "output_keys": node.output_keys,
                "llm_config": node.llm_config.model_dump() if node.llm_config else None,
                "output": None,
            }
            
            await save_node(node_db_data)
            print(f"Saved node: {node.node_id} (type: {internal_node_type}) to database with name '{node_db_data['name']}'.")
            return {"message": f"Node '{node.node_id}' added successfully to chain and database."}
        except Exception as db_e:
            print(f"Error saving node {node.node_id} to database: {db_e}")
            raise HTTPException(status_code=500, detail=f"Node added to chain but failed to save to database: {str(db_e)}")
            
    except Exception as chain_e:
        print(f"Error adding node {node.node_id} to chain: {chain_e}")
        raise HTTPException(status_code=400, detail=f"Failed to add node to chain: {str(chain_e)}")

@router.post("/add_edge", status_code=201)
async def add_edge_api(edge: EdgeInput, session_id: str = "default"):
    """Adds an edge to the script chain via API."""
    script_chain = get_script_chain(session_id)
    if edge.from_node not in script_chain.graph or edge.to_node not in script_chain.graph:
        raise HTTPException(status_code=404, detail=f"Node(s) not found: '{edge.from_node}' or '{edge.to_node}'")
    
    script_chain.graph.add_edge(edge.from_node, edge.to_node)
    try:
        if not nx.is_directed_acyclic_graph(script_chain.graph):
            script_chain.graph.remove_edge(edge.from_node, edge.to_node)
            raise HTTPException(status_code=400, detail="Adding this edge would create a cycle. Please check your node connections.")
    except ImportError:
        script_chain.graph.remove_edge(edge.from_node, edge.to_node)
        script_chain.add_edge(edge.from_node, edge.to_node)
    
    print(f"Added edge: {edge.from_node} -> {edge.to_node} for session {session_id}")
    return {"message": f"Edge from '{edge.from_node}' to '{edge.to_node}' added successfully."}

@router.post("/generate_text_node", response_model=GenerateTextNodeResponse)
async def generate_text_node_api(request: GenerateTextNodeRequest, session_id: str = "default"):
    """Executes a single text generation call based on provided prompt text."""
    script_chain = get_script_chain(session_id)
    
    try:
        print(f"\n=== RAW generate_text_node_api Request (Session: {session_id}) ===")
        print(f"Prompt Text: {request.prompt_text}")
        print(f"Context Data Received: {request.context_data}")
        print(f"LLM Config Received: {request.llm_config}")
        print("=== END RAW Request ===\n")
    except Exception as log_e:
        print(f"Error logging raw request: {log_e}")

    node_mapping = request.context_data.get('__node_mapping', {}) if request.context_data else {}
    current_node_id = None
    
    if request.context_data and '__current_node' in request.context_data:
        current_node_id = request.context_data['__current_node']
        print(f"Explicit current node ID: {current_node_id}")
        
    if current_node_id is None and request.context_data and 'node_id' in request.context_data:
        current_node_id = request.context_data['node_id']
        print(f"Found node_id in context: {current_node_id}")
    
    nodes_with_updated_content = []
    if node_mapping:
        for node_name, node_id in node_mapping.items():
            if node_id == current_node_id:
                continue
                
            if request.context_data and node_name in request.context_data:
                provided_content = request.context_data[node_name]
                stored_content = None
                
                if script_chain.storage.has_node(node_id):
                    node_data = script_chain.storage.get_node_output(node_id)
                    if node_data:
                        first_key = next(iter(node_data))
                        stored_content = node_data.get(first_key)
                
                if provided_content != stored_content:
                    print(f"Node {node_name} (ID: {node_id}) content has changed:")
                    print(f"  Old: {stored_content}")
                    print(f"  New: {provided_content}")
                    
                    script_chain.storage.store(node_id, {"generated_text": provided_content})
                    # script_chain.increment_node_version(node_id) # This method is on ScriptChain, not directly here
                    # Need to call it on the script_chain instance
                    script_chain_instance = get_script_chain(session_id) # Or pass it around
                    script_chain_instance._update_node_execution_record(node_id)

                    nodes_with_updated_content.append(node_id)
    
    if nodes_with_updated_content:
        print(f"Updated content for nodes: {nodes_with_updated_content}")
        
        for dependent_node_id, dependencies in script_chain.node_dependencies.items():
            for updated_node_id in nodes_with_updated_content:
                if updated_node_id in dependencies:
                    print(f"Marking node {dependent_node_id} for refresh due to dependency changes")
                    if script_chain.storage.has_node(dependent_node_id):
                        script_chain.storage.data[dependent_node_id] = {}

    processed_prompt, processed_node_values = template_processor.process_node_references(
        request.prompt_text,
        request.context_data or {}
    )
    
    node_config = default_llm_config
    if request.llm_config:
        node_config = LLMConfig(
            provider=request.llm_config.provider,
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens
        )
    
    print(f"Using model config: provider={node_config.provider}, model={node_config.model}, temp={node_config.temperature}, max_tokens={node_config.max_tokens}")
    
    context_content = "You are an expert AI assistant helping with data analysis and text generation."
    
    if request.context_data and '__node_mapping' in request.context_data:
        mapping_info = request.context_data['__node_mapping']
        context_content += f"\n\nYou have access to a graph of connected nodes with the following name-to-ID mapping: {mapping_info}"
    
    if request.context_data:
        context_keys = [k for k in request.context_data.keys() if k not in ['__node_mapping', '__current_node', 'node_id']]
        if context_keys:
            context_content += f"\n\nYou have access to information from the following nodes: {', '.join(context_keys)}."
            context_content += "\nUse this information to inform your response."

    temp_node = Node(
        node_id=current_node_id or "temp_generation",
        node_type="text_generation",
        model_config=node_config
    )

    node_inputs = {
        "context": context_content,
        "query": processed_prompt
    }

    try:
        print(f"=== Delegating to Node.process() ===")
        result = await temp_node.process(node_inputs)
        
        if not result or "generated_text" not in result:
            raise ValueError("Node.process() did not return expected 'generated_text' result")
        
        response_content = result["generated_text"]
        print(f"\n=== RESPONSE CONTENT ===\n{response_content}\n=== END RESPONSE CONTENT ===\n")
        
        if current_node_id:
            result_data = {
                "generated_text": response_content,
                "output": response_content,
                "content": response_content
            }
            script_chain.storage.store(current_node_id, result_data)
            # script_chain.increment_node_version(current_node_id)
            script_chain_instance = get_script_chain(session_id)
            script_chain_instance._update_node_execution_record(current_node_id)

            print(f"Updated node {current_node_id} with new content and incremented version")

            try:
                await update_node_output(current_node_id, response_content)
                print(f"Successfully saved output for node {current_node_id} to database.")
            except Exception as db_save_e:
                print(f"Error saving output for node {current_node_id} to database: {db_save_e}")

        token_usage = temp_node.token_usage
        
        return GenerateTextNodeResponse(
            generated_text=response_content,
            prompt_tokens=getattr(token_usage, 'prompt_tokens', None) if token_usage else None,
            completion_tokens=getattr(token_usage, 'completion_tokens', None) if token_usage else None,
            total_tokens=getattr(token_usage, 'total_tokens', None) if token_usage else None,
            cost=getattr(token_usage, 'cost', None) if token_usage else None,
            duration=round(token_usage.end_time - token_usage.start_time, 2) if token_usage and hasattr(token_usage, 'end_time') and hasattr(token_usage, 'start_time') else None
        )

    except Exception as e:
        print(f"Error during text generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed text generation: {str(e)}")

@router.post("/get_node_outputs")
async def get_node_outputs(request: Dict[str, List[str]], session_id: str = "default"):
    """Retrieves the current output values for specified nodes from the script chain."""
    script_chain = get_script_chain(session_id)
    try:
        node_ids = request.get("node_ids", [])
        if not node_ids:
            return {}
            
        result = {}
        print(f"Fetching outputs for nodes: {node_ids} (Session: {session_id})")
        
        print(f"Current storage state for session {session_id}:")
        for node_id, data in script_chain.storage.data.items():
            print(f"  Node {node_id}: {data}")
        
        for node_id in node_ids:
            if script_chain.storage.has_node(node_id):
                node_data = script_chain.storage.get_node_output(node_id)
                print(f"Node {node_id} data: {node_data}")
                
                if node_data:
                    output_keys = ["generated_text", "output", "content", "result"]
                    
                    found_output = False
                    for key in output_keys:
                        if key in node_data:
                            result[node_id] = node_data[key]
                            found_output = True
                            print(f"Node {node_id}: found output under key '{key}'")
                            break
                    
                    if not found_output and node_data:
                        first_key = next(iter(node_data))
                        result[node_id] = node_data[first_key]
                        print(f"Node {node_id}: used first available key '{first_key}'")
                    
                    if result.get(node_id) is None:
                        result[node_id] = ""
                else:
                    print(f"Node {node_id} exists but has no data")
            else:
                print(f"Node {node_id} not found in storage")
                    
        print(f"Returning outputs for {len(result)} nodes: {result}")
        return result
    except Exception as e:
        print(f"Error retrieving node outputs: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve node outputs: {str(e)}") 