import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Tuple, Union

# Import modular components
from llm import LLMConfig, default_llm_config, track_token_usage, get_client
from models import (
    Message, ModelConfigInput, NodeInput, EdgeInput, GenerateTextNodeRequest, 
    GenerateTextNodeResponse, NodeNameUpdate, TemplateValidationRequest, TemplateValidationResponse,
    NodePromptUpdate, NodeLLMConfigUpdate,
    NodeDocument, ChainDocument, LLMConfigDocument
)
from script_chain import ScriptChain, Node, get_script_chain
from callbacks import Callback, LoggingCallback
from templates import TemplateProcessor, template_processor
from utils import ContentParser, DataAccessor
from database import (
    save_node, 
    get_node, 
    save_chain, 
    get_chain, 
    get_all_chains, 
    get_all_nodes, 
    delete_node, # Renamed in database.py to avoid conflict, but main.py can use simple name
    update_node_name, 
    update_node_output, 
    update_chain, # Renamed from update_chain_data
    update_node_prompt,
    update_node_llm_config
    # Removed add_node_to_db, add_edge_to_db as they don't exist
    # Removed duplicate get_node, get_all_nodes, etc.
)

# Create FastAPI app
app = FastAPI()

# Configure CORS
# origins = [...] # Commenting out the static list
# Define a regex for allowed origins (localhost and 127.0.0.1 with any port)
allow_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allow_origin_regex, # Use regex instead of static list
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"], # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# --- API Routes ---
@app.get("/")
def read_root():
    return {"message": "ScriptChain Backend Running"}

@app.post("/add_node", status_code=201)
async def add_node_api(node: NodeInput, session_id: str = "default"):
    """Adds a node to the script chain via API and saves it to the database."""
    
    # Validate node_id
    if not node.node_id or not node.node_id.strip():
        raise HTTPException(status_code=400, detail="node_id cannot be empty or whitespace")
    
    # Map frontend node type 'llmNode' to backend 'text_generation'
    # For now, this is the only type we expect from the frontend.
    internal_node_type = "text_generation" # Default internal type
    if node.node_type == "llmNode":
        pass # Correctly mapped
    elif node.node_type == "text_generation":
        pass # Also acceptable if frontend sends it directly
    else:
        # If other types were intended to be supported directly from frontend, add them here
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
            node_type=internal_node_type, # Use the mapped internal type
            input_keys=node.input_keys,
            output_keys=node.output_keys,
            model_config=llm_config_for_node
        )
        print(f"Added node: {node.node_id} (type: {internal_node_type}) to in-memory chain for session {session_id}")
        
        # --- Add node to the database --- 
        try:
            node_db_data = {
                "node_id": node.node_id,
                "name": node.name if node.name else node.node_id,
                "node_type": internal_node_type, # Store the mapped internal type in DB
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

@app.post("/add_edge", status_code=201)
async def add_edge_api(edge: EdgeInput, session_id: str = "default"):
    """Adds an edge to the script chain via API."""
    script_chain = get_script_chain(session_id)
    # Basic validation: Check if nodes exist before adding edge
    if edge.from_node not in script_chain.graph or edge.to_node not in script_chain.graph:
        raise HTTPException(status_code=404, detail=f"Node(s) not found: '{edge.from_node}' or '{edge.to_node}'")
    
    # --- CYCLE PREVENTION ---
    # Temporarily add the edge and check for cycles
    script_chain.graph.add_edge(edge.from_node, edge.to_node)
    try:
        import networkx as nx
        if not nx.is_directed_acyclic_graph(script_chain.graph):
            script_chain.graph.remove_edge(edge.from_node, edge.to_node)
            raise HTTPException(status_code=400, detail="Adding this edge would create a cycle. Please check your node connections.")
    except ImportError:
        # Handle case where networkx is not imported in this file
        script_chain.graph.remove_edge(edge.from_node, edge.to_node)
        script_chain.add_edge(edge.from_node, edge.to_node)
    
    print(f"Added edge: {edge.from_node} -> {edge.to_node} for session {session_id}")
    return {"message": f"Edge from '{edge.from_node}' to '{edge.to_node}' added successfully."}

# --- Generate Text Node API Route ---
@app.post("/generate_text_node", response_model=GenerateTextNodeResponse)
async def generate_text_node_api(request: GenerateTextNodeRequest, session_id: str = "default"):
    """Executes a single text generation call based on provided prompt text."""
    script_chain = get_script_chain(session_id)
    
    # --- Log the raw incoming request data --- 
    try:
        print(f"\n=== RAW generate_text_node_api Request (Session: {session_id}) ===")
        print(f"Prompt Text: {request.prompt_text}")
        print(f"Context Data Received: {request.context_data}")
        print(f"LLM Config Received: {request.llm_config}")
        print("=== END RAW Request ===\n")
    except Exception as log_e:
        print(f"Error logging raw request: {log_e}")

    # Extract node mapping information from context data
    node_mapping = request.context_data.get('__node_mapping', {}) if request.context_data else {}
    current_node_id = None  # The ID of the node being executed
    
    # Identify the current node (if this is a node execution)
    if request.context_data and '__current_node' in request.context_data:
        current_node_id = request.context_data['__current_node']
        print(f"Explicit current node ID: {current_node_id}")
        
    # Detect if this is a simple node output update request
    if current_node_id is None and request.context_data and 'node_id' in request.context_data:
        current_node_id = request.context_data['node_id']
        print(f"Found node_id in context: {current_node_id}")
    
    # Handle node content updates from context data
    nodes_with_updated_content = []
    if node_mapping:
        for node_name, node_id in node_mapping.items():
            # Skip the node that is currently being executed
            if node_id == current_node_id:
                continue
                
            # Compare provided content with stored content
            if request.context_data and node_name in request.context_data:
                provided_content = request.context_data[node_name]
                stored_content = None
                
                # Get the stored content if the node exists in storage
                if script_chain.storage.has_node(node_id):
                    node_data = script_chain.storage.get_node_output(node_id)
                    if node_data:
                        first_key = next(iter(node_data))
                        stored_content = node_data.get(first_key)
                
                # If content has changed, update the storage
                if provided_content != stored_content:
                    print(f"Node {node_name} (ID: {node_id}) content has changed:")
                    print(f"  Old: {stored_content}")
                    print(f"  New: {provided_content}")
                    
                    # Update the storage directly
                    script_chain.storage.store(node_id, {"generated_text": provided_content})
                    script_chain.increment_node_version(node_id)
                    nodes_with_updated_content.append(node_id)
    
    if nodes_with_updated_content:
        print(f"Updated content for nodes: {nodes_with_updated_content}")
        
        # Clear dependent nodes that need refresh
        for dependent_node_id, dependencies in script_chain.node_dependencies.items():
            for updated_node_id in nodes_with_updated_content:
                if updated_node_id in dependencies:
                    print(f"Marking node {dependent_node_id} for refresh due to dependency changes")
                    if script_chain.storage.has_node(dependent_node_id):
                        script_chain.storage.data[dependent_node_id] = {}

    # --- Process the template ---
    processed_prompt, processed_node_values = template_processor.process_node_references(
        request.prompt_text,
        request.context_data or {}
    )
    
    # Configure model to use
    node_config = default_llm_config
    if request.llm_config:
        node_config = LLMConfig(
            provider=request.llm_config.provider,
            model=request.llm_config.model,
            temperature=request.llm_config.temperature,
            max_tokens=request.llm_config.max_tokens
        )
    
    print(f"Using model config: provider={node_config.provider}, model={node_config.model}, temp={node_config.temperature}, max_tokens={node_config.max_tokens}")
    
    # Build context for the Node.process() method
    # The Node expects 'context' and 'query' inputs for text_generation
    context_content = "You are an expert AI assistant helping with data analysis and text generation."
    
    # Add reference to mapping between node names and IDs if present
    if request.context_data and '__node_mapping' in request.context_data:
        mapping_info = request.context_data['__node_mapping']
        context_content += f"\n\nYou have access to a graph of connected nodes with the following name-to-ID mapping: {mapping_info}"
    
    # Add reference to all keys in context_data except for special system keys
    if request.context_data:
        context_keys = [k for k in request.context_data.keys() if k not in ['__node_mapping', '__current_node', 'node_id']]
        if context_keys:
            context_content += f"\n\nYou have access to information from the following nodes: {', '.join(context_keys)}."
            context_content += "\nUse this information to inform your response."

    # Create a temporary node for this generation request
    temp_node = Node(
        node_id=current_node_id or "temp_generation",
        node_type="text_generation",
        model_config=node_config
    )

    # Prepare inputs for the Node.process() method
    node_inputs = {
        "context": context_content,
        "query": processed_prompt
    }

    try:
        # Use the existing Node.process() logic
        print(f"=== Delegating to Node.process() ===")
        result = await temp_node.process(node_inputs)
        
        if not result or "generated_text" not in result:
            raise ValueError("Node.process() did not return expected 'generated_text' result")
        
        response_content = result["generated_text"]
        print(f"\n=== RESPONSE CONTENT ===\n{response_content}\n=== END RESPONSE CONTENT ===\n")
        
        # If we identified the current node, store the result and increment version
        if current_node_id:
            # Store the result in the global chain's storage
            result_data = {
                "generated_text": response_content,
                "output": response_content,
                "content": response_content
            }
            script_chain.storage.store(current_node_id, result_data)
            script_chain.increment_node_version(current_node_id)
            print(f"Updated node {current_node_id} with new content and incremented version")

            # Persist the output to the database
            try:
                await update_node_output(current_node_id, response_content)
                print(f"Successfully saved output for node {current_node_id} to database.")
            except Exception as db_save_e:
                print(f"Error saving output for node {current_node_id} to database: {db_save_e}")

        # Extract token usage information from the node
        token_usage = temp_node.token_usage
        
        # Prepare the response using the Pydantic model
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

@app.post("/get_node_outputs")
async def get_node_outputs(request: Dict[str, List[str]], session_id: str = "default"):
    """Retrieves the current output values for specified nodes from the script chain."""
    script_chain = get_script_chain(session_id)
    try:
        node_ids = request.get("node_ids", [])
        if not node_ids:
            return {}
            
        result = {}
        print(f"Fetching outputs for nodes: {node_ids} (Session: {session_id})")
        
        # For debugging: show all storage
        print(f"Current storage state for session {session_id}:")
        for node_id, data in script_chain.storage.data.items():
            print(f"  Node {node_id}: {data}")
        
        for node_id in node_ids:
            if script_chain.storage.has_node(node_id):
                node_data = script_chain.storage.get_node_output(node_id)
                print(f"Node {node_id} data: {node_data}")
                
                if node_data:
                    # Try several output key patterns in priority order
                    # This makes the endpoint more robust to different node types
                    output_keys = ["generated_text", "output", "content", "result"]
                    
                    # First try the standard output keys
                    found_output = False
                    for key in output_keys:
                        if key in node_data:
                            result[node_id] = node_data[key]
                            found_output = True
                            print(f"Node {node_id}: found output under key '{key}'")
                            break
                    
                    # If no standard key found, use the first available key
                    if not found_output and node_data:
                        first_key = next(iter(node_data))
                        result[node_id] = node_data[first_key]
                        print(f"Node {node_id}: used first available key '{first_key}'")
                    
                    # Convert None to empty string for consistency
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

@app.post("/execute")
async def execute_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain with concurrent execution (default behavior)."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            script_chain.storage.data[key] = value
        print(f"Initial storage set to: {script_chain.storage.data}")

    try:
        # Use concurrent execution by default for better performance
        results = await script_chain.execute_concurrent()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")

@app.post("/execute_concurrent")
async def execute_concurrent_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain with concurrent execution for better performance."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute_concurrent request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            script_chain.storage.data[key] = value
        print(f"Initial storage set to: {script_chain.storage.data}")

    try:
        results = await script_chain.execute_concurrent()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during concurrent chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Concurrent chain execution failed: {str(e)}")

@app.post("/execute_sequential")
async def execute_sequential_api(initial_inputs: Optional[Dict[str, Any]] = None, session_id: str = "default"):
    """Executes the AI-driven node chain sequentially (original behavior)."""
    script_chain = get_script_chain(session_id)
    print(f"--- Received /execute_sequential request (Session: {session_id}) ---")
    if initial_inputs:
        for key, value in initial_inputs.items():
            script_chain.storage.data[key] = value
        print(f"Initial storage set to: {script_chain.storage.data}")

    try:
        # Temporarily use the original execute method for sequential execution
        results = await script_chain.execute()
        if results and "error" in results:
             raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error during sequential chain execution: {e}")
        raise HTTPException(status_code=500, detail=f"Sequential chain execution failed: {str(e)}")

@app.post("/execute_chain/{chain_id}")
async def execute_chain_by_id(chain_id: str, execution_mode: str = "sequential", initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database by its ID."""
    try:
        # Get the chain from database
        chain_doc = await get_chain(chain_id)
        if not chain_doc:
            raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")
        
        # Create a temporary script chain and populate it with the chain's nodes
        temp_script_chain = ScriptChain()
        
        # Add nodes from the chain to the script chain
        for flow_node in chain_doc.flow_nodes:
            # Get the full node data from database
            node_doc = await get_node(flow_node.node_id)
            if node_doc:
                # Convert LLMConfigDocument to LLMConfig
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
        
        # Add edges from the chain
        for edge in chain_doc.edges:
            temp_script_chain.add_edge(edge["from_node"], edge["to_node"])
        
        # Set initial inputs if provided
        if initial_inputs:
            for key, value in initial_inputs.items():
                temp_script_chain.storage.data[key] = value
        
        # Execute the chain
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

@app.post("/execute_chain_sequential/{chain_id}")
async def execute_chain_sequential_by_id(chain_id: str, initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database sequentially."""
    return await execute_chain_by_id(chain_id, "sequential", initial_inputs)

@app.post("/execute_chain_concurrent/{chain_id}")
async def execute_chain_concurrent_by_id(chain_id: str, initial_inputs: Optional[Dict[str, Any]] = None):
    """Execute a specific chain from the database concurrently."""
    return await execute_chain_by_id(chain_id, "concurrent", initial_inputs)

# --- Debug Endpoints ---
@app.get("/debug/node_content")
async def debug_node_content(node_content: str):
    """Debug endpoint to test content parsing functionality."""
    result = {
        "original_content": node_content,
        "analysis": {},
        "parsed_data": {}
    }
    
    # Analyze using ContentParser
    parser = ContentParser()
    numbered_items = parser.parse_numbered_list(node_content)
    json_data = parser.try_parse_json(node_content)
    table_data = parser.extract_table(node_content)
    
    # Build analysis
    result["analysis"] = {
        "has_numbered_list": bool(numbered_items),
        "numbered_items_count": len(numbered_items) if numbered_items else 0,
        "has_json": json_data is not None,
        "has_table": table_data is not None
    }
    
    # Add parsed data
    result["parsed_data"] = {
        "numbered_items": numbered_items,
        "json_data": json_data,
        "table_data": table_data
    }
    
    return result

@app.post("/debug/process_template")
async def debug_process_template(request: dict):
    """Debug endpoint to test template processing directly."""
    if "prompt" not in request or "context_data" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'prompt' and 'context_data' fields")
    
    prompt = request["prompt"]
    context_data = request["context_data"]
    
    print(f"Debug process template request:")
    print(f"Prompt: {prompt}")
    print(f"Context data: {context_data}")
    
    # Process the template using our unified processor
    processed_prompt, processed_node_values = template_processor.process_node_references(
        prompt, context_data
    )
    
    # Return detailed results for debugging
    return {
        "original_prompt": prompt,
        "context_data": context_data,
        "processed_prompt": processed_prompt,
        "processed_node_values": processed_node_values,
        "validation": {
            "is_valid": len(template_processor.validate_node_references(prompt, context_data.keys())[1]) == 0,
            "missing_nodes": template_processor.validate_node_references(prompt, context_data.keys())[1],
            "found_nodes": template_processor.validate_node_references(prompt, context_data.keys())[2],
        }
    }

@app.post("/validate_template", response_model=TemplateValidationResponse)
async def validate_template_api(request: TemplateValidationRequest):
    """Validates that all node references in a template exist in the available nodes."""
    is_valid, missing_nodes, found_nodes = template_processor.validate_node_references(
        request.prompt_text, set(request.available_nodes)
    )
    
    warnings = []
    if not is_valid:
        for node in missing_nodes:
            warnings.append(f"Node reference '{node}' not found in available nodes.")
    
    return TemplateValidationResponse(
        is_valid=is_valid,
        missing_nodes=missing_nodes,
        found_nodes=found_nodes,
        warnings=warnings
    )

@app.post("/debug/test_reference")
async def debug_test_reference(request: dict):
    """Test endpoint for reference extraction."""
    if "content" not in request or "reference" not in request:
        raise HTTPException(status_code=400, detail="Request must include 'content' and 'reference' fields")
    
    content = request["content"]
    reference = request["reference"]
    
    # Create fake context with Node1 containing the content
    fake_context = {"Node1": content}
    data_accessor = DataAccessor(fake_context)
    
    # Try to parse the reference
    result = {
        "original_content": content,
        "reference": reference,
        "parsed_result": None,
        "details": {}
    }
    
    # Check if it's an item reference
    import re
    item_ref_pattern = r'\{([^:\}\[]+)(?:\[(\d+)\]|\:item\((\d+)\))\}'
    match = re.search(item_ref_pattern, reference)
    
    if match:
        node_name = match.group(1)
        item_num_str = match.group(2) or match.group(3)
        
        result["details"] = {
            "node_name": node_name,
            "item_num": item_num_str,
            "is_valid_node": data_accessor.has_node(node_name)
        }
        
        try:
            item_num = int(item_num_str)
            result["details"]["valid_item_num"] = True
            
            # Get the specific item
            item_content = data_accessor.get_item(node_name, item_num)
            result["parsed_result"] = item_content
            
        except ValueError:
            result["details"]["valid_item_num"] = False
    else:
        result["details"]["is_reference_pattern"] = False
    
    return result

# --- New Database CRUD Operations using Pydantic Models ---

# Original simple /nodes/ POST, marked for review or specific use
@app.post("/nodes/legacy_create", status_code=201, summary="Legacy: Create node with minimal data", deprecated=True)
async def create_node_legacy(node_id: str, output: Any):
    """
    Legacy endpoint to quickly save a node's output.
    Consider using the full POST /nodes/ endpoint with NodeDocument.
    """
    # This will likely fail if NodeDocument requires more fields (name, node_type) not provided here
    # and not defaulted by save_node for new entries.
    # For save_node to work, it needs a dict that can be parsed by NodeDocument.
    # Minimal example assuming save_node can handle very partial data for existing nodes or specific cases:
    # For a *new* node, this needs 'name' and 'node_type'.
    # This endpoint is problematic as is for creating new nodes with current save_node.
    # Placeholder for potential behavior or if save_node is adapted.
    # For now, this will likely cause an error if save_node tries to create a new NodeDocument.
    # It's better to create a full NodeDocument.
    # Let's assume this is for updating an existing node's output or a specific scenario.
    # To make it work for creation, it would need more params like name, node_type.
    node_data_for_save = {
        "node_id": node_id, 
        "output": output,
        # Required fields for new NodeDocument:
        "name": f"Legacy Node {node_id}", # Defaulting name
        "node_type": "unknown" # Defaulting node_type
    }
    try:
        created_node = await save_node(node_data_for_save)
        return created_node
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legacy node creation: {str(e)}")

@app.post("/nodes/", response_model=NodeDocument, status_code=201, summary="Create a new node document")
async def create_node_document(node_doc: NodeDocument):
    """
    Creates a new node in the database using the NodeDocument schema.
    The `id`, `created_at`, and `updated_at` fields will be auto-managed.
    """
    try:
        # save_node expects a dict. model_dump(by_alias=True) is good for _id.
        # exclude_unset=True or exclude_none=True can be useful.
        # Since save_node handles _id removal for new docs, this should be fine.
        node_data = node_doc.model_dump(by_alias=True, exclude={"id"}) # Exclude 'id' as DB will generate '_id'
        
        # save_node now returns the string ID of the inserted document
        new_node_id_str = await save_node(node_data) 
        if not new_node_id_str:
            raise HTTPException(status_code=500, detail="Failed to save node or retrieve ID.")

        # Fetch the full node document to return, matching the response_model
        # get_node expects the node_id (the string one, not Mongo ObjectId string)
        # We need to be careful here: save_node returns the MongoDB _id as a string.
        # get_node expects the logical node_id (e.g., "node_curl_1").
        # The NodeDocument that was passed in (node_doc) already has the node_id we need.
        created_node_doc = await get_node(node_doc.node_id) 
        if not created_node_doc:
            raise HTTPException(status_code=500, detail=f"Node created with logical ID {node_doc.node_id} (DB ID {new_node_id_str}) but could not be retrieved.")
        
        return created_node_doc
    except Exception as e:
        print(f"Error creating node document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/nodes/{node_id}", response_model=NodeDocument, summary="Get a node by its ID")
async def read_node(node_id: str):
    node = await get_node(node_id)
    if node:
        return node
    raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

@app.put("/nodes/{node_id}/output", summary="Update a node's output field")
async def update_node_output_field(node_id: str, output_data: Dict[str, Any]):
    """
    Updates the 'output' field of a specific node.
    The request body should be a JSON object, e.g., `{"output": "new output value"}`.
    """
    if "output" not in output_data:
        raise HTTPException(status_code=400, detail="Request body must contain 'output' field.")
    try:
        # update_node expects node_id and a dict of fields to update
        success = await update_node_output(node_id, output_data["output"])
        if success:
            # Fetch the updated node to return it, as update_node_output returns boolean
            updated_node_doc = await get_node(node_id)
            if updated_node_doc:
                return updated_node_doc
            else:
                # This case should ideally not happen if update was successful
                raise HTTPException(status_code=404, detail=f"Node '{node_id}' updated but could not be retrieved.")
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found for update or no changes made.")
    except Exception as e:
        print(f"Error updating node output for {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/nodes/{node_id}", status_code=200, summary="Delete a node by its ID")
async def remove_node(node_id: str):
    deleted_count = await delete_node(node_id)
    if deleted_count > 0:
        return {"message": f"Node '{node_id}' deleted successfully."}
    raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found or already deleted.")

@app.get("/nodes/", response_model=List[NodeDocument], summary="List all nodes")
async def list_nodes(q: Optional[str] = None): # Add optional query parameter 'q'
    # 'q' parameter is not used by get_all_nodes yet, but kept for potential future filtering
    nodes = await get_all_nodes()
    return nodes

# Original simple /chains/ POST, marked for review
@app.post("/chains/legacy_create", status_code=201, summary="Legacy: Create chain with dictionary", deprecated=True)
async def create_chain_legacy(chain_data: Dict[str, Any]):
    """
    Legacy endpoint to create a chain from a dictionary.
    Consider using POST /chains/ with ChainDocument.
    """
    try:
        # save_chain expects a dict that can be parsed by ChainDocument
        created_chain = await save_chain(chain_data)
        return created_chain
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in legacy chain creation: {str(e)}")

@app.post("/chains/", response_model=ChainDocument, status_code=201, summary="Create a new chain document")
async def create_chain_document(chain_doc: Union[ChainDocument, Dict[str, Any]]):
    """
    Creates a new chain in the database using the ChainDocument schema.
    Accepts either ChainDocument or dictionary input.
    `id`, `created_at`, and `updated_at` fields will be auto-managed.
    """
    try:
        # Handle both ChainDocument and dictionary input
        if isinstance(chain_doc, dict):
            # Convert dictionary to ChainDocument
            chain_data = chain_doc
        else:
            # Already a ChainDocument, convert to dict
            chain_data = chain_doc.model_dump(by_alias=True, exclude={"id"}) # Exclude 'id'
        
        new_chain_id = await save_chain(chain_data)
        if not new_chain_id:
            raise HTTPException(status_code=500, detail="Failed to save chain or retrieve ID.")
        
        # Fetch the full chain document to return, matching the response_model
        created_chain_doc = await get_chain(new_chain_id)
        if not created_chain_doc:
            raise HTTPException(status_code=500, detail=f"Chain created with ID {new_chain_id} but could not be retrieved.")
        return created_chain_doc
    except ValueError as ve: # Catch Pydantic validation errors from ChainDocument(**chain_data) in save_chain
        raise HTTPException(status_code=400, detail=f"Invalid chain data: {str(ve)}")
    except Exception as e:
        print(f"Error creating chain document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chains/{chain_id}", response_model=ChainDocument, summary="Get a chain by its ID")
async def read_chain(chain_id: str):
    chain = await get_chain(chain_id)
    if chain:
        return chain
    raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")

@app.get("/chains/", response_model=List[ChainDocument], summary="List all chains")
async def list_chains():
    return await get_all_chains()

@app.put("/chains/{chain_id}", response_model=ChainDocument, summary="Update an existing chain")
async def update_chain_api(chain_id: str, chain_data: ChainDocument):
    """Update an existing chain's properties (name, nodes, edges)."""
    # Ensure the ID in the path matches the ID in the body, if provided in body
    if chain_data.id and str(chain_data.id) != chain_id:
        raise HTTPException(
            status_code=400, 
            detail=f"Chain ID in path ({chain_id}) does not match ID in body ({chain_data.id})"
        )
    
    # # Ensure the incoming chain_data has its id field set to the chain_id from the path
    # # This is important because update_chain uses the ID from the object for some internal logic if present,
    # # though it primarily relies on the passed chain_id_str for the DB query.
    # # However, the ChainDocument model expects 'id' to be PyObjectId or None.
    # # We'll rely on update_chain in database.py to handle the ObjectId conversion from chain_id string.

    # The chain_data's own 'id' field might be None or already set.
    # update_chain will use the chain_id path parameter for the query.
    # We pass the full chain_data which might contain an ID, but it's ignored by model_dump(exclude={'id', '_id'})
    # in the update_chain database function for the $set payload.
    
    success = await update_chain(chain_id, chain_data) 
    if not success:
        # Attempt to fetch to see if it simply wasn't found vs. update failed for other reasons
        existing_chain = await get_chain(chain_id)
        if not existing_chain:
            raise HTTPException(status_code=404, detail=f"Chain with ID '{chain_id}' not found.")
        else:
            # If it exists but wasn't modified, it could be due to no actual changes or an issue
            # For simplicity, we'll say it was not updated. Could also return 200 OK with original if no change.
            raise HTTPException(status_code=304, detail=f"Chain with ID '{chain_id}' was not modified. The provided data might be identical or an update error occurred.")

    # Fetch and return the updated chain document to confirm changes
    updated_chain_doc = await get_chain(chain_id)
    if not updated_chain_doc:
        # This case should ideally not happen if success is true
        raise HTTPException(status_code=500, detail=f"Chain with ID '{chain_id}' was updated but could not be retrieved.")
    return updated_chain_doc

@app.put("/nodes/{node_id}/name", summary="Update a node's name")
async def update_node_name_api(node_id: str, update_data: NodeNameUpdate):
    """Update a single node's name in the database."""
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    if not update_data.name or not update_data.name.strip():
            raise HTTPException(status_code=400, detail="Node name cannot be empty.")

    # Check if the name is actually different before attempting an update
    if node_doc.name == update_data.name:
        return {"message": f"Node {node_id} name is already '{update_data.name}'. No change made.", "node": node_doc}

    success = await update_node_name(node_id, update_data.name) # from database.py
    
    if not success:
        # This path should ideally not be hit if the above checks are done,
        # unless there's another reason for update_node_name to fail (e.g., DB error unrelated to data itself)
        # or if modified_count was 0 for an unexpected reason despite different names (highly unlikely).
        raise HTTPException(status_code=500, detail=f"Failed to update node name for ID '{node_id}'. An unexpected error occurred.")
    
    updated_node_doc = await get_node(node_id) # Fetch the updated document to return
    return {"message": f"Node {node_id} name updated to {update_data.name}", "node": updated_node_doc}

@app.put("/nodes/{node_id}/prompt", response_model=NodeDocument, summary="Update a node's prompt field")
async def update_node_prompt_api(node_id: str, payload: NodePromptUpdate):
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    # Allow empty string for prompt, so no specific validation for empty payload.prompt here.
    if node_doc.prompt == payload.prompt:
        return node_doc # Return current document if no change

    success = await update_node_prompt(node_id, payload.prompt)
    if not success:
        # This could happen if modified_count is 0 for some other DB reason
        raise HTTPException(status_code=500, detail=f"Failed to update node prompt for '{node_id}'. Update was not applied.")
    
    updated_node_doc = await get_node(node_id)
    return updated_node_doc

@app.put("/nodes/{node_id}/llm_config", response_model=NodeDocument, summary="Update a node's LLM config field")
async def update_node_llm_config_api(node_id: str, payload: NodeLLMConfigUpdate):
    node_doc = await get_node(node_id)
    if not node_doc:
        raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found.")

    # Pydantic model LLMConfigDocument will ensure payload.llm_config is valid.
    # Check if the config is actually different before attempting an update.
    # Note: Direct comparison of Pydantic models or their dicts can be tricky if defaults are involved
    # or if order differs. A simple check might be sufficient if format is consistent.
    if node_doc.llm_config and node_doc.llm_config.model_dump() == payload.llm_config.model_dump():
        return node_doc # Return current document if no change
    # Handle case where current config is None and new one is also effectively None (or default)
    if not node_doc.llm_config and not payload.llm_config: # Or compare to a default LLMConfigDocument
        return node_doc

    success = await update_node_llm_config(node_id, payload.llm_config)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to update node LLM config for '{node_id}'. Update was not applied.")

    updated_node_doc = await get_node(node_id)
    return updated_node_doc

# --- Run Server ---
if __name__ == "__main__":
    print("Starting backend server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)