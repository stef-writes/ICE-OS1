# Comprehensive Guide: Implementing Load Saved Chains Functionality

## Overview
Currently, the system can save chains/flows to the database and load them visually in the frontend, but the loaded chains are not executable because they don't exist in the backend's in-memory `ScriptChain` instances. This guide provides a complete implementation plan to bridge this gap.

## Current State Analysis

### ✅ What Works Now
1. **Database Layer**: Complete CRUD operations for chains and nodes
   - `get_chain()`, `get_all_chains()` - Load chain metadata
   - `get_node()`, `get_all_nodes()` - Load individual nodes
   - API endpoints: `GET /chains/`, `GET /chains/{id}`, `GET /nodes/`

2. **Frontend Loading**: UI can load and display chains
   - `handleLoadFlow()` in `useFlowOperations.js` loads chain data
   - Recreates nodes and edges in ReactFlow canvas
   - Restores node positions, names, prompts, and LLM configs

### 🚧 What's Missing
1. **Backend**: Load chain into executable in-memory `ScriptChain`
2. **Frontend**: Trigger backend loading and sync state
3. **Node Prompts**: Ensure prompts are properly stored and loaded
4. **State Synchronization**: Keep frontend and backend in sync
5. **Error Handling**: Handle missing nodes, invalid chains, etc.

## Implementation Plan

### Phase 1: Backend Implementation

#### 1.1 Add Load Chain Into Memory Endpoint

Add this to `Backend/main.py`:

```python
@app.post("/load_chain_into_memory/{chain_id}")
async def load_chain_into_memory(chain_id: str, session_id: str = "default"):
    """Load a saved chain from database into the in-memory ScriptChain for execution."""
    try:
        # Get the chain from database
        chain_doc = await get_chain(chain_id)
        if not chain_doc:
            raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")
        
        # Get or create script chain for session
        script_chain = get_script_chain(session_id)
        
        # Clear existing chain state
        script_chain.graph.clear()
        script_chain.storage = NamespacedStorage()
        script_chain.node_versions = {}
        script_chain.node_dependencies = {}
        
        print(f"Loading chain '{chain_doc.name}' into memory for session {session_id}")
        
        # Reconstruct nodes from flow_nodes
        loaded_nodes = []
        for flow_node in chain_doc.flow_nodes:
            # Convert LLMConfigDocument back to LLMConfig
            llm_config = default_llm_config
            if flow_node.llm_config:
                llm_config = LLMConfig(
                    model=flow_node.llm_config.model,
                    temperature=flow_node.llm_config.temperature,
                    max_tokens=flow_node.llm_config.max_tokens
                )
            
            # Extract input_keys from prompt if not stored
            input_keys = flow_node.input_keys or []
            if flow_node.prompt and not input_keys:
                input_keys = extract_template_variables(flow_node.prompt)
            
            # Add node to script chain
            script_chain.add_node(
                node_id=flow_node.node_id,
                node_type=flow_node.node_type or "text_generation",
                input_keys=input_keys,
                output_keys=flow_node.output_keys or ["output"],
                model_config=llm_config
            )
            
            # Load any saved outputs from individual node documents
            node_doc = await get_node(flow_node.node_id)
            if node_doc and node_doc.output:
                script_chain.storage.store(flow_node.node_id, {
                    "output": node_doc.output,
                    "generated_text": node_doc.output,
                    "content": node_doc.output
                })
                # Set initial version to 1 if node has output
                script_chain.node_versions[flow_node.node_id] = 1
            
            loaded_nodes.append(flow_node.node_id)
        
        # Reconstruct edges
        loaded_edges = []
        for edge in chain_doc.edges:
            from_node = edge.get("from_node")
            to_node = edge.get("to_node")
            if from_node and to_node:
                script_chain.add_edge(from_node, to_node)
                loaded_edges.append(f"{from_node} -> {to_node}")
        
        return {
            "message": f"Chain '{chain_doc.name}' loaded into memory successfully",
            "chain_id": chain_id,
            "chain_name": chain_doc.name,
            "nodes_loaded": loaded_nodes,
            "edges_loaded": loaded_edges,
            "session_id": session_id
        }
        
    except Exception as e:
        print(f"Error loading chain {chain_id} into memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load chain: {str(e)}")
```

#### 1.2 Add Chain Status Endpoint

Add this to check if a chain is loaded in memory:

```python
@app.get("/chain_status/{session_id}")
async def get_chain_status(session_id: str = "default"):
    """Get the status of the current chain in memory for a session."""
    script_chain = get_script_chain(session_id)
    
    nodes_in_memory = list(script_chain.graph.nodes())
    edges_in_memory = [(u, v) for u, v in script_chain.graph.edges()]
    
    # Get storage status
    storage_status = {}
    for node_id in nodes_in_memory:
        storage_status[node_id] = {
            "has_data": script_chain.storage.has_node(node_id),
            "version": script_chain.node_versions.get(node_id, 0),
            "data_keys": list(script_chain.storage.get(node_id, {}).keys()) if script_chain.storage.has_node(node_id) else []
        }
    
    return {
        "session_id": session_id,
        "nodes_count": len(nodes_in_memory),
        "edges_count": len(edges_in_memory),
        "nodes": nodes_in_memory,
        "edges": edges_in_memory,
        "storage_status": storage_status
    }
```

#### 1.3 Enhance Save Operations

Modify the save operations to ensure prompts are stored in both places:

```python
# In the existing add_node_api function, add prompt storage:
async def add_node_api(node: NodeInput, session_id: str = "default"):
    # ... existing code ...
    
    # When saving to database, include prompt if available
    node_db_data = {
        "node_id": node.node_id,
        "name": node.name if node.name else node.node_id,
        "node_type": node.node_type,
        "input_keys": node.input_keys,
        "output_keys": node.output_keys,
        "llm_config": node.llm_config.model_dump() if node.llm_config else None,
        "prompt": getattr(node, 'prompt', ''),  # Add prompt field
        "output": None,
    }
    
    # ... rest of existing code ...
```

### Phase 2: Frontend Implementation

#### 2.1 Add Service Methods

Add to `Frontend/src/services/NodeService.js`:

```javascript
/**
 * Load a chain into backend memory for execution
 * @param {string} chainId - The ID of the chain to load
 * @param {string} sessionId - Optional session ID
 * @returns {Promise<object>} - Load result
 */
export const loadChainIntoMemory = async (chainId, sessionId = 'default') => {
  try {
    const response = await fetch(`${API_URL}/load_chain_into_memory/${chainId}?session_id=${sessionId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error loading chain into memory');
    }
    return response.json();
  } catch (error) {
    console.error('Error calling loadChainIntoMemory API:', error);
    throw error;
  }
};

/**
 * Get the status of the current chain in backend memory
 * @param {string} sessionId - Optional session ID
 * @returns {Promise<object>} - Chain status
 */
export const getChainStatus = async (sessionId = 'default') => {
  try {
    const response = await fetch(`${API_URL}/chain_status/${sessionId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error getting chain status');
    }
    return response.json();
  } catch (error) {
    console.error('Error calling getChainStatus API:', error);
    throw error;
  }
};
```

#### 2.2 Update Flow Operations

Modify `Frontend/src/hooks/useFlowOperations.js`:

```javascript
// Add to the existing handleLoadFlow function
const handleLoadFlow = async (chainToLoad) => {
  if (!chainToLoad || (!chainToLoad.id && !chainToLoad._id)) {
    alert("Invalid chain data provided for loading.");
    return;
  }
  const chainIdToLoad = chainToLoad.id || chainToLoad._id;
  console.log("Attempting to load flow with ID:", chainIdToLoad);

  try {
    // 1. Load the visual representation (existing code)
    const detailedChain = await NodeService.getChainDetails(chainIdToLoad);
    if (!detailedChain || !detailedChain.flow_nodes) {
      alert(`Could not fetch details for flow ID: ${chainIdToLoad}, or flow_nodes missing.`);
      console.error("Failed to load chain or flow_nodes missing:", detailedChain);
      return;
    }

    console.log("Detailed chain for loading:", detailedChain);

    // ... existing code to create newNodes and newEdges ...

    // 2. NEW: Load the chain into backend memory
    try {
      const loadResult = await NodeService.loadChainIntoMemory(chainIdToLoad);
      console.log("Chain loaded into backend memory:", loadResult);
      
      // 3. NEW: Load any existing node outputs
      const nodeIds = newNodes.map(node => node.id);
      if (nodeIds.length > 0) {
        const outputs = await NodeService.getNodeOutputs({ node_ids: nodeIds });
        console.log("Loaded node outputs:", outputs);
        
        // Update node data with loaded outputs
        newNodes.forEach(node => {
          if (outputs[node.id]) {
            node.data.output = outputs[node.id].output || outputs[node.id].generated_text || '';
          }
        });
        
        setNodeOutputs(outputs);
      }
      
    } catch (backendError) {
      console.error("Error loading chain into backend:", backendError);
      alert(`Warning: Chain loaded visually but failed to load into backend memory: ${backendError.message}`);
    }

    // 4. Update UI state (existing code)
    setNodes(newNodes);
    setEdges(newEdges);
    setCurrentFlowName(detailedChain.name);
    setCurrentFlowId(detailedChain.id || detailedChain._id);
    setIsLoadFlowModalOpen(false);
    alert(`Flow "${detailedChain.name}" loaded successfully.`);

  } catch (error) {
    console.error('Error loading flow:', error);
    alert(`Error loading flow: ${error.message}`);
  }
};
```

#### 2.3 Add Chain Status Indicator

Add to `Frontend/src/components/FlowControls.jsx`:

```javascript
// Add state for chain status
const [chainStatus, setChainStatus] = useState(null);
const [isChainLoaded, setIsChainLoaded] = useState(false);

// Add function to check chain status
const checkChainStatus = async () => {
  try {
    const status = await NodeService.getChainStatus();
    setChainStatus(status);
    setIsChainLoaded(status.nodes_count > 0);
  } catch (error) {
    console.error("Error checking chain status:", error);
    setIsChainLoaded(false);
  }
};

// Add status indicator to the UI
<div className="chain-status">
  <span className={`status-indicator ${isChainLoaded ? 'loaded' : 'not-loaded'}`}>
    {isChainLoaded ? '🟢 Chain Loaded' : '🔴 No Chain Loaded'}
  </span>
  {chainStatus && (
    <span className="status-details">
      ({chainStatus.nodes_count} nodes, {chainStatus.edges_count} edges)
    </span>
  )}
</div>
```

### Phase 3: Data Consistency Improvements

#### 3.1 Ensure Prompt Storage

Modify the save flow operation to ensure prompts are stored in individual nodes:

```javascript
// In handleSaveFlow, after saving the chain, also update individual nodes
const handleSaveFlow = async () => {
  // ... existing chain save code ...
  
  // NEW: Ensure individual nodes have their prompts stored
  try {
    for (const node of flowObject.nodes) {
      if (node.data.prompt) {
        await NodeService.updateNodePrompt(node.id, { prompt: node.data.prompt });
      }
    }
    console.log("Node prompts synchronized to database");
  } catch (error) {
    console.error("Error synchronizing node prompts:", error);
    // Don't fail the save operation for this
  }
};
```

#### 3.2 Add Node Prompt Update Service

Add to `Frontend/src/services/NodeService.js`:

```javascript
/**
 * Update a node's prompt
 * @param {string} nodeId - The node ID
 * @param {object} promptData - Object with prompt field
 * @returns {Promise<object>} - Updated node
 */
export const updateNodePrompt = async (nodeId, promptData) => {
  try {
    const response = await fetch(`${API_URL}/nodes/${nodeId}/prompt`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(promptData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error updating node prompt');
    }
    return response.json();
  } catch (error) {
    console.error('Error calling updateNodePrompt API:', error);
    throw error;
  }
};
```

### Phase 4: Testing and Validation

#### 4.1 Test Scenarios

1. **Basic Load Test**:
   - Save a simple 2-node chain
   - Load it back
   - Verify both visual and backend loading
   - Execute the loaded chain

2. **Complex Chain Test**:
   - Save a multi-node chain with dependencies
   - Load it back
   - Verify all edges and dependencies are correct
   - Test execution order

3. **Output Persistence Test**:
   - Create a chain, execute some nodes
   - Save the chain with outputs
   - Load it back
   - Verify outputs are restored

4. **Error Handling Test**:
   - Try loading non-existent chain
   - Try loading chain with missing nodes
   - Verify graceful error handling

#### 4.2 Validation Checklist

- [ ] Chain loads visually in frontend
- [ ] Chain loads into backend memory
- [ ] Node prompts are preserved
- [ ] Node outputs are restored
- [ ] Edges/dependencies are correct
- [ ] LLM configs are preserved
- [ ] Chain can be executed after loading
- [ ] Error handling works properly
- [ ] Multiple sessions work independently

### Phase 5: Additional Enhancements

#### 5.1 Auto-Load on Startup

Consider auto-loading the last used chain when the app starts:

```javascript
// In useEffect on app startup
useEffect(() => {
  const lastChainId = localStorage.getItem('lastLoadedChainId');
  if (lastChainId) {
    // Optionally auto-load the last chain
    handleLoadFlow({ id: lastChainId });
  }
}, []);
```

#### 5.2 Chain Validation

Add validation to ensure loaded chains are executable:

```python
@app.post("/validate_chain/{chain_id}")
async def validate_chain(chain_id: str):
    """Validate that a chain can be properly loaded and executed."""
    try:
        chain_doc = await get_chain(chain_id)
        if not chain_doc:
            return {"valid": False, "error": "Chain not found"}
        
        # Check for missing nodes
        missing_nodes = []
        for flow_node in chain_doc.flow_nodes:
            node_doc = await get_node(flow_node.node_id)
            if not node_doc:
                missing_nodes.append(flow_node.node_id)
        
        # Check for circular dependencies
        # ... add cycle detection logic ...
        
        return {
            "valid": len(missing_nodes) == 0,
            "missing_nodes": missing_nodes,
            "node_count": len(chain_doc.flow_nodes),
            "edge_count": len(chain_doc.edges)
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}
```

## Implementation Priority

1. **High Priority**: Backend load endpoint and basic frontend integration
2. **Medium Priority**: Prompt storage consistency and output restoration
3. **Low Priority**: Status indicators, validation, and auto-load features

## Estimated Effort

- **Backend Implementation**: 4-6 hours
- **Frontend Integration**: 3-4 hours  
- **Testing and Debugging**: 2-3 hours
- **Documentation and Polish**: 1-2 hours

**Total**: 10-15 hours for complete implementation

## Notes and Considerations

- **Session Management**: Consider how multiple users/sessions should work
- **Performance**: Loading large chains might be slow - consider pagination
- **Memory Usage**: In-memory chains consume server memory - consider cleanup
- **Concurrency**: Multiple users loading chains simultaneously
- **Backup Strategy**: Consider backing up in-memory state periodically