// NodeService.js - Handles API calls to the backend

const API_URL = 'http://127.0.0.1:8000';

// Create a unique session ID for this browser session
// In a production app, this might come from authentication or be stored in localStorage
const SESSION_ID = 'session_' + Math.random().toString(36).substring(2, 15);
console.log(`Using session ID: ${SESSION_ID}`);

/**
 * Sends a prompt to the backend API to generate text
 * @param {string} prompt - The prompt text
 * @param {object} config - Optional LLM config parameters
 * @param {object} contextData - Optional context data for the prompt (node outputs)
 * @returns {Promise<object>} - The API response with generated text
 */
export const generateText = async (prompt, config = null, contextData = null) => {
  try {
    console.log('Sending to backend - Prompt:', prompt);
    console.log('Sending to backend - Context Data:', contextData);
    
    const payload = {
      prompt_text: prompt,
      llm_config: config,
    };

    // If contextData is provided, add it to the payload
    if (contextData) {
      payload.context_data = contextData;
    }

    const response = await fetch(`${API_URL}/generate_text_node?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error generating text');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling generateText API:', error);
    throw error;
  }
};

/**
 * Adds a node to the backend script chain
 * @param {string} nodeId - Unique identifier for the node
 * @param {string} nodeType - Type of node (e.g., 'text_generation')
 * @param {Array<string>} inputKeys - Input keys required by the node
 * @param {Array<string>} outputKeys - Output keys produced by the node
 * @param {string} name - The name of the node.
 * @param {object} llmConfig - Optional LLM config parameters for the node.
 * @returns {Promise<object>} - The API response
 */
export const addNode = async (nodeId, nodeType, inputKeys = [], outputKeys = [], name = 'Untitled Node', llmConfig = null) => {
  try {
    const response = await fetch(`${API_URL}/add_node?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        node_id: nodeId,
        name: name,
        node_type: nodeType,
        input_keys: inputKeys,
        output_keys: outputKeys,
        llm_config: llmConfig,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding node');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addNode API:', error);
    throw error;
  }
};

/**
 * Adds an edge between two nodes in the backend script chain
 * @param {string} fromNode - Source node ID
 * @param {string} toNode - Target node ID
 * @returns {Promise<object>} - The API response
 */
export const addEdge = async (fromNode, toNode) => {
  try {
    const response = await fetch(`${API_URL}/add_edge?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from_node: fromNode,
        to_node: toNode,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error adding edge');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling addEdge API:', error);
    throw error;
  }
};

/**
 * Executes the script chain in the backend
 * @param {object} initialInputs - Optional initial inputs for the chain
 * @returns {Promise<object>} - The execution results
 */
export const executeChain = async (initialInputs = null) => {
  try {
    console.log('Executing chain with initial inputs:', initialInputs);
    
    const response = await fetch(`${API_URL}/execute?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(initialInputs),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error executing chain');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling executeChain API:', error);
    throw error;
  }
};

/**
 * Validates a template against available nodes
 * @param {string} promptText - The template text to validate
 * @param {Array<string>} availableNodes - List of available node names
 * @returns {Promise<object>} - Validation results
 */
export const validateTemplate = async (promptText, availableNodes = []) => {
  try {
    const response = await fetch(`${API_URL}/validate_template?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt_text: promptText,
        available_nodes: availableNodes,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error validating template');
    }

    return response.json();
  } catch (error) {
    console.error('Error calling validateTemplate API:', error);
    throw error;
  }
};

/**
 * Retrieves the latest output values for specified nodes
 * @param {Array<string>} nodeIds - Array of node IDs to fetch outputs for
 * @returns {Promise<object>} - Map of node IDs to their output values
 */
export const getNodeOutputs = async (nodeIds) => {
  try {
    // Since we don't have a dedicated endpoint for this yet,
    // this is a workaround to get node outputs from the backend's storage
    // In a production system, you'd implement a proper API endpoint
    
    // First try to get from the backend's storage using execute with empty input
    const response = await fetch(`${API_URL}/get_node_outputs?session_id=${SESSION_ID}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ node_ids: nodeIds }),
    });

    if (!response.ok) {
      // If the endpoint doesn't exist or fails, return empty object
      console.warn("get_node_outputs endpoint failed or doesn't exist, using local values");
      return {};
    }

    return response.json();
  } catch (error) {
    console.error('Error fetching node outputs:', error);
    return {}; // Return empty object on error
  }
};

// --- Chain Management API Calls ---

/**
 * Saves a new chain (flow) to the backend.
 * @param {object} chainData - The chain data (name, nodes, edges).
 * @returns {Promise<object>} - The API response (likely the saved chain with its ID).
 */
export const saveChain = async (chainData) => {
  try {
    const response = await fetch(`${API_URL}/chains/?session_id=${SESSION_ID}`, { // Added slash for consistency
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(chainData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error saving chain');
    }
    return response.json();
  } catch (error) {
    console.error('Error calling saveChain API:', error);
    throw error;
  }
};

/**
 * Updates an existing chain in the backend.
 * @param {string} chainId - The ID of the chain to update.
 * @param {object} chainData - The updated chain data.
 * @returns {Promise<object>} - The API response (likely the updated chain).
 */
export const updateChain = async (chainId, chainData) => {
  try {
    const response = await fetch(`${API_URL}/chains/${chainId}?session_id=${SESSION_ID}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(chainData),
    });
    if (!response.ok) {
      const errorData = await response.json();
      // Handle 304 Not Modified specifically if needed, or let it be an error for now
      if (response.status === 304) {
        console.warn(`Chain ${chainId} was not modified. Data might be identical.`);
        // Potentially return a specific indicator or the original data if passed differently
        return chainData; // Or throw an error as below if 304 should be treated as failure to change
      }
      throw new Error(errorData.detail || 'Error updating chain');
    }
    return response.json();
  } catch (error) {
    console.error(`Error calling updateChain API for ${chainId}:`, error);
    throw error;
  }
};

/**
 * Retrieves all saved chains from the backend.
 * @returns {Promise<Array<object>>} - A list of chain objects.
 */
export const getAllChains = async () => {
  try {
    const response = await fetch(`${API_URL}/chains/?session_id=${SESSION_ID}`, { // Added slash for consistency
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Error fetching all chains');
    }
    return response.json();
  } catch (error) {
    console.error('Error calling getAllChains API:', error);
    throw error;
  }
};

/**
 * Retrieves the details of a specific chain by its ID.
 * @param {string} chainId - The ID of the chain to retrieve.
 * @returns {Promise<object>} - The chain object.
 */
export const getChainDetails = async (chainId) => {
  try {
    const response = await fetch(`${API_URL}/chains/${chainId}?session_id=${SESSION_ID}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Error fetching chain details for ${chainId}`);
    }
    return response.json();
  } catch (error) {
    console.error(`Error calling getChainDetails API for ${chainId}:`, error);
    throw error;
  }
};

// --- End Chain Management API Calls ---

// --- Node Specific API Calls (other than addNode, addEdge which are higher up) ---

/**
 * Updates the name of a specific node.
 * @param {string} nodeId - The ID of the node to update.
 * @param {string} newName - The new name for the node.
 * @returns {Promise<object>} - The API response (likely the updated node data or a success message).
 */
export const updateNodeNameService = async (nodeId, newName) => {
  try {
    const response = await fetch(`${API_URL}/nodes/${nodeId}/name?session_id=${SESSION_ID}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: newName }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      // Consider if 400 (e.g. empty name) or specific codes should be handled differently or re-thrown
      throw new Error(errorData.detail || `Error updating node name for ${nodeId}`);
    }
    return response.json(); // Backend now returns { message, node } or { message } for no-change
  } catch (error) {
    console.error(`Error calling updateNodeNameService for ${nodeId}:`, error);
    throw error;
  }
};

/**
 * Updates the output of a specific node.
 * @param {string} nodeId - The ID of the node to update.
 * @param {any} outputData - The new output data for the node.
 * @returns {Promise<object>} - The API response.
 */
export const updateNodeOutputService = async (nodeId, outputData) => {
  try {
    const response = await fetch(`${API_URL}/nodes/${nodeId}/output?session_id=${SESSION_ID}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ output: outputData }), // Assuming backend expects { "output": ... }
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Error updating node output for ${nodeId}`);
    }
    return response.json(); // Assuming backend returns a success message or updated node
  } catch (error) {
    console.error(`Error calling updateNodeOutputService for ${nodeId}:`, error);
    throw error;
  }
};

/**
 * Updates the prompt of a specific node.
 * @param {string} nodeId - The ID of the node to update.
 * @param {string} newPrompt - The new prompt text for the node.
 * @returns {Promise<object>} - The API response (likely the updated node document).
 */
export const updateNodePrompt = async (nodeId, newPrompt) => {
  try {
    const response = await fetch(`${API_URL}/nodes/${nodeId}/prompt?session_id=${SESSION_ID}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt: newPrompt }),
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Error updating node prompt for ${nodeId}`);
    }
    return response.json(); // Backend returns the updated NodeDocument
  } catch (error) {
    console.error(`Error calling updateNodePrompt for ${nodeId}:`, error);
    throw error;
  }
};

/**
 * Updates the LLM configuration of a specific node.
 * @param {string} nodeId - The ID of the node to update.
 * @param {object} newConfig - The new LLM config object (e.g., { model, temperature, max_tokens }).
 * @returns {Promise<object>} - The API response (likely the updated node document).
 */
export const updateNodeLLMConfig = async (nodeId, newConfig) => {
  try {
    const response = await fetch(`${API_URL}/nodes/${nodeId}/llm_config?session_id=${SESSION_ID}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ llm_config: newConfig }), // Matches NodeLLMConfigUpdate model
    });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `Error updating node LLM config for ${nodeId}`);
    }
    return response.json(); // Backend returns the updated NodeDocument
  } catch (error) {
    console.error(`Error calling updateNodeLLMConfig for ${nodeId}:`, error);
    throw error;
  }
};

// Export all functions as a service object
const NodeService = {
  generateText,
  addNode,
  addEdge,
  executeChain,
  validateTemplate,
  getNodeOutputs,
  // Add new chain functions
  saveChain,
  updateChain,
  getAllChains,
  getChainDetails,
  // Node specific updates
  updateNodeNameService,
  updateNodeOutputService,
  updateNodePrompt,
  updateNodeLLMConfig,
};

export default NodeService; 