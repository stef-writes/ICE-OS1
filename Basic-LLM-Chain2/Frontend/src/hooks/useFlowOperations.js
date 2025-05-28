import { useCallback } from 'react';
import NodeService, { getAllChains } from '../services/NodeService';

export const useFlowOperations = (flowState, reactFlowInstance, createNodeHandlers) => {
  const {
    nodes,
    edges,
    nodeOutputs,
    currentFlowName,
    currentFlowId,
    setCurrentFlowName,
    setCurrentFlowId,
    setIsLoadFlowModalOpen,
    setAvailableChains,
    setNodes,
    setEdges,
    setNodeOutputs,
  } = flowState;

  // Execute the entire flow
  const executeFlow = useCallback(async () => {
    try {
      // Prepare initial data for the chain execution
      const initialInputs = {};
      
      // For each node, find if it has any connected input nodes
      edges.forEach(edge => {
        const sourceNode = nodes.find(n => n.id === edge.source);
        const targetNode = nodes.find(n => n.id === edge.target);
        
        if (sourceNode && targetNode) {
          const sourceOutput = nodeOutputs[sourceNode.id] || '';
          initialInputs[`context_${targetNode.id}`] = sourceOutput;
        }
      });
      
      // Execute the entire flow with these inputs
      const result = await NodeService.executeChain(initialInputs);
      console.log('Flow execution results:', result);
      
      // Update node outputs based on results
      if (result.results) {
        const newNodeOutputs = { ...nodeOutputs };
        
        Object.entries(result.results).forEach(([nodeId, nodeResult]) => {
          if (nodes.some(node => node.id === nodeId)) {
            const outputText = nodeResult.generated_text || 
                              nodeResult.decision_output || 
                              nodeResult.reasoning_result || 
                              nodeResult.retrieved_data || 
                              JSON.stringify(nodeResult);
            
            newNodeOutputs[nodeId] = outputText;
            
            // Update node display
            setNodes(prevNodes => 
              prevNodes.map(node => {
                if (node.id === nodeId) {
                  return {
                    ...node,
                    data: {
                      ...node.data,
                      output: outputText
                    }
                  };
                }
                return node;
              })
            );
          }
        });
        
        setNodeOutputs(newNodeOutputs);
      }
    } catch (error) {
      console.error('Error executing flow:', error);
    }
  }, [nodes, edges, nodeOutputs, setNodes, setNodeOutputs]);

  // Save flow
  const handleSaveFlow = async () => {
    console.log("Attempting to save flow:", currentFlowName, currentFlowId);
    if (!reactFlowInstance) {
      alert('ReactFlow instance not available.');
      return;
    }
    if (!currentFlowName || currentFlowName.trim() === '') {
      alert('Please enter a name for the flow.');
      return;
    }

    const flowObject = reactFlowInstance.toObject();
    
    // FIRST: Ensure all nodes are saved to the database
    console.log("Ensuring all nodes are saved to database...");
    try {
      for (const node of flowObject.nodes) {
        // Check if node exists in database, if not, create it
        try {
          // Try to get the node first
          const response = await fetch(`http://127.0.0.1:8000/nodes/${node.id}`);
          if (response.status === 404) {
            // Node doesn't exist, create it
            console.log(`Creating missing node ${node.id} in database...`);
            await NodeService.addNode(
              node.id,
              node.type || 'text_generation',
              ['context', 'query'],
              ['generated_text'],
              node.data.nodeName || node.id,
              node.data.llm_config || {
                model: 'gpt-4',
                temperature: 0.7,
                max_tokens: 1250
              }
            );
          } else if (response.ok) {
            // Node exists, update its prompt if needed
            if (node.data.prompt) {
              await NodeService.updateNodePrompt(node.id, node.data.prompt);
            }
          }
        } catch (nodeError) {
          console.error(`Error ensuring node ${node.id} exists:`, nodeError);
          // Continue with other nodes
        }
      }
      console.log("All nodes ensured in database");
    } catch (error) {
      console.error("Error ensuring nodes in database:", error);
      alert(`Warning: Some nodes may not be properly saved to database: ${error.message}`);
    }

    // SECOND: Save the chain
    const flowNodesData = flowObject.nodes.map(node => ({
      node_id: node.id,
      name: node.data.nodeName,
      node_type: node.type,
      prompt: node.data.prompt,
      position: node.position,
      llm_config: node.data.llm_config,
    }));

    const chainData = {
      name: currentFlowName,
      flow_nodes: flowNodesData,
      edges: flowObject.edges.map(edge => ({
        from_node: edge.source,
        to_node: edge.target,
      })),
    };

    try {
      let savedChain;
      if (currentFlowId) {
        console.log(`Updating existing flow with ID: ${currentFlowId}`);
        chainData._id = currentFlowId; 
        savedChain = await NodeService.updateChain(currentFlowId, chainData);
        alert('Flow updated successfully!');
      } else {
        console.log("Saving new flow.");
        savedChain = await NodeService.saveChain(chainData);
        setCurrentFlowId(savedChain.id || savedChain._id);
        alert('Flow saved successfully!');
      }
      console.log('Saved/Updated chain:', savedChain);
    } catch (error) {
      console.error('Error saving flow:', error);
      alert(`Error saving flow: ${error.message}`);
    }
  };

  // Save flow as new
  const handleSaveFlowAs = async () => {
    const newFlowName = prompt("Enter a name for the new flow:", currentFlowName + " Copy");
    if (!newFlowName || newFlowName.trim() === '') {
      return;
    }
    if (!reactFlowInstance) {
      alert('ReactFlow instance not available.');
      return;
    }

    const flowObject = reactFlowInstance.toObject();
    
    // FIRST: Ensure all nodes are saved to the database
    console.log("Ensuring all nodes are saved to database...");
    try {
      for (const node of flowObject.nodes) {
        // Check if node exists in database, if not, create it
        try {
          // Try to get the node first
          const response = await fetch(`http://127.0.0.1:8000/nodes/${node.id}`);
          if (response.status === 404) {
            // Node doesn't exist, create it
            console.log(`Creating missing node ${node.id} in database...`);
            await NodeService.addNode(
              node.id,
              node.type || 'text_generation',
              ['context', 'query'],
              ['generated_text'],
              node.data.nodeName || node.id,
              node.data.llm_config || {
                model: 'gpt-4',
                temperature: 0.7,
                max_tokens: 1250
              }
            );
          } else if (response.ok) {
            // Node exists, update its prompt if needed
            if (node.data.prompt) {
              await NodeService.updateNodePrompt(node.id, node.data.prompt);
            }
          }
        } catch (nodeError) {
          console.error(`Error ensuring node ${node.id} exists:`, nodeError);
          // Continue with other nodes
        }
      }
      console.log("All nodes ensured in database");
    } catch (error) {
      console.error("Error ensuring nodes in database:", error);
      alert(`Warning: Some nodes may not be properly saved to database: ${error.message}`);
    }

    // SECOND: Save the chain
    const flowNodesData = flowObject.nodes.map(node => ({
      node_id: node.id,
      name: node.data.nodeName,
      node_type: node.type,
      prompt: node.data.prompt,
      position: node.position,
      llm_config: node.data.llm_config,
    }));

    const chainData = {
      name: newFlowName,
      flow_nodes: flowNodesData,
      edges: flowObject.edges.map(edge => ({
        from_node: edge.source,
        to_node: edge.target,
      })),
    };

    try {
      console.log("Saving new flow (Save As):", newFlowName);
      const savedChain = await NodeService.saveChain(chainData);
      setCurrentFlowName(newFlowName);
      setCurrentFlowId(savedChain.id || savedChain._id);
      alert('Flow saved successfully as new!');
      console.log('Saved chain (Save As):', savedChain);
    } catch (error) {
      console.error('Error saving flow as new:', error);
      alert(`Error saving flow as new: ${error.message}`);
    }
  };
  
  // Open load flow modal
  const handleOpenLoadFlowModal = async () => {
    console.log("Attempting to open load flow modal");
    try {
      const chains = await getAllChains();
      setAvailableChains(chains || []);
      setIsLoadFlowModalOpen(true);
    } catch (error) {
      console.error("Failed to fetch chains:", error);
      alert(`Error fetching saved flows: ${error.message}`);
      setAvailableChains([]);
    }
  };

  // Load flow
  const handleLoadFlow = async (chainToLoad) => {
    if (!chainToLoad || (!chainToLoad.id && !chainToLoad._id)) {
      alert("Invalid chain data provided for loading.");
      return;
    }
    const chainIdToLoad = chainToLoad.id || chainToLoad._id;
    console.log("Attempting to load flow with ID:", chainIdToLoad);

    try {
      const detailedChain = await NodeService.getChainDetails(chainIdToLoad);
      if (!detailedChain || !detailedChain.flow_nodes) {
        alert(`Could not fetch details for flow ID: ${chainIdToLoad}, or flow_nodes missing.`);
        console.error("Failed to load chain or flow_nodes missing:", detailedChain);
        return;
      }

      console.log("Detailed chain for loading:", detailedChain);

      const newNodes = [];
      let yPos = 50; 
      const xOffset = 150;

      for (const flowNode of detailedChain.flow_nodes) {
        const nodeId = flowNode.node_id;
        
        const nodeLLMConfig = flowNode.llm_config || { 
          model: 'gpt-4', 
          temperature: 0.7, 
          max_tokens: 1250 
        };

        const position = flowNode.position && typeof flowNode.position.x === 'number' && typeof flowNode.position.y === 'number' 
          ? flowNode.position 
          : { x: xOffset, y: yPos };
        if (!(flowNode.position && typeof flowNode.position.x === 'number' && typeof flowNode.position.y === 'number')) {
            yPos += 100;
        }

        // Create proper handlers for the loaded node
        const nodeHandlers = createNodeHandlers ? createNodeHandlers(nodeId) : {
          onNameChange: () => {},
          onPromptChange: () => {},
          onSettingsChange: () => {},
          onOutputChange: () => {},
          onVariableSelect: () => {},
          onRun: () => {},
        };

        newNodes.push({
          id: nodeId,
          type: flowNode.node_type || 'llmNode',
          position: position,
          data: {
            nodeName: flowNode.name || nodeId,
            prompt: flowNode.prompt || '',
            output: '',
            variables: [],
            llm_config: nodeLLMConfig,
            ...nodeHandlers,
          },
        });
      }

      const newEdges = (detailedChain.edges || []).map((edge, index) => ({
        id: edge.id || `e${edge.from_node}-${edge.to_node}-${index}`,
        source: edge.from_node,
        target: edge.to_node,
      }));

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

  return {
    executeFlow,
    handleSaveFlow,
    handleSaveFlowAs,
    handleOpenLoadFlowModal,
    handleLoadFlow,
  };
}; 