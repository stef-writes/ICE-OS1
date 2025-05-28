import { useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import NodeService from '../services/NodeService';
import { debounce } from '../utils';
import { getCenterPosition } from '../utils/flowHelpers';

export const useNodeOperations = (flowState) => {
  const {
    nodes,
    edges,
    nodeOutputs,
    setNodes,
    setEdges,
    setNodeOutputs,
  } = flowState;

  // Refs to hold the latest state
  const latestNodesRef = useRef(nodes);
  const latestEdgesRef = useRef(edges);
  const latestNodeOutputsRef = useRef(nodeOutputs);

  useEffect(() => {
    latestNodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    latestEdgesRef.current = edges;
  }, [edges]);

  useEffect(() => {
    latestNodeOutputsRef.current = nodeOutputs;
  }, [nodeOutputs]);

  // Debounced function for saving prompt changes
  const debouncedSavePrompt = useCallback(
    debounce((nodeId, promptValue) => {
      NodeService.updateNodePrompt(nodeId, promptValue)
        .then(updatedNodeDoc => {
          console.log(`Node ${nodeId} prompt updated on backend via debounce.`);
        })
        .catch(err => console.error(`Error updating node ${nodeId} prompt on backend via debounce:`, err));
    }, 500),
    []
  );

  // Debounced function for saving LLM Config changes
  const debouncedSaveLLMConfig = useCallback(
    debounce((nodeId, newConfig) => {
      NodeService.updateNodeLLMConfig(nodeId, newConfig)
        .then(updatedNodeDoc => {
          console.log(`Node ${nodeId} LLM config updated on backend via debounce:`, updatedNodeDoc);
        })
        .catch(err => console.error(`Error updating node ${nodeId} LLM config on backend via debounce:`, err));
    }, 500),
    []
  );

  // Create node handlers for a specific node ID
  const createNodeHandlers = useCallback((nodeId) => ({
    onNameChange: (name) => {
      // Update node name in local React state
      // Ensure this update is comprehensive and immediately effective
      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          if (node.id === nodeId) {
            // This is the node whose name is changing
            console.log(`onNameChange: Updating name for node ${nodeId} from '${node.data.nodeName}' to '${name}'`);
            return {
              ...node,
              data: {
                ...node.data,
                nodeName: name, // Direct update
              },
            };
          } else {
            // This is for other nodes that might use the changed name in their variable lists
            // (This part seems less related to the current bug but good to keep consistent)
            if (node.data.variables) {
              const updatedVariables = node.data.variables.map(variable => {
                if (variable.id === nodeId) { // If this variable pointed to the node whose name changed
                  return {
                    ...variable,
                    name: name // Update the name in the list
                  };
                }
                return variable;
              });
              if (JSON.stringify(node.data.variables) !== JSON.stringify(updatedVariables)) {
                // console.log(`onNameChange: Updating variable list for node ${node.id} due to name change of ${nodeId}`);
                return {
                  ...node,
                  data: {
                    ...node.data,
                    variables: updatedVariables
                  }
                };
              }
            }
            return node; // No change to this node or its variables related to the current name change
          }
        })
      );
      
      // Call backend to update name in database
      NodeService.updateNodeNameService(nodeId, name)
        .then(response => {
          console.log(`Node ${nodeId} name updated on backend. Response:`, response.message);
          // Optionally, re-sync from backend if there's a chance of discrepancy,
          // but for now, we assume the frontend update is the source of truth for immediate UI.
          // if (response.node && response.node.name !== name) {
          //   console.warn(`Backend name for ${nodeId} (${response.node.name}) differs from frontend optimistic update (${name}). Re-syncing.`);
          //   setNodes(nds => nds.map(n => n.id === nodeId ? {...n, data: {...n.data, nodeName: response.node.name}} : n));
          // }
        })
        .catch(err => {
          console.error(`Failed to update node ${nodeId} name in database via NodeService:`, err);
        });
    },

    onPromptChange: (prompt) => {
      // Update local React state immediately for responsiveness
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                prompt,
              },
            };
          }
          return node;
        })
      );
      // Call the debounced function to save to backend
      debouncedSavePrompt(nodeId, prompt);
    },

    onSettingsChange: (newConfig) => {
      // Update local state for immediate UI feedback
      setNodes(nds => nds.map(n => {
        if (n.id === nodeId) {
          return { ...n, data: { ...n.data, llm_config: newConfig }};
        }
        return n;
      }));
      // Call backend
      debouncedSaveLLMConfig(nodeId, newConfig);
    },

    onOutputChange: (output) => {
      setNodeOutputs(prev => ({
        ...prev,
        [nodeId]: output
      }));
    },

    onVariableSelect: (variableIds) => {
      // Store the selected variable IDs for state management
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === nodeId) {
            return {
              ...node,
              data: {
                ...node.data,
                selectedVariableIds: variableIds,
              },
            };
          }
          return node;
        })
      );
    },

    onRun: async (prompt, providedActiveInputNodeIds) => {
      const currentNodes = latestNodesRef.current;
      const currentEdges = latestEdgesRef.current;
      const currentNodeOutputs = latestNodeOutputsRef.current;

      const currentNode = currentNodes.find(n => n.id === nodeId);
      const currentRunnerNodeName = currentNode?.data?.nodeName || nodeId;

      let activeInputNodeIds = providedActiveInputNodeIds;
      if (!activeInputNodeIds || activeInputNodeIds.length === 0) {
        console.log(`No activeInputNodeIds provided for node ${nodeId} (${currentRunnerNodeName}), determining from all incoming edges.`);
        const incomingEdges = currentEdges.filter(edge => edge.target === nodeId);
        activeInputNodeIds = incomingEdges.map(edge => edge.source);
        if (activeInputNodeIds.length > 0) {
          console.log(`Derived activeInputNodeIds for ${nodeId} (${currentRunnerNodeName}):`, activeInputNodeIds);
        }
      }

      console.log(`Running node ${nodeId} (current name: ${currentRunnerNodeName}) with actual active inputs:`, activeInputNodeIds);
      
      let latestBackendNodeOutputs = {}; // Renamed to avoid confusion with local latestNodeOutputsRef
      if (activeInputNodeIds && activeInputNodeIds.length > 0) {
        try {
          console.log("Fetching latest node outputs from backend for nodes:", activeInputNodeIds);
          latestBackendNodeOutputs = await NodeService.getNodeOutputs(activeInputNodeIds);
          console.log("Latest node outputs from backend:", latestBackendNodeOutputs);
        } catch (error) {
          console.warn("Error fetching node outputs:", error);
          latestBackendNodeOutputs = {}; 
        }
      }
      
      let contextData = {};
      contextData['__node_mapping'] = {};

      if (activeInputNodeIds && activeInputNodeIds.length > 0) {
        activeInputNodeIds.forEach(inputId => {
          let inputNodeOutput;
          if (latestBackendNodeOutputs[inputId]) { // Use fetched from backend first
            inputNodeOutput = latestBackendNodeOutputs[inputId];
          } else if (currentNodeOutputs[inputId]) { // Then local state via ref
            inputNodeOutput = currentNodeOutputs[inputId];
          } else {
            inputNodeOutput = '';
          }
          
          const sourceNode = currentNodes.find(n => n.id === inputId); // Use currentNodes from ref
          let resolvedSourceNodeNameForMapping;

          if (sourceNode) {
            if (sourceNode.data?.nodeName && sourceNode.data.nodeName.trim() !== '') {
              resolvedSourceNodeNameForMapping = sourceNode.data.nodeName.trim();
            } else {
              resolvedSourceNodeNameForMapping = `id:${inputId}`; 
              console.warn(`  WARNING: Source node ${inputId} IS FOUND, but its 'nodeName' is empty or whitespace. Using '${resolvedSourceNodeNameForMapping}' as the key in '__node_mapping'.`);
            }
          } else {
            resolvedSourceNodeNameForMapping = `id:${inputId}`; 
            console.error(`  CRITICAL: Source node ${inputId} (an active input) NOT FOUND in local 'nodes' state (via ref). Using '${resolvedSourceNodeNameForMapping}' as key in '__node_mapping'.`);
          }
          
          contextData['__node_mapping'][resolvedSourceNodeNameForMapping] = inputId;
          contextData[`id:${inputId}`] = inputNodeOutput;
          if (sourceNode && resolvedSourceNodeNameForMapping !== `id:${inputId}`) {
            contextData[resolvedSourceNodeNameForMapping] = inputNodeOutput;
          }
        });
      }
      
      contextData['__current_node'] = nodeId;
      console.log("Final context data being sent to backend for node " + nodeId + " ("+ currentRunnerNodeName +"):", JSON.stringify(contextData, null, 2));
      
      try {
        const response = await NodeService.generateText(prompt, null, contextData);
        const newOutput = response.generated_text;
        setNodeOutputs(prev => ({ ...prev, [nodeId]: newOutput }));
        setNodes(prevNodes => 
          prevNodes.map(node => {
            if (node.id === nodeId) return { ...node, data: { ...node.data, output: newOutput } };
            return node;
          })
        );
        
        const affectedTargetNodes = currentEdges // Use currentEdges from ref
          .filter(edge => edge.source === nodeId)
          .map(edge => edge.target);
            
        if (affectedTargetNodes.length > 0) {
          console.log(`Node ${nodeId} output changed, affected nodes: `, affectedTargetNodes);
          NodeService.updateNodeOutputService(nodeId, newOutput)
            .then(response => console.log(`Node ${nodeId} output updated on backend:`, response))
            .catch(err => console.error(`Error updating node ${nodeId} output on backend:`, err));
          setNodes(prevNodes => 
            prevNodes.map(node => {
              if (affectedTargetNodes.includes(node.id)) return { ...node, data: { ...node.data, needsRefresh: true } };
              return node;
            })
          );
        }
        return newOutput;
      } catch (error) {
        console.error('Caught Error:', error);
        const errorMessage = error?.message || 'An unknown error occurred';
        return `Error: ${errorMessage}`;
      }
    },
  }), [setNodes, setEdges, setNodeOutputs, debouncedSavePrompt, debouncedSaveLLMConfig]);

  // Add a new node
  const onAddNode = useCallback(async (reactFlowInstance) => {
    if (!reactFlowInstance) return;
    try {
      const centerPosition = getCenterPosition(reactFlowInstance);
      const id = uuidv4();
      // Use ref for current nodes length to make onAddNode itself stable
      const nodeName = `Node ${latestNodesRef.current.length + 1}`;
      const defaultLLMConfig = { model: 'gpt-4', temperature: 0.7, max_tokens: 1250 };

      await NodeService.addNode(id, 'llmNode', ['context', 'query'], ['generated_text'], nodeName, defaultLLMConfig);
      
      const nodeHandlers = createNodeHandlers(id);
      const newNode = {
        id, type: 'llmNode', position: centerPosition,
        data: { nodeName, prompt: '', output: '', variables: [], llm_config: defaultLLMConfig, ...nodeHandlers },
      };
      setNodes((nds) => [...nds, newNode]);
    } catch (error) {
      console.error('Failed to add node:', error);
    }
  }, [createNodeHandlers, setNodes]);

  return {
    onAddNode,
    createNodeHandlers,
  };
}; 