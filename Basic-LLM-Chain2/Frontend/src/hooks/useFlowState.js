import { useState, useCallback } from 'react';
import { applyNodeChanges, applyEdgeChanges, addEdge } from 'reactflow';
import NodeService from '../services/NodeService';

export const useFlowState = () => {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [nodeOutputs, setNodeOutputs] = useState({});
  const [isConnecting, setIsConnecting] = useState(false);
  
  // Flow metadata
  const [currentFlowName, setCurrentFlowName] = useState('Untitled Flow');
  const [currentFlowId, setCurrentFlowId] = useState(null);
  const [isLoadFlowModalOpen, setIsLoadFlowModalOpen] = useState(false);
  const [availableChains, setAvailableChains] = useState([]);

  // Handler for node changes (drag, select, remove)
  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  // Handler for edge changes (select, remove)
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  // Handler for connecting nodes
  const onConnect = useCallback(
    async (connection) => {
      try {
        setIsConnecting(true);
        // Create the edge in the UI
        setEdges((eds) => addEdge(connection, eds));
        const sourceNodeId = connection.source;
        const targetNodeId = connection.target;
        
        // Add the edge to the backend
        await NodeService.addEdge(sourceNodeId, targetNodeId);
        
        // Update available variables for the target node (multi-input support)
        setNodes((nds) => 
          nds.map((node) => {
            if (node.id === targetNodeId) {
              const sourceNode = nds.find(n => n.id === sourceNodeId);
              const sourceNodeName = sourceNode?.data?.nodeName || 'Unnamed Node';
              const updatedVariables = [
                ...(node.data.variables || []),
                {
                  id: sourceNodeId,
                  name: sourceNodeName
                }
              ];
              // Remove duplicates
              const uniqueVariables = updatedVariables.filter(
                (variable, index, self) => 
                  index === self.findIndex(v => v.id === variable.id)
              );
              return {
                ...node,
                data: {
                  ...node.data,
                  variables: uniqueVariables
                }
              };
            }
            return node;
          })
        );
      } catch (error) {
        // Show backend error (cycle prevention, etc.)
        alert(error.message || 'Failed to connect nodes.');
        // Revert the UI change on error
        setEdges((eds) => eds.filter(e => 
          !(e.source === connection.source && e.target === connection.target)
        ));
      } finally {
        setIsConnecting(false);
      }
    },
    []
  );

  return {
    // State
    nodes,
    edges,
    nodeOutputs,
    isConnecting,
    currentFlowName,
    currentFlowId,
    isLoadFlowModalOpen,
    availableChains,
    
    // Setters
    setNodes,
    setEdges,
    setNodeOutputs,
    setCurrentFlowName,
    setCurrentFlowId,
    setIsLoadFlowModalOpen,
    setAvailableChains,
    
    // Handlers
    onNodesChange,
    onEdgesChange,
    onConnect,
  };
}; 