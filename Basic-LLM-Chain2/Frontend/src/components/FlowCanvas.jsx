import React, { useRef, useCallback } from 'react';
import ReactFlow, {
  Controls,
  Background,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';

import nodeTypes from './CustomNodeTypes';
import FlowControls from './FlowControls';
import LoadFlowModal from './LoadFlowModal';
import { findDirectInputNodes } from '../utils/flowHelpers';
import { useFlowState } from '../hooks/useFlowState';
import { useNodeOperations } from '../hooks/useNodeOperations';
import { useFlowOperations } from '../hooks/useFlowOperations';

function Flow() {
  const reactFlowWrapper = useRef(null);
  const [reactFlowInstance, setReactFlowInstance] = React.useState(null);
  
  // Use custom hooks for state management
  const flowState = useFlowState();
  const {
    nodes,
    edges,
    currentFlowName,
    isLoadFlowModalOpen,
    availableChains,
    isConnecting,
    setCurrentFlowName,
    setIsLoadFlowModalOpen,
    onNodesChange,
    onEdgesChange,
    onConnect,
  } = flowState;

  // Use custom hooks for operations
  const { onAddNode, createNodeHandlers } = useNodeOperations(flowState);
  const {
    executeFlow,
    handleSaveFlow,
    handleSaveFlowAs,
    handleOpenLoadFlowModal,
    handleLoadFlow,
  } = useFlowOperations(flowState, reactFlowInstance, createNodeHandlers);

  // Handle adding a node with the current ReactFlow instance
  const handleAddNode = useCallback(() => {
    onAddNode(reactFlowInstance);
  }, [onAddNode, reactFlowInstance]);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Process nodes to add valid input nodes data
  const processedNodes = nodes.map(node => {
    const directInputNodes = findDirectInputNodes(node.id, nodes, edges);
    return {
      ...node,
      data: {
        ...node.data,
        validInputNodes: directInputNodes,
      },
      type: node.type,
    };
  });

  return (
    <div className="reactflow-wrapper" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={processedNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        connectionMode="loose"
        fitView
      >
        <Background />
        <Controls />
        <FlowControls
          currentFlowName={currentFlowName}
          setCurrentFlowName={setCurrentFlowName}
          onAddNode={handleAddNode}
          executeFlow={executeFlow}
          handleSaveFlow={handleSaveFlow}
          handleSaveFlowAs={handleSaveFlowAs}
          handleOpenLoadFlowModal={handleOpenLoadFlowModal}
          nodes={nodes}
          isConnecting={isConnecting}
        />
      </ReactFlow>

      <LoadFlowModal 
        isOpen={isLoadFlowModalOpen}
        onClose={() => setIsLoadFlowModalOpen(false)}
        chains={availableChains}
        onLoad={handleLoadFlow}
      />
    </div>
  );
}

function FlowCanvas() {
  return (
    <div className="app">
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </div>
  );
}

export default FlowCanvas; 