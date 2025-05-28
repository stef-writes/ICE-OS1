import React from 'react';
import { Panel } from 'reactflow';

const FlowControls = ({
  currentFlowName,
  setCurrentFlowName,
  onAddNode,
  executeFlow,
  handleSaveFlow,
  handleSaveFlowAs,
  handleOpenLoadFlowModal,
  nodes,
  isConnecting
}) => {
  return (
    <Panel position="top-right">
      <div className="flow-controls-panel">
        {/* Flow Name Input */}
        <div className="flow-name-container">
          <input
            type="text"
            value={currentFlowName}
            onChange={(evt) => setCurrentFlowName(evt.target.value)}
            placeholder="Flow Name"
            className="flow-name-input"
          />
        </div>

        {/* Action Buttons */}
        <div className="flow-action-buttons">
          <button className="add-node-button app-button" onClick={onAddNode}>
            Add Node
          </button>
          {nodes.length > 0 && (
            <button 
              className="execute-flow-button app-button" 
              onClick={executeFlow}
              disabled={isConnecting}
            >
              Execute Flow
            </button>
          )}
          <button onClick={handleSaveFlow} className="save-flow-button app-button">
            Save
          </button>
          <button onClick={handleSaveFlowAs} className="save-flow-as-button app-button">
            Save As
          </button>
          <button onClick={handleOpenLoadFlowModal} className="load-flow-button app-button">
            Load
          </button>
        </div>
      </div>
    </Panel>
  );
};

export default FlowControls; 