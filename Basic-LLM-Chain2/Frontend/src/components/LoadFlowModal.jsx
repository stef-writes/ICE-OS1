import React from 'react';

const LoadFlowModal = ({ isOpen, onClose, chains, onLoad }) => {
  if (!isOpen) return null;

  return (
    <div className="load-flow-modal">
      <div className="modal-content">
        <h2>Load Flow</h2>
        {chains.length === 0 && <p>No saved flows found.</p>}
        <ul>
          {chains.map(chain => (
            <li key={chain.id || chain._id} onClick={() => onLoad(chain)}>
              {chain.name} (ID: {chain.id || chain._id})
            </li>
          ))}
        </ul>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
};

export default LoadFlowModal; 