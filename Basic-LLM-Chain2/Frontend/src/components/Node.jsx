import React, { useState, useEffect, useRef } from 'react';
import { Handle, Position } from 'reactflow';
import './Node.css';

// Define available LLM models
const AVAILABLE_MODELS = [
  { id: 'gpt-4', name: 'GPT-4', provider: 'openai' },
  { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openai' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai' },
  // Add Anthropic Models
  { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', provider: 'anthropic' },
  { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet', provider: 'anthropic' },
  { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', provider: 'anthropic' },
  // Add DeepSeek Models
  { id: 'deepseek-chat', name: 'DeepSeek Chat', provider: 'deepseek' },
  // Add Gemini Models
  { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash', provider: 'gemini' },
  // Add other models here in the future, e.g.:
  // { id: 'deepseek-coder-v2', name: 'DeepSeek Coder V2', provider: 'deepseek' },
];

const Node = ({ data, isConnectable, selected }) => {
  const validInputNodes = data.validInputNodes || [];
  const [nodeName, setNodeName] = useState(data.nodeName || 'Node Name');
  const [prompt, setPrompt] = useState(data.prompt || '');
  const [output, setOutput] = useState(String(data.output || ''));
  const [isRunning, setIsRunning] = useState(false);
  const [selectedInputNodes, setSelectedInputNodes] = useState([]);
  const [showTooltip, setShowTooltip] = useState(false);
  // New state variables for UI enhancements
  const [isPromptExpanded, setIsPromptExpanded] = useState(false);
  const [isOutputExpanded, setIsOutputExpanded] = useState(false);
  const [promptHeight, setPromptHeight] = useState(120); // Default height
  const [outputHeight, setOutputHeight] = useState(120); // Default height
  const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false); // New state for config panel
  
  // Initialize LLM config from data, with defaults
  const [llmConfig, setLlmConfig] = useState(() => {
    const initialConfig = data.llm_config || {};
    return {
      provider: initialConfig.provider || 'openai',
      model: initialConfig.model || 'gpt-4',
      temperature: typeof initialConfig.temperature === 'number' ? initialConfig.temperature : 0.7,
      max_tokens: typeof initialConfig.max_tokens === 'number' ? initialConfig.max_tokens : 1000,
    };
  });
  
  const promptRef = useRef(null);
  const outputRef = useRef(null);
  const promptResizeRef = useRef(null);
  const outputResizeRef = useRef(null);
  
  // Handle node name change
  const handleNameChange = (e) => {
    const newName = e.target.value;
    setNodeName(newName);
    if (data.onNameChange) {
      data.onNameChange(newName);
    }
  };
  
  // Handle prompt change
  const handlePromptChange = (e) => {
    const newPrompt = e.target.value;
    setPrompt(newPrompt);
    if (data.onPromptChange) {
      data.onPromptChange(newPrompt);
    }
  };
  
  // Handle LLM config change
  const handleLLMConfigChange = (e) => {
    const { name, value, type } = e.target;
    // Ensure numeric values are stored as numbers
    const processedValue = type === 'number' || name === 'temperature' || name === 'max_tokens' 
      ? parseFloat(value) 
      : value;

    setLlmConfig(prevConfig => {
      const newConfig = {
        ...prevConfig,
        [name]: processedValue
      };

      // If the model is changing, also update the provider
      if (name === 'model') {
        const selectedModel = AVAILABLE_MODELS.find(m => m.id === processedValue);
        if (selectedModel) {
          newConfig.provider = selectedModel.provider;
        }
      }
      
      // Call the onSettingsChange passed from App.jsx to persist
      if (data.onSettingsChange) {
        data.onSettingsChange(newConfig); 
      }
      return newConfig;
    });
  };
  
  // Handle run button click
  const handleRun = async () => {
    if (isRunning) return;
    setIsRunning(true);
    if (data.onRun) {
      const result = await data.onRun(prompt, selectedInputNodes);
      const resultString = typeof result === 'string' ? result : JSON.stringify(result);
      setOutput(resultString);
      if (data.onOutputChange) {
        data.onOutputChange(resultString);
      }
    } else {
      // Mock response for testing
      setTimeout(() => {
        setOutput('AI generated output would appear here...');
      }, 1000);
    }
    setIsRunning(false);
  };
  
  // Insert variable template into prompt textarea
  const insertVariable = (varName) => {
    if (!promptRef.current) return;
    
    console.log(`Inserting variable: ${varName}`);
    
    const textarea = promptRef.current;
    const cursorPos = textarea.selectionStart;
    const textBefore = prompt.substring(0, cursorPos);
    const textAfter = prompt.substring(cursorPos);
    
    // Create template with spaces for better formatting
    const template = `{${varName}}`;
    console.log(`Created template: ${template}`);
    
    // Create new prompt with inserted template
    const newPrompt = textBefore + template + textAfter;
    console.log(`New prompt with template: ${newPrompt}`);
    
    // Update prompt state
    setPrompt(newPrompt);
    
    // Notify parent of the change
    if (data.onPromptChange) {
      data.onPromptChange(newPrompt);
    }
    
    // Set focus back to textarea and position cursor after inserted template
    setTimeout(() => {
      textarea.focus();
      const newCursorPos = cursorPos + template.length;
      textarea.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };
  
  // Toggle whether a node is selected for context
  const toggleInputNode = (nodeId) => {
    setSelectedInputNodes(prev => {
      const isSelected = prev.includes(nodeId);
      const newSelection = isSelected 
        ? prev.filter(id => id !== nodeId) 
        : [...prev, nodeId];
      
      // Notify parent of the change
    if (data.onVariableSelect) {
        data.onVariableSelect(newSelection);
    }
      
      return newSelection;
    });
  };
  
  // Handle prompt expansion toggle
  const togglePromptExpand = () => {
    setIsPromptExpanded(!isPromptExpanded);
  };
  
  // Handle output expansion toggle
  const toggleOutputExpand = () => {
    setIsOutputExpanded(!isOutputExpanded);
  };
  
  // Handle manual resizing of prompt area
  const handlePromptResize = (e) => {
    const startY = e.clientY;
    const startHeight = promptHeight;
    
    const onMouseMove = (moveEvent) => {
      const newHeight = startHeight + moveEvent.clientY - startY;
      if (newHeight >= 80) { // Minimum height
        setPromptHeight(newHeight);
      }
    };
    
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };
  
  // Handle manual resizing of output area
  const handleOutputResize = (e) => {
    const startY = e.clientY;
    const startHeight = outputHeight;
    
    const onMouseMove = (moveEvent) => {
      const newHeight = startHeight + moveEvent.clientY - startY;
      if (newHeight >= 80) { // Minimum height
        setOutputHeight(newHeight);
      }
    };
    
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  };
  
  // Auto-resize textareas
  useEffect(() => {
    if (promptRef.current) {
      promptRef.current.style.height = 'auto';
      promptRef.current.style.height = `${promptRef.current.scrollHeight}px`;
    }
    if (outputRef.current) {
      outputRef.current.style.height = 'auto';
      outputRef.current.style.height = `${outputRef.current.scrollHeight}px`;
    }
  }, [prompt, output]);
  
  // Initialize selected input nodes from prop
  useEffect(() => {
    if (data.selectedVariableIds && data.selectedVariableIds.length > 0) {
      setSelectedInputNodes(data.selectedVariableIds);
    }
  }, [data.selectedVariableIds]);
  
  // Update local llmConfig state if data.llm_config changes from parent
  useEffect(() => {
    if (data.llm_config) {
      setLlmConfig(prevConfig => ({ ...prevConfig, ...data.llm_config }));
    }
  }, [data.llm_config]);
  
  // Calculate CSS classes for expanded state
  const promptAreaClass = `prompt-area ${isPromptExpanded ? 'expanded' : ''}`;
  const outputAreaClass = `output-area ${isOutputExpanded ? 'expanded' : ''}`;
  
  return (
    <div className={`node ${isPromptExpanded || isOutputExpanded ? 'expanded-node' : ''} ${selected ? 'selected' : ''}`}>
      {/* Target handle (Input) - Top */}
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        style={{ background: 'var(--color-primary)', width: '10px', height: '10px' }}
        isConnectable={isConnectable}
      />
      
      {/* Header with node name and run button */}
      <div className="node-header">
        <input
          type="text"
          className="node-name-input"
          value={nodeName}
          onChange={handleNameChange}
          placeholder="Node Name"
        />
        <button 
          className={`run-button ${isRunning ? 'running' : ''}`}
          onClick={handleRun}
          disabled={isRunning}
        >
          {isRunning ? 'Running...' : 'Run'}
        </button>
      </div>
      
      {/* Configuration Toggle Button - only show if selected */}
      {selected && (
        <div className="node-section config-toggle-section">
          <button 
            onClick={() => setIsConfigPanelOpen(!isConfigPanelOpen)}
            className="config-toggle-button app-button-subtle"
          >
            {isConfigPanelOpen ? 'Hide Configuration' : 'Show LLM Configuration'}
          </button>
        </div>
      )}
      
      {/* LLM Configuration Section (Visible when node is selected AND panel is open) */}
      {selected && isConfigPanelOpen && (
        <div className="llm-config-section node-section">
          <div className="section-header">LLM Configuration</div>
          <div className="config-item">
            <label htmlFor={`model-${data.id}`}>Model:</label>
            <select
              id={`model-${data.id}`}
              name="model"
              value={llmConfig.model || 'gpt-4'}
              onChange={handleLLMConfigChange}
              className="llm-config-input llm-config-select"
            >
              {AVAILABLE_MODELS.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>
          <div className="config-item">
            <label htmlFor={`provider-${data.id}`}>Provider:</label>
            <input
              type="text"
              id={`provider-${data.id}`}
              name="provider"
              value={llmConfig.provider || 'openai'}
              readOnly
              className="llm-config-input"
            />
          </div>
          <div className="config-item slider-item">
            <label htmlFor={`temperature-${data.id}`}>Temperature: <span>{typeof llmConfig.temperature === 'number' ? llmConfig.temperature.toFixed(2) : '0.70'}</span></label>
            <input
              type="range"
              id={`temperature-${data.id}`}
              name="temperature"
              value={typeof llmConfig.temperature === 'number' ? llmConfig.temperature : 0.7}
              step="0.01"
              min="0"
              max="2"
              onChange={handleLLMConfigChange}
              className="llm-config-input llm-config-slider nodrag"
            />
          </div>
          <div className="config-item slider-item">
            <label htmlFor={`max_tokens-${data.id}`}>Max Tokens: <span>{typeof llmConfig.max_tokens === 'number' ? llmConfig.max_tokens : 1000}</span></label>
            <input
              type="range"
              id={`max_tokens-${data.id}`}
              name="max_tokens"
              value={typeof llmConfig.max_tokens === 'number' ? llmConfig.max_tokens : 1000}
              step="10"
              min="50"
              max="4096"
              onChange={handleLLMConfigChange}
              className="llm-config-input llm-config-slider nodrag"
            />
          </div>
        </div>
      )}
      
      {/* New Connected Nodes Section */}
      <div className="connected-nodes-section node-section">
        <div className="section-header">
          <span>Input from connected nodes:</span>
          <button 
            className="help-button" 
            onClick={() => setShowTooltip(!showTooltip)}
            type="button"
          >
            ?
          </button>
        </div>
        
        {showTooltip && (
          <div className="tooltip">
            Select connected nodes from the dropdown below.
            Only the selected nodes will have their output included
            as context when this node runs.
            Click on the variable tags below to insert them in your prompt.
          </div>
        )}
        
        {validInputNodes.length > 0 ? (
          <div className="connected-nodes-list">
            {validInputNodes.map(node => (
              <div key={node.id} className="connected-node-item">
                <label className="connected-node-checkbox">
                  <input
                    type="checkbox"
                    checked={selectedInputNodes.includes(node.id)}
                    onChange={() => toggleInputNode(node.id)}
                  />
                  <span>Use as input</span>
                </label>
                <button
                  className="insert-variable-button"
                  onClick={() => insertVariable(node.name)}
                  title={`Click to insert {${node.name}} at cursor position`}
                >
                  {node.name} <span className="insert-icon">→</span>
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-connections-message">
            No connected input nodes. Connect nodes to this node's input first.
          </div>
        )}
      </div>
      
      {/* Prompt area with resize and expand controls */}
      <div className={promptAreaClass}>
        <div className="area-header">
          <span>Prompt:</span>
          <div className="area-controls">
            <button 
              className="resize-handle" 
              ref={promptResizeRef}
              onMouseDown={handlePromptResize}
              title="Drag to resize"
            >
              ⣀
            </button>
            <button
              className="expand-button"
              onClick={togglePromptExpand}
              title={isPromptExpanded ? "Collapse" : "Expand"}
            >
              {isPromptExpanded ? '↙' : '↗'}
            </button>
          </div>
        </div>
      <textarea
        ref={promptRef}
        className="prompt-textarea"
        value={prompt}
        onChange={handlePromptChange}
          placeholder="Type your prompt here. Click on a connected node above to insert it into your prompt."
          style={{ 
            height: isPromptExpanded ? '400px' : `${promptHeight}px`,
            maxHeight: isPromptExpanded ? 'none' : `${promptHeight}px`
          }}
        />
      </div>
      
      {/* Output area with resize and expand controls */}
      <div className={outputAreaClass}>
        <div className="area-header">
          <span>Output:</span>
          <div className="area-controls">
            <button 
              className="resize-handle" 
              ref={outputResizeRef}
              onMouseDown={handleOutputResize}
              title="Drag to resize"
            >
              ⣀
            </button>
            <button
              className="expand-button"
              onClick={toggleOutputExpand}
              title={isOutputExpanded ? "Collapse" : "Expand"}
            >
              {isOutputExpanded ? '↙' : '↗'}
            </button>
          </div>
        </div>
      <textarea
        ref={outputRef}
        className="output-textarea"
        value={output}
        readOnly
        placeholder="AI Generated Output Here..."
          style={{ 
            height: isOutputExpanded ? '400px' : `${outputHeight}px`,
            maxHeight: isOutputExpanded ? 'none' : `${outputHeight}px`
          }}
      />
      </div>
      
      {/* Source handle (Output) - Bottom */}
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        style={{ background: 'var(--color-primary)', width: '10px', height: '10px' }}
        isConnectable={isConnectable}
      />
    </div>
  );
};

export default Node; 