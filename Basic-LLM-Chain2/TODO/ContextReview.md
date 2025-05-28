# Backend Context Management Review

This document provides a detailed review of how context is managed, passed, stored, retrieved, and used within the backend of the application. The primary files involved are `Backend/script_chain.py`, `Backend/main.py`, `Backend/templates.py`, and `Backend/database.py`.

## 1. Core Context Storage: `NamespacedStorage`

The `NamespacedStorage` class in `Backend/script_chain.py` is the heart of in-memory context management during the execution of a `ScriptChain`.

- **Structure**: `self.data` is a dictionary where keys are `node_id` and values are dictionaries representing the outputs of that node.
  ```python
  # self.data in NamespacedStorage
  # {
  #   "node_id_1": {"output_key_1": "value1", "output_key_2": "value2"},
  #   "node_id_2": {"generated_text": "some text", "output": "some text"}
  # }
  ```

- **Storing Context**:
  - The `store(self, node_id, data)` method updates the storage. It expects `data` to be a dictionary.
  - When a node in `ScriptChain` finishes processing, its results (a dictionary) are stored using `self.storage.store(node_id, node_result)`.
  - The `generate_text_node_api` in `main.py` also directly updates this storage if `context_data` from the frontend indicates a change in an input node's content before the current node is run.

- **Retrieving Context**:
  - `get(self, node_id, key=None)`: Retrieves data for a specific `node_id`. If `key` is provided, it returns a specific value; otherwise, it returns the entire dictionary of outputs for that node.
  - `get_node_output(self, node_id, key=None)`: A helper method that directly calls `get`.
  - `get_by_key(self, key)`: Scans all nodes for a specific `key` (non-namespaced) and returns the first value found. Used for backward compatibility.
  - `get_flattened(self)`: Returns a flattened view of all data, where key collisions are resolved by the last encountered value. Also for backward compatibility.
  - `get_all_data(self)`: Returns a flat dictionary with `node_id:key` as keys.

- **Usage in `ScriptChain.execute`**:
  - Before a node processes, `inputs_for_node` is prepared.
  - Outputs from `upstream_nodes` are collected using `self.storage.get_node_output(upstream_id)`.
  - These outputs are added to `inputs_for_node` both with namespaced keys (`f"{upstream_id}:{output_key}"`) and directly if the `output_key` matches one of the `node_instance.input_keys`.
  - For backward compatibility, `inputs_for_node["storage"]` is populated with `self.storage.get_flattened()`, and `inputs_for_node["get_node_output"]` is set to `self.storage.get_node_output` to allow templates to access specific node outputs.

## 2. Context Passing from Frontend to Backend

The primary way context (outputs of other nodes) is passed from the frontend to the backend for a single node execution is via the `context_data` field in the `GenerateTextNodeRequest` model, handled by the `/generate_text_node` endpoint in `Backend/main.py`.

- **`GenerateTextNodeRequest`**:
  ```python
  class GenerateTextNodeRequest(BaseModel):
      prompt_text: str
      llm_config: Optional[ModelConfigInput] = None
      context_data: Optional[Dict[str, Any]] = None # Key field for context
  ```

- **Frontend Assembly (`Frontend/src/hooks/useNodeOperations.js`)**:
  - When a node's `onRun` handler is triggered, it prepares `contextData`.
  - It fetches the `latestNodeOutputs` from the backend for `activeInputNodeIds` (nodes explicitly selected as inputs in the UI).
  - `contextData` includes:
    - `__node_mapping`: A dictionary mapping user-defined node names to their unique `node_id`s. This helps the backend resolve template references.
      ```javascript
      // contextData['__node_mapping'] example
      // { "My Input Node": "node_id_1", "Another Node": "node_id_2" }
      ```
    - `__current_node`: The `node_id` of the node currently being executed.
    - `id:{inputId}`: For each active input node, its output is stored with a key prefixed by `id:`. This is the **primary data storage** method.
      ```javascript
      // contextData['id:node_id_1'] = "Output of node_id_1";
      ```
    - `{resolvedSourceNodeName}`: For backward compatibility and easier template referencing by name, the output is also stored using the node's resolved name as the key.
      ```javascript
      // contextData['My Input Node'] = "Output of node_id_1";
      ```

- **Backend Reception (`Backend/main.py` - `generate_text_node_api`)**:
  - The `request.context_data` is received.
  - `node_mapping` and `current_node_id` are extracted from special keys (`__node_mapping`, `__current_node`).
  - **Context Synchronization**: Before processing the main prompt, the API checks if any input nodes (from `node_mapping`) have updated content in `request.context_data` compared to what's currently in the `script_chain.storage`. If so, it updates `script_chain.storage` and increments the version of those input nodes using `script_chain.increment_node_version(node_id)`. This is crucial for dynamic updates and ensuring nodes re-run if their dependencies change.

## 3. Context Usage in Prompt Templating

The `TemplateProcessor` class in `Backend/templates.py` is responsible for resolving template strings (e.g., `{NodeName}` or `{NodeName[item_index]}`) within prompts using the provided context.

- **`process_node_references(self, prompt_text, context_data, ...)`**: This is the key method used by `generate_text_node_api`.
  - It takes the `prompt_text` and the `context_data` (received from the frontend).
  - **Lookup Priority for Node References**:
    1.  **ID via `__node_mapping`**: If a `{NodeName}` is found in `prompt_text`, it first checks `context_data['__node_mapping']` to get the `node_id`. It then tries to find the output using `context_data[f"id:{node_id}"]`.
    2.  **Exact Name Match**: If not found by ID, it tries to find the output directly using `context_data[NodeName]`.
    3.  **Normalized Name Match**: If still not found, it tries a case-insensitive, stripped name match against keys in `context_data`.
  - **Item Access**: Handles references like `{NodeName[index]}` or `{NodeName:item(index)}` by using `DataAccessor` (from `utils.py`) to parse and retrieve specific items from list-like outputs.
  - **Output**: Returns the `processed_prompt` with references replaced by their actual values, and `processed_node_values` (a dictionary of which node references were successfully replaced).
  - **Error Handling**: If a reference cannot be resolved, it inserts an error message like `[CONTEXT_ERROR: Content for 'NodeName' not found in provided context_data]` into the prompt.

- **`process_node_template(self, template, inputs, node_id=None)`**: This method is used internally by `Node._apply_template` if a node has a predefined structured template (not just a single prompt string).
  - `inputs` here typically come from the `ScriptChain.execute` method's `inputs_for_node` dictionary.
  - It can use a `get_output(node_id, output_key=None)` function made available in `template_context` to fetch specific outputs from other nodes via `inputs["get_node_output"]` (which points to `NamespacedStorage.get_node_output`).

- **System Prompt Enhancement (`Backend/main.py` - `generate_text_node_api`)**:
  - The `system_content` passed to the LLM is dynamically built.
  - It includes the `__node_mapping` to inform the LLM about the available nodes and their IDs.
  - It also lists all keys from `request.context_data` (excluding special keys) to inform the LLM about the data it has access to.
  ```python
  # Example part of system_content
  # "You have access to a graph of connected nodes with the following name-to-ID mapping: {'My Input Node': 'node_id_1'}"
  # "You have access to information from the following nodes: id:node_id_1, My Input Node."
  # "Use this information to inform your response."
  ```

## 4. Persistence of Context (Node Outputs)

Node outputs, once generated, are persisted to the database.

- **`ScriptChain.Node.process`**:
  - After a node's specific processing logic (e.g., `generate_text`), if a `result` is obtained:
    - It calls `from database import update_node` (which seems to be an error, likely meant to be `update_node_output`).
    - It attempts to save the `result` to the database. *Correction: The `Node.process` method calls `database.update_node(self.node_id, result)`. The `database.py` file does not have an `update_node` function that takes these arguments. It has `save_node` which handles insert/update logic but expects `NodeDocument` or dict, and `update_node_output` which is specific to the output field.* This part needs review in the code.

- **`Backend/main.py` - `generate_text_node_api`**:
  - After the LLM generates `response_content`:
    - If `current_node_id` is known (meaning this API call is for executing a specific node in a chain):
      - The `result_data` (containing `generated_text`, `output`, `content`) is stored in the in-memory `script_chain.storage.store(current_node_id, result_data)`.
      - `script_chain.increment_node_version(current_node_id)` is called.
      - Crucially, `await update_node_output(current_node_id, response_content)` is called. This function in `database.py` updates *only* the `output` field and `updated_at` timestamp for the specified `node_id` in the `nodes` collection.

- **`Backend/database.py`**:
  - `update_node_output(node_id: str, output_data: Any) -> bool`: Updates the `output` field of a node in MongoDB. This is the primary mechanism for saving the result of a node's execution persistently.
  - `save_node(...)`: This function is more for creating or fully updating a node's document structure (name, type, keys, llm_config, etc.), not just its output from an execution. When a node is initially added via `add_node_api`, its `output` field is initialized to `None`.

## 5. Overall Context Flow for a Single Node Execution (e.g., via `/generate_text_node`)

1.  **Frontend (`useNodeOperations.js`)**:
    *   User triggers "Run" on a node.
    *   `contextData` is assembled:
        *   `__current_node`: ID of the node being run.
        *   `__node_mapping`: Names to IDs of connected input nodes.
        *   Outputs of active input nodes (fetched from backend or local state) are added, keyed by `id:{inputId}` and by `resolvedSourceNodeName`.
    *   `NodeService.generateText(prompt, llm_config, contextData)` is called.

2.  **Backend API (`main.py` - `/generate_text_node`)**:
    *   Receives `prompt_text`, `llm_config`, and `context_data`.
    *   Extracts `__node_mapping` and `__current_node`.
    *   **Context Sync**: Compares `context_data` values for input nodes with `script_chain.storage`. If different, updates `script_chain.storage` and increments versions of those input nodes.
    *   **Template Processing (`templates.py`)**: `template_processor.process_node_references(prompt_text, context_data)` is called. This resolves `{NodeReference}` in the `prompt_text` using values from `context_data`.
    *   **System Prompt Construction**: A system prompt is built, including the `__node_mapping` and available context keys, to guide the LLM.
    *   **LLM Call**: The processed prompt and system prompt are sent to the configured LLM.
    *   **Result Handling**:
        *   The LLM's `response_content` is received.
        *   If `current_node_id` is set:
            *   `script_chain.storage.store(current_node_id, result_data)`: Stores output in memory.
            *   `script_chain.increment_node_version(current_node_id)`: Updates version.
            *   `database.update_node_output(current_node_id, response_content)`: **Persists output to MongoDB.**
    *   Returns `GenerateTextNodeResponse` (containing `generated_text`) to the frontend.

3.  **Frontend (`useNodeOperations.js`)**:
    *   Receives the response.
    *   Updates local React state (`nodeOutputs`, and the node's `data.output`) with the `generated_text`.

## 6. Context Flow for Full Chain Execution (`ScriptChain.execute`)

This flow is primarily backend-internal once initiated.

1.  **Initiation**: `execute_api` in `main.py` calls `script_chain.execute()`. Initial inputs can be passed to populate `script_chain.storage`.
2.  **Topological Sort**: Nodes are ordered for execution.
3.  **Update Check**: `node_needs_update(node_id)` checks if a node or its dependencies have newer versions. If so, cached results for that node in `script_chain.storage` might be cleared.
4.  **Node Iteration**: For each node in execution order:
    *   `inputs_for_node` is prepared:
        *   Outputs from upstream nodes are fetched from `script_chain.storage.get_node_output()`.
        *   These are made available by namespaced keys, direct keys (if in `input_keys`), and via `storage` (flattened) and `get_node_output` accessors for templating.
    *   Node `process(inputs_for_node)` method is called:
        *   If the node has a `self.template` (structured template), `_apply_template` calls `template_processor.process_node_template` using `inputs_for_node`.
        *   The node performs its action (e.g., `generate_text` AI function). The `inputs` to these AI functions (like `generate_text`, `process_decision`) contain the resolved context. The `context` parameter specifically is often extracted from `inputs.get('context', '')`.
    *   **Result Storage**: The `node_result` (a dictionary) is stored in `script_chain.storage.store(node_id, node_result)`.
    *   **Versioning**: `self.increment_node_version(node_id)` is called.
    *   **Persistent Storage of Output**: *The `ScriptChain.execute()` flow itself does NOT appear to directly call `database.update_node_output()` for each node. The current implementation in `Node.process()` has a database call (`await update_node(self.node_id, result)`) that needs verification for its intended function and arguments, as it doesn't align with `database.py`'s `update_node_output` or `save_node` signatures correctly.*

## Key Observations and Potential Areas for Review

*   **Dual Context Paths**:
    *   **`/generate_text_node` (Single Node Execution)**: Context is explicitly passed from the frontend (`context_data`) and heavily processed for templating and system prompts. Node output is explicitly saved to the database via `update_node_output`.
    *   **`/execute` (Full Chain Execution)**: Context flows internally via `NamespacedStorage`. Prompt templating (if nodes have `self.template`) uses `process_node_template`. The mechanism for persisting individual node outputs to the DB during a full chain execution needs clarification (the call in `Node.process` seems problematic).

*   **Context for LLM**: The actual "context" string provided to the LLM in `generate_text` and other AI functions within `script_chain.py` is derived from `inputs.get('context', '')`. The construction of this specific `inputs['context']` field needs to be traced from how `inputs_for_node` is built in `ScriptChain.execute` or how `processed_inputs` is formed in `Node.process` after templating. For `/generate_text_node`, the system prompt is dynamically built using available keys from `request.context_data`.

*   **Database Persistence of Outputs**:
    *   Outputs are reliably saved to DB when `/generate_text_node` is used.
    *   The persistence of outputs during a full `/execute` call is less clear and the `Node.process` database call needs review. If outputs from a full chain run are not saved, then loading a saved chain might not show intermediate results unless each node was individually run.

*   **`__node_mapping`**: This is a crucial piece of metadata sent from the frontend to help the backend resolve node names to IDs, especially for template processing.

*   **Session Management**: Context (`ScriptChain` instances, including their `NamespacedStorage`) is session-specific, managed by `script_chain_store` in `main.py` using `session_id`.

This review provides a deep dive into backend context mechanisms. Clarifying the database persistence during full chain execution in `ScriptChain.execute` / `Node.process` is a key follow-up.