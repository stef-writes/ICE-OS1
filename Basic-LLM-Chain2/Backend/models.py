from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic_core import core_schema
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from bson import ObjectId # For MongoDB ObjectId

# --- Helper for MongoDB ObjectId (Pydantic V2 compatible) ---
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> core_schema.CoreSchema:
        """Return a Pydantic V2 CoreSchema for ObjectId validation."""
        # Validator to ensure the input is a valid ObjectId
        def validate_object_id(value: Any) -> ObjectId:
            if isinstance(value, ObjectId):
                return value
            if ObjectId.is_valid(value):
                return ObjectId(value)
            raise ValueError("Invalid ObjectId")

        # Define how the ObjectId should be serialized (e.g., to a string)
        # For JSON schema, represent as a string
        # For Python, it remains an ObjectId
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.no_info_plain_validator_function(validate_object_id),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: str(x)),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> Dict[str, Any]:
        """Modify the JSON schema to represent ObjectId as a string."""
        # For JSON schema purposes, ObjectId is represented as a string.
        json_schema = handler(core_schema.str_schema())
        return json_schema

# --- Prompt Templating System ---
@dataclass
class MessageTemplate:
    role: str
    template: str # String with placeholders like {user_input}

    def format(self, **kwargs):
        """Format the template string with provided key-value pairs."""
        # Returns a dictionary like {"role": "user", "content": "formatted text"}
        return {"role": self.role, "content": self.template.format(**kwargs)}

class PromptTemplate:
    def __init__(self, messages: List[MessageTemplate]):
        self.messages = messages

    def format_messages(self, **kwargs):
        """Format all MessageTemplates in the list."""
        return [message.format(**kwargs) for message in self.messages]

# --- API Models ---
class Message(BaseModel):
    role: str
    content: str

class ModelConfigInput(BaseModel):
    # Input model for specifying LLM config in API requests
    provider: Literal["openai", "anthropic", "deepseek", "gemini"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1250

class NodeInput(BaseModel):
    # Input model for adding/defining a node via API
    node_id: str
    name: Optional[str] = None # Added optional name field
    node_type: str
    input_keys: List[str] = []
    output_keys: List[str] = []
    # Renamed field from model_config to llm_config due to Pydantic V2 conflict
    llm_config: Optional[ModelConfigInput] = None

class EdgeInput(BaseModel):
    # Input model for adding an edge via API
    from_node: str
    to_node: str

class GenerateTextNodeRequest(BaseModel):
    # Input model for the NEW single-node text generation endpoint
    prompt_text: str # The final, already formatted prompt text
    # Renamed from model_config to avoid Pydantic v2 conflict
    llm_config: Optional[ModelConfigInput] = None # Optional config override
    # Change context_data to allow Any value type to accommodate the mapping object
    context_data: Optional[Dict[str, Any]] = None # Map of node names/ids to their outputs/mapping

class GenerateTextNodeResponse(BaseModel):
    # Output model for the NEW single-node text generation endpoint
    generated_text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None
    duration: Optional[float] = None

class NodeNameUpdate(BaseModel):
    name: str

class TemplateValidationRequest(BaseModel):
    prompt_text: str
    available_nodes: List[str]

class TemplateValidationResponse(BaseModel):
    is_valid: bool
    missing_nodes: List[str]
    found_nodes: List[str]
    warnings: Optional[List[str]] = None

# --- MongoDB Document Schemas ---

class LLMConfigDocument(BaseModel): # Renamed from ModelConfigInput to avoid confusion
    provider: Literal["openai", "anthropic", "deepseek", "gemini"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1250

# --- Pydantic models for specific field updates ---
class NodePromptUpdate(BaseModel):
    prompt: str

class NodeLLMConfigUpdate(BaseModel):
    llm_config: LLMConfigDocument # Reuses the existing LLMConfigDocument

class NodeDocument(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    node_id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="User-defined or default name for the node")
    node_type: str = Field(..., description="Type of the node (e.g., text_input, text_processor)")
    input_keys: List[str] = Field(default_factory=list, description="List of input keys the node expects")
    output_keys: List[str] = Field(default_factory=list, description="List of output keys the node produces")
    prompt: Optional[str] = Field(None, description="The prompt template associated with this node, if any")
    llm_config: Optional[LLMConfigDocument] = Field(None, description="LLM configuration for this node, if applicable")
    output: Any = Field(None, description="The output generated by this node")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of node creation")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last node update")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "node_id": "unique-node-abc-123",
                "name": "My Text Generation Node",
                "node_type": "text_generation",
                "input_keys": ["user_prompt"],
                "output_keys": ["generated_text"],
                "llm_config": {"model": "gpt-3.5-turbo", "temperature": 0.5},
                "output": "This is some AI generated text.",
                "created_at": "2023-10-26T10:00:00Z",
                "updated_at": "2023-10-26T10:05:00Z"
            }
        }

class EdgeDocument(BaseModel): # For embedding edge info in chains
    from_node: str
    from_node_name: str
    to_node: str
    to_node_name: str

# --- New Model for Storing Node Data within a Flow/Chain --- 
class FlowNodeData(BaseModel):
    node_id: str = Field(..., description="The unique ID of the node within the flow")
    name: str = Field(..., description="User-defined name of the node at the time of saving the flow")
    node_type: str = Field(..., description="Type of the node, e.g., 'llmNode'")
    prompt: Optional[str] = Field(None, description="The prompt text entered for the node")
    # Assuming LLMConfigDocument is suitable for storing LLM config per node in a flow
    llm_config: Optional[LLMConfigDocument] = Field(None, description="LLM configuration for this node in the flow")
    position: Optional[Dict[str, float]] = Field(None, description="x, y coordinates of the node in the UI")
    # Optional: Add other fields like selected_variables, output if they should be part of the saved flow state
    # variables: Optional[List[Dict[str, str]]] = None
    # selected_variable_ids: Optional[List[str]] = None
    # output: Optional[Any] = None 

class ChainDocument(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(..., description="Name of the chain/flow")
    
    # Replaces `nodes` and `nodes_info` with a more comprehensive list
    flow_nodes: List[FlowNodeData] = Field(default_factory=list, description="Detailed data for each node in the flow")
    
    edges: List[Dict[str,str]] = Field(default_factory=list, description="Simple connections: {\"from_node\": \"id1\", \"to_node\": \"id2\"}") 
    
    # enhanced_edges can still be populated by backend if useful for other queries, 
    # but flow_nodes will be primary for UI reconstruction.
    enhanced_edges: List[EdgeDocument] = Field(default_factory=list, description="Detailed info of edges in the chain, auto-populated if possible")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of chain creation")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last chain update")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "name": "My Advanced Flow",
                "flow_nodes": [
                    {
                        "node_id": "node-abc-123", 
                        "name": "Data Input Step", 
                        "node_type": "llmNode", 
                        "prompt": "User-entered prompt for this node...",
                        "position": {"x": 100, "y": 150},
                        "llm_config": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 500}
                    },
                    {
                        "node_id": "node-def-456", 
                        "name": "Analysis Step", 
                        "node_type": "llmNode", 
                        "prompt": "Another prompt...",
                        "position": {"x": 400, "y": 200}
                    }
                ],
                "edges": [{"from_node": "node-abc-123", "to_node": "node-def-456"}],
                "enhanced_edges": [], # Example: This might be auto-populated by backend
                "created_at": "2023-10-26T10:00:00Z",
                "updated_at": "2023-10-26T10:05:00Z"
            }
        } 

# Remove or comment out NodeInfo if it's no longer used elsewhere
# class NodeInfo(BaseModel): # For embedding node info in chains
#     node_id: str
#     node_name: str 