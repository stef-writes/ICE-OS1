from typing import Dict, Any, Tuple
from utils import InputValidator # Adjusted import
from llm import track_token_usage, LLMConfig, default_llm_config, get_client # Adjusted import
from models import MessageTemplate, PromptTemplate # Adjusted import
from templates import template_processor # Adjusted import
import traceback
import httpx # Import httpx for specific error catching
import json # Import json for pretty printing dicts

# --- Node Structure ---
class Node:
    def __init__(self, node_id, node_type, input_keys=None, output_keys=None, model_config=None, template=None):
        self.node_id = node_id        # Unique name for this node
        self.node_type = node_type      # Type of operation (e.g., "text_generation")
        self.input_keys = input_keys or [] # List of data keys this node needs from storage
        self.output_keys = output_keys or []# List of data keys this node will produce
        self.data = {}                  # Internal data for the node (not currently used)
        self.model_config = model_config or default_llm_config # Use node-specific or default LLM config
        self.token_usage = None         # To store token usage from the process method
        self.template = template        # Optional template configuration for the node

    async def process(self, inputs):
        """Processes input data based on node type. Calls specific AI functions."""
        print(f"--- Processing Node: {self.node_id} ({self.node_type}) ---")
        result = None
        api_response_for_tracking = None # Store API response here for the tracker

        # Apply template if available
        processed_inputs = self._apply_template(inputs)

        # The token tracker context manager is now placed *inside* relevant node types
        # to ensure it only runs when an actual API call is made.
        if self.node_type == "text_generation":
            with track_token_usage(provider=self.model_config.provider, model_name=self.model_config.model) as usage:
                result_data, api_response_for_tracking = await generate_text(processed_inputs, self.model_config)
                self.token_usage = usage # Store usage info
            result = result_data # Assign the content result

        elif self.node_type == "decision_making":
            with track_token_usage(provider=self.model_config.provider, model_name=self.model_config.model) as usage:
                result_data, api_response_for_tracking = await process_decision(processed_inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        elif self.node_type == "retrieval":
            # Retrieval doesn't use LLM/tokens, so no tracking here
            result = retrieve_data(processed_inputs)

        elif self.node_type == "logic_chain":
            with track_token_usage(provider=self.model_config.provider, model_name=self.model_config.model) as usage:
                result_data, api_response_for_tracking = await logical_reasoning(processed_inputs, self.model_config)
                self.token_usage = usage
            result = result_data

        else:
            print(f"Warning: Unknown node type '{self.node_type}' for node '{self.node_id}'")
            result = None

        # If an API call was made, update the tracker manually (since it's yielded)
        # The tracker printed automatically in its __exit__ method
        if self.token_usage and api_response_for_tracking:
             try:
                 # Ensure we have the dictionary form for update method
                 response_dict = api_response_for_tracking.model_dump()
                 self.token_usage.update(response_dict, self.model_config.provider)
             except Exception as e:
                 print(f"Error updating token tracker: {e}")

        print(f"--- Finished Node: {self.node_id} ---")
        
        # Save the node's output to the database
        if result is not None:
            try:
                # Ensure the result is serializable (e.g., convert complex objects if needed)
                # For now, assume result is a dict or basic type
                from database import update_node_output # Adjusted import
                # Extract the generated text from the result dict for database storage
                output_to_save = None
                if isinstance(result, dict):
                    # Try common output keys in priority order
                    for key in ["generated_text", "decision_output", "reasoning_result", "retrieved_data", "output"]:
                        if key in result:
                            output_to_save = result[key]
                            break
                    # If no standard key found, use the first value
                    if output_to_save is None and result:
                        output_to_save = next(iter(result.values()))
                else:
                    output_to_save = result
                
                if output_to_save is not None:
                    await update_node_output(self.node_id, output_to_save)
                    print(f"Saved output for node {self.node_id} to database.")
                else:
                    print(f"Warning: No suitable output found to save for node {self.node_id}")
            except Exception as e:
                print(f"Error saving node {self.node_id} output to database: {e}")

        return result
    
    def _apply_template(self, inputs):
        """Apply node template if defined, otherwise return original inputs."""
        if not self.template:
            return inputs
            
        # Use the global template processor for consistent processing
        return template_processor.process_node_template(self.template, inputs, self.node_id)

# --- AI Functions ---
async def generate_text(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Uses the configured LLM provider to generate structured text."""
    llm_client = get_client(config.provider)
    context = inputs.get('context', '')
    query = inputs.get('query', '')

    # Common structure for messages, adjust for provider specifics
    # OpenAI: messages=[{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    # Anthropic: system="...", messages=[{"role": "user", "content": ...}, {"role": "assistant", ...}] (though system can be part of messages too)

    if config.provider == "openai":
        system_message_content = f"You are an expert AI assistant. {context}"
        user_message_content = query
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        response = await llm_client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = response.choices[0].message.content
        return {"generated_text": generated_content}, response

    elif config.provider == "anthropic":
        system_prompt = f"You are an expert AI assistant. {context}"
        # Anthropic's API expects the system prompt separately if used,
        # or as the first message if included in the messages list.
        # For simplicity with Claude 3, using the `system` parameter is cleaner.
        
        # Anthropic expects messages to be a list of dicts with 'role' and 'content'
        # The first message must be 'user'.
        anthropic_messages = [{"role": "user", "content": query}]

        response = await llm_client.messages.create(
            model=config.model,
            system=system_prompt,
            messages=anthropic_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        # Anthropic's response structure for content (Claude 3+)
        # response.content is a list of ContentBlock objects.
        # We expect a single text block for typical generation.
        generated_content = ""
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            # Assuming the first content block is the one we want and it's a text block
            first_block = response.content[0]
            if hasattr(first_block, 'text'):
                generated_content = first_block.text
            else: # Fallback or handle other block types if necessary
                print(f"Warning: Anthropic response content block does not have text: {first_block}")
        else:
            print(f"Warning: Anthropic response content is empty or not as expected: {response.content}")
            
        return {"generated_text": generated_content}, response
    
    elif config.provider == "deepseek":
        # DeepSeek API is similar to OpenAI's chat completions
        system_message_content = f"You are an expert AI assistant. {context}"
        user_message_content = query
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        
        request_payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
        
        print("--- DeepSeek API Request --- ")
        try:
            print(f"URL: {llm_client.base_url}chat/completions") # Approximate URL, actual is internal to client
            print(f"Headers: {llm_client.default_headers}")
            print(f"Payload: {json.dumps(request_payload, indent=2)}")
        except Exception as e:
            print(f"Error printing DeepSeek request details: {e}")

        try:
            response = await llm_client.chat.completions.create(**request_payload)
            generated_content = response.choices[0].message.content
            return {"generated_text": generated_content}, response
        except httpx.RequestError as exc:
            print(f"DeepSeek API request failed (httpx.RequestError): {exc.request.method} {exc.request.url}")
            print(f"Underlying error: {exc}")
            raise ConnectionError(f"DeepSeek API connection error: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            print(f"DeepSeek API request failed (httpx.HTTPStatusError): {exc.response.status_code} for url {exc.request.url}")
            print(f"Response content: {exc.response.text}")
            raise ConnectionError(f"DeepSeek API HTTP status error: {exc.response.status_code} - {exc.response.text}") from exc
        except Exception as e:
            print(f"An unexpected error occurred during DeepSeek API call: {type(e).__name__} - {e}")
            traceback.print_exc()
            raise ConnectionError(f"Unexpected error during DeepSeek API call: {e}") from e
    
    elif config.provider == "gemini":
        # For Gemini, llm_client is the genai module.
        # We need to instantiate a GenerativeModel.
        model_name_for_gemini = config.model
        if not model_name_for_gemini.startswith("models/"):
            model_name_for_gemini = f"models/{model_name_for_gemini}"
        gemini_model_instance = llm_client.GenerativeModel(model_name_for_gemini)
        
        # Combine system and user messages for Gemini
        full_prompt_for_gemini = f"You are an expert AI assistant. {context}\n\n{query}"

        # Create generation config for Gemini
        generation_config = llm_client.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        # Make the async call to Gemini
        response = await gemini_model_instance.generate_content_async(
            contents=[full_prompt_for_gemini],
            generation_config=generation_config
        )
        
        # Extract text content from Gemini's response
        generated_content = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_content = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            # Fallback or error handling if response structure is not as expected
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
                raise ValueError(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
            print(f"Warning: Gemini response structure not as expected or empty: {response}")
            generated_content = ""
            
        return {"generated_text": generated_content}, response
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

async def process_decision(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """AI-powered ethical decision-making based on inputs."""
    llm_client = get_client(config.provider)
    scenario = inputs.get("situation", "")
    company_value = inputs.get("value", "")

    if config.provider == "openai":
        system_message_content = "Analyze this scenario based on ethical and company values."
        user_message_content = f"In the given scenario: {scenario}, how does it align with the value: {company_value}?"
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        response = await llm_client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = response.choices[0].message.content
        return {"decision_output": generated_content}, response

    elif config.provider == "anthropic":
        system_prompt = "Analyze this scenario based on ethical and company values."
        user_message_content = f"In the given scenario: {scenario}, how does it align with the value: {company_value}?"
        anthropic_messages = [{"role": "user", "content": user_message_content}]
        response = await llm_client.messages.create(
            model=config.model,
            system=system_prompt,
            messages=anthropic_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = ""
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, 'text'):
                generated_content = first_block.text
        return {"decision_output": generated_content}, response
    
    elif config.provider == "deepseek":
        system_message_content = "Analyze this scenario based on ethical and company values."
        user_message_content = f"In the given scenario: {scenario}, how does it align with the value: {company_value}?"
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        response = await llm_client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = response.choices[0].message.content
        return {"decision_output": generated_content}, response
    
    elif config.provider == "gemini":
        model_name_for_gemini = config.model
        if not model_name_for_gemini.startswith("models/"):
            model_name_for_gemini = f"models/{model_name_for_gemini}"
        gemini_model_instance = llm_client.GenerativeModel(model_name_for_gemini)
        
        full_prompt_for_gemini = f"Analyze this scenario based on ethical and company values.\n\nIn the given scenario: {scenario}, how does it align with the value: {company_value}?"

        generation_config = llm_client.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        response = await gemini_model_instance.generate_content_async(
            contents=[full_prompt_for_gemini],
            generation_config=generation_config
        )
        
        generated_content = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_content = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
                raise ValueError(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
            print(f"Warning: Gemini response structure not as expected or empty: {response}")
            generated_content = ""
            
        return {"decision_output": generated_content}, response
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

def retrieve_data(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves stored data from previous nodes."""
    # Expects 'storage' (the graph's data store) and 'key' (which data to get)
    storage = inputs.get("storage", {})
    key_to_retrieve = inputs.get("key", "")
    retrieved_value = storage.get(key_to_retrieve, "No data found.")
    # Return value using a standard output key for retrieval nodes
    return {"retrieved_data": retrieved_value}

async def logical_reasoning(inputs: Dict[str, Any], config: LLMConfig) -> Tuple[Dict[str, Any], Any]:
    """Processes multi-step logical AI reasoning chains."""
    llm_client = get_client(config.provider)
    premise = inputs.get("premise", "")
    supporting_evidence = inputs.get("supporting_evidence", "")

    if config.provider == "openai":
        system_message_content = "Perform structured logical reasoning."
        user_message_content = f"Given the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step."
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        response = await llm_client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = response.choices[0].message.content
        return {"reasoning_result": generated_content}, response

    elif config.provider == "anthropic":
        system_prompt = "Perform structured logical reasoning."
        user_message_content = f"Given the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step."
        anthropic_messages = [{"role": "user", "content": user_message_content}]
        response = await llm_client.messages.create(
            model=config.model,
            system=system_prompt,
            messages=anthropic_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = ""
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, 'text'):
                generated_content = first_block.text
        return {"reasoning_result": generated_content}, response
    
    elif config.provider == "deepseek":
        system_message_content = "Perform structured logical reasoning."
        user_message_content = f"Given the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step."
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]
        response = await llm_client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        generated_content = response.choices[0].message.content
        return {"reasoning_result": generated_content}, response
    
    elif config.provider == "gemini":
        model_name_for_gemini = config.model
        if not model_name_for_gemini.startswith("models/"):
            model_name_for_gemini = f"models/{model_name_for_gemini}"
        gemini_model_instance = llm_client.GenerativeModel(model_name_for_gemini)
        
        full_prompt_for_gemini = f"Perform structured logical reasoning.\n\nGiven the premise: {premise}, and supporting evidence: {supporting_evidence}, logically conclude the next step."

        generation_config = llm_client.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        response = await gemini_model_instance.generate_content_async(
            contents=[full_prompt_for_gemini],
            generation_config=generation_config
        )
        
        generated_content = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_content = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
                raise ValueError(f"Gemini content generation blocked: {response.prompt_feedback.block_reason_message}")
            print(f"Warning: Gemini response structure not as expected or empty: {response}")
            generated_content = ""
            
        return {"reasoning_result": generated_content}, response
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}") 