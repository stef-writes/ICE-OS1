import os
from openai import AsyncOpenAI
import anthropic
from dotenv import load_dotenv
from contextlib import contextmanager
import time
from typing import Dict, Any, Literal
import google.generativeai as genai # Added for Gemini

load_dotenv()

# --- LLM Configuration Class ---
class LLMConfig:
    def __init__(self, provider: Literal["openai", "anthropic", "deepseek", "gemini"] = "openai", model="gpt-4", temperature=0.7, max_tokens=1250):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

# --- Default Config Instance ---
# Create a default configuration instance to use
default_llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",        # Or "gpt-3.5-turbo", etc.
    temperature=0.7,      # Controls randomness (0.0=deterministic, 1.0=more random)
    max_tokens=1250        # Max length of the AI's generated response
)

# --- Token Usage Tracking ---
@contextmanager
def track_token_usage(provider: Literal["openai", "anthropic", "deepseek", "gemini"] = "openai", model_name: str = "gpt-4"):
    """Context manager to track API token usage."""
    class TokenUsage:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.start_time = time.time()
            self.end_time = None
            self.cost = 0 # Note: Based on approximate rates

        def update(self, response_dict, response_provider: Literal["openai", "anthropic", "deepseek", "gemini"]):
            """Update token counts from API response dictionary."""
            if response_provider == "openai":
                usage = response_dict.get("usage", {})
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
                # Approximate cost calculation (rates depend heavily on the actual model)
                prompt_cost = (self.prompt_tokens / 1000) * 0.0015  # Sample rate for OpenAI
                completion_cost = (self.completion_tokens / 1000) * 0.002 # Sample rate for OpenAI
                self.cost = prompt_cost + completion_cost
            elif response_provider == "anthropic":
                # Anthropic's API response for usage might be different.
                # Example: response_dict.get("usage", {}).get("input_tokens")
                #          response_dict.get("usage", {}).get("output_tokens")
                # This needs to be adjusted based on actual Anthropic API response.
                # For Claude 3, the response object has response.usage.input_tokens and response.usage.output_tokens
                usage = response_dict.get("usage", {})
                if usage: # Check if usage information is present
                    self.prompt_tokens += usage.get("input_tokens", 0)
                    self.completion_tokens += usage.get("output_tokens", 0)
                    self.total_tokens = self.prompt_tokens + self.completion_tokens

                # Placeholder for Anthropic cost calculation - update with actual rates
                # Rates for Claude 3 Opus (example): $15/million input, $75/million output (as of March 2024)
                # Rates for Claude 3 Sonnet (example): $3/million input, $15/million output
                # This needs to be model-specific if different Claude models have different rates.
                # For now, using Opus rates as a placeholder.
                prompt_cost_rate = 15 / 1000000
                completion_cost_rate = 75 / 1000000
                if "sonnet" in model_name.lower(): #粗略的检查
                    prompt_cost_rate = 3 / 1000000
                    completion_cost_rate = 15 / 1000000
                
                self.cost = (self.prompt_tokens * prompt_cost_rate) + (self.completion_tokens * completion_cost_rate)
            elif response_provider == "deepseek":
                usage = response_dict.get("usage", {})
                self.prompt_tokens += usage.get("total_tokens", 0) # Assuming DeepSeek uses 'total_tokens' for combined count
                self.completion_tokens += 0 # Assuming DeepSeek includes completion in 'total_tokens'
                self.total_tokens += usage.get("total_tokens", 0)
                # Placeholder for DeepSeek cost calculation - update with actual rates
                # Using a generic rate for now
                self.cost = (self.total_tokens / 1000) * 0.002 
            elif response_provider == "gemini":
                # Assuming Gemini's response structure for token usage is similar to OpenAI
                # This needs to be adjusted based on actual Gemini API response.
                # Gemini API typically returns total_tokens.
                # We will need to infer prompt and completion tokens if not directly provided.
                # For now, let's assume it provides total_tokens directly in a 'usage' like object or attribute
                usage = response_dict.get("usage_metadata", {}) # Example, adjust if different
                prompt_tokens = usage.get("prompt_token_count", 0)
                completion_tokens = usage.get("candidates_token_count", 0) # Or "completion_token_count"
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.total_tokens += usage.get("total_token_count", prompt_tokens + completion_tokens)
                
                # Placeholder for Gemini cost calculation - update with actual rates
                # Example rates for Gemini Pro (adjust as needed)
                # Input: $0.000125 / 1K characters, Output: $0.000375 / 1K characters
                # Or by tokens: $0.0025/1k tokens (placeholder)
                # This needs to be model-specific
                # Using a simplified token-based placeholder for now
                self.cost = (self.total_tokens / 1000) * 0.0025

        def finish(self):
            self.end_time = time.time()

        def __str__(self):
            duration = round(self.end_time - self.start_time, 2) if self.end_time else 0
            return (
                f"--- Token Usage ({provider}) ---\n"
                f"  Prompt Tokens:     {self.prompt_tokens}\n"
                f"  Completion Tokens: {self.completion_tokens}\n"
                f"  Total Tokens:      {self.total_tokens}\n"
                f"  Est. Cost (USD):   ${self.cost:.6f}\n"
                f"  API Call Duration: {duration}s\n"
                f"-------------------"
            )

    token_usage_tracker = TokenUsage()
    try:
        yield token_usage_tracker
    finally:
        token_usage_tracker.finish()
        # Ensure the full string is printed, especially if it's long
        # print(str(token_usage_tracker)) # Original print
        full_usage_string = str(token_usage_tracker)
        # Print in chunks if very long, or ensure your terminal/logging handles it
        # For now, direct print, but be mindful of potential truncation in some environments.
        print(full_usage_string)

# --- Client Setup ---
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") # Added for DeepSeek
gemini_api_key = os.getenv("GEMINI_API_KEY")     # Added for Gemini

openai_client = None
anthropic_client = None
deepseek_client = None # Added for DeepSeek
gemini_client = None   # Added for Gemini

if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found. OpenAI models will not be available.")
else:
    try:
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        print("--- OpenAI Client Initialized Successfully ---")
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")

if not anthropic_api_key:
    print("Warning: ANTHROPIC_API_KEY not found. Anthropic models will not be available.")
else:
    try:
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        print("--- Anthropic Client Initialized Successfully ---")
    except Exception as e:
        print(f"Error setting up Anthropic client: {e}")

if not deepseek_api_key:
    print("Warning: DEEPSEEK_API_KEY not found. DeepSeek models will not be available.")
else:
    try:
        # Use OpenAI's AsyncOpenAI client with DeepSeek's base_url
        deepseek_client = AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com" # Correct base URL for DeepSeek
        )
        print("--- DeepSeek Client Initialized Successfully ---")
    except Exception as e:
        print(f"Error setting up DeepSeek client: {e}")

if not gemini_api_key:
    print("Warning: GEMINI_API_KEY not found. Gemini models will not be available.")
else:
    try:
        genai.configure(api_key=gemini_api_key)
        # For Gemini, the client (model) is typically instantiated when making a call,
        # e.g., model = genai.GenerativeModel('gemini-pro')
        # So, we might not need a global client in the same way,
        # but we'll set a placeholder for consistency or if a specific client object is needed.
        # If direct model instantiation per call is the pattern, get_client will handle it.
        gemini_client = genai # Storing the configured module itself, or a dummy client if needed
        print("--- Gemini API Configured Successfully ---")
    except Exception as e:
        print(f"Error setting up Gemini client: {e}")

# Consolidate client access
# Deprecating the global 'client' in favor of provider-specific clients (openai_client, anthropic_client)
# Code using 'client' will need to be updated to select the correct client based on LLMConfig.provider 

def get_client(provider: Literal["openai", "anthropic", "deepseek", "gemini"]):
    if provider == "openai":
        if not openai_client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")
        return openai_client
    elif provider == "anthropic":
        if not anthropic_client:
            raise ValueError("Anthropic client not initialized. Check ANTHROPIC_API_KEY.")
        return anthropic_client
    elif provider == "deepseek":
        if not deepseek_client:
            raise ValueError("DeepSeek client not initialized. Check DEEPSEEK_API_KEY.")
        return deepseek_client # Placeholder
    elif provider == "gemini":
        if not gemini_client: # or check if API key is configured
            raise ValueError("Gemini API not configured. Check GEMINI_API_KEY.")
        # For Gemini, you might return the genai module or a specific model instance factory
        return gemini_client # Returning the configured genai module
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 