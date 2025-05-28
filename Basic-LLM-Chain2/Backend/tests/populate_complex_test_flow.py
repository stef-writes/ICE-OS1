#!/usr/bin/env python
import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = f"test_session_{int(time.time())}"

print(f"Using test session ID: {SESSION_ID}")
print(f"API Base URL: {API_BASE_URL}")

def make_api_request(method, endpoint, data=None):
    """Helper function to make API requests with proper error handling"""
    url = f"{API_BASE_URL}{endpoint}?session_id={SESSION_ID}"
    headers = {"Content-Type": "application/json"}
    
    try:
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=data)
        else:
            response = requests.get(url, headers=headers)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {e.response.text}")
        raise

def create_chained_math_test():
    """Create and execute a chained math test using different LLM providers"""
    
    print("\n=== CHAINED MATH TEST SCRIPT ===")
    print("Creating a chain: OpenAI -> Anthropic -> DeepSeek -> Gemini")
    print("Each node performs a math operation on the previous result\n")
    
    # Node configurations
    base_prompt = "You are a math expert. Only respond with the numerical answer to the given equation.\n\n"
    
    nodes = [
        {
            "id": f"math_openai_{int(time.time())}",
            "name": "Math Node 1",
            "prompt": base_prompt + "10 + 32 = ?",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 50
            },
            "expected_output": "42"
        },
        {
            "id": f"math_anthropic_{int(time.time())}",
            "name": "Math Node 2", 
            "prompt": base_prompt + "{Math Node 1} + 100 = ?",
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "temperature": 0.1,
                "max_tokens": 50
            },
            "expected_output": "142"
        },
        {
            "id": f"math_deepseek_{int(time.time())}",
            "name": "Math Node 3",
            "prompt": base_prompt + "10 x {Math Node 1} = ?",
            "llm_config": {
                "provider": "deepseek", 
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 50
            },
            "expected_output": "420"
        },
        {
            "id": f"math_gemini_{int(time.time())}",
            "name": "Math Node 4",
            "prompt": base_prompt + "{Math Node 3} - {Math Node 2} = ?",
            "llm_config": {
                "provider": "gemini",
                "model": "gemini-1.5-flash", 
                "temperature": 0.1,
                "max_tokens": 50
            },
            "expected_output": "278"  # 420 - 142 = 278
        }
    ]
    
    # Store actual outputs
    actual_outputs = {}
    
    try:
        # Step 1: Create all nodes
        print("--- STEP 1: Creating Nodes ---")
        for i, node in enumerate(nodes):
            print(f"Creating {node['name']} (ID: {node['id']}) using {node['llm_config']['provider']}")
            
            node_data = {
                "node_id": node["id"],
                "name": node["name"],
                "node_type": "text_generation",
                "input_keys": [],
                "output_keys": ["generated_text"],
                "llm_config": node["llm_config"]
            }
            
            result = make_api_request("POST", "/add_node", node_data)
            print(f"  ✓ Node created: {result.get('message', 'Success')}")
            
            # Set the prompt
            prompt_data = {"prompt": node["prompt"]}
            make_api_request("PUT", f"/nodes/{node['id']}/prompt", prompt_data)
            print(f"  ✓ Prompt set: {node['prompt'][:50]}...")
        
        print()
        
        # Step 2: Create edges
        print("--- STEP 2: Creating Edges ---")
        edges = [
            (nodes[0]["id"], nodes[1]["id"]),  # Node 1 -> Node 2
            (nodes[0]["id"], nodes[2]["id"]),  # Node 1 -> Node 3  
            (nodes[1]["id"], nodes[3]["id"]),  # Node 2 -> Node 4
            (nodes[2]["id"], nodes[3]["id"])   # Node 3 -> Node 4
        ]
        
        for from_node, to_node in edges:
            edge_data = {"from_node": from_node, "to_node": to_node}
            result = make_api_request("POST", "/add_edge", edge_data)
            print(f"  ✓ Edge created: {from_node[:15]}... -> {to_node[:15]}...")
        
        print()
        
        # Step 3: Execute nodes sequentially
        print("--- STEP 3: Executing Nodes ---")
        
        # Execute Node 1 (OpenAI) - no dependencies
        print(f"\nExecuting {nodes[0]['name']} ({nodes[0]['llm_config']['provider']})...")
        print(f"Prompt: {nodes[0]['prompt']}")
        
        context_data = {
            "__node_mapping": {},
            "__current_node": nodes[0]["id"]
        }
        
        payload = {
            "prompt_text": nodes[0]["prompt"],
            "llm_config": nodes[0]["llm_config"],
            "context_data": context_data
        }
        
        result = make_api_request("POST", "/generate_text_node", payload)
        actual_outputs[nodes[0]["id"]] = result["generated_text"].strip()
        print(f"Output: {actual_outputs[nodes[0]['id']]}")
        print(f"Expected: {nodes[0]['expected_output']}")
        
        # Execute Node 2 (Anthropic) - depends on Node 1
        print(f"\nExecuting {nodes[1]['name']} ({nodes[1]['llm_config']['provider']})...")
        print(f"Prompt: {nodes[1]['prompt']}")
        
        context_data = {
            "__node_mapping": {nodes[0]["name"]: nodes[0]["id"]},
            f"id:{nodes[0]['id']}": actual_outputs[nodes[0]["id"]],
            nodes[0]["name"]: actual_outputs[nodes[0]["id"]],
            "__current_node": nodes[1]["id"]
        }
        
        payload = {
            "prompt_text": nodes[1]["prompt"],
            "llm_config": nodes[1]["llm_config"], 
            "context_data": context_data
        }
        
        result = make_api_request("POST", "/generate_text_node", payload)
        actual_outputs[nodes[1]["id"]] = result["generated_text"].strip()
        print(f"Output: {actual_outputs[nodes[1]['id']]}")
        print(f"Expected: {nodes[1]['expected_output']}")
        
        # Execute Node 3 (DeepSeek) - depends on Node 1
        print(f"\nExecuting {nodes[2]['name']} ({nodes[2]['llm_config']['provider']})...")
        print(f"Prompt: {nodes[2]['prompt']}")
        
        context_data = {
            "__node_mapping": {nodes[0]["name"]: nodes[0]["id"]},
            f"id:{nodes[0]['id']}": actual_outputs[nodes[0]["id"]],
            nodes[0]["name"]: actual_outputs[nodes[0]["id"]],
            "__current_node": nodes[2]["id"]
        }
        
        payload = {
            "prompt_text": nodes[2]["prompt"],
            "llm_config": nodes[2]["llm_config"],
            "context_data": context_data
        }
        
        result = make_api_request("POST", "/generate_text_node", payload)
        actual_outputs[nodes[2]["id"]] = result["generated_text"].strip()
        print(f"Output: {actual_outputs[nodes[2]['id']]}")
        print(f"Expected: {nodes[2]['expected_output']}")
        
        # Execute Node 4 (Gemini) - depends on Node 2 and Node 3
        print(f"\nExecuting {nodes[3]['name']} ({nodes[3]['llm_config']['provider']})...")
        print(f"Prompt: {nodes[3]['prompt']}")
        
        context_data = {
            "__node_mapping": {
                nodes[1]["name"]: nodes[1]["id"],  # Math Node 2
                nodes[2]["name"]: nodes[2]["id"]   # Math Node 3
            },
            f"id:{nodes[1]['id']}": actual_outputs[nodes[1]["id"]],
            f"id:{nodes[2]['id']}": actual_outputs[nodes[2]["id"]],
            nodes[1]["name"]: actual_outputs[nodes[1]["id"]],
            nodes[2]["name"]: actual_outputs[nodes[2]["id"]],
            "__current_node": nodes[3]["id"]
        }
        
        payload = {
            "prompt_text": nodes[3]["prompt"],
            "llm_config": nodes[3]["llm_config"],
            "context_data": context_data
        }
        
        result = make_api_request("POST", "/generate_text_node", payload)
        actual_outputs[nodes[3]["id"]] = result["generated_text"].strip()
        print(f"Output: {actual_outputs[nodes[3]['id']]}")
        print(f"Expected: {nodes[3]['expected_output']}")
        
        # Step 4: Analyze results
        print("\n--- STEP 4: Test Results ---")
        print("\nExpected calculation chain:")
        print("  10 + 32 = 42 (OpenAI)")
        print("  42 + 100 = 142 (Anthropic)")  
        print("  10 × 42 = 420 (DeepSeek)")
        print("  420 - 142 = 278 (Gemini)")
        
        print("\nActual results:")
        all_correct = True
        for i, node in enumerate(nodes):
            actual = actual_outputs[node["id"]]
            expected = node["expected_output"]
            is_correct = actual == expected
            status = "✓ PASS" if is_correct else "✗ FAIL"
            print(f"  {node['name']} ({node['llm_config']['provider']}): {actual} {status}")
            if not is_correct:
                all_correct = False
        
        print(f"\n{'='*50}")
        if all_correct:
            print("🎉 CHAINED MATH TEST PASSED!")
            print("All LLM providers correctly processed node references!")
        else:
            print("❌ CHAINED MATH TEST FAILED!")
            print("Some nodes did not produce expected outputs.")
            print("Check the __node_mapping and context_data handling.")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    create_chained_math_test() 