#!/usr/bin/env python3
"""
Test script to demonstrate the performance improvement of concurrent node execution.
This script creates a test chain and compares sequential vs concurrent execution times.
"""

import asyncio
import time
import requests
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = f"perf_test_{int(time.time())}"

def make_api_request(method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Helper function to make API requests."""
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

def create_test_chain():
    """Create a test chain with parallel and sequential dependencies."""
    print("🔧 Creating test chain...")
    
    # Define nodes that can demonstrate concurrent execution benefits
    nodes = [
        {
            "node_id": "idea_generator",
            "name": "Idea Generator",
            "node_type": "text_generation",
            "prompt": "Generate a creative business idea in one sentence.",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100
            },
            "input_keys": [],
            "output_keys": ["generated_text"]
        },
        {
            "node_id": "market_research",
            "name": "Market Research",
            "node_type": "text_generation", 
            "prompt": "Research the market potential for this business idea: {generated_text}",
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3-haiku-20240307",
                "temperature": 0.5,
                "max_tokens": 200
            },
            "input_keys": ["generated_text"],
            "output_keys": ["generated_text"]
        },
        {
            "node_id": "technical_analysis",
            "name": "Technical Analysis",
            "node_type": "text_generation",
            "prompt": "Analyze the technical feasibility of this idea: {generated_text}",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 200
            },
            "input_keys": ["generated_text"],
            "output_keys": ["generated_text"]
        },
        {
            "node_id": "financial_projection",
            "name": "Financial Projection",
            "node_type": "text_generation",
            "prompt": "Create financial projections for this business: {generated_text}",
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.4,
                "max_tokens": 250
            },
            "input_keys": ["generated_text"],
            "output_keys": ["generated_text"]
        },
        {
            "node_id": "final_report",
            "name": "Final Report",
            "node_type": "text_generation",
            "prompt": (
                "Create a comprehensive business plan summary based on:\n"
                "Idea: {generated_text}\n"
                "Market: {generated_text}\n"
                "Technical: {generated_text}\n"
                "Financial: {generated_text}"
            ),
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.6,
                "max_tokens": 400
            },
            "input_keys": ["generated_text"],
            "output_keys": ["generated_text"]
        }
    ]
    
    # Create nodes
    for node in nodes:
        node_data = {
            "node_id": node["node_id"],
            "name": node["name"],
            "node_type": node["node_type"],
            "input_keys": node["input_keys"],
            "output_keys": node["output_keys"],
            "llm_config": node["llm_config"]
        }
        
        result = make_api_request("POST", "/api/graph/add_node", node_data)
        print(f"  ✅ Created node: {node['name']}")
        
        # Set the prompt
        prompt_data = {"prompt": node["prompt"]}
        make_api_request("PUT", f"/api/db/nodes/{node['node_id']}/prompt", prompt_data)
    
    # Create edges to define dependencies
    edges = [
        ("idea_generator", "market_research"),
        ("idea_generator", "technical_analysis"), 
        ("idea_generator", "financial_projection"),
        ("market_research", "final_report"),
        ("technical_analysis", "final_report"),
        ("financial_projection", "final_report")
    ]
    
    for from_node, to_node in edges:
        edge_data = {"from_node": from_node, "to_node": to_node}
        make_api_request("POST", "/api/graph/add_edge", edge_data)
        print(f"  🔗 Created edge: {from_node} → {to_node}")
    
    print("✅ Test chain created successfully!")
    print("\n📊 Chain structure:")
    print("  idea_generator")
    print("  ├── market_research ──┐")
    print("  ├── technical_analysis ─┼── final_report")
    print("  └── financial_projection ─┘")
    print("\n💡 Expected behavior:")
    print("  - Sequential: idea_generator → market_research → technical_analysis → financial_projection → final_report")
    print("  - Concurrent: idea_generator → [market_research, technical_analysis, financial_projection] → final_report")

def test_execution_performance():
    """Test both sequential and concurrent execution and compare performance."""
    print("\n" + "="*60)
    print("🚀 PERFORMANCE COMPARISON TEST")
    print("="*60)
    
    # Test sequential execution
    print("\n📈 Testing Sequential Execution...")
    start_time = time.time()
    try:
        sequential_result = make_api_request("POST", "/api/execute/execute_sequential")
        sequential_time = time.time() - start_time
        sequential_success = True
        print(f"✅ Sequential execution completed in {sequential_time:.2f} seconds")
        
        if "stats" in sequential_result:
            stats = sequential_result["stats"]
            print(f"   📊 Tokens: {stats.get('total_tokens', 'N/A')}")
            print(f"   💰 Cost: ${stats.get('total_cost', 0):.4f}")
            print(f"   🔄 Mode: {stats.get('execution_mode', 'unknown')}")
            
            # Print node outputs for debugging
            if "results" in sequential_result:
                print("\n   📝 Node Outputs:")
                for node_id, output in sequential_result["results"].items():
                    if isinstance(output, dict):
                        if "error" in output:
                            print(f"   - {node_id}: ❌ Error - {output['error']}")
                        elif "output" in output:
                            print(f"   - {node_id}: {output['output'][:100]}...")
                        elif "generated_text" in output:
                            print(f"   - {node_id}: {output['generated_text'][:100]}...")
                        else:
                            print(f"   - {node_id}: {str(output)[:100]}...")
                    else:
                        print(f"   - {node_id}: {str(output)[:100]}...")
    except Exception as e:
        print(f"❌ Sequential execution failed: {e}")
        sequential_success = False
        sequential_time = float('inf')
    
    # Wait a moment between tests
    time.sleep(2)
    
    # Test concurrent execution
    print("\n⚡ Testing Concurrent Execution...")
    start_time = time.time()
    try:
        concurrent_result = make_api_request("POST", "/api/execute/execute_concurrent")
        concurrent_time = time.time() - start_time
        concurrent_success = True
        print(f"✅ Concurrent execution completed in {concurrent_time:.2f} seconds")
        
        if "stats" in concurrent_result:
            stats = concurrent_result["stats"]
            print(f"   📊 Tokens: {stats.get('total_tokens', 'N/A')}")
            print(f"   💰 Cost: ${stats.get('total_cost', 0):.4f}")
            print(f"   🔄 Mode: {stats.get('execution_mode', 'unknown')}")
            print(f"   ⏱️ Execution time: {stats.get('execution_time', 'N/A')}s")
            print(f"   ✅ Nodes completed: {stats.get('nodes_completed', 'N/A')}/{stats.get('nodes_total', 'N/A')}")
            
            # Print node outputs for debugging
            if "results" in concurrent_result:
                print("\n   📝 Node Outputs:")
                for node_id, output in concurrent_result["results"].items():
                    if isinstance(output, dict):
                        if "error" in output:
                            print(f"   - {node_id}: ❌ Error - {output['error']}")
                        elif "output" in output:
                            print(f"   - {node_id}: {output['output'][:100]}...")
                        elif "generated_text" in output:
                            print(f"   - {node_id}: {output['generated_text'][:100]}...")
                        else:
                            print(f"   - {node_id}: {str(output)[:100]}...")
                    else:
                        print(f"   - {node_id}: {str(output)[:100]}...")
    except Exception as e:
        print(f"❌ Concurrent execution failed: {e}")
        concurrent_success = False
        concurrent_time = float('inf')
    
    # Performance comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE RESULTS")
    print("="*60)
    
    if sequential_success and concurrent_success:
        improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else float('inf')
        
        print(f"Sequential Time:  {sequential_time:.2f}s")
        print(f"Concurrent Time:  {concurrent_time:.2f}s")
        print(f"Time Saved:       {sequential_time - concurrent_time:.2f}s")
        print(f"Performance Gain: {improvement:.1f}% faster")
        print(f"Speedup Factor:   {speedup:.1f}x")
        
        if improvement > 0:
            print(f"\n🎉 SUCCESS! Concurrent execution is {improvement:.1f}% faster!")
            print(f"💡 This means your chains will complete {speedup:.1f}x faster with concurrent execution.")
        elif improvement < -5:  # Allow for small timing variations
            print(f"\n⚠️  Unexpected: Sequential was faster. This might be due to:")
            print("   - Network latency variations")
            print("   - API rate limiting")
            print("   - Small chain size")
        else:
            print(f"\n📊 Results are similar. For larger chains, concurrent execution will show bigger gains.")
    else:
        print("❌ Could not complete performance comparison due to execution failures.")
    
    print("\n" + "="*60)

def main():
    """Main test function."""
    print("🧪 Concurrent Execution Performance Test")
    print(f"🔗 API Base URL: {API_BASE_URL}")
    print(f"🆔 Session ID: {SESSION_ID}")
    
    try:
        # Create the test chain
        create_test_chain()
        
        # Run performance tests
        test_execution_performance()
        
        print("\n✅ Performance test completed!")
        print("\n💡 Key Takeaways:")
        print("   - Concurrent execution runs independent nodes in parallel")
        print("   - Performance gains increase with chain complexity")
        print("   - Your users will experience faster response times")
        print("   - You can handle more requests with the same infrastructure")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 