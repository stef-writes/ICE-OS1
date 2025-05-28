import networkx as nx
from typing import Dict, Any, List, Optional
from .nodes import Node  # Adjusted import
from .storage import NamespacedStorage # Adjusted import
from utils import InputValidator # Adjusted import
from llm import LLMConfig # Adjusted import
from callbacks import Callback # Adjusted import
import traceback
import asyncio
import time
from collections import defaultdict, deque

# --- ScriptChain class ---
class ScriptChain:
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.graph = nx.DiGraph()
        self.storage = NamespacedStorage()
        self.callbacks = callbacks or []
        self.node_versions = {} # Stores the version of the last execution for each node
        self.node_dependencies = defaultdict(set) # node_id -> set of predecessor_node_ids

    def add_node(self, node_id: str, node_type: str, input_keys: Optional[List[str]] = None, output_keys: Optional[List[str]] = None, model_config: Optional[LLMConfig] = None, template: Optional[Dict[str, Any]] = None):
        """Adds a node to the graph."""
        node_instance = Node(node_id, node_type, input_keys, output_keys, model_config, template)
        self.graph.add_node(node_id, node=node_instance)
        print(f"Added node: {node_id} ({node_type})")

    def add_edge(self, from_node: str, to_node: str):
        """Adds an edge between two nodes."""
        self.graph.add_edge(from_node, to_node)
        # Update node_dependencies immediately when an edge is added
        self.node_dependencies[to_node].add(from_node)
        print(f"Added edge: {from_node} -> {to_node}")

    def add_callback(self, callback: Callback):
        """Adds a callback to the chain."""
        self.callbacks.append(callback)

    def node_needs_update(self, node_id):
        """Check if a node needs to be re-executed due to dependency changes or if it was never run."""
        if node_id not in self.node_versions: # Never executed
            return True
            
        node_last_exec_version_for_deps = self.node_versions[node_id].get('depends_on_versions', {})
        
        for dep_node_id in self.node_dependencies[node_id]:
            # Get the current actual version of the dependency's output
            dep_current_output_version = self.node_versions.get(dep_node_id, {}).get('output_version', 0)
            # Get the version of the dependency that this node used last time it ran
            dep_version_at_last_exec = node_last_exec_version_for_deps.get(dep_node_id, 0)

            if dep_current_output_version > dep_version_at_last_exec:
                print(f"Node '{node_id}' needs update: Dependency '{dep_node_id}' changed (current: v{dep_current_output_version}, used: v{dep_version_at_last_exec})")
                return True
        return False

    def _update_node_execution_record(self, node_id):
        """Records the current output version of this node and the versions of its dependencies it used."""
        if node_id not in self.node_versions:
            self.node_versions[node_id] = {}
        
        # Increment/set the output version for this node
        current_output_version = self.node_versions[node_id].get('output_version', 0) + 1
        self.node_versions[node_id]['output_version'] = current_output_version
        print(f"Node '{node_id}' output version set to {current_output_version}")

        # Record the versions of dependencies it just used for this execution
        depends_on_versions = {}
        for dep_node_id in self.node_dependencies[node_id]:
            depends_on_versions[dep_node_id] = self.node_versions.get(dep_node_id, {}).get('output_version', 0)
        self.node_versions[node_id]['depends_on_versions'] = depends_on_versions
        if depends_on_versions:
            print(f"Node '{node_id}' recorded dependency versions: {depends_on_versions}")

    async def execute_concurrent(self) -> Dict[str, Any]:
        """
        Execute nodes concurrently when dependencies allow.
        This provides significant performance improvements over sequential execution.
        """
        print("--- Starting Concurrent Chain Execution ---")
        start_time = time.time()
        
        try:
            if not nx.is_directed_acyclic_graph(self.graph):
                return {"error": "Graph contains a cycle"}
        except Exception as e:
            return {"error": f"Graph validation failed: {e}"}        
        
        results = {}
        total_tokens = 0
        total_cost = 0.0
        
        in_degree = {node_id: self.graph.in_degree(node_id) for node_id in self.graph.nodes()}
        # Successors for each node to update in_degree later
        successors = {node_id: list(self.graph.successors(node_id)) for node_id in self.graph.nodes()}
        
        ready_queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        running_tasks = {}
        completed_nodes = set()
        
        print(f"Initial ready nodes: {list(ready_queue)}")
        print(f"Total nodes to execute: {len(self.graph.nodes())}")
        
        while ready_queue or running_tasks:
            while ready_queue:
                node_id = ready_queue.popleft()
                
                if self.node_needs_update(node_id) or not self.storage.has_node(node_id):
                    print(f"🚀 Starting execution of node: {node_id}")
                    task = asyncio.create_task(
                        self._execute_single_node_concurrent(node_id),
                        name=f"node-{node_id}"
                    )
                    running_tasks[node_id] = task
                else:
                    print(f"⏭️ Skipping node {node_id} - already up-to-date and results cached.")
                    # Simulate completion for dependency tracking if skipped
                    results[node_id] = self.storage.get_node_output(node_id)
                    completed_nodes.add(node_id)
                    # This node's output didn't change, so no need to update its version
                    # or versions of its dependents based on *this* node's execution.
                    # However, we must still propagate its completion to unlock dependent nodes.
                    for successor_id in successors[node_id]:
                        in_degree[successor_id] -= 1
                        if in_degree[successor_id] == 0:
                            ready_queue.append(successor_id)
                            print(f"🎯 Node {successor_id} is now ready to execute (triggered by skipped {node_id})")
            
            if not running_tasks and not ready_queue and len(completed_nodes) < len(self.graph.nodes()):
                # This can happen if all ready nodes were skipped, and no new tasks were started.
                # We need to check if there are still nodes that haven't been processed.
                # This state implies a bug in ready_queue logic or node_needs_update for cyclic dependencies or isolated components.
                print("Warning: No tasks running, but not all nodes completed. Checking remaining nodes.")
                all_graph_nodes = set(self.graph.nodes())
                remaining_nodes = all_graph_nodes - completed_nodes
                if not remaining_nodes:
                    break # All nodes processed
                # Try to re-populate ready_queue if any remaining node has in_degree 0
                was_repopulated = False
                for node_id in remaining_nodes:
                    if in_degree[node_id] == 0 and node_id not in ready_queue and node_id not in running_tasks:
                        ready_queue.append(node_id)
                        was_repopulated = True
                if not was_repopulated and remaining_nodes:
                    print(f"Error: Stuck! Remaining nodes {remaining_nodes} have non-zero in-degrees or are not being processed.")
                    break # Avoid infinite loop

            if not running_tasks: # If all were skipped or no tasks to run, and queue is empty, break.
                if not ready_queue: break
                else: continue # Go back to process ready_queue if it got repopulated

            print(f"⏳ Waiting for {len(running_tasks)} running tasks: {list(running_tasks.keys())}")
            done, pending = await asyncio.wait(running_tasks.values(), return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                node_id = next(nid for nid, t in running_tasks.items() if t == task)
                
                try:
                    result = await task
                    results[node_id] = result
                    completed_nodes.add(node_id)
                    
                    self._update_node_execution_record(node_id)
                    
                    node_instance = self.graph.nodes[node_id].get("node")
                    if node_instance and node_instance.token_usage:
                        try:
                            total_tokens += getattr(node_instance.token_usage, 'total_tokens', 0)
                            total_cost += getattr(node_instance.token_usage, 'cost', 0.0)
                        except AttributeError:
                            print(f"Warning: token_usage object for node {node_id} missing expected attributes.")
                    
                    for callback in self.callbacks:
                        try:
                            callback.on_node_complete(node_id, node_instance.node_type, result, node_instance.token_usage)
                        except Exception as e:
                            print(f"Error in callback {type(callback).__name__}.on_node_complete for node {node_id}: {e}")
                    
                    print(f"✅ Node {node_id} completed successfully")
                    
                    for successor_id in successors[node_id]:
                        in_degree[successor_id] -= 1
                        print(f"📊 Node {successor_id} dependency count updated: {in_degree[successor_id]}")
                        if in_degree[successor_id] == 0:
                            ready_queue.append(successor_id)
                            print(f"🎯 Node {successor_id} is now ready to execute")
                            
                except Exception as e:
                    print(f"❌ Node {node_id} failed: {e}")
                    traceback.print_exc()
                    results[node_id] = {"error": str(e)}
                    completed_nodes.add(node_id) # Mark as completed (failed) to prevent re-processing in this run
                    # Do not update execution record on failure
                    # Propagate completion to unlock dependents even on failure
                    for successor_id in successors[node_id]:
                        in_degree[successor_id] -= 1
                        if in_degree[successor_id] == 0:
                            ready_queue.append(successor_id)
                
                del running_tasks[node_id]
        
        execution_time = time.time() - start_time
        print(f"🏁 Concurrent Chain Execution Complete in {execution_time:.2f}s")
        print(f"📈 Completed nodes: {len(completed_nodes)}/{len(self.graph.nodes())}")
        
        for callback in self.callbacks:
            try:
                callback.on_chain_complete(results, total_tokens, total_cost)
            except Exception as e:
                print(f"Error in callback {type(callback).__name__}.on_chain_complete: {e}")
        
        return {
            "results": results,
            "stats": {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "execution_time": execution_time,
                "nodes_completed": len(completed_nodes),
                "nodes_total": len(self.graph.nodes()),
                "execution_mode": "concurrent"
            }
        }
    
    async def _execute_single_node_concurrent(self, node_id: str) -> Dict[str, Any]:
        """Execute a single node with proper input gathering for concurrent execution."""
        node_instance = self.graph.nodes[node_id].get("node")
        if not isinstance(node_instance, Node):
            raise ValueError(f"Node '{node_id}' does not contain a valid Node object")
        
        inputs_for_node = {}
        predecessor_nodes = list(self.graph.predecessors(node_id))
        
        for pred_id in predecessor_nodes:
            # Ensure predecessor results are available (they should be if this node is running)
            pred_outputs = self.storage.get_node_output(pred_id)
            if pred_outputs:
                for output_key, output_value in pred_outputs.items():
                    namespaced_key = f"{pred_id}:{output_key}"
                    inputs_for_node[namespaced_key] = output_value
                    if output_key in node_instance.input_keys:
                        inputs_for_node[output_key] = output_value
            else:
                # This case should ideally not happen if execution order is correct
                print(f"Warning: Output for predecessor '{pred_id}' not found for node '{node_id}'")

        for key in node_instance.input_keys:
            if key not in inputs_for_node:
                value = self.storage.get_by_key(key) # Fallback to non-namespaced
                if value is not None:
                    inputs_for_node[key] = value
        
        inputs_for_node["storage"] = self.storage.get_flattened()
        inputs_for_node["get_node_output"] = self.storage.get_node_output
        
        try:
            InputValidator.validate(node_instance, inputs_for_node)
        except ValueError as e:
            error_msg = f"Input validation error for node {node_id}: {e}"
            print(error_msg)
            error_result = {"error": str(e)}
            self.storage.store(node_id, error_result) # Store error in namespaced storage
            return error_result
        
        for callback in self.callbacks:
            try:
                callback.on_node_start(node_id, node_instance.node_type, inputs_for_node)
            except Exception as e:
                print(f"Error in callback {type(callback).__name__}.on_node_start for node {node_id}: {e}")
        
        try:
            print(f"⚡ Executing node {node_id}...")
            node_result = await node_instance.process(inputs_for_node)
            
            if isinstance(node_result, dict):
                self.storage.store(node_id, node_result)
                return node_result
            else:
                wrapped_result = {"output": node_result}
                self.storage.store(node_id, wrapped_result)
                return wrapped_result
                
        except Exception as e:
            error_msg = f"Error processing node {node_id}: {e}"
            print(error_msg)
            traceback.print_exc()
            error_result = {"error": str(e)}
            self.storage.store(node_id, error_result)
            return error_result

    async def execute(self):
        """Executes the graph nodes in topological (dependency) order."""
        try:
            execution_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            print("Error: Graph contains a cycle, cannot determine execution order.")
            return {"error": "Graph contains a cycle"}
        except Exception as e:
             print(f"Error during topological sort: {e}")
             return {"error": f"Failed to determine execution order: {e}"}

        results = {}
        total_tokens = 0
        total_cost = 0.0

        print(f"--- Executing Chain (Order: {execution_order}) ---")
        
        for node_id in execution_order:
            if node_id not in self.graph:
                print(f"Error: Node '{node_id}' found in execution order but not in graph.")
                continue

            node_instance = self.graph.nodes[node_id].get("node")
            if not isinstance(node_instance, Node):
                print(f"Error: Node '{node_id}' in graph does not contain a valid Node object.")
                continue
            
            # Check if this node needs to be re-run
            if not self.node_needs_update(node_id) and self.storage.has_node(node_id):
                print(f"⏭️ Skipping node {node_id} - already up-to-date and results cached.")
                results[node_id] = self.storage.get_node_output(node_id) # Load cached result
                # No token usage to add if skipped
                # Trigger on_node_complete with cached data if needed by callbacks
                for callback in self.callbacks:
                    try:
                        # Pass cached result and potentially empty token usage
                        callback.on_node_complete(node_id, node_instance.node_type, results[node_id], None) 
                    except Exception as e:
                        print(f"Error in callback {type(callback).__name__}.on_node_complete for skipped node {node_id}: {e}")
                continue # Move to the next node

            inputs_for_node = {}
            predecessor_nodes = list(self.graph.predecessors(node_id))
            
            for pred_id in predecessor_nodes:
                pred_outputs = self.storage.get_node_output(pred_id)
                if pred_outputs:
                    for output_key, output_value in pred_outputs.items():
                        namespaced_key = f"{pred_id}:{output_key}"
                        inputs_for_node[namespaced_key] = output_value
                        if output_key in node_instance.input_keys:
                            inputs_for_node[output_key] = output_value
                else:
                     print(f"Warning: Output for predecessor '{pred_id}' not found for node '{node_id}' during sequential execution.")
            
            for key in node_instance.input_keys:
                if key not in inputs_for_node:
                    value = self.storage.get_by_key(key)
                    if value is not None:
                        inputs_for_node[key] = value
            
            inputs_for_node["storage"] = self.storage.get_flattened()
            inputs_for_node["get_node_output"] = self.storage.get_node_output
            
            try:
                InputValidator.validate(node_instance, inputs_for_node)
            except ValueError as e:
                print(f"Input validation error for node {node_id}: {e}")
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue

            for callback in self.callbacks:
                try:
                    callback.on_node_start(node_id, node_instance.node_type, inputs_for_node)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_start for node {node_id}: {e}")

            try:
                node_result = await node_instance.process(inputs_for_node)
            except Exception as e:
                print(f"Error processing node {node_id}: {e}")
                traceback.print_exc()
                node_result = None
                results[node_id] = {"error": str(e)}
                self.storage.store(node_id, {"error": str(e)})
                continue

            if isinstance(node_result, dict):
                results[node_id] = node_result
                self.storage.store(node_id, node_result)
            else:
                results[node_id] = {"output": node_result}
                self.storage.store(node_id, {"output": node_result})
                
            self._update_node_execution_record(node_id)
            print(f"Node {node_id} execution complete, output version is now {self.node_versions[node_id]['output_version']}")

            if node_instance.token_usage:
                try:
                    total_tokens += getattr(node_instance.token_usage, 'total_tokens', 0)
                    total_cost += getattr(node_instance.token_usage, 'cost', 0.0)
                except AttributeError:
                     print(f"Warning: token_usage object for node {node_id} missing expected attributes.")

            for callback in self.callbacks:
                try:
                    callback.on_node_complete(node_id, node_instance.node_type, results.get(node_id), node_instance.token_usage)
                except Exception as e:
                    print(f"Error in callback {type(callback).__name__}.on_node_complete for node {node_id}: {e}")

        print("--- Chain Execution Finished ---")
        for callback in self.callbacks:
             try:
                 callback.on_chain_complete(results, total_tokens, total_cost)
             except Exception as e:
                 print(f"Error in callback {type(callback).__name__}.on_chain_complete: {e}")

        return {
            "results": results,
            "stats": {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "execution_mode": "sequential"
            }
        } 