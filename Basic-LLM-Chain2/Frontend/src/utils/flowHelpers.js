// Helper to find DIRECTLY connected input nodes
export const findDirectInputNodes = (nodeId, nodes, edges) => {
  const inputEdges = edges.filter(edge => edge.target === nodeId);
  const inputNodeIds = inputEdges.map(edge => edge.source);
  return nodes
    .filter(node => inputNodeIds.includes(node.id))
    .map(node => ({ id: node.id, name: node.data.nodeName || 'Unnamed Node' }));
};

// Helper: Find all downstream nodes from a given node (DFS)
export const findDownstreamNodes = (nodeId, edges) => {
  const visited = new Set();
  const stack = [nodeId];
  while (stack.length > 0) {
    const current = stack.pop();
    edges.forEach(edge => {
      if (edge.source === current && !visited.has(edge.target)) {
        visited.add(edge.target);
        stack.push(edge.target);
      }
    });
  }
  return visited;
};

// Helper to get center position of viewport
export const getCenterPosition = (reactFlowInstance) => {
  if (!reactFlowInstance) return { x: 100, y: 100 };
  
  const { x, y, zoom } = reactFlowInstance.getViewport();
  const centerX = (window.innerWidth / 2 - x) / zoom;
  const centerY = (window.innerHeight / 2 - y) / zoom;
  
  return { x: centerX, y: centerY };
}; 