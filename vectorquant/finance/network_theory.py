"""
Network Theory (Financial Markets)
"""
import math

def correlation_distance(rho):
    """
    distance = sqrt(2 * (1 - rho))
    """
    # Clip rho to [-1, 1] due to floating point
    rho_clean = max(min(rho, 1.0), -1.0)
    return math.sqrt(2.0 * (1.0 - rho_clean))

def minimum_spanning_tree(nodes, distance_matrix):
    """
    Computes MST using Prim's algorithm.
    nodes: list of n node identifiers
    distance_matrix: n x n symmetric matrix of distances
    Returns list of edges in the MST: [(node_u, node_v, weight), ...]
    """
    n = len(nodes)
    if n == 0: return []
    
    selected = [False] * n
    selected[0] = True
    num_edges = 0
    mst_edges = []
    
    while num_edges < n - 1:
        minimum = float('inf')
        u = -1
        v = -1
        
        for i in range(n):
            if selected[i]:
                for j in range(n):
                    if not selected[j] and distance_matrix[i][j] < minimum:
                        minimum = distance_matrix[i][j]
                        u = i
                        v = j
                        
        if u != -1 and v != -1:
            selected[v] = True
            mst_edges.append((nodes[u], nodes[v], distance_matrix[u][v]))
            num_edges += 1
            
    return mst_edges
