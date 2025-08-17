import os
import networkx as nx
import numpy as np

# Configuration - Set to True to use oversampled graphs
USE_OVERSAMPLED_GRAPHS = True

# Graph directories
GRAPHS_DIR = 'drive/training/graphs'
OVERSAMPLED_GRAPHS_DIR = 'drive/training/graphs_oversampled'
OVERSAMPLED_SPACING = 5  # Spacing used for oversampling


def load_oversampled_graph_npy(sample_id: str) -> nx.Graph:
    """Load oversampled graph from .npy file."""
    npy_file = os.path.join(OVERSAMPLED_GRAPHS_DIR, f"{sample_id}_oversampled_spacing{OVERSAMPLED_SPACING}.npy")
    
    if not os.path.exists(npy_file):
        print(f"⚠️ Oversampled graph not found: {npy_file}")
        return None
    
    try:
        # Load the .npy file
        data = np.load(npy_file, allow_pickle=True).item()
        
        # Extract nodes and edges
        nodes_array = data['nodes']
        edges_array = data['edges']
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with positions
        for i, (x, y) in enumerate(nodes_array):
            G.add_node(i, pos=(float(x), float(y)))
        
        # Add edges
        for u, v in edges_array:
            G.add_edge(int(u), int(v))
        
        print(f"✅ Loaded oversampled graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        return G
        
    except Exception as e:
        print(f"❌ Error loading oversampled graph: {e}")
        return None


def load_training_graph_by_id(sample_id: str) -> nx.Graph:
    """Load a training graph for given sample_id.
    
    If USE_OVERSAMPLED_GRAPHS is True, loads the oversampled graph.
    Otherwise, loads the original graph from manual files.
    
    Returns a NetworkX Graph with node attribute 'pos' (x,y) arrays.
    """
    # First try to load oversampled graph if enabled
    if USE_OVERSAMPLED_GRAPHS:
        oversampled_graph = load_oversampled_graph_npy(sample_id)
        if oversampled_graph is not None:
            return oversampled_graph
        else:
            print(f"⚠️ Falling back to original graph for sample {sample_id}")
    
    # Load original graph (fallback)
    npy_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.npy.graph")
    simple_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.graph")

    if os.path.exists(npy_graph):
        with open(npy_graph, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        import re
        float3 = re.compile(r'^[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*$')
        int2 = re.compile(r'^\d+\s+\d+$')
        G = nx.Graph()
        nodes = []
        # First pass: collect nodes
        for ln in lines:
            if float3.match(ln):
                x_str, y_str, _ = ln.split()
                nodes.append((float(x_str), float(y_str)))
            else:
                break
        # Add nodes with integer indices and pos
        for i, (x, y) in enumerate(nodes):
            G.add_node(i, pos=(x, y))
        # Second pass: edges
        for ln in lines[len(nodes):]:
            if int2.match(ln):
                a_str, b_str = ln.split()
                G.add_edge(int(a_str), int(b_str))
        return G

    if os.path.exists(simple_graph):
        with open(simple_graph, 'r') as f:
            raw = f.read().strip('\n')
        groups = [g for g in raw.split('\n\n') if g.strip()]
        if not groups:
            raise ValueError(f"Empty graph file: {simple_graph}")
        G = nx.Graph()
        # Nodes
        for ln in groups[0].splitlines():
            x_str, y_str = ln.split()
            i = len(G.nodes)
            G.add_node(i, pos=(float(x_str), float(y_str)))
        # Edges
        if len(groups) > 1:
            for ln in groups[1].splitlines():
                a_str, b_str = ln.split()
                G.add_edge(int(a_str), int(b_str))
        return G

    raise FileNotFoundError(f"No graph found for sample_id {sample_id}")


def get_graph_info(sample_id: str) -> dict:
    """Get information about available graphs for a sample."""
    info = {
        'sample_id': sample_id,
        'oversampled_available': False,
        'original_available': False,
        'oversampled_path': None,
        'original_paths': []
    }
    
    # Check oversampled graph
    oversampled_path = os.path.join(OVERSAMPLED_GRAPHS_DIR, f"{sample_id}_oversampled_spacing{OVERSAMPLED_SPACING}.npy")
    if os.path.exists(oversampled_path):
        info['oversampled_available'] = True
        info['oversampled_path'] = oversampled_path
    
    # Check original graphs
    npy_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.npy.graph")
    simple_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.graph")
    
    if os.path.exists(npy_graph):
        info['original_available'] = True
        info['original_paths'].append(npy_graph)
    if os.path.exists(simple_graph):
        info['original_available'] = True
        info['original_paths'].append(simple_graph)
    
    return info


def switch_to_oversampled_graphs():
    """Enable oversampled graphs for training."""
    global USE_OVERSAMPLED_GRAPHS
    USE_OVERSAMPLED_GRAPHS = True
    print(f"✅ Switched to oversampled graphs (spacing: {OVERSAMPLED_SPACING} pixels)")


def switch_to_original_graphs():
    """Disable oversampled graphs and use original graphs."""
    global USE_OVERSAMPLED_GRAPHS
    USE_OVERSAMPLED_GRAPHS = False
    print(f"✅ Switched to original graphs")


def get_current_graph_mode():
    """Get current graph loading mode."""
    if USE_OVERSAMPLED_GRAPHS:
        return f"oversampled (spacing: {OVERSAMPLED_SPACING} pixels)"
    else:
        return "original" 