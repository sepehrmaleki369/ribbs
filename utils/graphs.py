import os
import networkx as nx

GRAPHS_DIR = 'drive/training/graphs'


def load_training_graph_by_id(sample_id: str) -> nx.Graph:
    """Load a training graph for given sample_id (e.g., '21') from available files.
    Prefer detailed '*_manual1.npy.graph' if present, else fallback to simple '.graph'.
    Returns a NetworkX Graph with node attribute 'pos' (x,y) arrays.
    """
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