import os
import numpy as np
import networkx as nx
import pickle
from tqdm import tqdm
import sys

# Import after setting up paths
sys.path.append('.')

from utils.graphs import load_training_graph_by_id

# Configuration
GRAPHS_DIR = 'drive/training/graphs'
OUTPUT_DIR = 'drive/training/graphs_oversampled'
SPACING = 5  # Oversampling spacing used

def interpolate_new_nodes(pos1, pos2, spacing):
    """
    Interpolate new nodes between two positions with given spacing.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    
    # Calculate distance between points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # If distance is less than spacing, no interpolation needed
    if distance <= spacing:
        return []
    
    # Calculate number of interpolated points
    num_points = int(distance / spacing) - 1
    
    if num_points <= 0:
        return []
    
    # Interpolate points
    new_nodes_pos = []
    for i in range(1, num_points + 1):
        t = i / (num_points + 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        new_nodes_pos.append([x, y])
    
    return new_nodes_pos

def oversampling_graph(G, spacing=5):
    """
    Oversample graph by adding interpolated nodes along edges.
    """
    # Create a copy to avoid modifying original
    G_oversampled = G.copy()
    
    edges = list(G_oversampled.edges())
    for s, t in edges:
        # Get positions
        pos1 = G_oversampled.nodes[s]['pos']
        pos2 = G_oversampled.nodes[t]['pos']
        
        new_nodes_pos = interpolate_new_nodes(pos1, pos2, spacing)
        
        if len(new_nodes_pos) > 0:
            G_oversampled.remove_edge(s, t)
            n = max(G_oversampled.nodes()) + 1
            
            for i, n_pos in enumerate(new_nodes_pos):
                G_oversampled.add_node(n + i, pos=tuple(n_pos))
            
            G_oversampled.add_edge(s, n)
            for _ in range(len(new_nodes_pos) - 1):
                G_oversampled.add_edge(n, n + 1)
                n += 1
            G_oversampled.add_edge(n, t)
    
    return G_oversampled

def graph_to_numpy_format(graph):
    """
    Convert NetworkX graph to numpy format suitable for training.
    Returns:
        nodes_array: (N, 2) array of [x, y] positions
        edges_array: (M, 2) array of [u, v] node indices
    """
    if graph is None or len(graph.nodes()) == 0:
        return np.array([]), np.array([])
    
    # Extract node positions
    nodes = []
    node_id_to_index = {}
    
    for i, node in enumerate(graph.nodes()):
        node_attrs = graph.nodes[node]
        pos = None
        
        if 'pos' in node_attrs:
            pos = node_attrs['pos']
        elif 'position' in node_attrs:
            pos = node_attrs['position']
        elif 'coords' in node_attrs:
            pos = node_attrs['coords']
        
        if pos is not None:
            try:
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                elif hasattr(pos, '__getitem__') and len(pos) >= 2:
                    x, y = pos[0], pos[1]
                else:
                    continue
                    
                if hasattr(x, 'item'):
                    x = x.item()
                if hasattr(y, 'item'):
                    y = y.item()
                
                nodes.append([x, y])
                node_id_to_index[node] = i
                
            except Exception as e:
                continue
    
    # Extract edges
    edges = []
    for edge in graph.edges():
        try:
            u, v = edge
            if u in node_id_to_index and v in node_id_to_index:
                u_idx = node_id_to_index[u]
                v_idx = node_id_to_index[v]
                edges.append([u_idx, v_idx])
        except Exception as e:
            continue
    
    return np.array(nodes, dtype=np.float32), np.array(edges, dtype=np.int32)

def save_oversampled_graph_npy(sample_id, output_dir):
    """
    Load initial graph, oversample it, and save as .npy file.
    """
    try:
        # Load initial graph
        initial_graph = load_training_graph_by_id(str(sample_id))
        if initial_graph is None:
            print(f"⚠️ No graph available for sample {sample_id}, skipping...")
            return None
        
        print(f"✅ Loaded initial graph: {len(initial_graph.nodes())} nodes, {len(initial_graph.edges())} edges")
        
        # Apply oversampling
        oversampled_graph = oversampling_graph(initial_graph, SPACING)
        print(f"✅ Created oversampled graph: {len(oversampled_graph.nodes())} nodes, {len(oversampled_graph.edges())} edges")
        
        # Convert to numpy format
        nodes_array, edges_array = graph_to_numpy_format(oversampled_graph)
        
        if len(nodes_array) == 0:
            print(f"⚠️ Failed to convert graph for sample {sample_id}")
            return None
        
        # Create output filename
        output_filename = f"{sample_id}_oversampled_spacing{SPACING}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save as .npy file
        np.save(output_path, {
            'nodes': nodes_array,
            'edges': edges_array,
            'spacing': SPACING,
            'original_nodes': len(initial_graph.nodes()),
            'original_edges': len(initial_graph.edges()),
            'oversampled_nodes': len(oversampled_graph.nodes()),
            'oversampled_edges': len(oversampled_graph.edges())
        })
        
        print(f"✅ Saved: {output_filename}")
        
        return {
            'sample_id': sample_id,
            'original_nodes': len(initial_graph.nodes()),
            'original_edges': len(initial_graph.edges()),
            'oversampled_nodes': len(oversampled_graph.nodes()),
            'oversampled_edges': len(oversampled_graph.edges()),
            'node_increase': len(oversampled_graph.nodes()) - len(initial_graph.nodes()),
            'edge_increase': len(oversampled_graph.edges()) - len(initial_graph.edges()),
            'filename': output_filename
        }
        
    except Exception as e:
        print(f"❌ Failed on sample {sample_id}: {e}")
        return None

def get_all_available_samples():
    """Get all sample IDs that have graph files."""
    available_samples = []
    
    if os.path.exists(GRAPHS_DIR):
        graph_files = [f for f in os.listdir(GRAPHS_DIR) if f.endswith('.graph') or f.endswith('.npy.graph')]
        graph_samples = list(set([int(f.split('_')[0]) for f in graph_files]))
        print(f"📁 Found {len(graph_samples)} graph files: {sorted(graph_samples)}")
        available_samples = sorted(graph_samples)
    else:
        print(f"❌ Graphs directory not found: {GRAPHS_DIR}")
    
    return available_samples

def main():
    """Main function to save oversampled graphs as .npy files."""
    
    print("=== SAVING OVERSAMPLED GRAPHS TO TRAINING FOLDER ===")
    print(f"🎯 Using oversampling spacing: {SPACING} pixels")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Get all available samples
    available_samples = get_all_available_samples()
    
    if not available_samples:
        print("❌ No samples found with graph files!")
        return
    
    # Process all samples
    print(f"\n🎯 Processing {len(available_samples)} samples...")
    
    successful_samples = []
    failed_samples = []
    
    for sample_id in tqdm(available_samples, desc="Processing samples"):
        print(f"\n--- Processing Sample {sample_id} ---")
        
        result = save_oversampled_graph_npy(sample_id, OUTPUT_DIR)
        
        if result is not None:
            successful_samples.append(result)
        else:
            failed_samples.append(sample_id)
    
    # Summary
    print(f"\n=== SAVING SUMMARY ===")
    print(f"✅ Successfully processed: {len(successful_samples)} samples")
    if failed_samples:
        print(f"❌ Failed: {len(failed_samples)} samples: {failed_samples}")
    
    if successful_samples:
        # Calculate overall statistics
        total_original_nodes = sum(s['original_nodes'] for s in successful_samples)
        total_original_edges = sum(s['original_edges'] for s in successful_samples)
        total_oversampled_nodes = sum(s['oversampled_nodes'] for s in successful_samples)
        total_oversampled_edges = sum(s['oversampled_edges'] for s in successful_samples)
        
        total_node_increase = sum(s['node_increase'] for s in successful_samples)
        total_edge_increase = sum(s['edge_increase'] for s in successful_samples)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   • Original graphs: {total_original_nodes} nodes, {total_original_edges} edges")
        print(f"   • Oversampled graphs: {total_oversampled_nodes} nodes, {total_oversampled_edges} edges")
        print(f"   • Total node increase: {total_node_increase} nodes")
        print(f"   • Total edge increase: {total_edge_increase} edges")
        print(f"   • Average node increase: {total_node_increase/len(successful_samples):.1f} nodes per sample")
        print(f"   • Average edge increase: {total_edge_increase/len(successful_samples):.1f} edges per sample")
        
        # Save detailed summary
        summary_path = os.path.join(OUTPUT_DIR, 'oversampled_graphs_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("OVERsampled GRAPHS SAVING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Oversampling spacing: {SPACING} pixels\n")
            f.write(f"Output directory: {OUTPUT_DIR}\n")
            f.write(f"Total samples processed: {len(successful_samples)}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Original graphs: {total_original_nodes} nodes, {total_original_edges} edges\n")
            f.write(f"Oversampled graphs: {total_oversampled_nodes} nodes, {total_oversampled_edges} edges\n")
            f.write(f"Total node increase: {total_node_increase} nodes\n")
            f.write(f"Total edge increase: {total_edge_increase} edges\n\n")
            
            f.write("DETAILED RESULTS:\n")
            for sample in successful_samples:
                f.write(f"\nSample {sample['sample_id']}:\n")
                f.write(f"  Original: {sample['original_nodes']} nodes, {sample['original_edges']} edges\n")
                f.write(f"  Oversampled: {sample['oversampled_nodes']} nodes, {sample['oversampled_edges']} edges\n")
                f.write(f"  Increase: +{sample['node_increase']} nodes, +{sample['edge_increase']} edges\n")
                f.write(f"  Saved as: {sample['filename']}\n")
        
        print(f"\n✅ Detailed summary saved to: {summary_path}")
        
        # Show file structure
        print(f"\n📁 Files saved in {OUTPUT_DIR}:")
        for sample in successful_samples:
            print(f"   • {sample['filename']}")
    
    print(f"\n🎯 All oversampled graphs saved to: {OUTPUT_DIR}/")
    print(f"🎯 Ready for training! Use these .npy files in your training pipeline.")

if __name__ == '__main__':
    main()
