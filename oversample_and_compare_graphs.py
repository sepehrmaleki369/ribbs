import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
import networkx as nx
import yaml

# Import after setting up paths
import sys
sys.path.append('.')

from core.regression_dataset import RegressionDataset
from utils.graphs import load_training_graph_by_id

# Configuration
IMAGES_DIR = 'drive/training/images_npy'
GRAPHS_DIR = 'drive/training/graphs'
OUT_DIR = 'oversampled_graph_comparison'

def interpolate_new_nodes(pos1, pos2, spacing):
    """
    Interpolate new nodes between two positions with given spacing.
    
    Args:
        pos1: First position (x1, y1)
        pos2: Second position (x2, y2)
        spacing: Distance between interpolated nodes
        
    Returns:
        List of new node positions
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

def oversampling_graph(G, spacing=20):
    """
    Oversample graph by adding interpolated nodes along edges.
    
    Args:
        G: NetworkX graph
        spacing: Distance between interpolated nodes
        
    Returns:
        Oversampled NetworkX graph
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

def load_image(sample_id: int) -> np.ndarray:
    """Load training image."""
    path = os.path.join(IMAGES_DIR, f"{sample_id}_training.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.load(path)
    # Normalize for display if needed
    if img.dtype != np.float32 and img.max() > 1:
        img = img.astype(np.float32) / 255.0
    return img

def create_comparison_visualization(sample_id, sample, initial_graph, oversampled_graph, spacing=20):
    """
    Create a comparison visualization showing initial vs oversampled graphs.
    """
    
    # Create 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get image data
    original_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    if original_image.max() > 1.0:
        display_image = original_image / 255.0
    else:
        display_image = original_image
    
    # Panel 1: Training image
    axes[0].imshow(display_image)
    axes[0].set_title(f'Training Image {sample_id}', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # Panel 2: Initial graph overlay (BLUE)
    axes[1].imshow(display_image)
    if initial_graph is not None:
        # Draw initial graph nodes (BLUE)
        for node in initial_graph.nodes():
            node_attrs = initial_graph.nodes[node]
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
                        
                    if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
                        axes[1].plot(x, y, 'bo', markersize=4, alpha=0.8)
                except Exception as e:
                    continue
        
        # Draw initial graph edges (BLUE)
        for edge in initial_graph.edges():
            try:
                node1_attrs = initial_graph.nodes[edge[0]]
                node2_attrs = initial_graph.nodes[edge[1]]
                
                pos1 = None
                pos2 = None
                
                for attrs in [node1_attrs, node2_attrs]:
                    if 'pos' in attrs:
                        pos = attrs['pos']
                    elif 'position' in attrs:
                        pos = attrs['position']
                    elif 'coords' in attrs:
                        pos = attrs['coords']
                    else:
                        pos = None
                        
                    if pos is not None and hasattr(pos, '__getitem__') and len(pos) >= 2:
                        if attrs == node1_attrs:
                            pos1 = (pos[0].item() if hasattr(pos[0], 'item') else pos[0], 
                                   pos[1].item() if hasattr(pos[1], 'item') else pos[1])
                        else:
                            pos2 = (pos[0].item() if hasattr(pos[0], 'item') else pos[0], 
                                   pos[1].item() if hasattr(pos[1], 'item') else pos[1])
                
                if pos1 is not None and pos2 is not None:
                    x1, y1 = pos1
                    x2, y2 = pos2
                    
                    if (0 <= x1 < display_image.shape[1] and 0 <= y1 < display_image.shape[0] and 
                        0 <= x2 < display_image.shape[1] and 0 <= y2 < display_image.shape[0]):
                        axes[1].plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.8)
                        
            except Exception as e:
                continue
    
    axes[1].set_title(f'Initial Graph (BLUE)\n{len(initial_graph.nodes())} nodes, {len(initial_graph.edges())} edges', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # Panel 3: Oversampled graph overlay (RED)
    axes[2].imshow(display_image)
    if oversampled_graph is not None:
        # Draw oversampled graph nodes (RED)
        for node in oversampled_graph.nodes():
            node_attrs = oversampled_graph.nodes[node]
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
                        
                    if 0 <= x < display_image.shape[1] and 0 <= y < display_image.shape[0]:
                        axes[2].plot(x, y, 'ro', markersize=3, alpha=0.8)
                except Exception as e:
                    continue
        
        # Draw oversampled graph edges (RED)
        for edge in oversampled_graph.edges():
            try:
                node1_attrs = oversampled_graph.nodes[edge[0]]
                node2_attrs = oversampled_graph.nodes[edge[1]]
                
                pos1 = None
                pos2 = None
                
                for attrs in [node1_attrs, node2_attrs]:
                    if 'pos' in attrs:
                        pos = attrs['pos']
                    elif 'position' in attrs:
                        pos = attrs['position']
                    elif 'coords' in attrs:
                        pos = attrs['coords']
                    else:
                        pos = None
                        
                    if pos is not None and hasattr(pos, '__getitem__') and len(pos) >= 2:
                        if attrs == node1_attrs:
                            pos1 = (pos[0].item() if hasattr(pos[0], 'item') else pos[0], 
                                   pos[1].item() if hasattr(pos[1], 'item') else pos[1])
                        else:
                            pos2 = (pos[0].item() if hasattr(pos[0], 'item') else pos[0], 
                                   pos[1].item() if hasattr(pos[1], 'item') else pos[1])
                
                if pos1 is not None and pos2 is not None:
                    x1, y1 = pos1
                    x2, y2 = pos2
                    
                    if (0 <= x1 < display_image.shape[1] and 0 <= y1 < display_image.shape[0] and 
                        0 <= x2 < display_image.shape[1] and 0 <= y2 < display_image.shape[0]):
                        axes[2].plot([x1, x2], [y1, y2], 'r-', linewidth=1, alpha=0.8)
                        
            except Exception as e:
                continue
    
    axes[2].set_title(f'Oversampled Graph (RED)\n{len(oversampled_graph.nodes())} nodes, {len(oversampled_graph.edges())} edges\nSpacing: {spacing}', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    # Main title
    plt.suptitle(f'Graph Oversampling Comparison - Sample {sample_id}\nBLUE=Initial | RED=Oversampled (spacing={spacing})', 
                 fontsize=16, weight='bold')
    
    plt.tight_layout()
    return fig

def get_all_available_samples():
    """Get all sample IDs that have both image and graph files."""
    available_samples = []
    
    # Check what images are available
    if os.path.exists(IMAGES_DIR):
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('_training.npy')]
        image_samples = [int(f.split('_')[0]) for f in image_files]
        print(f"📁 Found {len(image_samples)} training images: {sorted(image_samples)}")
    else:
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return []
    
    # Check what graphs are available
    if os.path.exists(GRAPHS_DIR):
        graph_files = [f for f in os.listdir(GRAPHS_DIR) if f.endswith('.graph') or f.endswith('.npy.graph')]
        graph_samples = list(set([int(f.split('_')[0]) for f in graph_files]))
        print(f"📁 Found {len(graph_samples)} graph files: {sorted(graph_samples)}")
    else:
        print(f"❌ Graphs directory not found: {GRAPHS_DIR}")
        return []
    
    # Find intersection (samples with both image and graph)
    available_samples = sorted(list(set(image_samples) & set(graph_samples)))
    print(f"🎯 {len(available_samples)} samples have both image and graph: {available_samples}")
    
    return available_samples

def main():
    """Main function to oversample graphs and compare with initial graphs."""
    
    print("=== GRAPH OVERSAMPLING AND COMPARISON ===")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUT_DIR}")
    
    # Get all available samples
    available_samples = get_all_available_samples()
    
    if not available_samples:
        print("❌ No samples found with both image and graph files!")
        return
    
    # Oversampling parameters
    spacing = 5  # Distance between interpolated nodes
    print(f"🎯 Using oversampling spacing: {spacing} pixels")
    
    # Process all samples
    print(f"\n🎯 Processing {len(available_samples)} samples...")
    
    successful_samples = []
    failed_samples = []
    
    for sample_id in tqdm(available_samples, desc="Processing samples"):
        print(f"\n--- Processing Sample {sample_id} ---")
        
        try:
            # Load training sample
            sample = None
            try:
                # Load configuration and create dataset
                with open('configs/dataset/drive_regression.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                train_dataset = RegressionDataset(config, split='train')
                
                # Try to find sample by sample_id
                for idx in range(len(train_dataset)):
                    try:
                        temp_sample = train_dataset[idx]
                        if temp_sample.get('sample_id') == sample_id:
                            sample = temp_sample
                            break
                    except Exception as e:
                        continue
                
                # Fallback indexing if not found
                if sample is None:
                    try:
                        sample_index = sample_id - 21  # sample_id 21 -> index 0
                        if 0 <= sample_index < len(train_dataset):
                            sample = train_dataset[sample_index]
                    except Exception as e:
                        print(f"⚠️ Error with fallback indexing: {e}")
                        
            except Exception as e:
                print(f"⚠️ Error loading dataset: {e}")
                continue
            
            if sample is None:
                print(f"❌ Sample {sample_id} not found, skipping...")
                continue
            
            # Ensure sample has correct sample_id
            if 'sample_id' not in sample:
                sample['sample_id'] = sample_id
            
            # Load initial graph
            initial_graph = load_training_graph_by_id(str(sample_id))
            if initial_graph is None:
                print(f"⚠️ No graph available for sample {sample_id}, skipping...")
                continue
            
            print(f"✅ Loaded initial graph: {len(initial_graph.nodes())} nodes, {len(initial_graph.edges())} edges")
            
            # Apply oversampling
            oversampled_graph = oversampling_graph(initial_graph, spacing)
            print(f"✅ Created oversampled graph: {len(oversampled_graph.nodes())} nodes, {len(oversampled_graph.edges())} edges")
            
            # Create comparison visualization
            fig = create_comparison_visualization(sample_id, sample, initial_graph, oversampled_graph, spacing)
            
            # Save visualization
            save_name = f"sample_{sample_id:03d}_oversampling_comparison.png"
            save_path = os.path.join(OUT_DIR, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved comparison: {save_name}")
            
            # Store statistics
            successful_samples.append({
                'sample_id': sample_id,
                'initial_nodes': len(initial_graph.nodes()),
                'initial_edges': len(initial_graph.edges()),
                'oversampled_nodes': len(oversampled_graph.nodes()),
                'oversampled_edges': len(oversampled_graph.edges()),
                'node_increase': len(oversampled_graph.nodes()) - len(initial_graph.nodes()),
                'edge_increase': len(oversampled_graph.edges()) - len(initial_graph.edges())
            })
            
        except Exception as e:
            print(f"❌ Failed on sample {sample_id}: {e}")
            failed_samples.append(sample_id)
    
    # Summary
    print(f"\n=== OVERSAMPLING SUMMARY ===")
    print(f"✅ Successfully processed: {len(successful_samples)} samples")
    if failed_samples:
        print(f"❌ Failed: {len(failed_samples)} samples: {failed_samples}")
    
    if successful_samples:
        # Calculate overall statistics
        total_initial_nodes = sum(s['initial_nodes'] for s in successful_samples)
        total_initial_edges = sum(s['initial_edges'] for s in successful_samples)
        total_oversampled_nodes = sum(s['oversampled_nodes'] for s in successful_samples)
        total_oversampled_edges = sum(s['oversampled_edges'] for s in successful_samples)
        
        avg_node_increase = np.mean([s['node_increase'] for s in successful_samples])
        avg_edge_increase = np.mean([s['edge_increase'] for s in successful_samples])
        
        print(f"\n📊 Overall Statistics:")
        print(f"   • Initial graphs: {total_initial_nodes} nodes, {total_initial_edges} edges")
        print(f"   • Oversampled graphs: {total_oversampled_nodes} nodes, {total_oversampled_edges} edges")
        print(f"   • Average node increase: {avg_node_increase:.1f} nodes per sample")
        print(f"   • Average edge increase: {avg_edge_increase:.1f} edges per sample")
        print(f"   • Total node increase: {total_oversampled_nodes - total_initial_nodes} nodes")
        print(f"   • Total edge increase: {total_oversampled_edges - total_initial_edges} edges")
        
        # Save detailed summary
        summary_path = os.path.join(OUT_DIR, 'oversampling_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("GRAPH OVERSAMPLING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Oversampling spacing: {spacing} pixels\n")
            f.write(f"Total samples processed: {len(successful_samples)}\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Initial graphs: {total_initial_nodes} nodes, {total_initial_edges} edges\n")
            f.write(f"Oversampled graphs: {total_oversampled_nodes} nodes, {total_oversampled_edges} edges\n")
            f.write(f"Total node increase: {total_oversampled_nodes - total_initial_nodes} nodes\n")
            f.write(f"Total edge increase: {total_oversampled_edges - total_initial_edges} edges\n\n")
            
            f.write("DETAILED RESULTS:\n")
            for sample in successful_samples:
                f.write(f"\nSample {sample['sample_id']}:\n")
                f.write(f"  Initial: {sample['initial_nodes']} nodes, {sample['initial_edges']} edges\n")
                f.write(f"  Oversampled: {sample['oversampled_nodes']} nodes, {sample['oversampled_edges']} edges\n")
                f.write(f"  Increase: +{sample['node_increase']} nodes, +{sample['edge_increase']} edges\n")
        
        print(f"\n✅ Detailed summary saved to: {summary_path}")
    
    print(f"\n🎯 All comparisons saved to: {OUT_DIR}/")
    print(f"🎯 Oversampling complete! Check the output directory for results.")

if __name__ == '__main__':
    main()
