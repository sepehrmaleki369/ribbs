import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import yaml
import networkx as nx

# Import after setting up paths
import sys
sys.path.append('.')

from core.regression_dataset import RegressionDataset
from utils.graphs import load_training_graph_by_id

def create_simple_graph_overlay(sample_id, sample, initial_graph):
    """
    Create a simple visualization showing graph overlaid on training image.
    """
    
    # Create 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
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
    
    # Panel 2: Graph overlay (BLUE)
    axes[1].imshow(display_image)
    if initial_graph is not None:
        # Draw graph nodes (BLUE)
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
                        axes[1].plot(x, y, 'bo', markersize=6, alpha=0.8)
                except Exception as e:
                    continue
        
        # Draw graph edges (BLUE)
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
                        axes[1].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.8)
                        
            except Exception as e:
                continue
    
    axes[1].set_title(f'Graph Overlay (BLUE)', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # Main title
    plt.suptitle(f'Graph Overlay - Sample {sample_id}', fontsize=16, weight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to overlay graphs on training images."""
    
    print("=== GRAPH OVERLAY ON TRAINING IMAGES ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = RegressionDataset(config, split='train')
    print(f"✅ Found {len(train_dataset)} training samples")
    
    # Create output directory
    output_dir = 'graph_overlays'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample IDs that have epoch 110 data
    available_sample_ids = [22, 24, 25, 27, 31, 32, 33, 36]
    
    print(f"🎯 Overlaying graphs for {len(available_sample_ids)} samples")
    print(f"📁 Output directory: {output_dir}")
    
    # Process each sample
    for sample_id in tqdm(available_sample_ids, desc="Processing samples"):
        print(f"\n--- Processing Sample {sample_id} ---")
        
        # Get training sample
        sample = None
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
        
        if sample is None:
            print(f"❌ Sample {sample_id} not found, skipping...")
            continue
        
        # Ensure sample has correct sample_id
        if 'sample_id' not in sample:
            sample['sample_id'] = sample_id
        
        # Load graph from graph folder
        try:
            initial_graph = load_training_graph_by_id(str(sample_id))
            print(f"✅ Loaded graph for sample {sample_id}")
            
            if initial_graph is None:
                print(f"⚠️ No graph available for sample {sample_id}, skipping...")
                continue
                
        except Exception as e:
            print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
            continue
        
        # Create visualization
        fig = create_simple_graph_overlay(sample_id, sample, initial_graph)
        
        # Save visualization
        save_name = f"sample_{sample_id:03d}_graph_overlay.png"
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved overlay: {save_name}")
    
    print(f"\n✅ All graph overlays saved to: {output_dir}/")
    print(f"🎯 Overlay complete! Check the output directory for results.")

if __name__ == "__main__":
    main()
