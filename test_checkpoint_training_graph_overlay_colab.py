# Test Single Checkpoint on Training Data - Simple Visualization (COLAB VERSION)
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import yaml
import networkx as nx
import pickle

print("=== SIMPLE TRAINING DATA VISUALIZATION WITH GRAPHS (COLAB) ===")

# Simple visualization: 3 training images with graphs overlaid
def create_simple_visualization(sample_id, sample, pred_np, true_distance_map, initial_graph, updated_graph):
    """Create a simple 4-panel visualization showing image, label, prediction with initial graph, and prediction with updated graph."""
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Panel 1: Training image
    original_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    if original_image.max() > 1.0:
        display_image = original_image / 255.0
    else:
        display_image = original_image
    
    axes[0].imshow(display_image)
    axes[0].set_title(f'Training Image {sample_id}\nSample ID: {sample_id}', fontsize=12, weight='bold')
    axes[0].axis('off')
    
    # Panel 2: True label (distance map)
    im_true = axes[1].imshow(true_distance_map, cmap='hot', vmin=0, vmax=15)
    axes[1].set_title(f'True Distance Map\nSample {sample_id}\nRange: [{true_distance_map.min():.1f}, {true_distance_map.max():.1f}]', fontsize=12, weight='bold')
    axes[1].axis('off')
    plt.colorbar(im_true, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Prediction with initial graph (BLUE)
    axes[2].imshow(pred_np, cmap='hot', vmin=0, vmax=15)
    if initial_graph is not None:
        # Draw initial graph nodes (BLUE)
        for node in initial_graph.nodes():
            pos = initial_graph.nodes[node]['pos']
            axes[2].plot(pos[0], pos[1], 'bo', markersize=8, alpha=1.0, linewidth=2)
        
        # Draw initial graph edges (BLUE)
        for edge in initial_graph.edges():
            pos1 = initial_graph.nodes[edge[0]]['pos']
            pos2 = initial_graph.nodes[edge[1]]['pos']
            axes[2].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b-', linewidth=3, alpha=1.0)
    
    axes[2].set_title(f'Prediction + Initial Graph (BLUE)\nEpoch {checkpoint_epoch}\nRange: [{pred_np.min():.1f}, {pred_np.max():.1f}]', fontsize=12, weight='bold')
    axes[2].axis('off')
    
    # Panel 4: Prediction with updated graph (GREEN)
    axes[3].imshow(pred_np, cmap='hot', vmin=0, vmax=15)
    if updated_graph is not None:
        try:
            # Draw updated graph nodes (GREEN)
            nodes_drawn = 0
            for node in updated_graph.nodes():
                node_attrs = updated_graph.nodes[node]
                pos = None
                
                # Get position from different possible attributes
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
                            
                        # Convert to numeric values if needed
                        if hasattr(x, 'item'):
                            x = x.item()
                        if hasattr(y, 'item'):
                            y = y.item()
                            
                        # Check bounds and draw
                        if 0 <= x < pred_np.shape[1] and 0 <= y < pred_np.shape[0]:
                            axes[3].plot(x, y, 'go', markersize=8, alpha=1.0, linewidth=2)
                            nodes_drawn += 1
                    except Exception as e:
                        continue
            
            # Draw updated graph edges (GREEN)
            edges_drawn = 0
            for edge in updated_graph.edges():
                try:
                    node1_attrs = updated_graph.nodes[edge[0]]
                    node2_attrs = updated_graph.nodes[edge[1]]
                    
                    pos1 = None
                    pos2 = None
                    
                    # Get positions for both nodes
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
                        
                        # Check bounds and draw
                        if (0 <= x1 < pred_np.shape[1] and 0 <= y1 < pred_np.shape[0] and 
                            0 <= x2 < pred_np.shape[1] and 0 <= y2 < pred_np.shape[0]):
                            axes[3].plot([x1, x2], [y1, y2], 'g-', linewidth=3, alpha=1.0)
                            edges_drawn += 1
                            
                except Exception as e:
                    continue
                    
            print(f"✅ Drew {nodes_drawn} nodes and {edges_drawn} edges from updated graph")
            
        except Exception as e:
            print(f"⚠️ Error drawing updated graph: {e}")
            axes[3].text(0.5, 0.5, f'Error drawing\nupdated graph:\n{e}', 
                       ha='center', va='center', fontsize=10, color='red')
    
    axes[3].set_title(f'Prediction + Updated Graph (GREEN)\nEpoch {checkpoint_epoch}\nRange: [{pred_np.min():.1f}, {pred_np.max():.1f}]', fontsize=12, weight='bold')
    axes[3].axis('off')
    
    # Add color legend and sample info
    fig.text(0.02, 0.02, f'Sample {sample_id} | Epoch {checkpoint_epoch} | Snake Loss Training\n'
                          f'BLUE = Initial Graph | GREEN = Updated Graph from Training\n'
                          f'Prediction Range: [{pred_np.min():.1f}, {pred_np.max():.1f}]', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Main title
    plt.suptitle(f'Training Sample {sample_id} - Epoch {checkpoint_epoch} (Snake Simple Loss)\n'
                 f'Image | Label | Prediction+Initial | Prediction+Updated', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig

# Configuration
checkpoint_epoch = 110  # Epoch with snake loss
drive_checkpoint_dir = '/content/drive/MyDrive/ribbs/checkpoints_clipped_mse_simple'  # Correct path for your checkpoints
checkpoint_filename = 'checkpoint_epoch_110.pth'  # Exact filename from your path

print(f"🎯 Simple Visualization:")
print(f"   • Epoch {checkpoint_epoch}: Snake Simple Loss")
print(f"   • Show ALL training images with epoch {checkpoint_epoch} data")
print(f"   • Initial graphs (BLUE) and updated graphs (GREEN)")
print(f"   • Checkpoint path: {drive_checkpoint_dir}")
print(f"   • Checkpoint file: {checkpoint_filename}")

print(f"\n📊 EPOCH {checkpoint_epoch} DATA AVAILABILITY ANALYSIS:")
print(f"   • Total samples: 16 (21-36)")
print(f"   • Samples WITH epoch {checkpoint_epoch}: 9 samples")
print(f"   • Samples WITHOUT epoch {checkpoint_epoch}: 7 samples")
print(f"\n🔍 Why some samples don't have epoch {checkpoint_epoch}:")
print(f"   • Training may have stopped early for some samples")
print(f"   • Some samples may have converged before epoch {checkpoint_epoch}")
print(f"   • Memory/GPU constraints during training")
print(f"   • Different training schedules for different samples")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load configuration
with open('/content/ribbs/configs/dataset/drive_regression.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Import after setting up paths
import sys
sys.path.append('/content/ribbs')

from models.base_models import UNet
from core.regression_dataset import RegressionDataset
from utils.graphs import load_training_graph_by_id

# Load training dataset
print("Loading training dataset...")
train_dataset = RegressionDataset(config, split='train')
print(f"✅ Found {len(train_dataset)} training samples")

# Create model
model = UNet(
    in_channels=3,
    m_channels=64,
    out_channels=1,
    n_convs=2,
    n_levels=2,
    dropout=0.1,
    norm_type='batch',
    upsampling='bilinear',
    pooling="max",
    three_dimensional=False,
    apply_final_relu=False
).to(device)

# Load checkpoint from epoch 110 (with snake loss)
checkpoint_path = f'{drive_checkpoint_dir}/{checkpoint_filename}'
if os.path.exists(checkpoint_path):
    print(f"✅ Found checkpoint: {os.path.basename(checkpoint_path)} (with snake loss)")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from epoch {checkpoint_epoch} (with snake loss)")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

model.eval()

# Create output directory
os.makedirs('/content/predictions/simple_training_visualization', exist_ok=True)

def load_evolved_graph(sample_id, epoch, evolved_graphs_dir='/content/ribbs/graphs/evolved'):
    """Load evolved graph from snake loss training."""
    filename_pkl = f"evolved_sample_{sample_id}_epoch_{epoch}.pkl"
    filepath_pkl = os.path.join(evolved_graphs_dir, filename_pkl)
    
    if os.path.exists(filepath_pkl):
        try:
            with open(filepath_pkl, 'rb') as f:
                raw_data = pickle.load(f)
            print(f"✅ Loaded evolved graph: {filename_pkl}")
            
            # Try to get the graph using getGraph method
            if hasattr(raw_data, 'getGraph'):
                try:
                    evolved_graph = raw_data.getGraph()
                    if hasattr(evolved_graph, 'nodes'):
                        print(f"✅ Extracted graph with {len(evolved_graph.nodes())} nodes")
                        return evolved_graph
                except Exception as e:
                    print(f"⚠️ Error calling getGraph(): {e}")
            
            # If getGraph fails, create simple graph from control points
            if hasattr(raw_data, 's') and len(raw_data.s) > 0:
                print(f"🔄 Creating graph from control points")
                G = nx.Graph()
                
                # Add nodes with positions from control points
                for i, pos in enumerate(raw_data.s):
                    G.add_node(i, pos=(pos[0].item(), pos[1].item()))
                
                # Add edges between consecutive nodes
                for i in range(len(raw_data.s) - 1):
                    G.add_edge(i, i + 1)
                
                # Close the loop
                if len(raw_data.s) > 2:
                    G.add_edge(len(raw_data.s) - 1, 0)
                
                print(f"✅ Created graph with {len(G.nodes())} nodes")
                return G
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading evolved graph: {e}")
            return None
    
    print(f"❌ No evolved graph found for sample {sample_id} at epoch {epoch}")
    return None

# Select ALL training samples that have epoch 110 data available
# Based on the exploration, these 9 samples have epoch 110:
available_sample_ids = [22, 24, 25, 27, 31, 32, 33, 36]  # All samples with epoch 110 data
selected_sample_ids = available_sample_ids

print(f"\n=== Processing {len(selected_sample_ids)} Training Samples ===")
print(f"📋 Selected samples: {selected_sample_ids}")
print(f"🎯 These samples have epoch {checkpoint_epoch} data available")
print(f"📊 Will create {len(selected_sample_ids)} visualizations")

# Process each training sample
for sample_id in tqdm(selected_sample_ids, desc="Processing training samples"):
    print(f"\n--- Processing Training Sample {sample_id} ---")

    # Get training sample by ID
    sample = None
    for idx in range(len(train_dataset)):
        try:
            temp_sample = train_dataset[idx]
            if temp_sample.get('sample_id') == sample_id:
                sample = temp_sample
                print(f"✅ Found sample {sample_id} at dataset index {idx}")
                break
        except Exception as e:
            continue
    
    # Fallback indexing if not found
    if sample is None:
        try:
            sample_index = sample_id - 21  # sample_id 21 -> index 0
            if 0 <= sample_index < len(train_dataset):
                sample = train_dataset[sample_index]
                print(f"✅ Loaded sample {sample_id} using index {sample_index}")
        except Exception as e:
            print(f"⚠️ Error with fallback indexing: {e}")
    
    if sample is None:
        print(f"❌ Sample {sample_id} not found, skipping...")
        continue

    # Ensure sample has correct sample_id
    if 'sample_id' not in sample:
        sample['sample_id'] = sample_id

    # Get image and prediction
    image_tensor = sample['image'].unsqueeze(0).to(device)
    true_distance_map = sample['distance_map'].cpu().numpy()

    # Pad if needed
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4

    if pad_h > 0 or pad_w > 0:
        image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    # Get prediction
    with torch.no_grad():
        prediction = model(image_tensor)
        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]

    pred_np = prediction[0, 0].cpu().numpy()
    
    # Debug: Show prediction properties
    print(f"   📊 Prediction shape: {pred_np.shape}")
    print(f"   📊 Prediction range: [{pred_np.min():.3f}, {pred_np.max():.3f}]")
    print(f"   📊 Prediction mean: {pred_np.mean():.3f}")
    print(f"   📊 Prediction std: {pred_np.std():.3f}")
    
    # Check if prediction is all zeros or very small
    if pred_np.max() < 1e-6:
        print(f"   ⚠️ Warning: Prediction appears to be all zeros or very small!")
        print(f"   📊 Max value: {pred_np.max()}")
        print(f"   📊 Min value: {pred_np.min()}")
        print(f"   📊 Non-zero pixels: {np.count_nonzero(pred_np)}")

    # Load graphs
    try:
        initial_graph = load_training_graph_by_id(str(sample_id))
        print(f"✅ Loaded initial graph for sample {sample_id}")

        updated_graph = load_evolved_graph(sample_id, checkpoint_epoch, '/content/ribbs/graphs/evolved')
        
        if updated_graph is None:
            print(f"⚠️ No evolved graph available for sample {sample_id}, skipping...")
            continue

    except Exception as e:
        print(f"⚠️ Could not load graphs for sample {sample_id}: {e}")
        continue

    # Create visualization
    fig = create_simple_visualization(sample_id, sample, pred_np, true_distance_map, initial_graph, updated_graph)
    
    # Save visualization
    save_name = f"training_sample_{sample_id:03d}_simple.png"
    save_path = f'/content/predictions/simple_training_visualization/{save_name}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved visualization: {save_name}")

print(f"\n=== COMPLETED ===")
print(f"✅ Processed {len(selected_sample_ids)} training samples")
print(f"✅ Used checkpoint from epoch {checkpoint_epoch} (Snake Loss)")
print(f"📁 Results saved to: /content/predictions/simple_training_visualization/")
print(f"🖼️ Simple 3-panel visualizations created")
print(f"\n🎯 What you get:")
print(f"• Panel 1: Training image with true distance map")
print(f"• Panel 2: Initial graph (BLUE) overlaid on prediction")
print(f"• Panel 3: Updated graph (GREEN) overlaid on prediction")
print(f"• Simple and clean - no complex analysis")
