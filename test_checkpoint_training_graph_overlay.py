# Test Single Checkpoint on Training Data - With Graph Evolution
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.base_models import UNet
from core.regression_dataset import RegressionDataset
import os
from tqdm import tqdm
import yaml
from utils.graphs import load_training_graph_by_id

print("=== TESTING SINGLE CHECKPOINT ON TRAINING DATA WITH GRAPH EVOLUTION ===")

# Configuration
checkpoint_epoch = 125  # Epoch with snake loss
checkpoint_epoch_before = 100  # Epoch before snake loss (MSE only)
drive_checkpoint_dir = '/content/drive/MyDrive/ribbs/checkpoints_clipped_mse_simple'  # Update path as needed
num_training_samples = 5  # Number of training samples to visualize

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load configuration
with open('configs/dataset/drive_regression.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load training dataset
print("Loading training dataset...")
train_dataset = RegressionDataset(config, split='train')
print(f"✅ Found {len(train_dataset)} training samples")

# Check if checkpoints exist
checkpoint_path = os.path.join(drive_checkpoint_dir, f'checkpoint_epoch_{checkpoint_epoch}.pth')
checkpoint_path_before = os.path.join(drive_checkpoint_dir, f'checkpoint_epoch_{checkpoint_epoch_before}.pth')

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)
if not os.path.exists(checkpoint_path_before):
    print(f"❌ Checkpoint not found: {checkpoint_path_before}")
    exit(1)

print(f"✅ Found checkpoint: epoch_{checkpoint_epoch}.pth (with snake loss)")
print(f"✅ Found checkpoint: epoch_{checkpoint_epoch_before}.pth (MSE only)")

# Create models
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

model_before = UNet(
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

# Load checkpoints
checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint_before = torch.load(checkpoint_path_before, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

if 'model_state_dict' in checkpoint_before:
    model_before.load_state_dict(checkpoint_before['model_state_dict'])
else:
    model_before.load_state_dict(checkpoint_before)

model.eval()
model_before.eval()
print(f"✅ Model loaded from epoch {checkpoint_epoch} (with snake loss)")
print(f"✅ Model loaded from epoch {checkpoint_epoch_before} (MSE only)")

# Create output directory
os.makedirs('predictions/training_graph_evolution_test', exist_ok=True)

def create_updated_graph(initial_graph, prediction, sample_id, window_size=5):
    """
    Create updated graph based on prediction analysis.
    This simulates how the graph would evolve based on the model's output.
    
    Args:
        initial_graph: NetworkX graph from static file
        prediction: Model's distance map prediction
        sample_id: Sample identifier for debugging
        window_size: Search window size around each node (default: 5)
    
    Returns:
        updated_graph: Modified graph with adjusted node positions
    """
    updated_graph = initial_graph.copy()
    
    # Method 1: Adjust node positions based on prediction gradients
    for node in updated_graph.nodes():
        pos = updated_graph.nodes[node]['pos']
        x, y = int(pos[0]), int(pos[1])
        
        # Ensure coordinates are within prediction bounds
        if 0 <= x < prediction.shape[1] and 0 <= y < prediction.shape[0]:
            # Find local maximum in prediction near this node
            x_start = max(0, x - window_size)
            x_end = min(prediction.shape[1], x + window_size + 1)
            y_start = max(0, y - window_size)
            y_end = min(prediction.shape[0], y + window_size + 1)
            
            local_pred = prediction[y_start:y_end, x_start:x_end]
            
            if local_pred.size > 0:
                # Find position of maximum value in local window
                local_max_pos = np.unravel_index(np.argmax(local_pred), local_pred.shape)
                new_x = x_start + local_max_pos[1]
                new_y = y_start + local_max_pos[0]
                
                # Update node position to local maximum
                updated_graph.nodes[node]['pos'] = (new_x, new_y)
                
                # Optional: Add metadata about the movement
                updated_graph.nodes[node]['original_pos'] = pos
                updated_graph.nodes[node]['movement'] = (new_x - x, new_y - y)
    
    return updated_graph

# Select training samples to visualize
sample_indices = list(range(min(num_training_samples, len(train_dataset))))

print(f"\n=== Processing {len(sample_indices)} Training Samples ===")

# Process each training sample
for sample_idx in tqdm(sample_indices, desc="Processing training samples"):
    print(f"\n--- Processing Training Sample {sample_idx} ---")
    
    # Get training sample
    sample = train_dataset[sample_idx]
    image_tensor = sample['image'].unsqueeze(0).to(device)
    true_distance_map = sample['distance_map'].cpu().numpy()
    sample_id = sample.get('sample_id', sample_idx)
    
    # Pad if needed
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    
    if pad_h > 0 or pad_w > 0:
        image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # Get predictions from both models
    with torch.no_grad():
        # Prediction from epoch 100 (MSE only)
        prediction_before = model_before(image_tensor)
        if pad_h > 0 or pad_w > 0:
            prediction_before = prediction_before[:, :, :h, :w]
        
        # Prediction from epoch 125 (with snake loss)
        prediction = model(image_tensor)
        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]
    
    pred_before_np = prediction_before[0, 0].cpu().numpy()
    pred_np = prediction[0, 0].cpu().numpy()
    
    # Load graph for this sample
    try:
        initial_graph = load_training_graph_by_id(str(sample_id))
        print(f"✅ Loaded graph for sample {sample_id}")
        
        # Create updated graph based on prediction
        window_size = 5  # Define window size for graph evolution
        updated_graph = create_updated_graph(initial_graph, pred_np, sample_id, window_size)
        print(f"✅ Created evolved graph for sample {sample_id}")
        
    except Exception as e:
        print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
        initial_graph = None
        updated_graph = None
    
    # Create visualization - 2x4 layout to show both predictions
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # Row 1: Original image, True label, Epoch 100 prediction, Epoch 125 prediction
    # Original image
    original_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    if original_image.max() > 1.0:
        display_image = original_image / 255.0
    else:
        display_image = original_image
    
    axes[0, 0].imshow(display_image)
    axes[0, 0].set_title(f'Training Image {sample_idx}\nSample ID: {sample_id}', fontsize=12)
    axes[0, 0].axis('off')
    
    # True distance map
    im_true = axes[0, 1].imshow(true_distance_map, cmap='hot', vmin=0, vmax=15)
    axes[0, 1].set_title(f'True Distance Map\nRange: [{true_distance_map.min():.1f}, {true_distance_map.max():.1f}]', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im_true, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Epoch 100 prediction (MSE only)
    im_pred_before = axes[0, 2].imshow(pred_before_np, cmap='hot', vmin=0, vmax=15)
    axes[0, 2].set_title(f'Epoch {checkpoint_epoch_before} Prediction (MSE Only)\nRange: [{pred_before_np.min():.1f}, {pred_before_np.max():.1f}]', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im_pred_before, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Epoch 125 prediction (with snake loss)
    im_pred = axes[0, 3].imshow(pred_np, cmap='hot', vmin=0, vmax=15)
    axes[0, 3].set_title(f'Epoch {checkpoint_epoch} Prediction (Snake Loss)\nRange: [{pred_np.min():.1f}, {pred_np.max():.1f}]', fontsize=12)
    axes[0, 3].axis('off')
    plt.colorbar(im_pred, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Graph overlays and error maps (7 panels total)
    # Panel 5: Empty/Info panel
    axes[1, 0].text(0.5, 0.5, f'Comparison:\nEpoch {checkpoint_epoch_before} (MSE) vs\nEpoch {checkpoint_epoch} (Snake Loss)', 
                     ha='center', va='center', fontsize=14, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    axes[1, 0].set_title('Training Comparison', fontsize=12)
    axes[1, 0].axis('off')
    
    # Panel 6: Initial graph overlay on epoch 125 prediction
    axes[1, 1].imshow(pred_np, cmap='hot', vmin=0, vmax=15)
    if initial_graph is not None:
        # Draw graph nodes
        for node in initial_graph.nodes():
            pos = initial_graph.nodes[node]['pos']
            axes[1, 1].plot(pos[0], pos[1], 'bo', markersize=6, alpha=0.9, label='Initial Node')
        
        # Draw graph edges
        for edge in initial_graph.edges():
            pos1 = initial_graph.nodes[edge[0]]['pos']
            pos2 = initial_graph.nodes[edge[1]]['pos']
            axes[1, 1].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b-', linewidth=2.5, alpha=0.9)
    
    axes[1, 1].set_title(f'Epoch {checkpoint_epoch} + INITIAL Graph\n(Snake Loss)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Panel 7: Updated graph overlay on epoch 125 prediction (GREEN - evolved based on prediction)
    axes[1, 2].imshow(pred_np, cmap='hot', vmin=0, vmax=15)
    if updated_graph is not None:
        # Draw evolved graph nodes
        for node in updated_graph.nodes():
            pos = updated_graph.nodes[node]['pos']
            axes[1, 2].plot(pos[0], pos[1], 'go', markersize=6, alpha=0.9, label='Evolved Node')
            
            # Note: Movement tracking is still available in the data but not visually displayed
            # to keep the visualization clean and focused on the graph comparison
        
        # Draw evolved graph edges
        for edge in updated_graph.edges():
            pos1 = updated_graph.nodes[edge[0]]['pos']
            pos2 = updated_graph.nodes[edge[1]]['pos']
            axes[1, 2].plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'g-', linewidth=2.5, alpha=0.9)
    
    axes[1, 2].set_title(f'Epoch {checkpoint_epoch} + EVOLVED Graph\n(Modified by Prediction)', fontsize=12)
    axes[1, 2].axis('off')
    
    # Panel 8: Error improvement comparison (epoch 125 vs epoch 100)
    error_improvement = np.abs(true_distance_map - pred_before_np) - np.abs(true_distance_map - pred_np)
    im_error_improvement = axes[1, 3].imshow(error_improvement, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 3].set_title(f'Error Improvement\n(Epoch {checkpoint_epoch_before} → Epoch {checkpoint_epoch})\nBlue=Better, Red=Worse', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im_error_improvement, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Add legend for graph evolution
    if initial_graph is not None and updated_graph is not None:
        # Count movements
        total_nodes = len(initial_graph.nodes())
        moved_nodes = sum(1 for node in updated_graph.nodes() 
                         if 'movement' in updated_graph.nodes[node] and 
                         (updated_graph.nodes[node]['movement'][0] != 0 or 
                          updated_graph.nodes[node]['movement'][1] != 0))
        
        fig.text(0.02, 0.02, f'Graph Evolution Summary:\n'
                              f'• Total nodes: {total_nodes}\n'
                              f'• Moved nodes: {moved_nodes}\n'
                              f'• Movement threshold: ±{window_size} pixels\n'
                              f'• Evolution based on: Local maxima in prediction\n'
                              f'• Colors: BLUE=Initial, GREEN=Evolved',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Training Sample {sample_idx} - Epoch Comparison: {checkpoint_epoch_before} (MSE) vs {checkpoint_epoch} (Snake Loss)\n'
                 f'Graph Evolution: Static → Prediction-Based', fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    save_name = f"training_sample_{sample_idx:03d}_epoch_{checkpoint_epoch}_graph_evolution.png"
    save_path = f'predictions/training_graph_evolution_test/{save_name}'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization: {save_name}")
    
    # Save individual files
    np.save(f'predictions/training_graph_evolution_test/sample_{sample_idx:03d}_prediction.npy', pred_np)
    np.save(f'predictions/training_graph_evolution_test/sample_{sample_idx:03d}_error.npy', error_map)
    
    # Save graph evolution data
    if updated_graph is not None:
        graph_evolution_data = {
            'sample_id': sample_id,
            'epoch': checkpoint_epoch,
            'total_nodes': len(initial_graph.nodes()) if initial_graph else 0,
            'moved_nodes': sum(1 for node in updated_graph.nodes() 
                              if 'movement' in updated_graph.nodes[node] and 
                              (updated_graph.nodes[node]['movement'][0] != 0 or 
                               updated_graph.nodes[node]['movement'][1] != 0)),
            'node_movements': {}
        }
        
        for node in updated_graph.nodes():
            if 'movement' in updated_graph.nodes[node]:
                graph_evolution_data['node_movements'][node] = {
                    'original_pos': updated_graph.nodes[node]['original_pos'],
                    'new_pos': updated_graph.nodes[node]['pos'],
                    'movement': updated_graph.nodes[node]['movement']
                }
        
        np.save(f'predictions/training_graph_evolution_test/sample_{sample_idx:03d}_graph_evolution.npy', 
                graph_evolution_data)

print(f"\n=== COMPLETED ===")
print(f"✅ Processed {len(sample_indices)} training samples")
print(f"✅ Used checkpoint from epoch {checkpoint_epoch}")
print(f"📁 Results saved to: predictions/training_graph_evolution_test/")
print(f"🖼️ Visualizations: predictions/training_graph_evolution_test/training_sample_*_graph_evolution.png")
print(f"📊 Graph evolution data saved for each sample")
print(f"\n🎯 Key Features:")
print(f"• 8-panel layout: Shows before/after snake loss comparison")
print(f"• Epoch {checkpoint_epoch_before} (MSE only): Baseline performance")
print(f"• Epoch {checkpoint_epoch} (Snake loss): Improved performance")
print(f"• Initial graph (BLUE): Static skeleton overlay")
print(f"• Evolved graph (GREEN): Modified by prediction analysis")
print(f"• Error improvement: Shows how snake loss reduces errors")
print(f"• Better visibility: GREEN vs BLUE for clear comparison")
