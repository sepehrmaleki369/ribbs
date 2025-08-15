# Plot Discrete Distance Maps with Images and Skeletons
# Visualize discrete distance maps alongside original images and skeletons

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from core.general_dataset.io import load_array_from_file
import os
from tqdm import tqdm

def plot_discrete_distance_maps_with_images():
    """Plot discrete distance maps with original images and skeletons"""
    
    print("=== PLOTTING DISCRETE DISTANCE MAPS WITH IMAGES ===")
    
    # Check if discrete distance maps exist
    discrete_maps_dir = 'drive/training/discrete_distance_maps'
    if not os.path.exists(discrete_maps_dir):
        print(f"❌ Discrete distance maps directory not found: {discrete_maps_dir}")
        print("Please run create_discrete_distance_maps.py first")
        return
    
    # Get discrete distance map files
    discrete_map_files = [f for f in os.listdir(discrete_maps_dir) if f.endswith('.npy')]
    discrete_map_files.sort()
    
    print(f"Found {len(discrete_map_files)} discrete distance map files")
    
    # Create output directory
    os.makedirs('predictions/discrete_visualizations', exist_ok=True)
    
    # Process each file
    for i, filename in enumerate(tqdm(discrete_map_files, desc="Creating visualizations")):
        # Load discrete distance map
        discrete_map_path = os.path.join(discrete_maps_dir, filename)
        discrete_map = np.load(discrete_map_path)
        
        # Get corresponding image and label files
        stem = filename.replace('_discrete_distance_map.npy', '')
        # Extract just the number part (e.g., "21" from "21_manual1")
        image_number = stem.split('_')[0]
        image_filename = f"{image_number}_training.tif"  # Fixed: use number + _training.tif
        label_filename = f"{stem}.npy"
        
        image_path = os.path.join('drive/training/images', image_filename)
        label_path = os.path.join('drive/training/inverted_labels', label_filename)
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"⚠️  Label not found: {label_path}")
            continue
        
        # Load image and label
        image = load_array_from_file(image_path)
        vessel_label = load_array_from_file(label_path)
        
        # Create skeleton from label
        binary_vessels = (vessel_label == 0).astype(np.uint8)  # vessels are 0
        skeleton = skeletonize(binary_vessels)
        
        # Create visualization
        create_single_visualization(image, vessel_label, skeleton, discrete_map, binary_vessels, stem, i)
    
    # Create summary visualization
    create_summary_visualization(discrete_map_files)
    
    print(f"\n✅ VISUALIZATIONS CREATED SUCCESSFULLY!")
    print(f"Individual plots saved to: predictions/discrete_visualizations/")
    print(f"Summary plot saved to: predictions/discrete_summary.png")

def create_single_visualization(image, vessel_label, skeleton, discrete_map, binary_vessels, stem, index):
    """Create visualization for a single image"""
    
    # Create figure with 1x3 subplots (removed overlays)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Discrete Distance Map Analysis: {stem}', fontsize=16)
    
    # Original image (FIXED: Proper normalization for RGB images)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image - normalize to [0, 1] range
        normalized_image = image / 255.0
        axes[0].imshow(normalized_image)
        axes[0].set_title('Original Retinal Image (RGB)')
    else:
        # Grayscale image - normalize to [0, 1] range
        normalized_image = image / 255.0
        axes[0].imshow(normalized_image, cmap='gray')
        axes[0].set_title('Original Retinal Image (Grayscale)')
    axes[0].axis('off')
    
    # Skeleton
    axes[1].imshow(skeleton, cmap='Reds')
    axes[1].set_title('Skeleton (Centerlines)')
    axes[1].axis('off')
    
    # Discrete distance map
    im = axes[2].imshow(discrete_map, cmap='hot')
    axes[2].set_title('Discrete Distance Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    - Image shape: {image.shape}
    - Image range: [{image.min():.1f}, {image.max():.1f}]
    - Vessel pixels: {np.sum(binary_vessels):,}
    - Skeleton pixels: {np.sum(skeleton):,}
    - Distance range: [{discrete_map.min()}, {discrete_map.max()}]
    - Unique values: {len(np.unique(discrete_map))}
    - Mean distance: {discrete_map.mean():.1f}
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual plot
    output_path = f'predictions/discrete_visualizations/{stem}_discrete_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_visualization(discrete_map_files):
    """Create a summary visualization showing multiple examples"""
    
    # Select 6 examples for summary
    num_examples = min(6, len(discrete_map_files))
    selected_files = discrete_map_files[:num_examples]
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
    fig.suptitle('Discrete Distance Maps Summary', fontsize=16)
    
    for i, filename in enumerate(selected_files):
        # Load data
        discrete_map_path = os.path.join('drive/training/discrete_distance_maps', filename)
        discrete_map = np.load(discrete_map_path)
        
        stem = filename.replace('_discrete_distance_map.npy', '')
        # Extract just the number part (e.g., "21" from "21_manual1")
        image_number = stem.split('_')[0]
        image_filename = f"{image_number}_training.tif"  # Fixed: use number + _training.tif
        label_filename = f"{stem}.npy"
        
        image_path = os.path.join('drive/training/images', image_filename)
        label_path = os.path.join('drive/training/inverted_labels', label_filename)
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            image = load_array_from_file(image_path)
            vessel_label = load_array_from_file(label_path)
            binary_vessels = (vessel_label == 0).astype(np.uint8)
            skeleton = skeletonize(binary_vessels)
            
            # Original image (FIXED: Proper normalization for RGB images)
            if len(image.shape) == 3 and image.shape[2] == 3:
                normalized_image = image / 255.0
                axes[i, 0].imshow(normalized_image)
                axes[i, 0].set_title(f'{stem}\nOriginal Retinal Image (RGB)')
            else:
                normalized_image = image / 255.0
                axes[i, 0].imshow(normalized_image, cmap='gray')
                axes[i, 0].set_title(f'{stem}\nOriginal Retinal Image (Grayscale)')
            axes[i, 0].axis('off')
            
            # Skeleton
            axes[i, 1].imshow(skeleton, cmap='Reds')
            axes[i, 1].set_title('Skeleton')
            axes[i, 1].axis('off')
            
            # Discrete distance map
            im = axes[i, 2].imshow(discrete_map, cmap='hot')
            axes[i, 2].set_title('Discrete Distance Map')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('predictions/discrete_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_discrete_distance_maps_with_images()
 