import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from core.general_dataset.io import load_array_from_file

def plot_clipped_distance_maps():
    """Plot clipped distance maps (0-15 range) with original images"""
    
    print("=== PLOTTING CLIPPED DISTANCE MAPS ===")
    
    # Define paths
    distance_maps_dir = 'drive/training/distance_maps'
    images_dir = 'drive/training/images_npy'
    
    if not os.path.exists(distance_maps_dir):
        print(f"❌ Distance maps directory not found: {distance_maps_dir}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Get distance map files
    distance_map_files = [f for f in os.listdir(distance_maps_dir) if f.endswith('.npy')]
    distance_map_files.sort()
    
    print(f"Found {len(distance_map_files)} distance map files")
    
    # Create output directory
    os.makedirs('predictions/clipped_visualizations', exist_ok=True)
    
    # Plot first 6 samples
    samples_to_plot = min(6, len(distance_map_files))
    
    print(f"\nPlotting {samples_to_plot} samples...")
    
    for i, filename in enumerate(tqdm(distance_map_files[:samples_to_plot], desc="Creating plots")):
        # Load distance map
        distance_map_path = os.path.join(distance_maps_dir, filename)
        distance_map = np.load(distance_map_path)
        
        # Get corresponding image
        stem = filename.replace('_training_distance_map.npy', '')
        image_filename = f"{stem}_training.npy"
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        # Load image
        image = load_array_from_file(image_path)
        
        # Create visualization
        create_single_plot(image, distance_map, stem, i)
    
    # Create summary plot
    create_summary_plot(distance_map_files[:samples_to_plot])
    
    print(f"\n✅ VISUALIZATION COMPLETED!")
    print(f"Individual plots saved to: predictions/clipped_visualizations/")
    print(f"Summary plot saved to: predictions/clipped_summary.png")

def create_single_plot(image, distance_map, stem, index):
    """Create a single plot for one sample"""
    
    # Normalize image for display
    if image.max() > 1:
        image = image / 255.0
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Clipped Distance Map Analysis: {stem}', fontsize=16)
    
    # Original image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        axes[0].imshow(image)
        axes[0].set_title('Original Retinal Image (RGB)')
    else:
        # Grayscale image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Retinal Image (Grayscale)')
    axes[0].axis('off')
    
    # Clipped distance map
    im = axes[1].imshow(distance_map, cmap='hot')
    axes[1].set_title('Clipped Distance Map (0-15)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Add statistics text
    stats_text = f"""
    Statistics:
    - Image shape: {image.shape}
    - Image range: [{image.min():.1f}, {image.max():.1f}]
    - Distance map shape: {distance_map.shape}
    - Distance range: [{distance_map.min():.1f}, {distance_map.max():.1f}]
    - Distance mean: {distance_map.mean():.2f}
    - Distance std: {distance_map.std():.2f}
    - Unique values: {len(np.unique(distance_map))}
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save individual plot
    output_path = f'predictions/clipped_visualizations/{stem}_clipped_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plot(distance_map_files):
    """Create a summary plot showing all clipped distance maps"""
    
    print("\nCreating summary plot...")
    
    # Load all distance maps for summary
    distance_maps = []
    filenames = []
    
    for filename in distance_map_files:
        distance_map_path = os.path.join('drive/training/distance_maps', filename)
        distance_map = np.load(distance_map_path)
        distance_maps.append(distance_map)
        filenames.append(filename.replace('_training_distance_map.npy', ''))
    
    # Create summary figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Clipped Distance Maps Summary (0-15 Range)', fontsize=16)
    
    # Plot each distance map
    for i, (distance_map, filename) in enumerate(zip(distance_maps, filenames)):
        row = i // 3
        col = i % 3
        
        im = axes[row, col].imshow(distance_map, cmap='hot')
        axes[row, col].set_title(f'Sample {filename}')
        axes[row, col].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(distance_maps), 6):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions/clipped_summary.png', dpi=300, bbox_inches='tight')
    print(f"✅ Summary plot saved as: predictions/clipped_summary.png")

def print_summary_statistics(distance_map_files):
    """Print summary statistics of all clipped distance maps"""
    
    print("\n=== SUMMARY STATISTICS ===")
    
    all_ranges = []
    all_means = []
    all_stds = []
    
    for filename in distance_map_files:
        distance_map_path = os.path.join('drive/training/distance_maps', filename)
        distance_map = np.load(distance_map_path)
        
        all_ranges.append([distance_map.min(), distance_map.max()])
        all_means.append(distance_map.mean())
        all_stds.append(distance_map.std())
    
    # Calculate global statistics
    global_min = min([r[0] for r in all_ranges])
    global_max = max([r[1] for r in all_ranges])
    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)
    
    print(f"Global range: [{global_min:.1f}, {global_max:.1f}]")
    print(f"Average mean: {avg_mean:.2f}")
    print(f"Average std: {avg_std:.2f}")
    print(f"Total files: {len(distance_map_files)}")
    
    # Show individual file statistics
    print(f"\nIndividual file statistics:")
    print(f"{'Filename':<25} {'Range':<15} {'Mean':<8} {'Std':<8}")
    print(f"{'-'*25} {'-'*15} {'-'*8} {'-'*8}")
    
    for filename, (min_val, max_val), mean_val, std_val in zip(distance_map_files, all_ranges, all_means, all_stds):
        filename_short = filename.replace('_training_distance_map.npy', '')
        print(f"{filename_short:<25} [{min_val:.1f}, {max_val:.1f}] {mean_val:.2f} {std_val:.2f}")

if __name__ == "__main__":
    plot_clipped_distance_maps() 