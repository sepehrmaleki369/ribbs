import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def plot_all_distance_maps():
    """Plot all distance maps from the training folder"""
    
    print("=== PLOTTING ALL DISTANCE MAPS ===")
    
    # Get distance map files
    distance_map_files = [f for f in os.listdir('drive/training/distance_maps') if f.endswith('.npy')]
    distance_map_files.sort()
    
    print(f"Found {len(distance_map_files)} distance map files")
    
    # Calculate grid layout
    num_files = len(distance_map_files)
    cols = 5  # 5 columns
    rows = (num_files + cols - 1) // cols  # Calculate rows needed
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    fig.suptitle('All Distance Maps from Training Dataset (Clean Skeletonization)', fontsize=16)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Plot each distance map
    for i, filename in enumerate(tqdm(distance_map_files, desc="Plotting distance maps")):
        distance_path = os.path.join('drive/training/distance_maps', filename)
        distance_map = np.load(distance_path)
        
        # Get statistics
        skeleton_pixels = np.sum(distance_map == 0)
        distance_mean = distance_map.mean()
        distance_range = [distance_map.min(), distance_map.max()]
        
        # Plot
        im = axes[i].imshow(distance_map, cmap='hot')
        axes[i].set_title(f'{filename.replace("_distance_map.npy", "")}\n'
                         f'Skeleton: {skeleton_pixels}\n'
                         f'Mean: {distance_mean:.1f}\n'
                         f'Range: [{distance_range[0]:.1f}, {distance_range[1]:.1f}]')
        axes[i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(num_files, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions/all_distance_maps_training.png', dpi=300, bbox_inches='tight')
    print(f"\nAll distance maps plot saved as predictions/all_distance_maps_training.png")
    
    # Create summary statistics
    create_summary_statistics(distance_map_files)
    
    print(f"\n✅ All distance maps plotted successfully!")

def create_summary_statistics(distance_map_files):
    """Create summary statistics for all distance maps"""
    
    print(f"\n=== CREATING SUMMARY STATISTICS ===")
    
    skeleton_pixels_list = []
    distance_means_list = []
    distance_ranges_list = []
    
    for filename in distance_map_files:
        distance_path = os.path.join('drive/training/distance_maps', filename)
        distance_map = np.load(distance_path)
        
        skeleton_pixels = np.sum(distance_map == 0)
        distance_mean = distance_map.mean()
        distance_range = distance_map.max()
        
        skeleton_pixels_list.append(skeleton_pixels)
        distance_means_list.append(distance_mean)
        distance_ranges_list.append(distance_range)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distance Maps Summary Statistics', fontsize=16)
    
    # Plot 1: Skeleton pixel counts
    axes[0, 0].hist(skeleton_pixels_list, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Skeleton Pixel Counts')
    axes[0, 0].set_xlabel('Skeleton Pixels')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(skeleton_pixels_list), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(skeleton_pixels_list):.0f}')
    axes[0, 0].legend()
    
    # Plot 2: Distance means
    axes[0, 1].hist(distance_means_list, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Distance Map Means')
    axes[0, 1].set_xlabel('Mean Distance')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(distance_means_list), color='red', linestyle='--',
                       label=f'Mean: {np.mean(distance_means_list):.2f}')
    axes[0, 1].legend()
    
    # Plot 3: Distance ranges
    axes[1, 0].hist(distance_ranges_list, bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Distance Map Ranges (Max Distance)')
    axes[1, 0].set_xlabel('Max Distance')
    axes[1, 0].set_ylabel('Number of Images')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(distance_ranges_list), color='red', linestyle='--',
                       label=f'Mean: {np.mean(distance_ranges_list):.2f}')
    axes[1, 0].legend()
    
    # Plot 4: Scatter plot of skeleton pixels vs distance mean
    axes[1, 1].scatter(skeleton_pixels_list, distance_means_list, alpha=0.7, color='purple')
    axes[1, 1].set_title('Skeleton Pixels vs Distance Mean')
    axes[1, 1].set_xlabel('Skeleton Pixels')
    axes[1, 1].set_ylabel('Distance Mean')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(skeleton_pixels_list, distance_means_list)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predictions/distance_maps_summary_statistics.png', dpi=300, bbox_inches='tight')
    print(f"Summary statistics saved as predictions/distance_maps_summary_statistics.png")
    
    # Print summary
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total files: {len(distance_map_files)}")
    print(f"Average skeleton pixels: {np.mean(skeleton_pixels_list):.0f} ± {np.std(skeleton_pixels_list):.0f}")
    print(f"Average distance mean: {np.mean(distance_means_list):.2f} ± {np.std(distance_means_list):.2f}")
    print(f"Average distance range: {np.mean(distance_ranges_list):.2f} ± {np.std(distance_ranges_list):.2f}")
    print(f"Correlation (skeleton pixels vs distance mean): {correlation:.3f}")

if __name__ == "__main__":
    plot_all_distance_maps() 