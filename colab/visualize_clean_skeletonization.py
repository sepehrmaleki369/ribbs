import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from core.general_dataset.io import load_array_from_file
import os
from tqdm import tqdm

def visualize_clean_skeletonization():
    """Visualize clean skeletonization results - NO PRE/POST PROCESSING"""
    
    print("=== VISUALIZING CLEAN SKELETONIZATION RESULTS ===")
    print("NO PRE/POST PROCESSING: Direct skeletonization for clean results")
    
    # Get training label files
    training_label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    training_label_files.sort()
    
    print(f"Found {len(training_label_files)} training label files")
    
    # Process first 6 images for visualization
    num_images = min(6, len(training_label_files))
    
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5*num_images))
    fig.suptitle('Clean Skeletonization Results - NO PRE/POST PROCESSING', fontsize=16)
    
    for i, filename in enumerate(training_label_files[:num_images]):
        print(f"\nProcessing {filename}...")
        
        # Load label
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        
        # FIXED: Properly handle inverted labels
        binary_vessels = (vessel_label == 0).astype(np.uint8)  # vessels are 0
        
        # NO PREPROCESSING: Direct skeletonization
        skeleton = skeletonize(binary_vessels)
        
        # NO POSTPROCESSING: Use skeleton directly
        final_skeleton = skeleton
        
        # Create distance map
        distance_map = create_distance_map_clean(final_skeleton)
        
        # Visualize results
        if num_images == 1:
            ax_row = axes
        else:
            ax_row = axes[i]
        
        # Original label
        ax_row[0].imshow(vessel_label, cmap='gray')
        ax_row[0].set_title(f'{filename}\nOriginal Label\nRange: [{vessel_label.min()}, {vessel_label.max()}]')
        ax_row[0].axis('off')
        
        # Binary vessels (corrected)
        ax_row[1].imshow(binary_vessels, cmap='gray')
        ax_row[1].set_title(f'Binary Vessels (CLEAN)\n{np.sum(binary_vessels)} pixels\n({np.sum(binary_vessels)/vessel_label.size*100:.1f}%)')
        ax_row[1].axis('off')
        
        # Final skeleton
        ax_row[2].imshow(final_skeleton, cmap='hot')
        ax_row[2].set_title(f'Clean Skeleton\n{np.sum(final_skeleton)} pixels\n({np.sum(final_skeleton)/np.sum(binary_vessels)*100:.1f}% of vessels)')
        ax_row[2].axis('off')
        
        # Distance map
        im = ax_row[3].imshow(distance_map, cmap='hot')
        ax_row[3].set_title(f'Distance Map\nRange: [{distance_map.min():.1f}, {distance_map.max():.1f}]\nMean: {distance_map.mean():.1f}')
        ax_row[3].axis('off')
        plt.colorbar(im, ax=ax_row[3], fraction=0.046, pad=0.04)
        
        print(f"  Vessel pixels: {np.sum(binary_vessels)}")
        print(f"  Skeleton pixels: {np.sum(final_skeleton)}")
        print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
    
    plt.tight_layout()
    plt.savefig('predictions/clean_skeletonization_results.png', dpi=300, bbox_inches='tight')
    print("\nClean skeletonization results saved as predictions/clean_skeletonization_results.png")
    
    # Create detailed analysis for one image
    create_detailed_analysis_clean(training_label_files[0])
    
    # Create comparison with old vs new approach
    create_comparison_visualization_clean(training_label_files[0])
    
    print(f"\n✅ Clean skeletonization visualization completed!")

def create_distance_map_clean(skeleton):
    """Create distance map from skeleton with CLEAN approach"""
    
    # Distance from skeleton pixels (not background)
    # Skeleton pixels should have distance 0, everything else shows distance to skeleton
    background_mask = skeleton == 0
    distance_map = distance_transform_edt(background_mask)
    
    return distance_map

def create_detailed_analysis_clean(filename):
    """Create detailed analysis for one image with clean approach"""
    
    print(f"\nCreating detailed analysis for {filename}...")
    
    # Load and process
    label_path = os.path.join('drive/training/inverted_labels', filename)
    vessel_label = load_array_from_file(label_path)
    binary_vessels = (vessel_label == 0).astype(np.uint8)
    skeleton = skeletonize(binary_vessels)
    final_skeleton = skeleton  # No postprocessing
    distance_map = create_distance_map_clean(final_skeleton)
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Detailed Analysis: {filename} (Clean Version - NO PRE/POST PROCESSING)', fontsize=16)
    
    # Row 1: Processing steps
    axes[0, 0].imshow(vessel_label, cmap='gray')
    axes[0, 0].set_title('Original Label')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary_vessels, cmap='gray')
    axes[0, 1].set_title(f'Binary Vessels\n{np.sum(binary_vessels)} pixels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(skeleton, cmap='hot')
    axes[0, 2].set_title(f'Raw Skeleton\n{np.sum(skeleton)} pixels')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(final_skeleton, cmap='hot')
    axes[0, 3].set_title(f'Final Skeleton (No Postprocessing)\n{np.sum(final_skeleton)} pixels')
    axes[0, 3].axis('off')
    
    # Row 2: Final results and analysis
    im1 = axes[1, 0].imshow(distance_map, cmap='hot')
    axes[1, 0].set_title(f'Distance Map\nRange: [{distance_map.min():.1f}, {distance_map.max():.1f}]')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay skeleton on vessels
    overlay = np.zeros((*final_skeleton.shape, 3))
    overlay[binary_vessels > 0] = [0.5, 0.5, 0.5]  # Gray for vessels
    overlay[final_skeleton > 0] = [1, 0, 0]  # Red for skeleton
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Skeleton Overlay\nRed = Skeleton, Gray = Vessels')
    axes[1, 1].axis('off')
    
    # Distance map histogram
    axes[1, 2].hist(distance_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 2].set_title('Distance Map Histogram')
    axes[1, 2].set_xlabel('Distance Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Skeleton connectivity analysis
    from skimage.measure import label as skimage_label
    skeleton_labels = skimage_label(final_skeleton)
    num_components = skeleton_labels.max()
    axes[1, 3].imshow(skeleton_labels, cmap='tab20')
    axes[1, 3].set_title(f'Skeleton Components\n{num_components} connected components')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'predictions/detailed_analysis_clean_{filename.replace(".npy", "")}.png', dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved as predictions/detailed_analysis_clean_{filename.replace('.npy', '')}.png")

def create_comparison_visualization_clean(filename):
    """Create comparison between processed and clean approaches"""
    
    print(f"\nCreating comparison visualization for {filename}...")
    
    # Load label
    label_path = os.path.join('drive/training/inverted_labels', filename)
    vessel_label = load_array_from_file(label_path)
    binary_vessels = (vessel_label == 0).astype(np.uint8)
    
    # PROCESSED approach (with pre/post processing)
    from skimage.morphology import remove_small_objects, binary_closing, binary_opening
    from skimage.filters import gaussian
    
    # Preprocess vessels
    cleaned_vessels = remove_small_objects(binary_vessels.astype(bool), min_size=5)
    cleaned_vessels = binary_closing(cleaned_vessels, footprint=np.ones((3, 3)))
    cleaned_vessels = binary_opening(cleaned_vessels, footprint=np.ones((2, 2)))
    cleaned_vessels = gaussian(cleaned_vessels.astype(float), sigma=0.5) > 0.5
    cleaned_vessels = cleaned_vessels.astype(np.uint8)
    
    # Create skeleton
    skeleton_processed = skeletonize(cleaned_vessels)
    
    # Post-process skeleton
    final_skeleton_processed = remove_small_objects(skeleton_processed.astype(bool), min_size=5)
    dist_transform = distance_transform_edt(cleaned_vessels)
    skeleton_coords = np.where(final_skeleton_processed > 0)
    for i, j in zip(skeleton_coords[0], skeleton_coords[1]):
        if dist_transform[i, j] < 1.0:
            final_skeleton_processed[i, j] = 0
    final_skeleton_processed = final_skeleton_processed.astype(np.uint8)
    
    distance_map_processed = create_distance_map_clean(final_skeleton_processed)
    
    # CLEAN approach (no pre/post processing)
    skeleton_clean = skeletonize(binary_vessels)
    final_skeleton_clean = skeleton_clean
    distance_map_clean = create_distance_map_clean(final_skeleton_clean)
    
    # Create comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Comparison: Processed vs Clean Approach - {filename}', fontsize=16)
    
    # Row 1: Processed approach
    axes[0, 0].imshow(cleaned_vessels, cmap='gray')
    axes[0, 0].set_title('PROCESSED: Cleaned Vessels\nWith preprocessing')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(final_skeleton_processed, cmap='hot')
    axes[0, 1].set_title(f'PROCESSED: Skeleton\n{np.sum(final_skeleton_processed)} pixels')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(distance_map_processed, cmap='hot')
    axes[0, 2].set_title(f'PROCESSED: Distance Map\nRange: [{distance_map_processed.min():.1f}, {distance_map_processed.max():.1f}]')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    axes[0, 3].hist(distance_map_processed.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 3].set_title('PROCESSED: Distance Histogram')
    axes[0, 3].set_xlabel('Distance Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_yscale('log')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Clean approach
    axes[1, 0].imshow(binary_vessels, cmap='gray')
    axes[1, 0].set_title('CLEAN: Binary Vessels\nNo preprocessing')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_skeleton_clean, cmap='hot')
    axes[1, 1].set_title(f'CLEAN: Skeleton\n{np.sum(final_skeleton_clean)} pixels')
    axes[1, 1].axis('off')
    
    im2 = axes[1, 2].imshow(distance_map_clean, cmap='hot')
    axes[1, 2].set_title(f'CLEAN: Distance Map\nRange: [{distance_map_clean.min():.1f}, {distance_map_clean.max():.1f}]')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    axes[1, 3].hist(distance_map_clean.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 3].set_title('CLEAN: Distance Histogram')
    axes[1, 3].set_xlabel('Distance Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].set_yscale('log')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'predictions/comparison_processed_vs_clean_{filename.replace(".npy", "")}.png', dpi=300, bbox_inches='tight')
    print(f"Comparison saved as predictions/comparison_processed_vs_clean_{filename.replace('.npy', '')}.png")
    
    # Print comparison statistics
    print(f"\n=== COMPARISON STATISTICS ===")
    print(f"PROCESSED approach:")
    print(f"  Vessel pixels: {np.sum(cleaned_vessels)}")
    print(f"  Skeleton pixels: {np.sum(final_skeleton_processed)}")
    print(f"  Distance map range: [{distance_map_processed.min():.2f}, {distance_map_processed.max():.2f}]")
    print(f"  Distance map mean: {distance_map_processed.mean():.2f}")
    
    print(f"\nCLEAN approach:")
    print(f"  Vessel pixels: {np.sum(binary_vessels)}")
    print(f"  Skeleton pixels: {np.sum(final_skeleton_clean)}")
    print(f"  Distance map range: [{distance_map_clean.min():.2f}, {distance_map_clean.max():.2f}]")
    print(f"  Distance map mean: {distance_map_clean.mean():.2f}")

if __name__ == "__main__":
    visualize_clean_skeletonization() 