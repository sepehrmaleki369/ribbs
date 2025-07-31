import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import gaussian
from skimage.morphology import binary_closing, binary_opening
from scipy.ndimage import distance_transform_edt
from core.general_dataset.io import load_array_from_file
import os
from tqdm import tqdm

def visualize_fixed_skeletonization():
    """Visualize skeletonization results from the fixed version"""
    
    print("=== VISUALIZING FIXED SKELETONIZATION RESULTS ===")
    
    # Get training label files
    training_label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    training_label_files.sort()
    
    print(f"Found {len(training_label_files)} training label files")
    
    # Process first 6 images for visualization
    num_images = min(6, len(training_label_files))
    
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5*num_images))
    fig.suptitle('Fixed Skeletonization Results - Proper Vessel Processing', fontsize=16)
    
    for i, filename in enumerate(training_label_files[:num_images]):
        print(f"\nProcessing {filename}...")
        
        # Load label
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        
        # FIXED: Properly handle inverted labels
        binary_vessels = (vessel_label == 0).astype(np.uint8)  # vessels are 0
        
        # Preprocess vessels
        cleaned_vessels = preprocess_vessels_fixed(binary_vessels)
        
        # Create skeleton
        skeleton = skeletonize(cleaned_vessels)
        
        # Post-process skeleton
        final_skeleton = postprocess_skeleton_fixed(skeleton, cleaned_vessels)
        
        # Create distance map
        distance_map = create_distance_map_fixed(final_skeleton)
        
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
        ax_row[1].set_title(f'Binary Vessels (FIXED)\n{np.sum(binary_vessels)} pixels\n({np.sum(binary_vessels)/vessel_label.size*100:.1f}%)')
        ax_row[1].axis('off')
        
        # Final skeleton
        ax_row[2].imshow(final_skeleton, cmap='hot')
        ax_row[2].set_title(f'Final Skeleton\n{np.sum(final_skeleton)} pixels\n({np.sum(final_skeleton)/np.sum(binary_vessels)*100:.1f}% of vessels)')
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
    plt.savefig('predictions/fixed_skeletonization_results.png', dpi=300, bbox_inches='tight')
    print("\nSkeletonization results saved as predictions/fixed_skeletonization_results.png")
    
    # Create detailed analysis for one image
    create_detailed_analysis(training_label_files[0])
    
    # Create comparison with old vs new approach
    create_comparison_visualization(training_label_files[0])
    
    print(f"\n✅ Fixed skeletonization visualization completed!")

def preprocess_vessels_fixed(binary_vessels):
    """Preprocess vessels with FIXED approach for inverted labels"""
    
    # Remove very small noise
    cleaned = remove_small_objects(binary_vessels.astype(bool), min_size=5)
    
    # Apply morphological operations to clean up vessels
    # Close small gaps
    cleaned = binary_closing(cleaned, footprint=np.ones((3, 3)))
    
    # Remove small holes
    cleaned = binary_opening(cleaned, footprint=np.ones((2, 2)))
    
    # Smooth the vessels slightly
    cleaned = gaussian(cleaned.astype(float), sigma=0.5) > 0.5
    
    return cleaned.astype(np.uint8)

def postprocess_skeleton_fixed(skeleton, binary_vessels):
    """Post-process skeleton with FIXED approach"""
    
    # Remove very small skeleton components
    cleaned_skeleton = remove_small_objects(skeleton.astype(bool), min_size=5)
    
    # Remove skeleton pixels that are too close to vessel boundaries
    # This helps eliminate artifacts from thin vessel walls
    dist_transform = distance_transform_edt(binary_vessels)
    skeleton_coords = np.where(cleaned_skeleton > 0)
    
    for i, j in zip(skeleton_coords[0], skeleton_coords[1]):
        if dist_transform[i, j] < 1.0:  # Remove if too close to boundary
            cleaned_skeleton[i, j] = 0
    
    return cleaned_skeleton.astype(np.uint8)

def create_distance_map_fixed(skeleton):
    """Create distance map from skeleton with FIXED approach"""
    
    # Distance from skeleton pixels (not background)
    # Skeleton pixels should have distance 0, everything else shows distance to skeleton
    background_mask = skeleton == 0
    distance_map = distance_transform_edt(background_mask)
    
    return distance_map

def create_detailed_analysis(filename):
    """Create detailed analysis for one image"""
    
    print(f"\nCreating detailed analysis for {filename}...")
    
    # Load and process
    label_path = os.path.join('drive/training/inverted_labels', filename)
    vessel_label = load_array_from_file(label_path)
    binary_vessels = (vessel_label == 0).astype(np.uint8)
    cleaned_vessels = preprocess_vessels_fixed(binary_vessels)
    skeleton = skeletonize(cleaned_vessels)
    final_skeleton = postprocess_skeleton_fixed(skeleton, cleaned_vessels)
    distance_map = create_distance_map_fixed(final_skeleton)
    
    # Create detailed visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Detailed Analysis: {filename} (Fixed Version)', fontsize=16)
    
    # Row 1: Processing steps
    axes[0, 0].imshow(vessel_label, cmap='gray')
    axes[0, 0].set_title('Original Label')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary_vessels, cmap='gray')
    axes[0, 1].set_title(f'Binary Vessels\n{np.sum(binary_vessels)} pixels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cleaned_vessels, cmap='gray')
    axes[0, 2].set_title(f'Cleaned Vessels\n{np.sum(cleaned_vessels)} pixels')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(skeleton, cmap='hot')
    axes[0, 3].set_title(f'Raw Skeleton\n{np.sum(skeleton)} pixels')
    axes[0, 3].axis('off')
    
    # Row 2: Final results and analysis
    axes[1, 0].imshow(final_skeleton, cmap='hot')
    axes[1, 0].set_title(f'Final Skeleton\n{np.sum(final_skeleton)} pixels')
    axes[1, 0].axis('off')
    
    im1 = axes[1, 1].imshow(distance_map, cmap='hot')
    axes[1, 1].set_title(f'Distance Map\nRange: [{distance_map.min():.1f}, {distance_map.max():.1f}]')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay skeleton on vessels
    overlay = np.zeros((*final_skeleton.shape, 3))
    overlay[cleaned_vessels > 0] = [0.5, 0.5, 0.5]  # Gray for vessels
    overlay[final_skeleton > 0] = [1, 0, 0]  # Red for skeleton
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Skeleton Overlay\nRed = Skeleton, Gray = Vessels')
    axes[1, 2].axis('off')
    
    # Distance map histogram
    axes[1, 3].hist(distance_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 3].set_title('Distance Map Histogram')
    axes[1, 3].set_xlabel('Distance Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].set_yscale('log')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'predictions/detailed_analysis_{filename.replace(".npy", "")}.png', dpi=300, bbox_inches='tight')
    print(f"Detailed analysis saved as predictions/detailed_analysis_{filename.replace('.npy', '')}.png")

def create_comparison_visualization(filename):
    """Create comparison between old (wrong) and new (fixed) approaches"""
    
    print(f"\nCreating comparison visualization for {filename}...")
    
    # Load label
    label_path = os.path.join('drive/training/inverted_labels', filename)
    vessel_label = load_array_from_file(label_path)
    
    # OLD (WRONG) approach
    binary_vessels_old = (vessel_label > 0).astype(np.uint8)  # Wrong: treating 16 as vessels
    skeleton_old = skeletonize(binary_vessels_old)
    distance_map_old = distance_transform_edt(skeleton_old == 0)
    
    # NEW (FIXED) approach
    binary_vessels_new = (vessel_label == 0).astype(np.uint8)  # Correct: vessels are 0
    cleaned_vessels_new = preprocess_vessels_fixed(binary_vessels_new)
    skeleton_new = skeletonize(cleaned_vessels_new)
    final_skeleton_new = postprocess_skeleton_fixed(skeleton_new, cleaned_vessels_new)
    distance_map_new = create_distance_map_fixed(final_skeleton_new)
    
    # Create comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Comparison: Old (Wrong) vs New (Fixed) Approach - {filename}', fontsize=16)
    
    # Row 1: Old (Wrong) approach
    axes[0, 0].imshow(binary_vessels_old, cmap='gray')
    axes[0, 0].set_title('OLD: Binary Vessels (WRONG)\nTreating 16 as vessels')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(skeleton_old, cmap='hot')
    axes[0, 1].set_title(f'OLD: Skeleton (WRONG)\n{np.sum(skeleton_old)} pixels')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(distance_map_old, cmap='hot')
    axes[0, 2].set_title(f'OLD: Distance Map (WRONG)\nRange: [{distance_map_old.min():.1f}, {distance_map_old.max():.1f}]')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    axes[0, 3].hist(distance_map_old.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 3].set_title('OLD: Distance Histogram (WRONG)')
    axes[0, 3].set_xlabel('Distance Value')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_yscale('log')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: New (Fixed) approach
    axes[1, 0].imshow(binary_vessels_new, cmap='gray')
    axes[1, 0].set_title('NEW: Binary Vessels (FIXED)\nTreating 0 as vessels')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_skeleton_new, cmap='hot')
    axes[1, 1].set_title(f'NEW: Skeleton (FIXED)\n{np.sum(final_skeleton_new)} pixels')
    axes[1, 1].axis('off')
    
    im2 = axes[1, 2].imshow(distance_map_new, cmap='hot')
    axes[1, 2].set_title(f'NEW: Distance Map (FIXED)\nRange: [{distance_map_new.min():.1f}, {distance_map_new.max():.1f}]')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    axes[1, 3].hist(distance_map_new.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 3].set_title('NEW: Distance Histogram (FIXED)')
    axes[1, 3].set_xlabel('Distance Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].set_yscale('log')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'predictions/comparison_old_vs_new_{filename.replace(".npy", "")}.png', dpi=300, bbox_inches='tight')
    print(f"Comparison saved as predictions/comparison_old_vs_new_{filename.replace('.npy', '')}.png")
    
    # Print comparison statistics
    print(f"\n=== COMPARISON STATISTICS ===")
    print(f"OLD (WRONG) approach:")
    print(f"  Vessel pixels: {np.sum(binary_vessels_old)}")
    print(f"  Skeleton pixels: {np.sum(skeleton_old)}")
    print(f"  Distance map range: [{distance_map_old.min():.2f}, {distance_map_old.max():.2f}]")
    print(f"  Distance map mean: {distance_map_old.mean():.2f}")
    
    print(f"\nNEW (FIXED) approach:")
    print(f"  Vessel pixels: {np.sum(binary_vessels_new)}")
    print(f"  Skeleton pixels: {np.sum(final_skeleton_new)}")
    print(f"  Distance map range: [{distance_map_new.min():.2f}, {distance_map_new.max():.2f}]")
    print(f"  Distance map mean: {distance_map_new.mean():.2f}")

if __name__ == "__main__":
    visualize_fixed_skeletonization() 