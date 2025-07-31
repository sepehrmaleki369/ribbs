import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import seaborn as sns

def skeletonize_and_distance_maps():
    """Skeletonize vessels, create distance maps, and prepare dataset"""
    
    print("=== SKELETONIZE VESSELS AND CREATE DISTANCE MAPS ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("skeleton_analysis", exist_ok=True)
    
    # Define paths
    images_dir = "drive/training/images_npy"
    labels_dir = "drive/training/inverted_labels"
    
    # Get all files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    # Process all samples
    all_data = []
    
    for i, (img_file, label_file) in enumerate(zip(image_files, label_files)):
        print(f"\nProcessing sample {i+1}: {img_file}")
        
        # Load data
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        
        image = load_array_from_file(img_path)
        label = load_array_from_file(label_path)
        
        # Convert to binary (0 for background, 1 for vessels)
        binary_vessels = (label > 0).astype(np.uint8)
        
        # Skeletonize vessels
        skeleton = skeletonize(binary_vessels).astype(np.uint8)
        
        # Create distance map from skeleton
        distance_map = distance_transform_edt(skeleton == 0)  # Distance to skeleton
        
        # Store data
        all_data.append({
            'index': i,
            'image_file': img_file,
            'label_file': label_file,
            'image': image,
            'original_label': label,
            'binary_vessels': binary_vessels,
            'skeleton': skeleton,
            'distance_map': distance_map,
            'stats': {
                'vessel_pixels': binary_vessels.sum(),
                'skeleton_pixels': skeleton.sum(),
                'reduction_ratio': skeleton.sum() / binary_vessels.sum() if binary_vessels.sum() > 0 else 0,
                'distance_range': [distance_map.min(), distance_map.max()],
                'distance_mean': distance_map.mean(),
                'distance_std': distance_map.std()
            }
        })
        
        print(f"  Original vessels: {binary_vessels.sum()} pixels")
        print(f"  Skeleton: {skeleton.sum()} pixels")
        print(f"  Reduction ratio: {skeleton.sum() / binary_vessels.sum():.3f}")
        print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
    
    print(f"\nSuccessfully processed {len(all_data)} samples")
    
    # Create comprehensive visualization
    print("\n=== CREATING COMPREHENSIVE VISUALIZATION ===")
    
    num_samples = min(8, len(all_data))
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))
    fig.suptitle('Vessel Skeletonization and Distance Maps', fontsize=16)
    
    for i in range(num_samples):
        data = all_data[i]
        
        # Original image
        axes[i, 0].imshow(data['image'] / 255.0)
        axes[i, 0].set_title(f"Original Image\n{data['image_file']}")
        axes[i, 0].axis('off')
        
        # Original binary vessels
        axes[i, 1].imshow(data['binary_vessels'], cmap='gray')
        axes[i, 1].set_title(f"Binary Vessels\n{data['stats']['vessel_pixels']} pixels")
        axes[i, 1].axis('off')
        
        # Skeleton
        axes[i, 2].imshow(data['skeleton'], cmap='gray')
        axes[i, 2].set_title(f"Skeleton\n{data['stats']['skeleton_pixels']} pixels")
        axes[i, 2].axis('off')
        
        # Distance map
        im_dist = axes[i, 3].imshow(data['distance_map'], cmap='hot')
        axes[i, 3].set_title(f"Distance Map\nRange: [{data['distance_map'].min():.1f}, {data['distance_map'].max():.1f}]")
        axes[i, 3].axis('off')
        plt.colorbar(im_dist, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # Overlay skeleton on original image
        overlay = data['image'] / 255.0
        overlay[data['skeleton'] > 0] = [1, 0, 0]  # Red skeleton
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title(f"Skeleton Overlay\nRed = centerlines")
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('skeleton_analysis/skeletonization_results.png', dpi=300, bbox_inches='tight')
    print("Skeletonization results saved as: skeleton_analysis/skeletonization_results.png")
    
    # Create detailed analysis for first few samples
    print("\n=== CREATING DETAILED ANALYSIS ===")
    
    num_detailed = min(4, len(all_data))
    fig, axes = plt.subplots(num_detailed, 4, figsize=(20, 5 * num_detailed))
    fig.suptitle('Detailed Skeleton Analysis', fontsize=16)
    
    for i in range(num_detailed):
        data = all_data[i]
        
        # Original vessels
        axes[i, 0].imshow(data['binary_vessels'], cmap='gray')
        axes[i, 0].set_title(f"Sample {i+1}: Original Vessels")
        axes[i, 0].axis('off')
        
        # Skeleton
        axes[i, 1].imshow(data['skeleton'], cmap='gray')
        axes[i, 1].set_title(f"Sample {i+1}: Skeleton")
        axes[i, 1].axis('off')
        
        # Distance map
        axes[i, 2].imshow(data['distance_map'], cmap='hot')
        axes[i, 2].set_title(f"Sample {i+1}: Distance Map")
        axes[i, 2].axis('off')
        
        # Histogram
        axes[i, 3].hist(data['distance_map'].flatten(), bins=50, alpha=0.7)
        axes[i, 3].set_title(f"Sample {i+1}: Distance Distribution")
        axes[i, 3].set_xlabel('Distance to Skeleton')
        axes[i, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('skeleton_analysis/detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: skeleton_analysis/detailed_analysis.png")
    
    # Create statistics summary
    print("\n=== CREATING STATISTICS SUMMARY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skeletonization Statistics Summary', fontsize=16)
    
    # Vessel pixel counts
    vessel_counts = [d['stats']['vessel_pixels'] for d in all_data]
    skeleton_counts = [d['stats']['skeleton_pixels'] for d in all_data]
    reduction_ratios = [d['stats']['reduction_ratio'] for d in all_data]
    
    axes[0, 0].bar(range(len(vessel_counts)), vessel_counts, alpha=0.7, label='Vessel Pixels')
    axes[0, 0].bar(range(len(skeleton_counts)), skeleton_counts, alpha=0.7, label='Skeleton Pixels')
    axes[0, 0].set_title('Pixel Counts per Sample')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Number of Pixels')
    axes[0, 0].legend()
    
    axes[0, 1].bar(range(len(reduction_ratios)), reduction_ratios, alpha=0.7)
    axes[0, 1].set_title('Reduction Ratio (Skeleton/Vessels)')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Ratio')
    
    # Distance map statistics
    distance_ranges = [d['stats']['distance_range'] for d in all_data]
    distance_means = [d['stats']['distance_mean'] for d in all_data]
    distance_stds = [d['stats']['distance_std'] for d in all_data]
    
    axes[0, 2].bar(range(len(distance_means)), distance_means, alpha=0.7, label='Mean')
    axes[0, 2].bar(range(len(distance_stds)), distance_stds, alpha=0.7, label='Std')
    axes[0, 2].set_title('Distance Map Statistics')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Distance Value')
    axes[0, 2].legend()
    
    # Overall statistics
    all_distances = np.concatenate([d['distance_map'].flatten() for d in all_data])
    all_vessel_pixels = sum([d['stats']['vessel_pixels'] for d in all_data])
    all_skeleton_pixels = sum([d['stats']['skeleton_pixels'] for d in all_data])
    
    axes[1, 0].hist(all_distances, bins=100, alpha=0.7)
    axes[1, 0].set_title('Overall Distance Distribution')
    axes[1, 0].set_xlabel('Distance to Skeleton')
    axes[1, 0].set_ylabel('Frequency')
    
    # Pie chart of pixel reduction
    axes[1, 1].pie([all_vessel_pixels - all_skeleton_pixels, all_skeleton_pixels], 
                   labels=['Removed', 'Skeleton'], autopct='%1.1f%%')
    axes[1, 1].set_title('Overall Pixel Reduction')
    
    # Distance range distribution
    distance_maxs = [d['stats']['distance_range'][1] for d in all_data]
    axes[1, 2].bar(range(len(distance_maxs)), distance_maxs, alpha=0.7)
    axes[1, 2].set_title('Maximum Distance per Sample')
    axes[1, 2].set_xlabel('Sample Index')
    axes[1, 2].set_ylabel('Max Distance')
    
    plt.tight_layout()
    plt.savefig('skeleton_analysis/statistics_summary.png', dpi=300, bbox_inches='tight')
    print("Statistics summary saved as: skeleton_analysis/statistics_summary.png")
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(f"Total samples: {len(all_data)}")
    print(f"Image shape: {all_data[0]['image'].shape}")
    print(f"Distance map shape: {all_data[0]['distance_map'].shape}")
    
    print(f"\nSkeletonization statistics:")
    print(f"  Total vessel pixels: {all_vessel_pixels}")
    print(f"  Total skeleton pixels: {all_skeleton_pixels}")
    print(f"  Overall reduction ratio: {all_skeleton_pixels / all_vessel_pixels:.3f}")
    print(f"  Average reduction per sample: {np.mean(reduction_ratios):.3f}")
    
    print(f"\nDistance map statistics:")
    print(f"  Global distance range: [{all_distances.min():.2f}, {all_distances.max():.2f}]")
    print(f"  Global distance mean: {all_distances.mean():.2f}")
    print(f"  Global distance std: {all_distances.std():.2f}")
    
    print(f"\nDataset prepared:")
    print(f"  Input: Original images ({len(all_data)} samples)")
    print(f"  Ground truth: Distance maps from skeletons")
    print(f"  Each sample: Image (584×565×3) + Distance map (584×565)")
    
    print(f"\nAll skeleton analysis files saved to: skeleton_analysis/")

if __name__ == "__main__":
    skeletonize_and_distance_maps() 