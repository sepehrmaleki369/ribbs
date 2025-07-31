import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from skimage.morphology import skeletonize, thin, remove_small_objects, binary_closing
from skimage.filters import gaussian
from skimage.measure import label as skimage_label, regionprops
import seaborn as sns

def remove_short_branches(skeleton, min_branch_length=10):
    """Remove short branches from skeleton"""
    # This is a simplified version - you can implement more sophisticated branch removal
    return remove_small_objects(skeleton, min_size=min_branch_length)

def clean_skeleton_comprehensive(skeleton, binary_vessels):
    """Comprehensive skeleton cleaning"""
    
    # Step 1: Remove very small components
    labeled = skimage_label(skeleton)
    props = regionprops(labeled)
    
    cleaned = np.zeros_like(skeleton)
    for prop in props:
        if prop.area >= 15:  # Minimum branch length
            cleaned[labeled == prop.label] = 1
    
    # Step 2: Remove thin connections
    dist_transform = distance_transform_edt(binary_vessels)
    skeleton_coords = np.where(cleaned > 0)
    
    for i, j in zip(skeleton_coords[0], skeleton_coords[1]):
        if dist_transform[i, j] < 1.5:  # Remove if original vessel was too thin
            cleaned[i, j] = 0
    
    # Step 3: Remove short branches
    cleaned = remove_short_branches(cleaned, min_branch_length=10)
    
    return cleaned.astype(np.uint8)

def create_better_skeleton(binary_vessels):
    """Create skeleton that better preserves vessel connectivity"""
    
    # Use morphological thinning instead of skeletonize for better connectivity
    from skimage.morphology import thin
    
    # Apply thinning to ensure better connectivity
    skeleton = thin(binary_vessels)
    
    # Clean up small branches
    skeleton = remove_small_objects(skeleton, min_size=5)
    
    return skeleton.astype(np.uint8)

def remove_thin_branches_from_labels(binary_vessels, min_thickness=5):
    """Remove very thin branches from labels before skeletonization"""
    
    # Create distance transform to identify thin areas
    dist_transform = distance_transform_edt(binary_vessels)
    
    # Remove areas that are too thin (increased from 3 to 5)
    cleaned = binary_vessels.copy()
    cleaned[dist_transform < min_thickness] = 0
    
    # Remove small disconnected components that might be created (increased from 50 to 100)
    cleaned = remove_small_objects(cleaned.astype(bool), min_size=100)
    
    return cleaned.astype(np.uint8)

def improved_skeletonization():
    """Improved skeletonization with better preprocessing and post-processing"""
    
    print("=== IMPROVED SKELETONIZATION ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("improved_skeleton_analysis", exist_ok=True)
    
    # Define paths
    images_dir = "drive/training/images_npy"
    labels_dir = "drive/training/inverted_labels"
    
    # Get all files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    def preprocess_vessels(binary_vessels):
        """Remove thin branches from labels before skeletonization"""
        # Remove very thin branches (thickness < 5 pixels, increased from 3)
        cleaned = remove_thin_branches_from_labels(binary_vessels, min_thickness=5)
        return cleaned
    
    def create_simple_skeleton(binary_vessels):
        """Original skeletonization without any post-processing"""
        # Just use skeletonize - no additional processing
        skeleton = skeletonize(binary_vessels > 0)
        return skeleton.astype(np.uint8)
    
    def create_distance_map(skeleton):
        """Correct distance map creation"""
        # Distance from skeleton pixels (not background)
        background_mask = skeleton == 0
        distance_map = distance_transform_edt(background_mask)
        return distance_map
    
    def create_skeleton_methods(binary_vessels):
        """Create skeletons using different methods"""
        # Method 1: Original skeletonize (no preprocessing)
        skeleton1 = skeletonize(binary_vessels > 0).astype(np.uint8)
        
        # Method 2: Preprocessed + skeletonize (removed thin branches)
        preprocessed = preprocess_vessels(binary_vessels)
        skeleton2 = skeletonize(preprocessed > 0).astype(np.uint8)
        
        # Method 3: Simple approach (recommended)
        skeleton3 = create_simple_skeleton(binary_vessels)
        
        # Method 4: Comprehensive cleaning
        skeleton4 = clean_skeleton_comprehensive(skeleton3, binary_vessels)
        
        # Method 5: Preprocessed + comprehensive cleaning (best method)
        skeleton5 = clean_skeleton_comprehensive(skeleton2, preprocessed)
        
        return {
            'original': skeleton1,
            'preprocessed': skeleton2,
            'simple': skeleton3,
            'comprehensive': skeleton4,
            'preprocessed_comprehensive': skeleton5
        }
    
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
        
        # Create skeletons using different methods
        skeletons = create_skeleton_methods(binary_vessels)
        
        # Create distance maps from best method (preprocessed + comprehensive)
        distance_map = create_distance_map(skeletons['preprocessed_comprehensive'])
        
        # Store data
        all_data.append({
            'index': i,
            'image_file': img_file,
            'label_file': label_file,
            'image': image,
            'original_label': label,
            'binary_vessels': binary_vessels,
            'skeletons': skeletons,
            'distance_map': distance_map,
            'stats': {
                'vessel_pixels': binary_vessels.sum(),
                'original_skeleton_pixels': skeletons['original'].sum(),
                'preprocessed_skeleton_pixels': skeletons['preprocessed'].sum(),
                'simple_skeleton_pixels': skeletons['simple'].sum(),
                'comprehensive_skeleton_pixels': skeletons['comprehensive'].sum(),
                'preprocessed_comprehensive_skeleton_pixels': skeletons['preprocessed_comprehensive'].sum(),
                'distance_range': [distance_map.min(), distance_map.max()],
                'distance_mean': distance_map.mean(),
                'distance_std': distance_map.std()
            }
        })
        
        print(f"  Original vessels: {binary_vessels.sum()} pixels")
        print(f"  Original skeleton: {skeletons['original'].sum()} pixels")
        print(f"  Preprocessed skeleton: {skeletons['preprocessed'].sum()} pixels")
        print(f"  Comprehensive skeleton: {skeletons['comprehensive'].sum()} pixels")
        print(f"  Preprocessed+Comprehensive skeleton: {skeletons['preprocessed_comprehensive'].sum()} pixels")
        print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
    
    print(f"\nSuccessfully processed {len(all_data)} samples")
    
    # Create comprehensive visualization
    print("\n=== CREATING COMPREHENSIVE VISUALIZATION ===")
    
    num_samples = min(4, len(all_data))
    fig, axes = plt.subplots(num_samples, 7, figsize=(35, 5 * num_samples))
    fig.suptitle('Skeletonization Methods Comparison with Thin Branch Removal', fontsize=16)
    
    for i in range(num_samples):
        data = all_data[i]
        
        # Original image
        axes[i, 0].imshow(data['image'] / 255.0)
        axes[i, 0].set_title(f"Original Image\n{data['image_file']}")
        axes[i, 0].axis('off')
        
        # Binary vessels
        axes[i, 1].imshow(data['binary_vessels'], cmap='gray')
        axes[i, 1].set_title(f"Original Vessels\n{data['stats']['vessel_pixels']} pixels")
        axes[i, 1].axis('off')
        
        # Preprocessed vessels (thin branches removed)
        preprocessed_vessels = preprocess_vessels(data['binary_vessels'])
        axes[i, 2].imshow(preprocessed_vessels, cmap='gray')
        axes[i, 2].set_title(f"Preprocessed Vessels\n{preprocessed_vessels.sum()} pixels")
        axes[i, 2].axis('off')
        
        # Original skeleton
        axes[i, 3].imshow(data['skeletons']['original'], cmap='gray')
        axes[i, 3].set_title(f"Original Skeleton\n{data['stats']['original_skeleton_pixels']} pixels")
        axes[i, 3].axis('off')
        
        # Preprocessed skeleton
        axes[i, 4].imshow(data['skeletons']['preprocessed'], cmap='gray')
        axes[i, 4].set_title(f"Preprocessed Skeleton\n{data['stats']['preprocessed_skeleton_pixels']} pixels")
        axes[i, 4].axis('off')
        
        # Best skeleton (preprocessed + comprehensive)
        axes[i, 5].imshow(data['skeletons']['preprocessed_comprehensive'], cmap='gray')
        axes[i, 5].set_title(f"Best Skeleton\n{data['stats']['preprocessed_comprehensive_skeleton_pixels']} pixels")
        axes[i, 5].axis('off')
        
        # Distance map
        im_dist = axes[i, 6].imshow(data['distance_map'], cmap='hot')
        axes[i, 6].set_title(f"Distance Map\nRange: [{data['distance_map'].min():.1f}, {data['distance_map'].max():.1f}]")
        axes[i, 6].axis('off')
        plt.colorbar(im_dist, ax=axes[i, 6], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('improved_skeleton_analysis/methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Methods comparison saved as: improved_skeleton_analysis/methods_comparison.png")
    
    # Create detailed analysis for first sample
    print("\n=== CREATING DETAILED ANALYSIS ===")
    
    data = all_data[0]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Detailed Analysis - Sample 1', fontsize=16)
    
    # Row 1: Vessel preprocessing
    axes[0, 0].imshow(data['binary_vessels'], cmap='gray')
    axes[0, 0].set_title('Original Vessels')
    axes[0, 0].axis('off')
    
    preprocessed_vessels = preprocess_vessels(data['binary_vessels'])
    axes[0, 1].imshow(preprocessed_vessels, cmap='gray')
    axes[0, 1].set_title('Preprocessed Vessels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(data['skeletons']['original'], cmap='gray')
    axes[0, 2].set_title('Original Skeleton')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(data['skeletons']['preprocessed_comprehensive'], cmap='gray')
    axes[0, 3].set_title('Best Skeleton')
    axes[0, 3].axis('off')
    
    # Row 2: Overlays and distance map
    # Overlay original skeleton
    overlay1 = data['image'] / 255.0
    overlay1[data['skeletons']['original'] > 0] = [1, 0, 0]
    axes[1, 0].imshow(overlay1)
    axes[1, 0].set_title('Original Skeleton Overlay')
    axes[1, 0].axis('off')
    
    # Overlay best skeleton
    overlay2 = data['image'] / 255.0
    overlay2[data['skeletons']['preprocessed_comprehensive'] > 0] = [0, 1, 0]
    axes[1, 1].imshow(overlay2)
    axes[1, 1].set_title('Best Skeleton Overlay')
    axes[1, 1].axis('off')
    
    # Distance map
    axes[1, 2].imshow(data['distance_map'], cmap='hot')
    axes[1, 2].set_title('Distance Map')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(data['distance_map'].flatten(), bins=50, alpha=0.7)
    axes[1, 3].set_title('Distance Distribution')
    axes[1, 3].set_xlabel('Distance to Skeleton')
    axes[1, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('improved_skeleton_analysis/detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: improved_skeleton_analysis/detailed_analysis.png")
    
    # Create statistics summary
    print("\n=== CREATING STATISTICS SUMMARY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skeletonization Statistics with Thin Branch Removal', fontsize=16)
    
    # Pixel counts comparison
    original_counts = [d['stats']['original_skeleton_pixels'] for d in all_data]
    preprocessed_counts = [d['stats']['preprocessed_skeleton_pixels'] for d in all_data]
    best_counts = [d['stats']['preprocessed_comprehensive_skeleton_pixels'] for d in all_data]
    
    x = np.arange(len(all_data))
    width = 0.25
    
    axes[0, 0].bar(x - width, original_counts, width, label='Original', alpha=0.7)
    axes[0, 0].bar(x, preprocessed_counts, width, label='Preprocessed', alpha=0.7)
    axes[0, 0].bar(x + width, best_counts, width, label='Best', alpha=0.7)
    axes[0, 0].set_title('Skeleton Pixel Counts Comparison')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Number of Pixels')
    axes[0, 0].legend()
    
    # Reduction ratios
    vessel_counts = [d['stats']['vessel_pixels'] for d in all_data]
    original_ratios = [orig / vessel for orig, vessel in zip(original_counts, vessel_counts)]
    best_ratios = [best / vessel for best, vessel in zip(best_counts, vessel_counts)]
    
    axes[0, 1].bar(x - width/2, original_ratios, width, label='Original', alpha=0.7)
    axes[0, 1].bar(x + width/2, best_ratios, width, label='Best', alpha=0.7)
    axes[0, 1].set_title('Reduction Ratios')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Skeleton/Vessel Ratio')
    axes[0, 1].legend()
    
    # Distance map statistics
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
    total_vessel_pixels = sum([d['stats']['vessel_pixels'] for d in all_data])
    total_original_skeleton = sum([d['stats']['original_skeleton_pixels'] for d in all_data])
    total_best_skeleton = sum([d['stats']['preprocessed_comprehensive_skeleton_pixels'] for d in all_data])
    
    axes[1, 0].hist(all_distances, bins=100, alpha=0.7)
    axes[1, 0].set_title('Overall Distance Distribution')
    axes[1, 0].set_xlabel('Distance to Skeleton')
    axes[1, 0].set_ylabel('Frequency')
    
    # Pie chart comparison
    axes[1, 1].pie([total_vessel_pixels - total_original_skeleton, total_original_skeleton], 
                   labels=['Removed', 'Original Skeleton'], autopct='%1.1f%%')
    axes[1, 1].set_title('Original Skeleton Reduction')
    
    axes[1, 2].pie([total_vessel_pixels - total_best_skeleton, total_best_skeleton], 
                   labels=['Removed', 'Best Skeleton'], autopct='%1.1f%%')
    axes[1, 2].set_title('Best Skeleton Reduction')
    
    plt.tight_layout()
    plt.savefig('improved_skeleton_analysis/statistics_summary.png', dpi=300, bbox_inches='tight')
    print("Statistics summary saved as: improved_skeleton_analysis/statistics_summary.png")
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(f"Total samples: {len(all_data)}")
    print(f"Image shape: {all_data[0]['image'].shape}")
    print(f"Distance map shape: {all_data[0]['distance_map'].shape}")
    
    print(f"\nSkeletonization statistics:")
    print(f"  Total vessel pixels: {total_vessel_pixels}")
    print(f"  Total original skeleton pixels: {total_original_skeleton}")
    print(f"  Total best skeleton pixels: {total_best_skeleton}")
    print(f"  Original reduction ratio: {total_original_skeleton / total_vessel_pixels:.3f}")
    print(f"  Best reduction ratio: {total_best_skeleton / total_vessel_pixels:.3f}")
    
    print(f"\nDistance map statistics:")
    print(f"  Global distance range: [{all_distances.min():.2f}, {all_distances.max():.2f}]")
    print(f"  Global distance mean: {all_distances.mean():.2f}")
    print(f"  Global distance std: {all_distances.std():.2f}")
    
    print(f"\nThin branch removal improvements:")
    print(f"  Original skeleton: {total_original_skeleton} pixels")
    print(f"  Best skeleton: {total_best_skeleton} pixels")
    print(f"  Reduction: {total_original_skeleton - total_best_skeleton} pixels ({((total_original_skeleton - total_best_skeleton) / total_original_skeleton * 100):.1f}%)")
    
    print(f"\nAll improved skeleton analysis files saved to: improved_skeleton_analysis/")

if __name__ == "__main__":
    improved_skeletonization() 