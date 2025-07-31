import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as skimage_label, regionprops
import seaborn as sns

def debug_labels():
    """Debug binary labels to identify sources of spurious skeleton branches"""
    
    print("=== DEBUGGING BINARY LABELS ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("label_debug", exist_ok=True)
    
    # Define paths
    images_dir = "drive/training/images_npy"
    labels_dir = "drive/training/inverted_labels"
    
    # Get all files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    def debug_single_label(binary_vessels, sample_name):
        """Debug a single binary label"""
        
        print(f"\n=== DEBUGGING {sample_name} ===")
        
        # Check for noise and connected components
        labeled = skimage_label(binary_vessels)
        props = regionprops(labeled)
        
        print(f"Number of connected components: {len(props)}")
        print(f"Total vessel pixels: {binary_vessels.sum()}")
        
        # Analyze component sizes
        areas = [prop.area for prop in props]
        print(f"Component areas: {sorted(areas, reverse=True)[:10]}...")  # Top 10 largest
        
        # Check for small components (potential noise)
        small_components = [prop for prop in props if prop.area < 50]
        print(f"Small components (< 50 pixels): {len(small_components)}")
        
        # Check boundary smoothness
        boundaries = binary_vessels - binary_erosion(binary_vessels)
        print(f"Boundary pixels: {boundaries.sum()}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Label Debug Analysis - {sample_name}', fontsize=16)
        
        # Row 1: Original analysis
        axes[0, 0].imshow(binary_vessels, cmap='gray')
        axes[0, 0].set_title('Original Binary')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(labeled, cmap='tab20')
        axes[0, 1].set_title(f'Connected Components\n({len(props)} components)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(boundaries, cmap='gray')
        axes[0, 2].set_title('Boundaries')
        axes[0, 2].axis('off')
        
        # Skeleton
        skeleton = skeletonize(binary_vessels > 0)
        axes[0, 3].imshow(skeleton, cmap='gray')
        axes[0, 3].set_title(f'Skeleton\n({skeleton.sum()} pixels)')
        axes[0, 3].axis('off')
        
        # Row 2: Noise analysis and cleaning
        # Remove small components
        cleaned = remove_small_objects(binary_vessels.astype(bool), min_size=50)
        axes[1, 0].imshow(cleaned, cmap='gray')
        axes[1, 0].set_title('After Removing Small Objects')
        axes[1, 0].axis('off')
        
        # Cleaned skeleton
        cleaned_skeleton = skeletonize(cleaned)
        axes[1, 1].imshow(cleaned_skeleton, cmap='gray')
        axes[1, 1].set_title(f'Cleaned Skeleton\n({cleaned_skeleton.sum()} pixels)')
        axes[1, 1].axis('off')
        
        # Overlay original skeleton on vessels
        overlay1 = binary_vessels.copy()
        overlay1[skeleton > 0] = 2  # Mark skeleton in different color
        axes[1, 2].imshow(overlay1, cmap='gray')
        axes[1, 2].set_title('Skeleton Overlay')
        axes[1, 2].axis('off')
        
        # Overlay cleaned skeleton on vessels
        overlay2 = binary_vessels.copy()
        overlay2[cleaned_skeleton > 0] = 2
        axes[1, 3].imshow(overlay2, cmap='gray')
        axes[1, 3].set_title('Cleaned Skeleton Overlay')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'label_debug/debug_{sample_name}.png', dpi=300, bbox_inches='tight')
        print(f"Debug visualization saved as: label_debug/debug_{sample_name}.png")
        
        return {
            'original_vessels': binary_vessels,
            'original_skeleton': skeleton,
            'cleaned_vessels': cleaned,
            'cleaned_skeleton': cleaned_skeleton,
            'num_components': len(props),
            'small_components': len(small_components),
            'boundary_pixels': boundaries.sum(),
            'original_skeleton_pixels': skeleton.sum(),
            'cleaned_skeleton_pixels': cleaned_skeleton.sum()
        }
    
    # Debug first few samples
    num_debug_samples = min(5, len(image_files))
    debug_results = []
    
    for i in range(num_debug_samples):
        img_file = image_files[i]
        label_file = label_files[i]
        
        # Load data
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        
        image = load_array_from_file(img_path)
        label = load_array_from_file(label_path)
        
        # Convert to binary
        binary_vessels = (label > 0).astype(np.uint8)
        
        # Debug this sample
        sample_name = img_file.replace('.npy', '')
        result = debug_single_label(binary_vessels, sample_name)
        debug_results.append(result)
    
    # Create summary analysis
    print("\n=== SUMMARY ANALYSIS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Label Debug Summary', fontsize=16)
    
    # Component counts
    component_counts = [r['num_components'] for r in debug_results]
    small_component_counts = [r['small_components'] for r in debug_results]
    
    x = np.arange(len(debug_results))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, component_counts, width, label='Total Components', alpha=0.7)
    axes[0, 0].bar(x + width/2, small_component_counts, width, label='Small Components', alpha=0.7)
    axes[0, 0].set_title('Component Analysis')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Number of Components')
    axes[0, 0].legend()
    
    # Skeleton pixel counts
    original_skeleton_pixels = [r['original_skeleton_pixels'] for r in debug_results]
    cleaned_skeleton_pixels = [r['cleaned_skeleton_pixels'] for r in debug_results]
    
    axes[0, 1].bar(x - width/2, original_skeleton_pixels, width, label='Original Skeleton', alpha=0.7)
    axes[0, 1].bar(x + width/2, cleaned_skeleton_pixels, width, label='Cleaned Skeleton', alpha=0.7)
    axes[0, 1].set_title('Skeleton Pixel Counts')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Number of Pixels')
    axes[0, 1].legend()
    
    # Boundary pixels
    boundary_pixels = [r['boundary_pixels'] for r in debug_results]
    axes[0, 2].bar(x, boundary_pixels, alpha=0.7)
    axes[0, 2].set_title('Boundary Pixels')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Number of Pixels')
    
    # Reduction ratios
    vessel_pixels = [r['original_vessels'].sum() for r in debug_results]
    original_ratios = [orig / vessel for orig, vessel in zip(original_skeleton_pixels, vessel_pixels)]
    cleaned_ratios = [cleaned / vessel for cleaned, vessel in zip(cleaned_skeleton_pixels, vessel_pixels)]
    
    axes[1, 0].bar(x - width/2, original_ratios, width, label='Original', alpha=0.7)
    axes[1, 0].bar(x + width/2, cleaned_ratios, width, label='Cleaned', alpha=0.7)
    axes[1, 0].set_title('Skeleton/Vessel Ratios')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    
    # Improvement percentages
    improvements = [(orig - cleaned) / orig * 100 for orig, cleaned in zip(original_skeleton_pixels, cleaned_skeleton_pixels)]
    axes[1, 1].bar(x, improvements, alpha=0.7)
    axes[1, 1].set_title('Skeleton Reduction (%)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Reduction (%)')
    
    # Component size distribution
    all_areas = []
    for r in debug_results:
        labeled = skimage_label(r['original_vessels'])
        props = regionprops(labeled)
        all_areas.extend([prop.area for prop in props])
    
    axes[1, 2].hist(all_areas, bins=50, alpha=0.7)
    axes[1, 2].set_title('Component Size Distribution')
    axes[1, 2].set_xlabel('Component Area (pixels)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('label_debug/summary_analysis.png', dpi=300, bbox_inches='tight')
    print("Summary analysis saved as: label_debug/summary_analysis.png")
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(f"Analyzed {len(debug_results)} samples")
    
    print(f"\nComponent analysis:")
    print(f"  Average components per sample: {np.mean(component_counts):.1f}")
    print(f"  Average small components per sample: {np.mean(small_component_counts):.1f}")
    print(f"  Small components as % of total: {np.mean(small_component_counts) / np.mean(component_counts) * 100:.1f}%")
    
    print(f"\nSkeleton analysis:")
    print(f"  Average original skeleton pixels: {np.mean(original_skeleton_pixels):.0f}")
    print(f"  Average cleaned skeleton pixels: {np.mean(cleaned_skeleton_pixels):.0f}")
    print(f"  Average reduction: {np.mean(improvements):.1f}%")
    
    print(f"\nBoundary analysis:")
    print(f"  Average boundary pixels: {np.mean(boundary_pixels):.0f}")
    print(f"  Boundary pixels as % of vessels: {np.mean(boundary_pixels) / np.mean(vessel_pixels) * 100:.1f}%")
    
    print(f"\nAll debug files saved to: label_debug/")

if __name__ == "__main__":
    debug_labels() 