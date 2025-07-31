import numpy as np
import matplotlib.pyplot as plt
from core.general_dataset.io import load_array_from_file
import os
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm

def analyze_original_labels():
    """Analyze original vessel labels for noise, artifacts, and quality issues"""
    
    print("=== ANALYZING ORIGINAL VESSEL LABELS ===")
    
    # Get all label files
    label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    label_files.sort()
    
    print(f"Found {len(label_files)} label files")
    
    # Analyze each label
    all_stats = []
    
    for i, filename in enumerate(tqdm(label_files, desc="Analyzing labels")):
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        
        # Convert to binary
        binary_vessels = (vessel_label > 0).astype(np.uint8)
        
        # Basic statistics
        stats = {
            'filename': filename,
            'shape': vessel_label.shape,
            'original_range': [vessel_label.min(), vessel_label.max()],
            'total_pixels': vessel_label.size,
            'vessel_pixels': np.sum(binary_vessels),
            'background_pixels': np.sum(binary_vessels == 0),
            'vessel_percentage': np.sum(binary_vessels) / vessel_label.size * 100,
            'unique_values': len(np.unique(vessel_label))
        }
        
        # Analyze connected components
        labeled_vessels = label(binary_vessels)
        vessel_props = regionprops(labeled_vessels)
        
        # Component analysis
        component_areas = [prop.area for prop in vessel_props]
        stats['num_components'] = len(vessel_props)
        stats['largest_component'] = max(component_areas) if component_areas else 0
        stats['smallest_component'] = min(component_areas) if component_areas else 0
        stats['avg_component_area'] = np.mean(component_areas) if component_areas else 0
        
        # Noise analysis - count very small components
        small_components = [area for area in component_areas if area < 10]
        stats['small_components'] = len(small_components)
        stats['small_component_pixels'] = sum(small_components)
        
        # Check for isolated pixels
        isolated_pixels = np.sum(component_areas == 1)
        stats['isolated_pixels'] = isolated_pixels
        
        # Check for thin artifacts
        thin_components = [area for area in component_areas if 1 < area < 5]
        stats['thin_components'] = len(thin_components)
        
        all_stats.append(stats)
        
        # Create detailed visualization for first few images
        if i < 3:
            create_detailed_analysis(vessel_label, binary_vessels, labeled_vessels, filename, stats)
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total images analyzed: {len(all_stats)}")
    
    # Value range analysis
    all_ranges = [stats['original_range'] for stats in all_stats]
    print(f"Value ranges: {all_ranges[:5]}...")  # Show first 5
    
    # Component analysis
    avg_components = np.mean([stats['num_components'] for stats in all_stats])
    avg_vessel_percentage = np.mean([stats['vessel_percentage'] for stats in all_stats])
    total_small_components = sum([stats['small_components'] for stats in all_stats])
    total_isolated_pixels = sum([stats['isolated_pixels'] for stats in all_stats])
    
    print(f"Average components per image: {avg_components:.1f}")
    print(f"Average vessel percentage: {avg_vessel_percentage:.2f}%")
    print(f"Total small components (<10 pixels): {total_small_components}")
    print(f"Total isolated pixels: {total_isolated_pixels}")
    
    # Create summary plots
    create_summary_plots(all_stats)
    
    # Check for potential issues
    print(f"\n=== POTENTIAL ISSUES DETECTED ===")
    
    # Check for noise (many small components)
    noisy_images = [stats for stats in all_stats if stats['small_components'] > 50]
    if noisy_images:
        print(f"⚠️  {len(noisy_images)} images have many small components (potential noise)")
        for stats in noisy_images[:3]:  # Show first 3
            print(f"    {stats['filename']}: {stats['small_components']} small components")
    
    # Check for isolated pixels
    isolated_images = [stats for stats in all_stats if stats['isolated_pixels'] > 10]
    if isolated_images:
        print(f"⚠️  {len(isolated_images)} images have many isolated pixels")
        for stats in isolated_images[:3]:  # Show first 3
            print(f"    {stats['filename']}: {stats['isolated_pixels']} isolated pixels")
    
    # Check for unusual value ranges
    unusual_ranges = [stats for stats in all_stats if stats['original_range'][1] > 255]
    if unusual_ranges:
        print(f"⚠️  {len(unusual_ranges)} images have unusual value ranges")
        for stats in unusual_ranges[:3]:  # Show first 3
            print(f"    {stats['filename']}: range {stats['original_range']}")
    
    # Check for very small or large vessel percentages
    extreme_percentages = [stats for stats in all_stats 
                          if stats['vessel_percentage'] < 1 or stats['vessel_percentage'] > 50]
    if extreme_percentages:
        print(f"⚠️  {len(extreme_percentages)} images have extreme vessel percentages")
        for stats in extreme_percentages[:3]:  # Show first 3
            print(f"    {stats['filename']}: {stats['vessel_percentage']:.2f}% vessels")
    
    print(f"\n✅ Original labels analysis completed!")
    print(f"📊 Summary:")
    print(f"  - Detailed analysis plots saved for first 3 images")
    print(f"  - Summary plots saved")

def create_detailed_analysis(vessel_label, binary_vessels, labeled_vessels, filename, stats):
    """Create detailed analysis for a single image"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Detailed Analysis: {filename}', fontsize=16)
    
    # Original label
    axes[0, 0].imshow(vessel_label, cmap='gray')
    axes[0, 0].set_title(f'Original Label\nRange: {stats["original_range"]}')
    axes[0, 0].axis('off')
    
    # Binary vessels
    axes[0, 1].imshow(binary_vessels, cmap='gray')
    axes[0, 1].set_title(f'Binary Vessels\n{stats["vessel_pixels"]} pixels ({stats["vessel_percentage"]:.1f}%)')
    axes[0, 1].axis('off')
    
    # Connected components
    axes[0, 2].imshow(labeled_vessels, cmap='nipy_spectral')
    axes[0, 2].set_title(f'Connected Components\n{stats["num_components"]} components')
    axes[0, 2].axis('off')
    
    # Histogram of component areas
    component_areas = [prop.area for prop in regionprops(labeled_vessels)]
    if component_areas:
        axes[0, 3].hist(component_areas, bins=50, alpha=0.7, color='blue')
        axes[0, 3].set_title('Component Area Distribution')
        axes[0, 3].set_xlabel('Area (pixels)')
        axes[0, 3].set_ylabel('Frequency')
        axes[0, 3].set_yscale('log')
        axes[0, 3].grid(True, alpha=0.3)
    
    # Zoom in on small regions to check for noise
    # Find a region with potential noise
    small_components = [prop for prop in regionprops(labeled_vessels) if prop.area < 10]
    if small_components:
        # Show a region with small components
        small_prop = small_components[0]
        bbox = small_prop.bbox
        y1, x1, y2, x2 = bbox
        # Expand the region slightly
        y1 = max(0, y1 - 20)
        x1 = max(0, x1 - 20)
        y2 = min(binary_vessels.shape[0], y2 + 20)
        x2 = min(binary_vessels.shape[1], x2 + 20)
        
        axes[1, 0].imshow(binary_vessels[y1:y2, x1:x2], cmap='gray')
        axes[1, 0].set_title('Zoom: Potential Noise Region')
        axes[1, 0].axis('off')
    
    # Show isolated pixels
    isolated_mask = np.zeros_like(binary_vessels)
    for prop in regionprops(labeled_vessels):
        if prop.area == 1:
            isolated_mask[labeled_vessels == prop.label] = 1
    
    axes[1, 1].imshow(isolated_mask, cmap='hot')
    axes[1, 1].set_title(f'Isolated Pixels\n{stats["isolated_pixels"]} pixels')
    axes[1, 1].axis('off')
    
    # Show small components
    small_mask = np.zeros_like(binary_vessels)
    for prop in regionprops(labeled_vessels):
        if 1 < prop.area < 10:
            small_mask[labeled_vessels == prop.label] = 1
    
    axes[1, 2].imshow(small_mask, cmap='hot')
    axes[1, 2].set_title(f'Small Components (2-9 pixels)\n{stats["small_components"]} components')
    axes[1, 2].axis('off')
    
    # Value distribution
    unique_vals, counts = np.unique(vessel_label, return_counts=True)
    axes[1, 3].bar(unique_vals, counts, alpha=0.7, color='green')
    axes[1, 3].set_title('Value Distribution')
    axes[1, 3].set_xlabel('Pixel Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].set_yscale('log')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'predictions/original_label_analysis_{filename.replace(".npy", "")}.png', 
                dpi=300, bbox_inches='tight')

def create_summary_plots(all_stats):
    """Create summary plots across all images"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Original Labels Summary Analysis', fontsize=16)
    
    # Vessel percentage distribution
    vessel_percentages = [stats['vessel_percentage'] for stats in all_stats]
    axes[0, 0].hist(vessel_percentages, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Vessel Percentage Distribution')
    axes[0, 0].set_xlabel('Vessel Percentage (%)')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Number of components distribution
    num_components = [stats['num_components'] for stats in all_stats]
    axes[0, 1].hist(num_components, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Number of Components Distribution')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Small components distribution
    small_components = [stats['small_components'] for stats in all_stats]
    axes[0, 2].hist(small_components, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].set_title('Small Components (<10 pixels) Distribution')
    axes[0, 2].set_xlabel('Number of Small Components')
    axes[0, 2].set_ylabel('Number of Images')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Isolated pixels distribution
    isolated_pixels = [stats['isolated_pixels'] for stats in all_stats]
    axes[1, 0].hist(isolated_pixels, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Isolated Pixels Distribution')
    axes[1, 0].set_xlabel('Number of Isolated Pixels')
    axes[1, 0].set_ylabel('Number of Images')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average component area
    avg_areas = [stats['avg_component_area'] for stats in all_stats]
    axes[1, 1].hist(avg_areas, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Average Component Area Distribution')
    axes[1, 1].set_xlabel('Average Component Area (pixels)')
    axes[1, 1].set_ylabel('Number of Images')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Largest component size
    largest_components = [stats['largest_component'] for stats in all_stats]
    axes[1, 2].hist(largest_components, bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[1, 2].set_title('Largest Component Size Distribution')
    axes[1, 2].set_xlabel('Largest Component Area (pixels)')
    axes[1, 2].set_ylabel('Number of Images')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/original_labels_summary.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    analyze_original_labels() 