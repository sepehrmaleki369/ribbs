import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from core.general_dataset.io import load_array_from_file
import os

def test_zhang_vs_lee():
    """Compare Zhang's method vs Lee's method for skeletonization"""
    
    print("=== COMPARING ZHANG'S METHOD VS LEE'S METHOD ===")
    
    # Load a sample vessel label
    label_path = 'drive/training/inverted_labels/21_manual1.npy'
    if os.path.exists(label_path):
        vessel_label = load_array_from_file(label_path)
        print(f"✅ Loaded vessel label from {label_path}")
        print(f"Label shape: {vessel_label.shape}")
        print(f"Label range: [{vessel_label.min()}, {vessel_label.max()}]")
    else:
        print(f"❌ Label not found at {label_path}")
        return
    
    # Convert to binary (vessels = 1, background = 0)
    binary_vessels = (vessel_label > 0).astype(np.uint8)
    print(f"Binary vessels - True pixels: {np.sum(binary_vessels)}")
    
    # Test both methods
    print("\n=== TESTING BOTH METHODS ===")
    
    # Zhang's method (default)
    print("Zhang's method (default)")
    skeleton_zhang = skeletonize(binary_vessels, method='zhang')
    print(f"Zhang skeleton - True pixels: {np.sum(skeleton_zhang)}")
    
    # Lee's method
    print("Lee's method")
    skeleton_lee = skeletonize(binary_vessels, method='lee')
    print(f"Lee skeleton - True pixels: {np.sum(skeleton_lee)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Zhang vs Lee Skeletonization Methods', fontsize=16)
    
    # Original vessel label
    axes[0, 0].imshow(vessel_label, cmap='gray')
    axes[0, 0].set_title('Original Vessel Label')
    axes[0, 0].axis('off')
    
    # Binary vessels
    axes[0, 1].imshow(binary_vessels, cmap='gray')
    axes[0, 1].set_title('Binary Vessels')
    axes[0, 1].axis('off')
    
    # Difference between methods
    diff = skeleton_zhang.astype(int) - skeleton_lee.astype(int)
    axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 2].set_title('Difference (Zhang - Lee)\nRed=Zhang only, Blue=Lee only')
    axes[0, 2].axis('off')
    
    # Zhang's skeleton
    axes[1, 0].imshow(skeleton_zhang, cmap='gray')
    axes[1, 0].set_title(f'Zhang Method\nPixels: {np.sum(skeleton_zhang)}')
    axes[1, 0].axis('off')
    
    # Lee's skeleton
    axes[1, 1].imshow(skeleton_lee, cmap='gray')
    axes[1, 1].set_title(f'Lee Method\nPixels: {np.sum(skeleton_lee)}')
    axes[1, 1].axis('off')
    
    # Overlay comparison
    overlay = np.zeros((*skeleton_zhang.shape, 3))
    overlay[skeleton_zhang] = [1, 0, 0]  # Red for Zhang
    overlay[skeleton_lee] = [0, 0, 1]    # Blue for Lee
    overlay[np.logical_and(skeleton_zhang, skeleton_lee)] = [0.5, 0, 0.5]  # Purple for both
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay Comparison\nRed=Zhang, Blue=Lee, Purple=Both')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions/zhang_vs_lee_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved as predictions/zhang_vs_lee_comparison.png")
    
    # Test on multiple images
    print("\n=== TESTING ON MULTIPLE IMAGES ===")
    
    label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    label_files.sort()
    
    results = []
    
    for i, filename in enumerate(label_files[:5]):  # Test first 5 images
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        binary_vessels = (vessel_label > 0).astype(np.uint8)
        
        # Zhang's method
        skeleton_zhang = skeletonize(binary_vessels, method='zhang')
        
        # Lee's method
        skeleton_lee = skeletonize(binary_vessels, method='lee')
        
        results.append({
            'filename': filename,
            'original_pixels': np.sum(binary_vessels),
            'zhang_pixels': np.sum(skeleton_zhang),
            'lee_pixels': np.sum(skeleton_lee),
            'zhang_reduction': (np.sum(binary_vessels) - np.sum(skeleton_zhang)) / np.sum(binary_vessels) * 100,
            'lee_reduction': (np.sum(binary_vessels) - np.sum(skeleton_lee)) / np.sum(binary_vessels) * 100
        })
        
        print(f"Image {i+1}: {filename}")
        print(f"  Original: {np.sum(binary_vessels)} pixels")
        print(f"  Zhang: {np.sum(skeleton_zhang)} pixels ({results[-1]['zhang_reduction']:.1f}% reduction)")
        print(f"  Lee: {np.sum(skeleton_lee)} pixels ({results[-1]['lee_reduction']:.1f}% reduction)")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pixel count comparison
    x_pos = np.arange(len(results))
    width = 0.35
    
    zhang_counts = [r['zhang_pixels'] for r in results]
    lee_counts = [r['lee_pixels'] for r in results]
    
    axes[0].bar(x_pos - width/2, zhang_counts, width, label='Zhang Method', alpha=0.7, color='red')
    axes[0].bar(x_pos + width/2, lee_counts, width, label='Lee Method', alpha=0.7, color='blue')
    
    axes[0].set_title('Skeleton Pixel Count Comparison')
    axes[0].set_xlabel('Image')
    axes[0].set_ylabel('Skeleton Pixels')
    axes[0].legend()
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"Img {i+1}" for i in range(len(results))])
    axes[0].grid(True, alpha=0.3)
    
    # Reduction percentage
    zhang_reductions = [r['zhang_reduction'] for r in results]
    lee_reductions = [r['lee_reduction'] for r in results]
    
    axes[1].bar(x_pos - width/2, zhang_reductions, width, label='Zhang Method', alpha=0.7, color='red')
    axes[1].bar(x_pos + width/2, lee_reductions, width, label='Lee Method', alpha=0.7, color='blue')
    
    axes[1].set_title('Pixel Reduction Percentage')
    axes[1].set_xlabel('Image')
    axes[1].set_ylabel('Reduction (%)')
    axes[1].legend()
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Img {i+1}" for i in range(len(results))])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/zhang_vs_lee_summary.png', dpi=300, bbox_inches='tight')
    print("Summary saved as predictions/zhang_vs_lee_summary.png")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Zhang Method - Average pixels: {np.mean(zhang_counts):.0f}")
    print(f"Lee Method - Average pixels: {np.mean(lee_counts):.0f}")
    print(f"Zhang Method - Average reduction: {np.mean(zhang_reductions):.1f}%")
    print(f"Lee Method - Average reduction: {np.mean(lee_reductions):.1f}%")
    
    # Check which method we've been using
    print(f"\n=== CURRENT METHOD CHECK ===")
    print("In our create_distance_maps_dataset.py, we use:")
    print("skeleton = skeletonize(binary_vessels)")
    print("This defaults to Zhang's method!")
    print("So we ARE using Zhang's method in our current approach.")
    
    print(f"\n✅ Zhang vs Lee comparison completed!")
    print(f"📊 Summary:")
    print(f"  - Comparison plot: predictions/zhang_vs_lee_comparison.png")
    print(f"  - Summary plot: predictions/zhang_vs_lee_summary.png")

if __name__ == "__main__":
    test_zhang_vs_lee() 