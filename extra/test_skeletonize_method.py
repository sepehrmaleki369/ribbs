import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert
from core.general_dataset.io import load_array_from_file
import os

def test_skeletonize_method():
    """Test skeletonize method on our vessel images"""
    
    print("=== TESTING SKELETONIZE METHOD ON VESSEL IMAGES ===")
    
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
    
    # Test different skeletonization approaches
    print("\n=== TESTING DIFFERENT SKELETONIZATION METHODS ===")
    
    # Method 1: Direct skeletonize on binary vessels
    print("Method 1: Direct skeletonize on binary vessels")
    skeleton1 = skeletonize(binary_vessels)
    print(f"Skeleton 1 - True pixels: {np.sum(skeleton1)}")
    
    # Method 2: Invert and skeletonize (like the example)
    print("Method 2: Invert and skeletonize")
    inverted_vessels = invert(binary_vessels)
    skeleton2 = skeletonize(inverted_vessels)
    print(f"Skeleton 2 - True pixels: {np.sum(skeleton2)}")
    
    # Method 3: Our current approach (for comparison)
    print("Method 3: Our current approach")
    from skimage.morphology import skeletonize as skeletonize_current
    skeleton3 = skeletonize_current(binary_vessels)
    print(f"Skeleton 3 - True pixels: {np.sum(skeleton3)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Skeletonization Methods Comparison', fontsize=16)
    
    # Original vessel label
    axes[0, 0].imshow(vessel_label, cmap='gray')
    axes[0, 0].set_title('Original Vessel Label')
    axes[0, 0].axis('off')
    
    # Binary vessels
    axes[0, 1].imshow(binary_vessels, cmap='gray')
    axes[0, 1].set_title('Binary Vessels')
    axes[0, 1].axis('off')
    
    # Inverted vessels
    axes[0, 2].imshow(inverted_vessels, cmap='gray')
    axes[0, 2].set_title('Inverted Vessels')
    axes[0, 2].axis('off')
    
    # Method 1: Direct skeletonize
    axes[1, 0].imshow(skeleton1, cmap='gray')
    axes[1, 0].set_title(f'Method 1: Direct\nPixels: {np.sum(skeleton1)}')
    axes[1, 0].axis('off')
    
    # Method 2: Invert and skeletonize
    axes[1, 1].imshow(skeleton2, cmap='gray')
    axes[1, 1].set_title(f'Method 2: Invert\nPixels: {np.sum(skeleton2)}')
    axes[1, 1].axis('off')
    
    # Method 3: Our current approach
    axes[1, 2].imshow(skeleton3, cmap='gray')
    axes[1, 2].set_title(f'Method 3: Current\nPixels: {np.sum(skeleton3)}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions/skeletonization_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved as predictions/skeletonization_methods_comparison.png")
    
    # Test on multiple images
    print("\n=== TESTING ON MULTIPLE IMAGES ===")
    
    label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    label_files.sort()
    
    results = []
    
    for i, filename in enumerate(label_files[:5]):  # Test first 5 images
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        binary_vessels = (vessel_label > 0).astype(np.uint8)
        
        # Method 1: Direct
        skeleton1 = skeletonize(binary_vessels)
        
        # Method 2: Invert
        inverted_vessels = invert(binary_vessels)
        skeleton2 = skeletonize(inverted_vessels)
        
        # Method 3: Current
        skeleton3 = skeletonize_current(binary_vessels)
        
        results.append({
            'filename': filename,
            'original_pixels': np.sum(binary_vessels),
            'method1_pixels': np.sum(skeleton1),
            'method2_pixels': np.sum(skeleton2),
            'method3_pixels': np.sum(skeleton3)
        })
        
        print(f"Image {i+1}: {filename}")
        print(f"  Original: {np.sum(binary_vessels)} pixels")
        print(f"  Method 1: {np.sum(skeleton1)} pixels")
        print(f"  Method 2: {np.sum(skeleton2)} pixels")
        print(f"  Method 3: {np.sum(skeleton3)} pixels")
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pixel count comparison
    x_pos = np.arange(len(results))
    width = 0.25
    
    method1_counts = [r['method1_pixels'] for r in results]
    method2_counts = [r['method2_pixels'] for r in results]
    method3_counts = [r['method3_pixels'] for r in results]
    
    axes[0].bar(x_pos - width, method1_counts, width, label='Method 1: Direct', alpha=0.7)
    axes[0].bar(x_pos, method2_counts, width, label='Method 2: Invert', alpha=0.7)
    axes[0].bar(x_pos + width, method3_counts, width, label='Method 3: Current', alpha=0.7)
    
    axes[0].set_title('Skeleton Pixel Count Comparison')
    axes[0].set_xlabel('Image')
    axes[0].set_ylabel('Skeleton Pixels')
    axes[0].legend()
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"Img {i+1}" for i in range(len(results))])
    axes[0].grid(True, alpha=0.3)
    
    # Reduction percentage
    reduction1 = [(r['original_pixels'] - r['method1_pixels']) / r['original_pixels'] * 100 for r in results]
    reduction2 = [(r['original_pixels'] - r['method2_pixels']) / r['original_pixels'] * 100 for r in results]
    reduction3 = [(r['original_pixels'] - r['method3_pixels']) / r['original_pixels'] * 100 for r in results]
    
    axes[1].bar(x_pos - width, reduction1, width, label='Method 1: Direct', alpha=0.7)
    axes[1].bar(x_pos, reduction2, width, label='Method 2: Invert', alpha=0.7)
    axes[1].bar(x_pos + width, reduction3, width, label='Method 3: Current', alpha=0.7)
    
    axes[1].set_title('Pixel Reduction Percentage')
    axes[1].set_xlabel('Image')
    axes[1].set_ylabel('Reduction (%)')
    axes[1].legend()
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"Img {i+1}" for i in range(len(results))])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/skeletonization_methods_summary.png', dpi=300, bbox_inches='tight')
    print("Summary saved as predictions/skeletonization_methods_summary.png")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Method 1 (Direct) - Average pixels: {np.mean(method1_counts):.0f}")
    print(f"Method 2 (Invert) - Average pixels: {np.mean(method2_counts):.0f}")
    print(f"Method 3 (Current) - Average pixels: {np.mean(method3_counts):.0f}")
    print(f"Method 1 (Direct) - Average reduction: {np.mean(reduction1):.1f}%")
    print(f"Method 2 (Invert) - Average reduction: {np.mean(reduction2):.1f}%")
    print(f"Method 3 (Current) - Average reduction: {np.mean(reduction3):.1f}%")
    
    print(f"\n✅ Skeletonization testing completed!")
    print(f"📊 Summary:")
    print(f"  - Comparison plot: predictions/skeletonization_methods_comparison.png")
    print(f"  - Summary plot: predictions/skeletonization_methods_summary.png")

if __name__ == "__main__":
    test_skeletonize_method() 