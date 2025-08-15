import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def clip_distance_maps_to_15():
    """Clip distance maps to 0-15 range and replace original files"""
    
    print("=== CLIPPING DISTANCE MAPS TO 0-15 RANGE ===")
    
    distance_maps_dir = 'drive/training/distance_maps'
    if not os.path.exists(distance_maps_dir):
        print(f"❌ Directory not found: {distance_maps_dir}")
        return
    
    files = [f for f in os.listdir(distance_maps_dir) if f.endswith('.npy')]
    if not files:
        print("❌ No distance map files found")
        return
    
    print(f"Found {len(files)} distance map files")
    print("Clipping all distance maps to range [0, 15]...")
    
    # Statistics collection
    all_stats = []
    total_pixels_clipped = 0
    total_pixels = 0
    
    for filename in tqdm(files, desc="Clipping distance maps"):
        filepath = os.path.join(distance_maps_dir, filename)
        distance_map = np.load(filepath)
        
        # Store original stats
        original_min = distance_map.min()
        original_max = distance_map.max()
        original_mean = distance_map.mean()
        
        # Clip to 0-15 range
        clipped_map = np.clip(distance_map, 0, 15)
        
        # Calculate clipping statistics
        pixels_clipped = np.sum(distance_map > 15)
        total_pixels_clipped += pixels_clipped
        total_pixels += distance_map.size
        
        # Save clipped map (replace original)
        np.save(filepath, clipped_map)
        
        # Collect stats
        stats = {
            'filename': filename,
            'original_range': [original_min, original_max],
            'clipped_range': [clipped_map.min(), clipped_map.max()],
            'original_mean': original_mean,
            'clipped_mean': clipped_map.mean(),
            'pixels_clipped': pixels_clipped,
            'clipping_percentage': (pixels_clipped / distance_map.size) * 100
        }
        all_stats.append(stats)
        
        print(f"  {filename}: {original_min:.1f}-{original_max:.1f} → {clipped_map.min():.1f}-{clipped_map.max():.1f} (clipped {pixels_clipped} pixels)")
    
    # Create visualization
    create_clipping_visualization(all_stats)
    
    # Print summary
    print(f"\n=== CLIPPING SUMMARY ===")
    print(f"Total files processed: {len(files)}")
    print(f"Total pixels clipped: {total_pixels_clipped:,}")
    print(f"Overall clipping percentage: {(total_pixels_clipped / total_pixels) * 100:.2f}%")
    
    # Show top 5 files with most clipping
    sorted_stats = sorted(all_stats, key=lambda x: x['pixels_clipped'], reverse=True)
    print(f"\nTop 5 files with most clipping:")
    for i, stats in enumerate(sorted_stats[:5]):
        print(f"  {stats['filename']}: {stats['clipping_percentage']:.2f}% clipped")
    
    print(f"\n✅ CLIPPING COMPLETED!")
    print(f"All distance maps now in range [0, 15]")
    print(f"Original files replaced with clipped versions")

def create_clipping_visualization(all_stats):
    """Create visualization of clipping results"""
    
    print("\n=== CREATING VISUALIZATION ===")
    
    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Extract data for plotting
    filenames = [stats['filename'] for stats in all_stats]
    original_maxes = [stats['original_range'][1] for stats in all_stats]
    clipping_percentages = [stats['clipping_percentage'] for stats in all_stats]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distance Map Clipping Analysis', fontsize=16)
    
    # Plot 1: Original max values
    axes[0, 0].bar(range(len(filenames)), original_maxes, alpha=0.7, color='red')
    axes[0, 0].axhline(y=15, color='green', linestyle='--', label='Clipping threshold')
    axes[0, 0].set_title('Original Maximum Values')
    axes[0, 0].set_ylabel('Maximum Distance Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Clipping percentages
    axes[0, 1].bar(range(len(filenames)), clipping_percentages, alpha=0.7, color='orange')
    axes[0, 1].set_title('Percentage of Pixels Clipped')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Before vs After comparison
    before_means = [stats['original_mean'] for stats in all_stats]
    after_means = [stats['clipped_mean'] for stats in all_stats]
    
    x = np.arange(len(filenames))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, before_means, width, label='Before Clipping', alpha=0.7, color='blue')
    axes[1, 0].bar(x + width/2, after_means, width, label='After Clipping', alpha=0.7, color='green')
    axes[1, 0].set_title('Mean Values: Before vs After')
    axes[1, 0].set_ylabel('Mean Distance Value')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Histogram of clipping percentages
    axes[1, 1].hist(clipping_percentages, bins=10, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Distribution of Clipping Percentages')
    axes[1, 1].set_xlabel('Clipping Percentage (%)')
    axes[1, 1].set_ylabel('Number of Files')
    
    plt.tight_layout()
    plt.savefig('predictions/distance_maps_clipping_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved as: predictions/distance_maps_clipping_analysis.png")

if __name__ == "__main__":
    clip_distance_maps_to_15() 