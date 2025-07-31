import numpy as np
import os
from core.general_dataset.io import load_array_from_file

def verify_new_distance_maps():
    """Verify the new distance maps have correct properties"""
    
    print("=== VERIFYING NEW DISTANCE MAPS ===")
    
    # Get distance map files
    distance_map_files = [f for f in os.listdir('drive/training/distance_maps') if f.endswith('.npy')]
    distance_map_files.sort()
    
    print(f"Found {len(distance_map_files)} distance map files")
    
    # Analyze first few files
    print("\n=== ANALYSIS OF NEW DISTANCE MAPS ===")
    
    total_skeleton_pixels = 0
    total_distance_mean = 0
    distance_ranges = []
    
    for i, filename in enumerate(distance_map_files[:5]):  # Check first 5 files
        distance_path = os.path.join('drive/training/distance_maps', filename)
        distance_map = np.load(distance_path)
        
        # Count skeleton pixels (distance = 0)
        skeleton_pixels = np.sum(distance_map == 0)
        distance_mean = distance_map.mean()
        distance_range = [distance_map.min(), distance_map.max()]
        
        print(f"\n{filename}:")
        print(f"  Skeleton pixels: {skeleton_pixels}")
        print(f"  Distance mean: {distance_mean:.2f}")
        print(f"  Distance range: [{distance_range[0]:.2f}, {distance_range[1]:.2f}]")
        print(f"  Shape: {distance_map.shape}")
        
        total_skeleton_pixels += skeleton_pixels
        total_distance_mean += distance_mean
        distance_ranges.append(distance_range)
    
    # Summary statistics
    avg_skeleton_pixels = total_skeleton_pixels / 5
    avg_distance_mean = total_distance_mean / 5
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Average skeleton pixels (first 5 files): {avg_skeleton_pixels:.0f}")
    print(f"Average distance mean (first 5 files): {avg_distance_mean:.2f}")
    print(f"Distance ranges: {distance_ranges}")
    
    # Compare with expected values from clean skeletonization
    print(f"\n=== COMPARISON WITH EXPECTED VALUES ===")
    print(f"Expected average skeleton pixels: ~8,537 (from clean skeletonization)")
    print(f"Expected average distance mean: ~21.47 (from clean skeletonization)")
    print(f"Actual average skeleton pixels: {avg_skeleton_pixels:.0f}")
    print(f"Actual average distance mean: {avg_distance_mean:.2f}")
    
    # Check if values are reasonable
    if avg_skeleton_pixels > 5000:  # Should be much higher than old processed version
        print("✅ Skeleton pixel count looks good (much higher than old processed version)")
    else:
        print("⚠️  Skeleton pixel count seems low")
    
    if avg_distance_mean < 30:  # Should be lower than old processed version
        print("✅ Distance mean looks good (lower than old processed version)")
    else:
        print("⚠️  Distance mean seems high")
    
    print(f"\n✅ New distance maps verification completed!")
    print(f"📊 Files in drive/training/distance_maps/: {len(distance_map_files)}")

if __name__ == "__main__":
    verify_new_distance_maps() 