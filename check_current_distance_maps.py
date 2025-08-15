import numpy as np
import os

def check_current_distance_maps():
    """Check the current range of distance maps"""
    
    print("=== CHECKING CURRENT DISTANCE MAP RANGES ===")
    
    distance_maps_dir = 'drive/training/distance_maps'
    if not os.path.exists(distance_maps_dir):
        print(f"❌ Directory not found: {distance_maps_dir}")
        return
    
    files = [f for f in os.listdir(distance_maps_dir) if f.endswith('.npy')]
    if not files:
        print("❌ No distance map files found")
        return
    
    print(f"Found {len(files)} distance map files")
    print("\nChecking first 5 files:")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for i, filename in enumerate(files[:5]):
        filepath = os.path.join(distance_maps_dir, filename)
        distance_map = np.load(filepath)
        
        min_val = distance_map.min()
        max_val = distance_map.max()
        
        global_min = min(global_min, min_val)
        global_max = max(global_max, max_val)
        
        print(f"{filename}: range [{min_val:.1f}, {max_val:.1f}]")
    
    print(f"\n=== SUMMARY ===")
    print(f"Global range: [{global_min:.1f}, {global_max:.1f}]")
    
    if global_max <= 15.0:
        print("✅ Distance maps are CLIPPED to 0-15 range")
    else:
        print("❌ Distance maps are in ORIGINAL range (not clipped)")
    
    # Check if any values are above 15
    above_15_count = 0
    for filename in files[:5]:
        filepath = os.path.join(distance_maps_dir, filename)
        distance_map = np.load(filepath)
        if distance_map.max() > 15.0:
            above_15_count += 1
    
    if above_15_count > 0:
        print(f"⚠️  {above_15_count} files have values above 15.0")
    else:
        print("✅ All checked files are within 0-15 range")

if __name__ == "__main__":
    check_current_distance_maps() 