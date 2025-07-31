import os
import shutil

def rename_distance_maps():
    """Rename distance maps to match image naming pattern"""
    
    print("=== RENAMING DISTANCE MAPS ===")
    
    # Get distance map files
    distance_map_files = [f for f in os.listdir('drive/training/distance_maps') if f.endswith('.npy')]
    distance_map_files.sort()
    
    print(f"Found {len(distance_map_files)} distance map files")
    
    # Rename each file
    for filename in distance_map_files:
        # Extract the number (e.g., "21" from "21_manual1_distance_map.npy")
        number = filename.split('_')[0]
        
        # Create new filename to match image pattern
        new_filename = f"{number}_training_distance_map.npy"
        
        # Full paths
        old_path = os.path.join('drive/training/distance_maps', filename)
        new_path = os.path.join('drive/training/distance_maps', new_filename)
        
        # Rename
        shutil.move(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
    
    print(f"\n✅ Renamed {len(distance_map_files)} distance map files")
    
    # Verify the new names
    print("\n=== VERIFICATION ===")
    new_files = [f for f in os.listdir('drive/training/distance_maps') if f.endswith('.npy')]
    new_files.sort()
    
    print("New distance map files:")
    for filename in new_files:
        print(f"  {filename}")

if __name__ == "__main__":
    rename_distance_maps() 