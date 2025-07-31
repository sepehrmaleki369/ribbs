import os

def debug_stems():
    """Debug stem extraction from filenames"""
    
    print("=== DEBUGGING STEM EXTRACTION ===")
    
    # Check image files
    image_dir = "drive/training/images_npy"
    print(f"\nImage files in {image_dir}:")
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.npy'):
            stem = os.path.splitext(fname)[0]
            print(f"  {fname} -> stem: {stem}")
    
    # Check distance map files
    distance_dir = "drive/training/distance_maps"
    print(f"\nDistance map files in {distance_dir}:")
    for fname in sorted(os.listdir(distance_dir)):
        if fname.endswith('.npy'):
            stem = os.path.splitext(fname)[0]
            print(f"  {fname} -> stem: {stem}")
    
    # Check if stems match
    image_stems = set()
    for fname in os.listdir(image_dir):
        if fname.endswith('.npy'):
            stem = os.path.splitext(fname)[0]
            image_stems.add(stem)
    
    distance_stems = set()
    for fname in os.listdir(distance_dir):
        if fname.endswith('.npy'):
            stem = os.path.splitext(fname)[0]
            distance_stems.add(stem)
    
    print(f"\nStem comparison:")
    print(f"  Image stems: {len(image_stems)}")
    print(f"  Distance stems: {len(distance_stems)}")
    print(f"  Common stems: {len(image_stems & distance_stems)}")
    print(f"  Image only: {len(image_stems - distance_stems)}")
    print(f"  Distance only: {len(distance_stems - image_stems)}")
    
    if image_stems != distance_stems:
        print(f"\nMismatched stems:")
        print(f"  In image but not distance: {sorted(image_stems - distance_stems)}")
        print(f"  In distance but not image: {sorted(distance_stems - image_stems)}")

if __name__ == "__main__":
    debug_stems() 