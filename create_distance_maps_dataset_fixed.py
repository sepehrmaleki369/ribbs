import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from core.general_dataset.io import load_array_from_file
import os
from tqdm import tqdm

def create_distance_maps_dataset_fixed():
    """Create distance maps dataset with FIXED label processing - NO PRE/POST PROCESSING"""
    
    print("=== CREATING DISTANCE MAPS DATASET (FIXED VERSION - NO PRE/POST PROCESSING) ===")
    print("FIXED: Properly handling inverted labels (vessels=0, background=16)")
    print("NO PRE/POST PROCESSING: Direct skeletonization for clean results")
    
    # Create output directories
    os.makedirs('drive/training/distance_maps', exist_ok=True)
    
    # Check if test labels exist
    test_labels_dir = 'drive/test/inverted_labels'
    if os.path.exists(test_labels_dir):
        os.makedirs('drive/test/distance_maps', exist_ok=True)
        process_test = True
        print("✅ Test labels found - will process test data too")
    else:
        process_test = False
        print("⚠️  Test labels not found - skipping test data")
    
    # Get training label files
    training_label_files = [f for f in os.listdir('drive/training/inverted_labels') if f.endswith('.npy')]
    training_label_files.sort()
    
    print(f"Found {len(training_label_files)} training label files")
    
    # Process training data
    print("\n=== PROCESSING TRAINING DATA ===")
    training_stats = []
    
    for i, filename in enumerate(tqdm(training_label_files, desc="Processing training labels")):
        label_path = os.path.join('drive/training/inverted_labels', filename)
        vessel_label = load_array_from_file(label_path)
        
        # FIXED: Properly handle inverted labels
        # Vessels are 0, background is 16
        binary_vessels = (vessel_label == 0).astype(np.uint8)  # FIXED: vessels are 0
        
        print(f"\nProcessing {filename}:")
        print(f"  Original range: [{vessel_label.min()}, {vessel_label.max()}]")
        print(f"  Vessel pixels: {np.sum(binary_vessels)}")
        print(f"  Background pixels: {np.sum(binary_vessels == 0)}")
        print(f"  Vessel percentage: {np.sum(binary_vessels) / vessel_label.size * 100:.2f}%")
        
        # NO PREPROCESSING: Direct skeletonization
        skeleton = skeletonize(binary_vessels)
        
        # NO POSTPROCESSING: Use skeleton directly
        final_skeleton = skeleton
        
        # Create distance map
        distance_map = create_distance_map_fixed(final_skeleton)
        
        # Save distance map
        stem = filename.replace('.npy', '')
        distance_filename = f"{stem}_distance_map.npy"
        distance_path = os.path.join('drive/training/distance_maps', distance_filename)
        np.save(distance_path, distance_map)
        
        # Collect statistics
        stats = {
            'filename': filename,
            'original_vessel_pixels': np.sum(binary_vessels),
            'skeleton_pixels': np.sum(final_skeleton),
            'distance_map_range': [distance_map.min(), distance_map.max()],
            'distance_map_mean': distance_map.mean(),
            'distance_map_std': distance_map.std()
        }
        training_stats.append(stats)
        
        print(f"  Final skeleton: {np.sum(final_skeleton)} pixels")
        print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
    
    # Process test data if available
    test_stats = []
    if process_test:
        print("\n=== PROCESSING TEST DATA ===")
        test_label_files = [f for f in os.listdir(test_labels_dir) if f.endswith('.npy')]
        test_label_files.sort()
        
        for i, filename in enumerate(tqdm(test_label_files, desc="Processing test labels")):
            label_path = os.path.join(test_labels_dir, filename)
            vessel_label = load_array_from_file(label_path)
            
            # FIXED: Properly handle inverted labels
            binary_vessels = (vessel_label == 0).astype(np.uint8)  # FIXED: vessels are 0
            
            # NO PREPROCESSING: Direct skeletonization
            skeleton = skeletonize(binary_vessels)
            
            # NO POSTPROCESSING: Use skeleton directly
            final_skeleton = skeleton
            
            # Create distance map
            distance_map = create_distance_map_fixed(final_skeleton)
            
            # Save distance map
            stem = filename.replace('.npy', '')
            distance_filename = f"{stem}_distance_map.npy"
            distance_path = os.path.join('drive/test/distance_maps', distance_filename)
            np.save(distance_path, distance_map)
            
            # Collect statistics
            stats = {
                'filename': filename,
                'original_vessel_pixels': np.sum(binary_vessels),
                'skeleton_pixels': np.sum(final_skeleton),
                'distance_map_range': [distance_map.min(), distance_map.max()],
                'distance_map_mean': distance_map.mean(),
                'distance_map_std': distance_map.std()
            }
            test_stats.append(stats)
    
    # Create visualizations
    create_visualizations_fixed(training_stats, test_stats)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Training data processed: {len(training_stats)} files")
    if test_stats:
        print(f"Test data processed: {len(test_stats)} files")
    
    # Training statistics
    avg_training_skeleton = np.mean([stats['skeleton_pixels'] for stats in training_stats])
    avg_training_distance = np.mean([stats['distance_map_mean'] for stats in training_stats])
    
    print(f"Average training skeleton pixels: {avg_training_skeleton:.0f}")
    print(f"Average training distance map mean: {avg_training_distance:.2f}")
    
    if test_stats:
        avg_test_skeleton = np.mean([stats['skeleton_pixels'] for stats in test_stats])
        avg_test_distance = np.mean([stats['distance_map_mean'] for stats in test_stats])
        print(f"Average test skeleton pixels: {avg_test_skeleton:.0f}")
        print(f"Average test distance map mean: {avg_test_distance:.2f}")
    
    print(f"\n✅ Distance maps dataset created successfully!")
    print(f"📊 Files saved:")
    print(f"  - Training: drive/training/distance_maps/")
    if test_stats:
        print(f"  - Test: drive/test/distance_maps/")

def create_distance_map_fixed(skeleton):
    """Create distance map from skeleton with FIXED approach"""
    
    # Distance from skeleton pixels (not background)
    # Skeleton pixels should have distance 0, everything else shows distance to skeleton
    background_mask = skeleton == 0
    distance_map = distance_transform_edt(background_mask)
    
    return distance_map

def create_visualizations_fixed(training_stats, test_stats):
    """Create visualizations for the FIXED approach"""
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distance Maps Dataset Creation (FIXED Version - NO PRE/POST PROCESSING)', fontsize=16)
    
    # Training statistics
    training_skeleton_pixels = [stats['skeleton_pixels'] for stats in training_stats]
    training_distance_means = [stats['distance_map_mean'] for stats in training_stats]
    training_distance_ranges = [stats['distance_map_range'][1] for stats in training_stats]
    
    # Plot 1: Skeleton pixel counts
    axes[0, 0].hist(training_skeleton_pixels, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Training Skeleton Pixel Counts')
    axes[0, 0].set_xlabel('Skeleton Pixels')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distance map means
    axes[0, 1].hist(training_distance_means, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Training Distance Map Means')
    axes[0, 1].set_xlabel('Mean Distance')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distance map ranges
    axes[0, 2].hist(training_distance_ranges, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].set_title('Training Distance Map Ranges')
    axes[0, 2].set_xlabel('Max Distance')
    axes[0, 2].set_ylabel('Number of Images')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Test statistics (if available)
    if test_stats:
        test_skeleton_pixels = [stats['skeleton_pixels'] for stats in test_stats]
        test_distance_means = [stats['distance_map_mean'] for stats in test_stats]
        test_distance_ranges = [stats['distance_map_range'][1] for stats in test_stats]
        
        # Plot 4: Test skeleton pixel counts
        axes[1, 0].hist(test_skeleton_pixels, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Test Skeleton Pixel Counts')
        axes[1, 0].set_xlabel('Skeleton Pixels')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Test distance map means
        axes[1, 1].hist(test_distance_means, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Test Distance Map Means')
        axes[1, 1].set_xlabel('Mean Distance')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Test distance map ranges
        axes[1, 2].hist(test_distance_ranges, bins=20, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 2].set_title('Test Distance Map Ranges')
        axes[1, 2].set_xlabel('Max Distance')
        axes[1, 2].set_ylabel('Number of Images')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # If no test data, show skeleton vs vessel ratio
        training_vessel_pixels = [stats['original_vessel_pixels'] for stats in training_stats]
        
        x_pos = np.arange(len(training_stats))
        
        # Skeleton vs vessel ratio
        skeleton_ratios = [skeleton / vessel for skeleton, vessel in zip(training_skeleton_pixels, training_vessel_pixels)]
        axes[1, 0].bar(x_pos, skeleton_ratios, alpha=0.7, color='cyan')
        axes[1, 0].set_title('Training: Skeleton/Vessel Ratio')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Skeleton pixel counts
        axes[1, 1].bar(x_pos, training_skeleton_pixels, alpha=0.7, color='magenta')
        axes[1, 1].set_title('Training: Skeleton Pixel Counts')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Skeleton Pixels')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Distance map means
        axes[1, 2].bar(x_pos, training_distance_means, alpha=0.7, color='yellow')
        axes[1, 2].set_title('Training: Distance Map Means')
        axes[1, 2].set_xlabel('Image Index')
        axes[1, 2].set_ylabel('Mean Distance')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/distance_maps_creation_fixed.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as predictions/distance_maps_creation_fixed.png")

if __name__ == "__main__":
    create_distance_maps_dataset_fixed() 