# Create Discrete Distance Maps
# Generate discrete distance maps (0, 1, 2, 3, 4, 5...) without clipping

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from core.general_dataset.io import load_array_from_file
import os
from tqdm import tqdm

def create_discrete_distance_maps():
    """Create discrete distance maps dataset - NO CLIPPING, NO LIMITS"""
    
    print("=== CREATING DISCRETE DISTANCE MAPS DATASET ===")
    print("DISCRETE: Integer values (0, 1, 2, 3, 4, 5...)")
    print("NO CLIPPING: All distance values preserved")
    print("NO LIMITS: Maximum distance can be any value")
    
    # Create output directories
    os.makedirs('drive/training/discrete_distance_maps', exist_ok=True)
    
    # Check if test labels exist
    test_labels_dir = 'drive/test/inverted_labels'
    if os.path.exists(test_labels_dir):
        os.makedirs('drive/test/discrete_distance_maps', exist_ok=True)
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
        
        # Handle inverted labels
        binary_vessels = (vessel_label == 0).astype(np.uint8)  # vessels are 0
        
        print(f"\nProcessing {filename}:")
        print(f"  Original range: [{vessel_label.min()}, {vessel_label.max()}]")
        print(f"  Vessel pixels: {np.sum(binary_vessels)}")
        print(f"  Background pixels: {np.sum(binary_vessels == 0)}")
        print(f"  Vessel percentage: {np.sum(binary_vessels) / vessel_label.size * 100:.2f}%")
        
        # Skeletonize
        skeleton = skeletonize(binary_vessels)
        
        # Create discrete distance map
        distance_map = create_discrete_distance_map(skeleton)
        
        # Save discrete distance map
        stem = filename.replace('.npy', '')
        distance_filename = f"{stem}_discrete_distance_map.npy"
        distance_path = os.path.join('drive/training/discrete_distance_maps', distance_filename)
        np.save(distance_path, distance_map)
        
        # Collect statistics
        unique_values = np.unique(distance_map)
        stats = {
            'filename': filename,
            'original_vessel_pixels': np.sum(binary_vessels),
            'skeleton_pixels': np.sum(skeleton),
            'discrete_range': [distance_map.min(), distance_map.max()],
            'unique_values': unique_values,
            'max_distance': distance_map.max(),
            'distance_map_mean': distance_map.mean(),
            'distance_map_std': distance_map.std()
        }
        training_stats.append(stats)
        
        print(f"  Skeleton: {np.sum(skeleton)} pixels")
        print(f"  Discrete range: [{distance_map.min()}, {distance_map.max()}]")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Values: {unique_values}")
    
    # Process test data if available
    test_stats = []
    if process_test:
        print("\n=== PROCESSING TEST DATA ===")
        test_label_files = [f for f in os.listdir(test_labels_dir) if f.endswith('.npy')]
        test_label_files.sort()
        
        for i, filename in enumerate(tqdm(test_label_files, desc="Processing test labels")):
            label_path = os.path.join(test_labels_dir, filename)
            vessel_label = load_array_from_file(label_path)
            
            # Handle inverted labels
            binary_vessels = (vessel_label == 0).astype(np.uint8)  # vessels are 0
            
            # Skeletonize
            skeleton = skeletonize(binary_vessels)
            
            # Create discrete distance map
            distance_map = create_discrete_distance_map(skeleton)
            
            # Save discrete distance map
            stem = filename.replace('.npy', '')
            distance_filename = f"{stem}_discrete_distance_map.npy"
            distance_path = os.path.join('drive/test/discrete_distance_maps', distance_filename)
            np.save(distance_path, distance_map)
            
            # Collect statistics
            unique_values = np.unique(distance_map)
            stats = {
                'filename': filename,
                'original_vessel_pixels': np.sum(binary_vessels),
                'skeleton_pixels': np.sum(skeleton),
                'discrete_range': [distance_map.min(), distance_map.max()],
                'unique_values': unique_values,
                'max_distance': distance_map.max(),
                'distance_map_mean': distance_map.mean(),
                'distance_map_std': distance_map.std()
            }
            test_stats.append(stats)
    
    # Create visualizations
    create_discrete_visualizations(training_stats, test_stats)
    
    print(f"\n✅ DISCRETE DISTANCE MAPS CREATED SUCCESSFULLY!")
    print(f"Training maps saved to: drive/training/discrete_distance_maps/")
    if process_test:
        print(f"Test maps saved to: drive/test/discrete_distance_maps/")

def create_discrete_distance_map(skeleton):
    """Create discrete distance map from skeleton - NO CLIPPING"""
    
    # Create distance transform
    background_mask = skeleton == 0
    distance_map = distance_transform_edt(background_mask)
    
    # Convert to discrete integers - NO CLIPPING
    discrete_map = distance_map.astype(np.int32)
    
    return discrete_map

def create_discrete_visualizations(training_stats, test_stats):
    """Create visualizations for discrete distance maps"""
    
    # Create summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Discrete Distance Maps Dataset Creation (NO CLIPPING)', fontsize=16)
    
    # Training statistics
    training_skeleton_pixels = [stats['skeleton_pixels'] for stats in training_stats]
    training_max_distances = [stats['max_distance'] for stats in training_stats]
    training_unique_counts = [len(stats['unique_values']) for stats in training_stats]
    
    # Plot 1: Skeleton pixel counts
    axes[0, 0].hist(training_skeleton_pixels, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Training Skeleton Pixel Counts')
    axes[0, 0].set_xlabel('Skeleton Pixels')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Maximum distances (NO CLIPPING)
    axes[0, 1].hist(training_max_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Training Maximum Distances (NO CLIPPING)')
    axes[0, 1].set_xlabel('Maximum Distance')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Unique value counts
    axes[0, 2].hist(training_unique_counts, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 2].set_title('Training Unique Distance Values')
    axes[0, 2].set_xlabel('Number of Unique Values')
    axes[0, 2].set_ylabel('Number of Images')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Test statistics (if available)
    if test_stats:
        test_skeleton_pixels = [stats['skeleton_pixels'] for stats in test_stats]
        test_max_distances = [stats['max_distance'] for stats in test_stats]
        test_unique_counts = [len(stats['unique_values']) for stats in test_stats]
        
        # Plot 4: Test skeleton pixel counts
        axes[1, 0].hist(test_skeleton_pixels, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Test Skeleton Pixel Counts')
        axes[1, 0].set_xlabel('Skeleton Pixels')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Test maximum distances
        axes[1, 1].hist(test_max_distances, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Test Maximum Distances (NO CLIPPING)')
        axes[1, 1].set_xlabel('Maximum Distance')
        axes[1, 1].set_ylabel('Number of Images')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Test unique value counts
        axes[1, 2].hist(test_unique_counts, bins=20, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 2].set_title('Test Unique Distance Values')
        axes[1, 2].set_xlabel('Number of Unique Values')
        axes[1, 2].set_ylabel('Number of Images')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # If no test data, show detailed training stats
        training_vessel_pixels = [stats['original_vessel_pixels'] for stats in training_stats]
        
        x_pos = np.arange(len(training_stats))
        
        # Skeleton vs vessel ratio
        skeleton_ratios = [skeleton / vessel for skeleton, vessel in zip(training_skeleton_pixels, training_vessel_pixels)]
        axes[1, 0].bar(x_pos, skeleton_ratios, alpha=0.7, color='cyan')
        axes[1, 0].set_title('Training: Skeleton/Vessel Ratio')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Maximum distances
        axes[1, 1].bar(x_pos, training_max_distances, alpha=0.7, color='magenta')
        axes[1, 1].set_title('Training: Maximum Distances (NO CLIPPING)')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Max Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Unique value counts
        axes[1, 2].bar(x_pos, training_unique_counts, alpha=0.7, color='yellow')
        axes[1, 2].set_title('Training: Unique Distance Values')
        axes[1, 2].set_xlabel('Image Index')
        axes[1, 2].set_ylabel('Unique Values')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions/discrete_distance_maps_creation.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as predictions/discrete_distance_maps_creation.png")

if __name__ == "__main__":
    create_discrete_distance_maps() 