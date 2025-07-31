import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as skimage_label, regionprops
import seaborn as sns

def remove_thin_branches_from_labels(binary_vessels, min_thickness=5):
    """Remove very thin branches from labels before skeletonization"""
    
    # Create distance transform to identify thin areas
    dist_transform = distance_transform_edt(binary_vessels)
    
    # Remove areas that are too thin
    cleaned = binary_vessels.copy()
    cleaned[dist_transform < min_thickness] = 0
    
    # Remove small disconnected components
    cleaned = remove_small_objects(cleaned.astype(bool), min_size=100)
    
    return cleaned.astype(np.uint8)

def clean_skeleton_comprehensive(skeleton, binary_vessels):
    """Comprehensive skeleton cleaning"""
    
    # Step 1: Remove very small components
    labeled = skimage_label(skeleton)
    props = regionprops(labeled)
    
    cleaned = np.zeros_like(skeleton)
    for prop in props:
        if prop.area >= 15:  # Minimum branch length
            cleaned[labeled == prop.label] = 1
    
    # Step 2: Remove thin connections
    dist_transform = distance_transform_edt(binary_vessels)
    skeleton_coords = np.where(cleaned > 0)
    
    for i, j in zip(skeleton_coords[0], skeleton_coords[1]):
        if dist_transform[i, j] < 1.5:  # Remove if original vessel was too thin
            cleaned[i, j] = 0
    
    # Step 3: Remove short branches
    cleaned = remove_small_objects(cleaned, min_size=10)
    
    return cleaned.astype(np.uint8)

def create_distance_map(skeleton):
    """Create distance map from skeleton"""
    # Distance from skeleton pixels (not background)
    background_mask = skeleton == 0
    distance_map = distance_transform_edt(background_mask)
    return distance_map

def preprocess_vessels(binary_vessels):
    """Remove thin branches from labels before skeletonization"""
    # Remove very thin branches (thickness < 5 pixels)
    cleaned = remove_thin_branches_from_labels(binary_vessels, min_thickness=5)
    return cleaned

def create_best_skeleton(binary_vessels):
    """Create the best skeleton using our optimized method"""
    # Preprocess vessels
    preprocessed = preprocess_vessels(binary_vessels)
    
    # Create skeleton
    skeleton = skeletonize(preprocessed > 0).astype(np.uint8)
    
    # Apply comprehensive cleaning
    skeleton = clean_skeleton_comprehensive(skeleton, preprocessed)
    
    return skeleton

def create_distance_maps_dataset():
    """Create distance maps dataset for both training and test data"""
    
    print("=== CREATING DISTANCE MAPS DATASET ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Define paths
    train_images_dir = "drive/training/images_npy"
    train_labels_dir = "drive/training/inverted_labels"
    test_images_dir = "drive/test/images_npy"
    test_labels_dir = "drive/test/inverted_labels"
    
    # Create output directories
    train_distance_dir = "drive/training/distance_maps"
    test_distance_dir = "drive/test/distance_maps"
    os.makedirs(train_distance_dir, exist_ok=True)
    os.makedirs(test_distance_dir, exist_ok=True)
    
    # Process training data
    print("\n=== PROCESSING TRAINING DATA ===")
    train_image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith('.npy')])
    train_label_files = sorted([f for f in os.listdir(train_labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(train_image_files)} training images and {len(train_label_files)} training labels")
    
    train_stats = []
    
    for i, (img_file, label_file) in enumerate(zip(train_image_files, train_label_files)):
        print(f"\nProcessing training sample {i+1}: {img_file}")
        
        # Load data
        img_path = os.path.join(train_images_dir, img_file)
        label_path = os.path.join(train_labels_dir, label_file)
        
        image = load_array_from_file(img_path)
        label = load_array_from_file(label_path)
        
        # Convert to binary (0 for background, 1 for vessels)
        binary_vessels = (label > 0).astype(np.uint8)
        
        # Create best skeleton
        skeleton = create_best_skeleton(binary_vessels)
        
        # Create distance map
        distance_map = create_distance_map(skeleton)
        
        # Save distance map
        distance_filename = img_file.replace('.npy', '_distance_map.npy')  # Changed from _distance to _distance_map
        distance_path = os.path.join(train_distance_dir, distance_filename)
        np.save(distance_path, distance_map)
        
        # Store statistics
        train_stats.append({
            'image_file': img_file,
            'label_file': label_file,
            'distance_file': distance_filename,
            'vessel_pixels': binary_vessels.sum(),
            'skeleton_pixels': skeleton.sum(),
            'distance_range': [distance_map.min(), distance_map.max()],
            'distance_mean': distance_map.mean(),
            'distance_std': distance_map.std()
        })
        
        print(f"  Original vessels: {binary_vessels.sum()} pixels")
        print(f"  Skeleton: {skeleton.sum()} pixels")
        print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
        print(f"  Saved: {distance_path}")
    
    # Process test data
    print("\n=== PROCESSING TEST DATA ===")
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith('.npy')])
    
    # Check if test labels exist
    if os.path.exists(test_labels_dir):
        test_label_files = sorted([f for f in os.listdir(test_labels_dir) if f.endswith('.npy')])
        print(f"Found {len(test_image_files)} test images and {len(test_label_files)} test labels")
        
        test_stats = []
        
        for i, (img_file, label_file) in enumerate(zip(test_image_files, test_label_files)):
            print(f"\nProcessing test sample {i+1}: {img_file}")
            
            # Load data
            img_path = os.path.join(test_images_dir, img_file)
            label_path = os.path.join(test_labels_dir, label_file)
            
            image = load_array_from_file(img_path)
            label = load_array_from_file(label_path)
            
            # Convert to binary (0 for background, 1 for vessels)
            binary_vessels = (label > 0).astype(np.uint8)
            
            # Create best skeleton
            skeleton = create_best_skeleton(binary_vessels)
            
            # Create distance map
            distance_map = create_distance_map(skeleton)
            
            # Save distance map
            distance_filename = img_file.replace('.npy', '_distance_map.npy')  # Changed from _distance to _distance_map
            distance_path = os.path.join(test_distance_dir, distance_filename)
            np.save(distance_path, distance_map)
            
            # Store statistics
            test_stats.append({
                'image_file': img_file,
                'label_file': label_file,
                'distance_file': distance_filename,
                'vessel_pixels': binary_vessels.sum(),
                'skeleton_pixels': skeleton.sum(),
                'distance_range': [distance_map.min(), distance_map.max()],
                'distance_mean': distance_map.mean(),
                'distance_std': distance_map.std()
            })
            
            print(f"  Original vessels: {binary_vessels.sum()} pixels")
            print(f"  Skeleton: {skeleton.sum()} pixels")
            print(f"  Distance map range: [{distance_map.min():.2f}, {distance_map.max():.2f}]")
            print(f"  Saved: {distance_path}")
    else:
        print(f"Found {len(test_image_files)} test images but no test labels directory found")
        print("Test distance maps will not be created (no ground truth available)")
        test_stats = []
    
    # Create visualization
    print("\n=== CREATING VISUALIZATION ===")
    
    # Create output directory for visualizations
    os.makedirs("distance_maps_analysis", exist_ok=True)
    
    # Visualize first few samples from training
    num_samples = min(4, len(train_stats))
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    fig.suptitle('Distance Maps Dataset - Training Samples', fontsize=16)
    
    for i in range(num_samples):
        stats = train_stats[i]
        
        # Load data for visualization
        img_path = os.path.join(train_images_dir, stats['image_file'])
        label_path = os.path.join(train_labels_dir, stats['label_file'])
        distance_path = os.path.join(train_distance_dir, stats['distance_file'])
        
        image = load_array_from_file(img_path)
        label = load_array_from_file(label_path)
        distance_map = np.load(distance_path)
        
        # Original image
        axes[i, 0].imshow(image / 255.0)
        axes[i, 0].set_title(f"Original Image\n{stats['image_file']}")
        axes[i, 0].axis('off')
        
        # Binary vessels
        binary_vessels = (label > 0).astype(np.uint8)
        axes[i, 1].imshow(binary_vessels, cmap='gray')
        axes[i, 1].set_title(f"Binary Vessels\n{stats['vessel_pixels']} pixels")
        axes[i, 1].axis('off')
        
        # Distance map
        im_dist = axes[i, 2].imshow(distance_map, cmap='hot')
        axes[i, 2].set_title(f"Distance Map\nRange: [{distance_map.min():.1f}, {distance_map.max():.1f}]")
        axes[i, 2].axis('off')
        plt.colorbar(im_dist, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Distance map histogram
        axes[i, 3].hist(distance_map.flatten(), bins=50, alpha=0.7)
        axes[i, 3].set_title('Distance Distribution')
        axes[i, 3].set_xlabel('Distance to Skeleton')
        axes[i, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('distance_maps_analysis/training_samples.png', dpi=300, bbox_inches='tight')
    print("Training samples visualization saved as: distance_maps_analysis/training_samples.png")
    
    # Create statistics summary
    print("\n=== CREATING STATISTICS SUMMARY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distance Maps Dataset Statistics', fontsize=16)
    
    # Training statistics
    train_vessel_pixels = [s['vessel_pixels'] for s in train_stats]
    train_skeleton_pixels = [s['skeleton_pixels'] for s in train_stats]
    train_distance_means = [s['distance_mean'] for s in train_stats]
    train_distance_stds = [s['distance_std'] for s in train_stats]
    
    # Test statistics
    test_vessel_pixels = [s['vessel_pixels'] for s in test_stats] if test_stats else []
    test_skeleton_pixels = [s['skeleton_pixels'] for s in test_stats] if test_stats else []
    test_distance_means = [s['distance_mean'] for s in test_stats] if test_stats else []
    test_distance_stds = [s['distance_std'] for s in test_stats] if test_stats else []
    
    x_train = np.arange(len(train_stats))
    x_test = np.arange(len(test_stats)) if test_stats else np.array([])
    width = 0.35
    
    # Vessel pixels comparison
    axes[0, 0].bar(x_train - width/2, train_vessel_pixels, width, label='Training', alpha=0.7)
    if test_stats:
        axes[0, 0].bar(x_test + width/2, test_vessel_pixels, width, label='Test', alpha=0.7)
    axes[0, 0].set_title('Vessel Pixels Comparison')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Number of Pixels')
    axes[0, 0].legend()
    
    # Skeleton pixels comparison
    axes[0, 1].bar(x_train - width/2, train_skeleton_pixels, width, label='Training', alpha=0.7)
    if test_stats:
        axes[0, 1].bar(x_test + width/2, test_skeleton_pixels, width, label='Test', alpha=0.7)
    axes[0, 1].set_title('Skeleton Pixels Comparison')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Number of Pixels')
    axes[0, 1].legend()
    
    # Distance statistics
    axes[0, 2].bar(x_train - width/2, train_distance_means, width, label='Training Mean', alpha=0.7)
    if test_stats:
        axes[0, 2].bar(x_test + width/2, test_distance_means, width, label='Test Mean', alpha=0.7)
    axes[0, 2].set_title('Distance Map Statistics')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Distance Value')
    axes[0, 2].legend()
    
    # Overall statistics
    all_train_distances = []
    all_test_distances = []
    
    for stats in train_stats:
        distance_path = os.path.join(train_distance_dir, stats['distance_file'])
        distance_map = np.load(distance_path)
        all_train_distances.extend(distance_map.flatten())
    
    for stats in test_stats:
        distance_path = os.path.join(test_distance_dir, stats['distance_file'])
        distance_map = np.load(distance_path)
        all_test_distances.extend(distance_map.flatten())
    
    axes[1, 0].hist(all_train_distances, bins=100, alpha=0.7, label='Training', density=True)
    if all_test_distances:
        axes[1, 0].hist(all_test_distances, bins=100, alpha=0.7, label='Test', density=True)
    axes[1, 0].set_title('Overall Distance Distribution')
    axes[1, 0].set_xlabel('Distance to Skeleton')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # Pie chart for training
    total_train_vessels = sum(train_vessel_pixels)
    total_train_skeleton = sum(train_skeleton_pixels)
    axes[1, 1].pie([total_train_vessels - total_train_skeleton, total_train_skeleton], 
                   labels=['Removed', 'Skeleton'], autopct='%1.1f%%')
    axes[1, 1].set_title('Training Skeleton Reduction')
    
    # Pie chart for test (only if test data exists)
    if test_stats:
        total_test_vessels = sum(test_vessel_pixels)
        total_test_skeleton = sum(test_skeleton_pixels)
        axes[1, 2].pie([total_test_vessels - total_test_skeleton, total_test_skeleton], 
                       labels=['Removed', 'Skeleton'], autopct='%1.1f%%')
        axes[1, 2].set_title('Test Skeleton Reduction')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Test Data\nAvailable', 
                        ha='center', va='center', transform=axes[1, 2].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Test Skeleton Reduction')
    
    plt.tight_layout()
    plt.savefig('distance_maps_analysis/statistics_summary.png', dpi=300, bbox_inches='tight')
    print("Statistics summary saved as: distance_maps_analysis/statistics_summary.png")
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(f"Training samples: {len(train_stats)}")
    print(f"Test samples: {len(test_stats)}")
    print(f"Image shape: {image.shape}")
    print(f"Distance map shape: {distance_map.shape}")
    
    print(f"\nTraining statistics:")
    print(f"  Total vessel pixels: {total_train_vessels}")
    print(f"  Total skeleton pixels: {total_train_skeleton}")
    print(f"  Reduction ratio: {total_train_skeleton / total_train_vessels:.3f}")
    print(f"  Distance range: [{min(all_train_distances):.2f}, {max(all_train_distances):.2f}]")
    print(f"  Distance mean: {np.mean(all_train_distances):.2f}")
    print(f"  Distance std: {np.std(all_train_distances):.2f}")
    
    if test_stats:
        print(f"\nTest statistics:")
        total_test_vessels = sum(test_vessel_pixels)
        total_test_skeleton = sum(test_skeleton_pixels)
        print(f"  Total vessel pixels: {total_test_vessels}")
        print(f"  Total skeleton pixels: {total_test_skeleton}")
        print(f"  Reduction ratio: {total_test_skeleton / total_test_vessels:.3f}")
        print(f"  Distance range: [{min(all_test_distances):.2f}, {max(all_test_distances):.2f}]")
        print(f"  Distance mean: {np.mean(all_test_distances):.2f}")
        print(f"  Distance std: {np.std(all_test_distances):.2f}")
    else:
        print(f"\nTest statistics: No test data available")
    
    print(f"\nDistance maps saved to:")
    print(f"  Training: {train_distance_dir}")
    if test_stats:
        print(f"  Test: {test_distance_dir}")
    else:
        print(f"  Test: Not created (no ground truth available)")
    print(f"  Analysis files: distance_maps_analysis/")

if __name__ == "__main__":
    create_distance_maps_dataset() 