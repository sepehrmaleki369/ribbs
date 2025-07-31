import numpy as np
import matplotlib.pyplot as plt
import os
from core.general_dataset.io import load_array_from_file
import seaborn as sns

def plot_approach2_dataset():
    """Plot Approach 2 dataset: images with their ground truth labels"""
    
    print("=== PLOTTING APPROACH 2 DATASET ===")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("approach2_visualization", exist_ok=True)
    
    # Define paths for Approach 2
    images_dir = "drive/training/images_npy"
    labels_dir = "drive/training/inverted_labels"
    masks_dir = "drive/training/mask_npy"
    graphs_dir = "drive/training/graphs"
    
    # Get all files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.npy')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.npy')])
    graph_files = sorted([f for f in os.listdir(graphs_dir) if f.endswith('.graph')])
    
    print(f"Found {len(image_files)} images, {len(label_files)} labels, {len(mask_files)} masks, {len(graph_files)} graphs")
    
    # Load and analyze all data
    all_data = []
    
    for i, (img_file, label_file, mask_file) in enumerate(zip(image_files, label_files, mask_files)):
        try:
            # Load data
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, label_file)
            mask_path = os.path.join(masks_dir, mask_file)
            
            image = load_array_from_file(img_path)
            label = load_array_from_file(label_path)
            mask = load_array_from_file(mask_path)
            
            # Store data
            all_data.append({
                'index': i,
                'image_file': img_file,
                'label_file': label_file,
                'mask_file': mask_file,
                'image': image,
                'label': label,
                'mask': mask,
                'image_stats': {
                    'shape': image.shape,
                    'min': image.min(),
                    'max': image.max(),
                    'mean': image.mean(),
                    'std': image.std()
                },
                'label_stats': {
                    'shape': label.shape,
                    'min': label.min(),
                    'max': label.max(),
                    'mean': label.mean(),
                    'unique_values': np.unique(label),
                    'class_counts': np.bincount(label.flatten().astype(int))
                },
                'mask_stats': {
                    'shape': mask.shape,
                    'min': mask.min(),
                    'max': mask.max(),
                    'mean': mask.mean(),
                    'unique_values': np.unique(mask),
                    'vessel_percentage': (mask > 0).sum() / mask.size * 100
                }
            })
            
            print(f"Loaded {img_file}: Image {image.shape}, Label {label.shape} (classes: {np.unique(label)}), Mask {mask.shape}")
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    print(f"\nSuccessfully loaded {len(all_data)} samples")
    
    # Create comprehensive visualization
    print("\n=== CREATING COMPREHENSIVE VISUALIZATION ===")
    
    # Plot all samples in a grid
    num_samples = len(all_data)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 15, rows * 5))
    fig.suptitle('Approach 2 Dataset: Images, Labels, and Masks', fontsize=16)
    
    for i, data in enumerate(all_data):
        row = i // cols
        col = (i % cols) * 3
        
        # Image
        axes[row, col].imshow(data['image'] / 255.0)
        axes[row, col].set_title(f"Image {i+1}\n{data['image_file']}\nRange: [{data['image_stats']['min']:.0f}-{data['image_stats']['max']:.0f}]")
        axes[row, col].axis('off')
        
        # Label (multi-class)
        im_label = axes[row, col + 1].imshow(data['label'], cmap='tab20')
        axes[row, col + 1].set_title(f"Label {i+1}\n{data['label_file']}\nClasses: {len(data['label_stats']['unique_values'])}")
        axes[row, col + 1].axis('off')
        
        # Mask (binary)
        axes[row, col + 2].imshow(data['mask'], cmap='gray')
        axes[row, col + 2].set_title(f"Mask {i+1}\n{data['mask_file']}\nVessels: {data['mask_stats']['vessel_percentage']:.1f}%")
        axes[row, col + 2].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = (i % cols) * 3
        for j in range(3):
            axes[row, col + j].axis('off')
    
    plt.tight_layout()
    plt.savefig('approach2_visualization/all_samples_overview.png', dpi=300, bbox_inches='tight')
    print("All samples overview saved as: approach2_visualization/all_samples_overview.png")
    
    # Create detailed analysis for first few samples
    print("\n=== CREATING DETAILED ANALYSIS ===")
    
    num_detailed = min(6, len(all_data))
    
    fig, axes = plt.subplots(num_detailed, 4, figsize=(20, 5 * num_detailed))
    fig.suptitle('Detailed Analysis: First 6 Samples', fontsize=16)
    
    for i in range(num_detailed):
        data = all_data[i]
        
        # Image
        axes[i, 0].imshow(data['image'] / 255.0)
        axes[i, 0].set_title(f"Sample {i+1}: Image\n{data['image_file']}")
        axes[i, 0].axis('off')
        
        # Label
        axes[i, 1].imshow(data['label'], cmap='tab20')
        axes[i, 1].set_title(f"Sample {i+1}: Multi-class Label\nClasses: {data['label_stats']['unique_values']}")
        axes[i, 1].axis('off')
        
        # Mask
        axes[i, 2].imshow(data['mask'], cmap='gray')
        axes[i, 2].set_title(f"Sample {i+1}: Binary Mask\nVessels: {data['mask_stats']['vessel_percentage']:.1f}%")
        axes[i, 2].axis('off')
        
        # Histogram
        axes[i, 3].hist(data['label'].flatten(), bins=20, alpha=0.7, color='blue', label='Label')
        axes[i, 3].hist(data['mask'].flatten(), bins=10, alpha=0.7, color='red', label='Mask')
        axes[i, 3].set_title(f"Sample {i+1}: Value Distribution")
        axes[i, 3].set_xlabel('Pixel Value')
        axes[i, 3].set_ylabel('Frequency')
        axes[i, 3].legend()
    
    plt.tight_layout()
    plt.savefig('approach2_visualization/detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("Detailed analysis saved as: approach2_visualization/detailed_analysis.png")
    
    # Create statistics summary
    print("\n=== CREATING STATISTICS SUMMARY ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Statistics Summary', fontsize=16)
    
    # Image statistics
    img_means = [d['image_stats']['mean'] for d in all_data]
    img_stds = [d['image_stats']['std'] for d in all_data]
    
    axes[0, 0].bar(range(len(img_means)), img_means, alpha=0.7, label='Mean')
    axes[0, 0].bar(range(len(img_stds)), img_stds, alpha=0.7, label='Std')
    axes[0, 0].set_title('Image Statistics')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Pixel Value')
    axes[0, 0].legend()
    
    # Label statistics
    label_unique_counts = [len(d['label_stats']['unique_values']) for d in all_data]
    label_means = [d['label_stats']['mean'] for d in all_data]
    
    axes[0, 1].bar(range(len(label_unique_counts)), label_unique_counts, alpha=0.7)
    axes[0, 1].set_title('Number of Classes per Sample')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Number of Classes')
    
    axes[0, 2].bar(range(len(label_means)), label_means, alpha=0.7)
    axes[0, 2].set_title('Label Mean Values')
    axes[0, 2].set_xlabel('Sample Index')
    axes[0, 2].set_ylabel('Mean Label Value')
    
    # Mask statistics
    mask_means = [d['mask_stats']['mean'] for d in all_data]
    vessel_percentages = [d['mask_stats']['vessel_percentage'] for d in all_data]
    
    axes[1, 0].bar(range(len(mask_means)), mask_means, alpha=0.7)
    axes[1, 0].set_title('Mask Mean Values')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Mean Mask Value')
    
    axes[1, 1].bar(range(len(vessel_percentages)), vessel_percentages, alpha=0.7, color='red')
    axes[1, 1].set_title('Vessel Percentage per Sample')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Vessel Percentage (%)')
    
    # Class distribution across all samples
    all_classes = set()
    for data in all_data:
        all_classes.update(data['label_stats']['unique_values'])
    
    class_counts = {cls: 0 for cls in all_classes}
    for data in all_data:
        for cls in data['label_stats']['unique_values']:
            class_counts[cls] += 1
    
    axes[1, 2].bar(class_counts.keys(), class_counts.values(), alpha=0.7)
    axes[1, 2].set_title('Class Frequency Across Samples')
    axes[1, 2].set_xlabel('Class Value')
    axes[1, 2].set_ylabel('Number of Samples')
    
    plt.tight_layout()
    plt.savefig('approach2_visualization/statistics_summary.png', dpi=300, bbox_inches='tight')
    print("Statistics summary saved as: approach2_visualization/statistics_summary.png")
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE SUMMARY ===")
    print(f"Total samples: {len(all_data)}")
    print(f"Image shape: {all_data[0]['image_stats']['shape']}")
    print(f"Label shape: {all_data[0]['label_stats']['shape']}")
    print(f"Mask shape: {all_data[0]['mask_stats']['shape']}")
    
    print(f"\nImage statistics:")
    print(f"  Mean range: [{min([d['image_stats']['mean'] for d in all_data]):.1f}, {max([d['image_stats']['mean'] for d in all_data]):.1f}]")
    print(f"  All images: RGB format, range [0, 255]")
    
    print(f"\nLabel statistics:")
    all_label_classes = set()
    for data in all_data:
        all_label_classes.update(data['label_stats']['unique_values'])
    print(f"  Total unique classes across all samples: {len(all_label_classes)}")
    print(f"  Class values: {sorted(all_label_classes)}")
    print(f"  Mean classes per sample: {np.mean([len(d['label_stats']['unique_values']) for d in all_data]):.1f}")
    
    print(f"\nMask statistics:")
    print(f"  All masks: Binary format (0/255)")
    print(f"  Vessel percentage range: [{min([d['mask_stats']['vessel_percentage'] for d in all_data]):.1f}%, {max([d['mask_stats']['vessel_percentage'] for d in all_data]):.1f}%]")
    print(f"  Average vessel percentage: {np.mean([d['mask_stats']['vessel_percentage'] for d in all_data]):.1f}%")
    
    print(f"\nAll visualization files saved to: approach2_visualization/")

if __name__ == "__main__":
    plot_approach2_dataset() 