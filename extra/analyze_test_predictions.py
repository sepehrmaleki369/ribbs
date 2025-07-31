import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import seaborn as sns

def analyze_test_predictions():
    """Analyze the test predictions to check distance map values"""
    
    print("=== ANALYZING TEST PREDICTIONS ===")
    
    # Get all distance map files
    test_predictions_dir = 'predictions/test_predictions'
    distance_map_files = [f for f in os.listdir(test_predictions_dir) if f.endswith('_distance_map.npy')]
    distance_map_files.sort()
    
    print(f"Found {len(distance_map_files)} distance map files")
    
    # Load and analyze each distance map
    all_values = []
    all_stats = []
    
    for i, filename in enumerate(tqdm(distance_map_files, desc="Analyzing distance maps")):
        filepath = os.path.join(test_predictions_dir, filename)
        distance_map = np.load(filepath)
        
        # Basic statistics
        stats = {
            'filename': filename,
            'min': float(distance_map.min()),
            'max': float(distance_map.max()),
            'mean': float(distance_map.mean()),
            'std': float(distance_map.std()),
            'median': float(np.median(distance_map)),
            'shape': distance_map.shape,
            'total_pixels': distance_map.size,
            'zero_pixels': np.sum(distance_map == 0),
            'near_zero_pixels': np.sum(distance_map < 0.1),
            'small_values': np.sum(distance_map < 1.0),
            'medium_values': np.sum((distance_map >= 1.0) & (distance_map < 5.0)),
            'large_values': np.sum(distance_map >= 5.0)
        }
        
        all_stats.append(stats)
        all_values.extend(distance_map.flatten())
    
    # Convert to numpy array for analysis
    all_values = np.array(all_values)
    
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total values analyzed: {len(all_values):,}")
    print(f"Global min: {all_values.min():.4f}")
    print(f"Global max: {all_values.max():.4f}")
    print(f"Global mean: {all_values.mean():.4f}")
    print(f"Global std: {all_values.std():.4f}")
    print(f"Global median: {np.median(all_values):.4f}")
    
    # Check for expected pattern
    print(f"\n=== PATTERN ANALYSIS ===")
    zero_count = np.sum(all_values == 0)
    near_zero_count = np.sum(all_values < 0.1)
    small_count = np.sum(all_values < 1.0)
    
    print(f"Values exactly 0: {zero_count:,} ({zero_count/len(all_values)*100:.2f}%)")
    print(f"Values < 0.1: {near_zero_count:,} ({near_zero_count/len(all_values)*100:.2f}%)")
    print(f"Values < 1.0: {small_count:,} ({small_count/len(all_values)*100:.2f}%)")
    
    # Check value distribution
    print(f"\n=== VALUE DISTRIBUTION ===")
    unique_values = np.unique(all_values)
    print(f"Unique values: {len(unique_values)}")
    print(f"Value range: {unique_values.min():.4f} to {unique_values.max():.4f}")
    
    # Show some sample unique values
    print(f"Sample unique values: {unique_values[:20]}")
    
    # Create detailed analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Test Predictions Analysis - Distance Map Values', fontsize=16)
    
    # 1. Global histogram
    axes[0, 0].hist(all_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Global Value Distribution')
    axes[0, 0].set_xlabel('Distance Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Zoom in on small values
    small_values = all_values[all_values < 10]
    axes[0, 1].hist(small_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Small Values Distribution (< 10)')
    axes[0, 1].set_xlabel('Distance Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot of individual files
    file_stats = []
    file_names = []
    for stats in all_stats:
        file_stats.append([stats['min'], stats['max'], stats['mean'], stats['std']])
        file_names.append(stats['filename'].replace('_distance_map.npy', ''))
    
    file_stats = np.array(file_stats)
    axes[0, 2].boxplot([all_values[all_values < 5]], labels=['All Values < 5'])
    axes[0, 2].set_title('Box Plot of Small Values')
    axes[0, 2].set_ylabel('Distance Value')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Individual file statistics
    mins = [stats['min'] for stats in all_stats]
    maxs = [stats['max'] for stats in all_stats]
    means = [stats['mean'] for stats in all_stats]
    
    x_pos = np.arange(len(file_names))
    axes[1, 0].plot(x_pos, mins, 'o-', label='Min', color='red')
    axes[1, 0].plot(x_pos, maxs, 's-', label='Max', color='blue')
    axes[1, 0].plot(x_pos, means, '^-', label='Mean', color='green')
    axes[1, 0].set_title('Statistics per Test Image')
    axes[1, 0].set_xlabel('Test Image')
    axes[1, 0].set_ylabel('Distance Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(x_pos[::2])
    axes[1, 0].set_xticklabels([f"Test {i+1}" for i in range(0, len(file_names), 2)], rotation=45)
    
    # 5. Value range analysis
    value_ranges = [stats['max'] - stats['min'] for stats in all_stats]
    axes[1, 1].bar(x_pos, value_ranges, alpha=0.7, color='orange')
    axes[1, 1].set_title('Value Range per Test Image')
    axes[1, 1].set_xlabel('Test Image')
    axes[1, 1].set_ylabel('Max - Min')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(x_pos[::2])
    axes[1, 1].set_xticklabels([f"Test {i+1}" for i in range(0, len(file_names), 2)], rotation=45)
    
    # 6. Zero/near-zero analysis
    zero_percentages = [stats['zero_pixels'] / stats['total_pixels'] * 100 for stats in all_stats]
    near_zero_percentages = [stats['near_zero_pixels'] / stats['total_pixels'] * 100 for stats in all_stats]
    
    x_pos = np.arange(len(file_names))
    width = 0.35
    axes[1, 2].bar(x_pos - width/2, zero_percentages, width, label='Exactly 0', alpha=0.7, color='red')
    axes[1, 2].bar(x_pos + width/2, near_zero_percentages, width, label='< 0.1', alpha=0.7, color='blue')
    axes[1, 2].set_title('Zero/Near-Zero Percentage per Test Image')
    axes[1, 2].set_xlabel('Test Image')
    axes[1, 2].set_ylabel('Percentage (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xticks(x_pos[::2])
    axes[1, 2].set_xticklabels([f"Test {i+1}" for i in range(0, len(file_names), 2)], rotation=45)
    
    plt.tight_layout()
    plt.savefig('predictions/test_predictions_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create summary table
    print(f"\n=== INDIVIDUAL FILE STATISTICS ===")
    print(f"{'File':<20} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Zero%':<8}")
    print("-" * 70)
    for stats in all_stats:
        zero_pct = stats['zero_pixels'] / stats['total_pixels'] * 100
        print(f"{stats['filename'][:18]:<20} {stats['min']:<8.3f} {stats['max']:<8.3f} {stats['mean']:<8.3f} {stats['std']:<8.3f} {zero_pct:<8.2f}")
    
    # Check if values match expectations
    print(f"\n=== EXPECTATION CHECK ===")
    print("Expected pattern: Centers at 0, increasing distances (1, 2, 3, ...)")
    
    if zero_count > 0:
        print(f"✅ Found {zero_count:,} zero values (centers)")
    else:
        print(f"❌ No zero values found - centers not at 0")
    
    if all_values.min() >= 0:
        print(f"✅ All values are non-negative")
    else:
        print(f"❌ Found negative values: {all_values.min():.4f}")
    
    if all_values.max() > 1:
        print(f"✅ Found values > 1 (increasing distances): max = {all_values.max():.4f}")
    else:
        print(f"❌ All values <= 1 - no increasing distance pattern")
    
    # Check for reasonable distance progression
    unique_sorted = np.sort(unique_values)
    if len(unique_sorted) > 1:
        diffs = np.diff(unique_sorted)
        avg_diff = np.mean(diffs)
        print(f"Average difference between consecutive values: {avg_diff:.4f}")
        
        if avg_diff < 0.1:
            print(f"⚠️  Very small differences - might be continuous rather than discrete")
        elif avg_diff > 2.0:
            print(f"⚠️  Large differences - might be too coarse")
        else:
            print(f"✅ Reasonable progression between values")
    
    print(f"\n✅ Analysis completed!")
    print(f"📊 Summary:")
    print(f"  - Analysis plot: predictions/test_predictions_analysis.png")
    print(f"  - Processed {len(distance_map_files)} distance maps")
    print(f"  - Total values analyzed: {len(all_values):,}")

if __name__ == "__main__":
    analyze_test_predictions() 