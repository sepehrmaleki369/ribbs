# Verify Discrete Distance Maps
# Check that discrete distance maps contain only integer values, no continuous values

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def verify_discrete_distance_maps():
    """Verify that discrete distance maps are truly discrete (integer values only)"""
    
    print("=== VERIFYING DISCRETE DISTANCE MAPS ===")
    print("Checking for integer values only - no continuous values")
    
    # Check if discrete distance maps exist
    discrete_maps_dir = 'drive/training/discrete_distance_maps'
    if not os.path.exists(discrete_maps_dir):
        print(f"❌ Discrete distance maps directory not found: {discrete_maps_dir}")
        return
    
    # Get discrete distance map files
    discrete_map_files = [f for f in os.listdir(discrete_maps_dir) if f.endswith('.npy')]
    discrete_map_files.sort()
    
    print(f"Found {len(discrete_map_files)} discrete distance map files")
    
    # Statistics collection
    verification_stats = []
    
    # Check each file
    for filename in tqdm(discrete_map_files, desc="Verifying discrete maps"):
        discrete_map_path = os.path.join(discrete_maps_dir, filename)
        discrete_map = np.load(discrete_map_path)
        
        # Check data type
        dtype = discrete_map.dtype
        
        # Check if all values are integers
        unique_values = np.unique(discrete_map)
        is_integer = all(isinstance(val, (int, np.integer)) for val in unique_values)
        
        # Check for any decimal values
        has_decimals = any(val % 1 != 0 for val in unique_values)
        
        # Check value range
        min_val = discrete_map.min()
        max_val = discrete_map.max()
        
        # Check if values are consecutive integers
        expected_values = set(range(int(min_val), int(max_val) + 1))
        actual_values = set(unique_values)
        is_consecutive = actual_values == expected_values
        
        # Statistics
        stats = {
            'filename': filename,
            'dtype': dtype,
            'is_integer': is_integer,
            'has_decimals': has_decimals,
            'min_val': min_val,
            'max_val': max_val,
            'unique_count': len(unique_values),
            'is_consecutive': is_consecutive,
            'unique_values': unique_values,
            'value_range': max_val - min_val + 1
        }
        verification_stats.append(stats)
        
        # Print results for this file
        print(f"\n📁 {filename}:")
        print(f"  Data type: {dtype}")
        print(f"  Integer values only: {'✅' if is_integer else '❌'}")
        print(f"  Has decimal values: {'❌' if not has_decimals else '⚠️'}")
        print(f"  Value range: [{min_val}, {max_val}]")
        print(f"  Unique values: {len(unique_values)}")
        print(f"  Consecutive integers: {'✅' if is_consecutive else '❌'}")
        print(f"  Sample values: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
    
    # Create verification summary
    create_verification_summary(verification_stats)
    
    # Check overall results
    all_integer = all(stats['is_integer'] for stats in verification_stats)
    all_no_decimals = all(not stats['has_decimals'] for stats in verification_stats)
    
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"✅ All files have integer values only: {'YES' if all_integer else 'NO'}")
    print(f"✅ No decimal values found: {'YES' if all_no_decimals else 'NO'}")
    print(f"📊 Total files checked: {len(verification_stats)}")
    
    if all_integer and all_no_decimals:
        print(f"🎉 SUCCESS: All discrete distance maps are truly discrete!")
    else:
        print(f"⚠️  WARNING: Some files may contain continuous values!")

def create_verification_summary(verification_stats):
    """Create summary visualization of verification results"""
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Discrete Distance Maps Verification Results', fontsize=16)
    
    # Extract data
    filenames = [stats['filename'].replace('_discrete_distance_map.npy', '') for stats in verification_stats]
    min_vals = [stats['min_val'] for stats in verification_stats]
    max_vals = [stats['max_val'] for stats in verification_stats]
    unique_counts = [stats['unique_count'] for stats in verification_stats]
    value_ranges = [stats['value_range'] for stats in verification_stats]
    is_consecutive = [stats['is_consecutive'] for stats in verification_stats]
    
    x_pos = np.arange(len(verification_stats))
    
    # Plot 1: Minimum values
    axes[0, 0].bar(x_pos, min_vals, alpha=0.7, color='blue')
    axes[0, 0].set_title('Minimum Values (Should be 0)')
    axes[0, 0].set_xlabel('File Index')
    axes[0, 0].set_ylabel('Min Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Maximum values
    axes[0, 1].bar(x_pos, max_vals, alpha=0.7, color='red')
    axes[0, 1].set_title('Maximum Values (Discrete Range)')
    axes[0, 1].set_xlabel('File Index')
    axes[0, 1].set_ylabel('Max Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Unique value counts
    axes[0, 2].bar(x_pos, unique_counts, alpha=0.7, color='green')
    axes[0, 2].set_title('Number of Unique Values')
    axes[0, 2].set_xlabel('File Index')
    axes[0, 2].set_ylabel('Unique Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Value ranges
    axes[1, 0].bar(x_pos, value_ranges, alpha=0.7, color='orange')
    axes[1, 0].set_title('Value Ranges (Max - Min + 1)')
    axes[1, 0].set_xlabel('File Index')
    axes[1, 0].set_ylabel('Range')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Consecutive check
    consecutive_colors = ['green' if consecutive else 'red' for consecutive in is_consecutive]
    axes[1, 1].bar(x_pos, [1 if consecutive else 0 for consecutive in is_consecutive], 
                   color=consecutive_colors, alpha=0.7)
    axes[1, 1].set_title('Consecutive Integers Check')
    axes[1, 1].set_xlabel('File Index')
    axes[1, 1].set_ylabel('Consecutive (1=Yes, 0=No)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    axes[1, 2].text(0.1, 0.9, f"Total Files: {len(verification_stats)}", fontsize=12)
    axes[1, 2].text(0.1, 0.8, f"All Integer: {'✅' if all(stats['is_integer'] for stats in verification_stats) else '❌'}", fontsize=12)
    axes[1, 2].text(0.1, 0.7, f"No Decimals: {'✅' if all(not stats['has_decimals'] for stats in verification_stats) else '❌'}", fontsize=12)
    axes[1, 2].text(0.1, 0.6, f"All Consecutive: {'✅' if all(stats['is_consecutive'] for stats in verification_stats) else '❌'}", fontsize=12)
    axes[1, 2].text(0.1, 0.5, f"Avg Min: {np.mean(min_vals):.1f}", fontsize=12)
    axes[1, 2].text(0.1, 0.4, f"Avg Max: {np.mean(max_vals):.1f}", fontsize=12)
    axes[1, 2].text(0.1, 0.3, f"Avg Unique: {np.mean(unique_counts):.1f}", fontsize=12)
    axes[1, 2].set_title('Verification Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions/discrete_verification_results.png', dpi=300, bbox_inches='tight')
    print(f"Verification results saved as predictions/discrete_verification_results.png")

if __name__ == "__main__":
    verify_discrete_distance_maps() 