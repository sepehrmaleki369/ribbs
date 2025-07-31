import numpy as np
import os
from core.general_dataset.io import load_array_from_file

def check_label_values():
    """Check if labels have negative values"""
    
    print("=== CHECKING LABEL VALUES ===")
    
    labels_dir = "drive/training/inverted_labels"
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.npy')])
    
    print(f"Found {len(label_files)} label files")
    
    all_values = set()
    negative_values = []
    
    for i, label_file in enumerate(label_files):
        label_path = os.path.join(labels_dir, label_file)
        label = load_array_from_file(label_path)
        
        unique_vals = np.unique(label)
        all_values.update(unique_vals)
        
        # Check for negative values
        negative_vals = unique_vals[unique_vals < 0]
        if len(negative_vals) > 0:
            negative_values.append({
                'file': label_file,
                'negative_values': negative_vals,
                'all_unique': unique_vals
            })
        
        print(f"Label {i+1}: {label_file}")
        print(f"  Shape: {label.shape}")
        print(f"  Min value: {label.min()}")
        print(f"  Max value: {label.max()}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Has negative values: {len(negative_vals) > 0}")
        print()
    
    print("=== SUMMARY ===")
    print(f"All unique values across all labels: {sorted(all_values)}")
    print(f"Number of labels with negative values: {len(negative_values)}")
    
    if negative_values:
        print("\nLabels with negative values:")
        for item in negative_values:
            print(f"  {item['file']}: {item['negative_values']}")
    else:
        print("\n✓ No negative values found in any labels!")
    
    # Check if all labels have the same value range
    all_mins = []
    all_maxs = []
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        label = load_array_from_file(label_path)
        all_mins.append(label.min())
        all_maxs.append(label.max())
    
    print(f"\nValue ranges across all labels:")
    print(f"  Min values: {all_mins}")
    print(f"  Max values: {all_maxs}")
    print(f"  Global min: {min(all_mins)}")
    print(f"  Global max: {max(all_maxs)}")

if __name__ == "__main__":
    check_label_values() 