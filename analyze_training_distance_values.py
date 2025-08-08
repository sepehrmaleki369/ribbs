import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from core.regression_dataset import RegressionDataset


def analyze_training_distance_values():
    # Ensure output dir exists
    os.makedirs('predictions', exist_ok=True)

    # Load config
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Build train dataset (current branch decides discrete vs continuous)
    dataset = RegressionDataset(config, split='train')
    if len(dataset) == 0:
        print('❌ No training samples found')
        return

    # Aggregate stats
    global_min = float('inf')
    global_max = float('-inf')
    value_counts = {}
    any_decimals = False

    print('\nScanning training distance maps...')
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        dm = item['distance_map'].numpy()
        # Update min/max
        dmin = float(dm.min())
        dmax = float(dm.max())
        global_min = min(global_min, dmin)
        global_max = max(global_max, dmax)
        # Count values
        vals, counts = np.unique(dm, return_counts=True)
        # Check decimals
        if np.any(vals % 1 != 0):
            any_decimals = True
        # Accumulate
        for v, c in zip(vals, counts):
            value_counts[v] = value_counts.get(v, 0) + int(c)

    # Prepare histogram/bar data
    all_values = np.array(sorted(value_counts.keys()))
    all_counts = np.array([value_counts[v] for v in all_values])

    total_pixels = int(all_counts.sum())
    unique_count = len(all_values)

    print('\n=== Training Distance Map Stats ===')
    print(f'Samples: {len(dataset)}')
    print(f'Value range: [{global_min:.1f}, {global_max:.1f}]')
    print(f'Unique values: {unique_count}')
    print(f'Has decimal values: {"YES" if any_decimals else "NO"}')

    head_vals = all_values[:10]
    print('First 10 unique values:', head_vals)

    # Plot
    plt.figure(figsize=(12, 5))
    if unique_count <= 512:
        # Bar plot for discrete-ish distributions
        plt.bar(all_values, all_counts, width=0.8)
        plt.xlabel('Distance value')
        plt.ylabel('Pixel count')
        plt.title('Training Distance Map Value Distribution')
    else:
        # Too many unique values, use histogram
        flat_vals = np.repeat(all_values, all_counts)
        plt.hist(flat_vals, bins=100)
        plt.xlabel('Distance value')
        plt.ylabel('Frequency')
        plt.title('Training Distance Map Value Distribution (Histogram)')

    plt.tight_layout()
    out_path = 'predictions/training_distance_values_hist.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'✅ Saved plot: {out_path}')


if __name__ == '__main__':
    analyze_training_distance_values() 