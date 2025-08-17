import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm

SAMPLES = [21, 27, 34]  # change/extend as needed
IMAGES_DIR = 'drive/training/images_npy'
GRAPHS_DIR = 'drive/training/graphs'
OUT_DIR = 'all_graph_overlays'


def load_image(sample_id: int) -> np.ndarray:
    path = os.path.join(IMAGES_DIR, f"{sample_id}_training.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.load(path)
    # Normalize for display if needed
    if img.dtype != np.float32 and img.max() > 1:
        img = img.astype(np.float32) / 255.0
    return img


def parse_graph(sample_id: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Parse graph from either detailed .npy.graph or compact .graph.
    Returns:
      nodes_xy: ndarray of shape (N, 2) with (x, y)
      edges: list of (u, v) index pairs
    """
    npy_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.npy.graph")
    simple_graph = os.path.join(GRAPHS_DIR, f"{sample_id}_manual1.graph")

    if os.path.exists(npy_graph):
        with open(npy_graph, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        nodes: List[Tuple[float, float]] = []
        edges: List[Tuple[int, int]] = []
        float3 = re.compile(r'^[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*\s+[-+]?\d+\.?\d*$')
        int2 = re.compile(r'^\d+\s+\d+$')
        for ln in lines:
            if float3.match(ln):
                x_str, y_str, _ = ln.split()
                nodes.append((float(x_str), float(y_str)))
            elif int2.match(ln):
                a_str, b_str = ln.split()
                edges.append((int(a_str), int(b_str)))
            # else: ignore
        return np.array(nodes, dtype=np.float32), edges

    elif os.path.exists(simple_graph):
        with open(simple_graph, 'r') as f:
            raw = f.read().strip('\n')
        groups = [g for g in raw.split('\n\n') if g.strip()]
        if not groups:
            raise ValueError(f"Empty graph file: {simple_graph}")
        # First group: nodes (two numbers per line)
        node_lines = [ln for ln in groups[0].splitlines() if ln.strip()]
        nodes: List[Tuple[float, float]] = []
        for ln in node_lines:
            x_str, y_str = ln.split()
            nodes.append((float(x_str), float(y_str)))
        # Second group (if present): edges (two int indices)
        edges: List[Tuple[int, int]] = []
        if len(groups) > 1:
            edge_lines = [ln for ln in groups[1].splitlines() if ln.strip()]
            for ln in edge_lines:
                a_str, b_str = ln.split()
                edges.append((int(a_str), int(b_str)))
        return np.array(nodes, dtype=np.float32), edges

    else:
        raise FileNotFoundError(f"Graph not found for sample {sample_id}: {npy_graph} or {simple_graph}")


def validate_graph(nodes_xy: np.ndarray, edges: List[Tuple[int, int]], img_shape: Tuple[int, int, int]):
    h, w = img_shape[0], img_shape[1]
    x = nodes_xy[:, 0]
    y = nodes_xy[:, 1]

    # Bounds
    out_of_bounds = np.sum((x < 0) | (x >= w) | (y < 0) | (y >= h))

    # Edge validity
    num_nodes = nodes_xy.shape[0]
    invalid_edges = sum(1 for (u, v) in edges if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes)

    # Degree stats
    degrees = np.zeros(num_nodes, dtype=int)
    for (u, v) in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            degrees[u] += 1
            degrees[v] += 1

    stats = {
        'num_nodes': int(num_nodes),
        'num_edges': int(len(edges)),
        'min_degree': int(degrees.min()) if num_nodes > 0 else 0,
        'max_degree': int(degrees.max()) if num_nodes > 0 else 0,
        'mean_degree': float(degrees.mean()) if num_nodes > 0 else 0.0,
        'out_of_bounds_nodes': int(out_of_bounds),
        'invalid_edges': int(invalid_edges),
    }
    return stats, degrees


def plot_graph_on_image(sample_id: int):
    img = load_image(sample_id)
    nodes_xy, edges = parse_graph(sample_id)

    stats, _ = validate_graph(nodes_xy, edges, img.shape)

    # Plot overlay
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    # Draw edges in blue
    for (u, v) in edges:
        if 0 <= u < len(nodes_xy) and 0 <= v < len(nodes_xy):
            x0, y0 = nodes_xy[u]
            x1, y1 = nodes_xy[v]
            ax.plot([x0, x1], [y0, y1], color='blue', linewidth=1.0, alpha=0.8)
    # Draw nodes in blue
    ax.scatter(nodes_xy[:, 0], nodes_xy[:, 1], s=6, c='blue', alpha=0.9)
    ax.set_title(f"Graph Overlay - Sample {sample_id}")
    ax.axis('off')

    out_path = os.path.join(OUT_DIR, f"graph_overlay_{sample_id:03d}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Sample {sample_id}: {stats['num_nodes']} nodes, {stats['num_edges']} edges -> {out_path}")
    return stats


def get_all_available_samples():
    """Get all sample IDs that have both image and graph files."""
    available_samples = []
    
    # Check what images are available
    if os.path.exists(IMAGES_DIR):
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('_training.npy')]
        image_samples = [int(f.split('_')[0]) for f in image_files]
        print(f"📁 Found {len(image_samples)} training images: {sorted(image_samples)}")
    else:
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return []
    
    # Check what graphs are available
    if os.path.exists(GRAPHS_DIR):
        graph_files = [f for f in os.listdir(GRAPHS_DIR) if f.endswith('.graph') or f.endswith('.npy.graph')]
        graph_samples = list(set([int(f.split('_')[0]) for f in graph_files]))
        print(f"📁 Found {len(graph_samples)} graph files: {sorted(graph_samples)}")
    else:
        print(f"❌ Graphs directory not found: {GRAPHS_DIR}")
        return []
    
    # Find intersection (samples with both image and graph)
    available_samples = sorted(list(set(image_samples) & set(graph_samples)))
    print(f"🎯 {len(available_samples)} samples have both image and graph: {available_samples}")
    
    return available_samples


def main():
    print("=== OVERLAY ALL GRAPHS ON ALL TRAINING IMAGES ===")
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUT_DIR}")
    
    # Get all available samples
    available_samples = get_all_available_samples()
    
    if not available_samples:
        print("❌ No samples found with both image and graph files!")
        return
    
    # Process all samples
    print(f"\n🎯 Processing {len(available_samples)} samples...")
    
    successful_samples = []
    failed_samples = []
    
    for sample_id in tqdm(available_samples, desc="Processing samples"):
        try:
            stats = plot_graph_on_image(sample_id)
            successful_samples.append((sample_id, stats))
        except Exception as e:
            print(f"❌ Failed on sample {sample_id}: {e}")
            failed_samples.append(sample_id)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"✅ Successfully processed: {len(successful_samples)} samples")
    if failed_samples:
        print(f"❌ Failed: {len(failed_samples)} samples: {failed_samples}")
    
    if successful_samples:
        # Calculate overall statistics
        total_nodes = sum(stats['num_nodes'] for _, stats in successful_samples)
        total_edges = sum(stats['num_edges'] for _, stats in successful_samples)
        avg_nodes = total_nodes / len(successful_samples)
        avg_edges = total_edges / len(successful_samples)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   • Total nodes across all graphs: {total_nodes}")
        print(f"   • Total edges across all graphs: {total_edges}")
        print(f"   • Average nodes per graph: {avg_nodes:.1f}")
        print(f"   • Average edges per graph: {avg_edges:.1f}")
    
    print(f"\n🎯 All overlays saved to: {OUT_DIR}/")


if __name__ == '__main__':
    main() 