#!/usr/bin/env python3
"""
Test script to verify oversampled graphs are being loaded correctly.
"""

import sys
sys.path.append('.')

from utils.graphs import (
    load_training_graph_by_id, 
    get_graph_info, 
    switch_to_oversampled_graphs,
    switch_to_original_graphs,
    get_current_graph_mode
)

def test_graph_loading():
    """Test loading both original and oversampled graphs."""
    
    print("=== TESTING GRAPH LOADING ===")
    
    # Test samples
    test_samples = [21, 22, 23, 24, 25]
    
    print(f"\n🎯 Current graph mode: {get_current_graph_mode()}")
    
    # Test oversampled graphs
    print(f"\n🔵 Testing OVERSAMPLED graphs:")
    switch_to_oversampled_graphs()
    
    for sample_id in test_samples:
        print(f"\n--- Sample {sample_id} ---")
        
        # Get graph info
        info = get_graph_info(str(sample_id))
        print(f"   📊 Graph info: {info}")
        
        # Try to load graph
        try:
            graph = load_training_graph_by_id(str(sample_id))
            if graph is not None:
                print(f"   ✅ Loaded graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
            else:
                print(f"   ❌ Failed to load graph")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test original graphs
    print(f"\n🔴 Testing ORIGINAL graphs:")
    switch_to_original_graphs()
    
    for sample_id in test_samples:
        print(f"\n--- Sample {sample_id} ---")
        
        # Get graph info
        info = get_graph_info(str(sample_id))
        print(f"   📊 Graph info: {info}")
        
        # Try to load graph
        try:
            graph = load_training_graph_by_id(str(sample_id))
            if graph is not None:
                print(f"   ✅ Loaded graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
            else:
                print(f"   ❌ Failed to load graph")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Switch back to oversampled for training
    print(f"\n🔄 Switching back to oversampled graphs for training...")
    switch_to_oversampled_graphs()
    print(f"✅ Ready for training with: {get_current_graph_mode()}")


if __name__ == '__main__':
    test_graph_loading()
