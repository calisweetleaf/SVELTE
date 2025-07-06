#!/usr/bin/env python3
"""
Test runner for AttentionTopologySystem
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from src.model_architecture.attention_topology import AttentionTopologySystem, CurvatureMethod

def test_attention_topology():
    print("Testing AttentionTopologySystem...")
    
    # Create test data
    tensors = {
        'layer1': np.random.rand(4, 4),
        'layer2': np.random.rand(3, 3)
    }
    
    try:
        # Initialize system
        ats = AttentionTopologySystem(tensors)
        print("✓ System initialized successfully")
        
        # Test curvature computation
        curvature = ats.compute_curvature()
        print(f"✓ Curvature computed for {len(curvature)} layers")
        
        # Test different curvature methods
        for method in CurvatureMethod:
            try:
                curv = ats.compute_curvature(method)
                print(f"✓ {method.value} curvature computed successfully")
            except Exception as e:
                print(f"✗ {method.value} curvature failed: {e}")
        
        # Test topology analysis
        for layer_name in tensors.keys():
            try:
                metrics = ats.analyze_topology(layer_name)
                print(f"✓ Topology analysis for {layer_name}: entropy={metrics.entropy:.4f}")
            except Exception as e:
                print(f"✗ Topology analysis for {layer_name} failed: {e}")
        
        # Test statistics
        try:
            stats = ats.get_curvature_statistics('layer1')
            print(f"✓ Statistics computed: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        except Exception as e:
            print(f"✗ Statistics computation failed: {e}")
        
        print("\nAll tests completed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_attention_topology()
    sys.exit(0 if success else 1)
