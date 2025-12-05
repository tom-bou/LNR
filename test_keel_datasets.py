#!/usr/bin/env python3
"""
Test script for KEEL dataset loading.
Run this after downloading the datasets to verify everything works.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.keel import KEEL_LT, get_keel_dataset

def test_keel_dataset(dataset_name='yeast1'):
    """Test loading a KEEL dataset."""
    print(f"Testing KEEL dataset: {dataset_name}")
    print("=" * 50)
    
    try:
        # Try to load the dataset
        dataset_loader = get_keel_dataset(dataset_name)
        
        print(f"\n✓ Successfully loaded {dataset_name}")
        print(f"  Number of classes: {dataset_loader.num_classes}")
        print(f"  Class names: {dataset_loader.class_names}")
        print(f"  Class distribution: {dataset_loader.cls_num_list}")
        
        # Test DataLoader
        print(f"\nTesting DataLoaders...")
        print(f"  Train batches: {len(dataset_loader.train_instance)}")
        print(f"  Test batches: {len(dataset_loader.eval)}")
        
        # Get a sample batch
        sample_batch = next(iter(dataset_loader.train_instance))
        index, features, target = sample_batch
        print(f"\n  Sample batch shape:")
        print(f"    Features: {features.shape}")
        print(f"    Targets: {target.shape}")
        print(f"    Indices: {index.shape}")
        
        print(f"\n✓ All tests passed for {dataset_name}!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"  Please run: python scripts/download_keel_datasets.py")
        return False
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Test all datasets
    datasets = ['yeast1', 'yeast3', 'yeast4', 'yeast5', 'yeast6']
    
    print("KEEL Dataset Test Suite")
    print("=" * 50)
    
    results = {}
    for dataset_name in datasets:
        results[dataset_name] = test_keel_dataset(dataset_name)
        print("\n" + "=" * 50 + "\n")
    
    # Summary
    print("Test Summary:")
    print("=" * 50)
    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {dataset_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")




