#!/usr/bin/env python3
"""
Test script to verify the evaluation function works with dict predictions.
"""

import sys
sys.path.insert(0, '/data/veteran/project/dataE/Attribute_AKS')

from evaluation import compute_fuzzy_f1_scores

def test_dict_predictions():
    """Test evaluation with dict predictions like {'Color': 'black'}"""
    
    print("=" * 80)
    print("Testing Evaluation with Dict Predictions")
    print("=" * 80)
    
    # Test case 1: Simple dict predictions
    print("\n【Test 1】Simple dict predictions")
    print("-" * 80)
    
    predictions = [
        {'Color': 'black'},
        {'Color': 'white'},
        {'Color': 'red'},
        {'Color': ''},  # Empty prediction
    ]
    
    labels = [
        'black',
        'white',
        'blue',
        'green',
    ]
    
    attr_names = ['Color', 'Color', 'Color', 'Color']
    
    precision, recall, f1, attr_f1_scores = compute_fuzzy_f1_scores(
        predictions, labels, attr_names, threshold=0.5
    )
    
    print(f"Predictions: {predictions}")
    print(f"Labels:      {labels}")
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Per-Attr:  {attr_f1_scores}")
    
    # Expected: 2 TP (black, white), 1 FN (red vs blue), 1 FN (empty)
    # Precision = 2 / (2 + 0) = 1.0
    # Recall = 2 / (2 + 2) = 0.5
    # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.6667
    
    # Test case 2: Mixed dict and string predictions
    print("\n【Test 2】Mixed dict and string predictions")
    print("-" * 80)
    
    predictions = [
        {'Size': 'large'},
        'medium',  # String instead of dict
        {'Size': 'small'},
    ]
    
    labels = [
        'large',
        'medium',
        'small',
    ]
    
    attr_names = ['Size', 'Size', 'Size']
    
    precision, recall, f1, attr_f1_scores = compute_fuzzy_f1_scores(
        predictions, labels, attr_names, threshold=0.5
    )
    
    print(f"Predictions: {predictions}")
    print(f"Labels:      {labels}")
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Per-Attr:  {attr_f1_scores}")
    
    # Expected: 3 TP (all match)
    # Precision = 3 / 3 = 1.0
    # Recall = 3 / 3 = 1.0
    # F1 = 1.0
    
    # Test case 3: Multiple attributes
    print("\n【Test 3】Multiple attributes")
    print("-" * 80)
    
    predictions = [
        {'Color': 'black', 'Size': 'large'},
        {'Color': 'white', 'Size': 'medium'},
        {'Color': 'red', 'Size': 'small'},
    ]
    
    labels = [
        'black',
        'white',
        'blue',
    ]
    
    attr_names = ['Color', 'Color', 'Color']
    
    precision, recall, f1, attr_f1_scores = compute_fuzzy_f1_scores(
        predictions, labels, attr_names, threshold=0.5
    )
    
    print(f"Predictions: {predictions}")
    print(f"Labels:      {labels}")
    print(f"Attr Names:  {attr_names}")
    print(f"\nResults:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Per-Attr:  {attr_f1_scores}")
    
    # Expected: 2 TP (black, white), 1 FN (red vs blue)
    # Precision = 2 / 2 = 1.0
    # Recall = 2 / 3 = 0.6667
    # F1 = 2 * 1.0 * 0.6667 / (1.0 + 0.6667) = 0.8
    
    # Test case 4: Real-world example from main.py
    print("\n【Test 4】Real-world example (from main.py)")
    print("-" * 80)
    
    results = [
        {
            'product_id': 'B07JLCR327',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'black'},
            'label': 'White',
            'num_keyframes': 8,
        },
        {
            'product_id': 'B07JLCR328',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'white'},
            'label': 'white',
            'num_keyframes': 6,
        },
        {
            'product_id': 'B07JLCR329',
            'category': 'beauty',
            'attr_name': 'Size',
            'pred': {'Size': 'large'},
            'label': 'large',
            'num_keyframes': 5,
        },
    ]
    
    # Extract data
    predictions = [r['pred'] for r in results if 'error' not in r]
    labels = [r['label'] for r in results if 'error' not in r]
    attr_names = [r['attr_name'] for r in results if 'error' not in r]
    
    precision, recall, f1, attr_f1_scores = compute_fuzzy_f1_scores(
        predictions, labels, attr_names, threshold=0.5
    )
    
    print(f"Results count: {len(results)}")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Per-Attr:  {attr_f1_scores}")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_dict_predictions()
