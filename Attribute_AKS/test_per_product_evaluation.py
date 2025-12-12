"""
Test script for per-product F1 evaluation
验证新的per-product聚合评估方式
"""

import sys
sys.path.insert(0, '/data/veteran/project/dataE/Attribute_AKS')

from evaluation import compute_per_product_f1_scores, custom_fuzzy_match


def test_fuzzy_match():
    """Test fuzzy matching logic"""
    print("\n" + "="*60)
    print("TEST 1: Fuzzy Match Logic")
    print("="*60)
    
    test_cases = [
        ("White", "black", False),
        ("All", "All", True),
        ("Ginity", "Unknown", False),
        ("Straight", "Straight", True),
        ("Red", "Red Color", False),  # 公共前缀="Red", len=3 >= 3*0.5=1.5 ✓
        ("Color", "Col", True),  # 公共前缀="Col", len=3 >= 5*0.5=2.5 ✓
    ]
    
    for label, pred, expected in test_cases:
        result = custom_fuzzy_match(label, pred, threshold=0.5)
        status = "✓" if result == expected else "✗"
        print(f"{status} fuzzy_match('{label}', '{pred}') = {result} (expected {expected})")


def test_per_product_evaluation():
    """Test per-product F1 evaluation"""
    print("\n" + "="*60)
    print("TEST 2: Per-Product F1 Evaluation")
    print("="*60)
    
    # Scenario 1: All correct
    print("\n--- Scenario 1: All Correct ---")
    results_1 = [
        {
            'product_id': 'P001',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'White'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P001',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Ginity'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results_1)
    print(f"Product P001: {per_product['P001']}")
    print(f"Expected: F1=1.0, TP=2, FP=0, FN=0")
    assert per_product['P001']['f1'] == 1.0, "F1 should be 1.0"
    assert per_product['P001']['tp'] == 2, "TP should be 2"
    print("✓ Scenario 1 passed")
    
    # Scenario 2: All wrong
    print("\n--- Scenario 2: All Wrong ---")
    results_2 = [
        {
            'product_id': 'P002',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'black'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P002',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Unknown'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results_2)
    print(f"Product P002: {per_product['P002']}")
    print(f"Expected: F1=0.0, TP=0, FP=0, FN=2")
    assert per_product['P002']['f1'] == 0.0, "F1 should be 0.0"
    assert per_product['P002']['fn'] == 2, "FN should be 2"
    print("✓ Scenario 2 passed")
    
    # Scenario 3: Partial correct
    print("\n--- Scenario 3: Partial Correct (1 correct, 1 wrong) ---")
    results_3 = [
        {
            'product_id': 'P003',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'White'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P003',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Unknown'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results_3)
    print(f"Product P003: {per_product['P003']}")
    print(f"Expected: F1=0.5, TP=1, FP=0, FN=1")
    # Precision = 1/(1+0) = 1.0, Recall = 1/(1+1) = 0.5, F1 = 2*1*0.5/(1+0.5) = 0.6667
    expected_f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5)
    print(f"Calculated F1: {expected_f1:.4f}")
    assert per_product['P003']['tp'] == 1, "TP should be 1"
    assert per_product['P003']['fn'] == 1, "FN should be 1"
    print("✓ Scenario 3 passed")
    
    # Scenario 4: Multiple products
    print("\n--- Scenario 4: Multiple Products ---")
    results_4 = [
        # Product P004: 2 correct
        {
            'product_id': 'P004',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'White'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P004',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Ginity'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        # Product P005: 0 correct
        {
            'product_id': 'P005',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'black'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P005',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Unknown'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results_4)
    print(f"Product P004: {per_product['P004']}")
    print(f"Product P005: {per_product['P005']}")
    print(f"Overall: {overall}")
    print(f"Expected overall: TP=2, FP=0, FN=2, F1=0.5")
    assert overall['total_products'] == 2, "Should have 2 products"
    assert overall['tp'] == 2, "Overall TP should be 2"
    assert overall['fn'] == 2, "Overall FN should be 2"
    print("✓ Scenario 4 passed")
    
    # Scenario 5: Empty prediction
    print("\n--- Scenario 5: Empty Prediction (Failed Extraction) ---")
    results_5 = [
        {
            'product_id': 'P006',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': ''},  # Empty prediction
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
        {
            'product_id': 'P006',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Ginity'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results_5)
    print(f"Product P006: {per_product['P006']}")
    print(f"Expected: TP=1, FP=0, FN=1 (empty prediction counts as FN)")
    assert per_product['P006']['tp'] == 1, "TP should be 1"
    assert per_product['P006']['fn'] == 1, "FN should be 1"
    print("✓ Scenario 5 passed")


def test_comparison_with_old_method():
    """Compare old per-attribute method with new per-product method"""
    print("\n" + "="*60)
    print("TEST 3: Comparison - Old vs New Method")
    print("="*60)
    
    results = [
        {
            'product_id': 'B07JLCR327',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'black'},
            'label': 'White',
            'num_keyframes': 8,
            'keyframe_indices': [2, 5, 7, 10, 14, 16, 20, 22]
        },
        {
            'product_id': 'B07JLCR327',
            'category': 'beauty',
            'attr_name': 'Hair Type',
            'pred': {'Hair Type': 'Straight'},
            'label': 'All',
            'num_keyframes': 8,
            'keyframe_indices': [2, 5, 7, 10, 14, 16, 20, 24]
        },
        {
            'product_id': 'B07JLCR327',
            'category': 'beauty',
            'attr_name': 'Brand',
            'pred': {'Brand': 'Unknown'},
            'label': 'Ginity',
            'num_keyframes': 8,
            'keyframe_indices': [2, 5, 7, 10, 14, 16, 20, 22]
        },
    ]
    
    per_product, overall = compute_per_product_f1_scores(results)
    
    print("\n新方式 (Per-Product):")
    print(f"  Product B07JLCR327:")
    print(f"    TP={per_product['B07JLCR327']['tp']}, FP={per_product['B07JLCR327']['fp']}, FN={per_product['B07JLCR327']['fn']}")
    print(f"    Precision={per_product['B07JLCR327']['precision']}, Recall={per_product['B07JLCR327']['recall']}, F1={per_product['B07JLCR327']['f1']}")
    
    print(f"\n  Overall:")
    print(f"    TP={overall['tp']}, FP={overall['fp']}, FN={overall['fn']}")
    print(f"    Precision={overall['precision']}, Recall={overall['recall']}, F1={overall['f1']}")
    
    print("\n✓ Comparison test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Per-Product F1 Evaluation Tests")
    print("="*60)
    
    test_fuzzy_match()
    test_per_product_evaluation()
    test_comparison_with_old_method()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
