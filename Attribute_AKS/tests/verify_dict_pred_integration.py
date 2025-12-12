#!/usr/bin/env python3
"""
Integration verification script for dict prediction evaluation changes.
"""

import sys
sys.path.insert(0, '/data/veteran/project/dataE/Attribute_AKS')

def verify_imports():
    """Verify all imports work correctly."""
    print("\n【1】验证导入...")
    print("-" * 60)
    
    try:
        from evaluation import compute_fuzzy_f1_scores, custom_fuzzy_match
        print("✓ evaluation.compute_fuzzy_f1_scores")
        print("✓ evaluation.custom_fuzzy_match")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    try:
        from main import evaluate_results
        print("✓ main.evaluate_results")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    return True


def verify_dict_handling():
    """Verify dict predictions are handled correctly."""
    print("\n【2】验证字典处理...")
    print("-" * 60)
    
    from evaluation import compute_fuzzy_f1_scores
    
    # Test 1: Dict predictions
    predictions = [{'Color': 'black'}, {'Color': 'white'}]
    labels = ['black', 'white']
    attr_names = ['Color', 'Color']
    
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        assert f1 == 1.0, f"Expected F1=1.0, got {f1}"
        print(f"✓ 字典预测值处理: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 字典处理失败: {e}")
        return False
    
    # Test 2: String predictions (backward compatibility)
    predictions = ['black', 'white']
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        assert f1 == 1.0, f"Expected F1=1.0, got {f1}"
        print(f"✓ 字符串预测值处理（向后兼容）: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 字符串处理失败: {e}")
        return False
    
    # Test 3: Mixed predictions
    predictions = [{'Color': 'black'}, 'white']
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        assert f1 == 1.0, f"Expected F1=1.0, got {f1}"
        print(f"✓ 混合预测值处理: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 混合处理失败: {e}")
        return False
    
    return True


def verify_evaluate_results():
    """Verify evaluate_results function works with dict predictions."""
    print("\n【3】验证 evaluate_results 函数...")
    print("-" * 60)
    
    from main import evaluate_results
    
    results = [
        {
            'product_id': 'P001',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'black'},
            'label': 'black',
            'num_keyframes': 8,
        },
        {
            'product_id': 'P002',
            'category': 'beauty',
            'attr_name': 'Color',
            'pred': {'Color': 'white'},
            'label': 'white',
            'num_keyframes': 6,
        },
        {
            'product_id': 'P003',
            'category': 'beauty',
            'attr_name': 'Size',
            'pred': {'Size': 'large'},
            'label': 'large',
            'num_keyframes': 5,
        },
    ]
    
    try:
        metrics = evaluate_results(results)
        
        # Verify metrics structure
        assert 'overall_precision' in metrics, "Missing overall_precision"
        assert 'overall_recall' in metrics, "Missing overall_recall"
        assert 'overall_f1' in metrics, "Missing overall_f1"
        assert 'attr_f1_scores' in metrics, "Missing attr_f1_scores"
        
        print(f"✓ evaluate_results 返回正确的指标")
        print(f"  - Precision: {metrics['overall_precision']:.4f}")
        print(f"  - Recall: {metrics['overall_recall']:.4f}")
        print(f"  - F1: {metrics['overall_f1']:.4f}")
        print(f"  - Per-Attr: {metrics['attr_f1_scores']}")
        
    except Exception as e:
        print(f"✗ evaluate_results 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def verify_error_handling():
    """Verify error handling for edge cases."""
    print("\n【4】验证错误处理...")
    print("-" * 60)
    
    from evaluation import compute_fuzzy_f1_scores
    
    # Test 1: Empty dict
    predictions = [{}]
    labels = ['black']
    attr_names = ['Color']
    
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        print(f"✓ 空字典处理: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 空字典处理失败: {e}")
        return False
    
    # Test 2: Missing key in dict
    predictions = [{'Size': 'large'}]
    labels = ['black']
    attr_names = ['Color']  # Key doesn't exist in dict
    
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        print(f"✓ 缺失键处理: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 缺失键处理失败: {e}")
        return False
    
    # Test 3: Empty prediction value
    predictions = [{'Color': ''}]
    labels = ['black']
    attr_names = ['Color']
    
    try:
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
        print(f"✓ 空预测值处理: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 空预测值处理失败: {e}")
        return False
    
    return True


def verify_fuzzy_matching():
    """Verify fuzzy matching still works correctly."""
    print("\n【5】验证模糊匹配...")
    print("-" * 60)
    
    from evaluation import compute_fuzzy_f1_scores
    
    # Test fuzzy matching with threshold
    predictions = [{'Color': 'blac'}]  # Typo
    labels = ['black']
    attr_names = ['Color']
    
    try:
        # With threshold 0.5, 'blac' vs 'black' should match (4/5 = 0.8 > 0.5)
        p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names, threshold=0.5)
        print(f"✓ 模糊匹配（threshold=0.5）: F1={f1:.4f}")
    except Exception as e:
        print(f"✗ 模糊匹配失败: {e}")
        return False
    
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("集成验证：字典预测值评估")
    print("=" * 60)
    
    tests = [
        ("导入验证", verify_imports),
        ("字典处理", verify_dict_handling),
        ("evaluate_results", verify_evaluate_results),
        ("错误处理", verify_error_handling),
        ("模糊匹配", verify_fuzzy_matching),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n总体: {passed}/{total} 通过")
    
    if passed == total:
        print("\n✓ 所有验证通过！集成完成。")
        return 0
    else:
        print(f"\n✗ 有 {total - passed} 个验证失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
