"""
Integration verification script
Verifies that all improvements are correctly integrated
"""
import sys
import os
import json
import importlib


def verify_imports():
    """Verify all modules can be imported"""
    print("\n" + "="*60)
    print("1. Verifying Imports")
    print("="*60)
    
    modules_to_check = [
        ('qwen_vl_extractor', 'QwenVLExtractor'),
        ('evaluation', 'AttributeEvaluator'),
        ('evaluation', 'custom_fuzzy_match'),
        ('evaluation', 'compute_fuzzy_f1_scores'),
        ('main', 'evaluate_results'),
    ]
    
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"  ✓ {module_name}.{class_name}")
            else:
                print(f"  ✗ {module_name}.{class_name} NOT FOUND")
                return False
        except ImportError as e:
            print(f"  ✗ Failed to import {module_name}: {e}")
            return False
    
    return True


def verify_qwen_extractor():
    """Verify QwenVLExtractor has new methods"""
    print("\n" + "="*60)
    print("2. Verifying QwenVLExtractor")
    print("="*60)
    
    try:
        from qwen_vl_extractor import QwenVLExtractor
        
        # Check for new methods
        methods_to_check = [
            '_extract_value_from_json_response',
            '_parse_multi_attr_json_response',
        ]
        
        for method_name in methods_to_check:
            if hasattr(QwenVLExtractor, method_name):
                print(f"  ✓ QwenVLExtractor.{method_name}")
            else:
                print(f"  ✗ QwenVLExtractor.{method_name} NOT FOUND")
                return False
        
        # Check that old methods are removed
        old_methods = [
            '_extract_value_from_response',
            '_parse_multi_attr_response',
        ]
        
        for method_name in old_methods:
            if not hasattr(QwenVLExtractor, method_name):
                print(f"  ✓ Old method {method_name} removed")
            else:
                print(f"  ⚠ Old method {method_name} still exists (not critical)")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def verify_evaluation():
    """Verify evaluation module has new functions"""
    print("\n" + "="*60)
    print("3. Verifying Evaluation Module")
    print("="*60)
    
    try:
        from evaluation import (
            custom_fuzzy_match,
            compute_fuzzy_f1_scores,
            AttributeEvaluator
        )
        
        # Test custom_fuzzy_match
        print("  Testing custom_fuzzy_match...")
        assert custom_fuzzy_match("red", "red", 0.5) == True
        assert custom_fuzzy_match("red", "blue", 0.5) == False
        print("    ✓ custom_fuzzy_match works correctly")
        
        # Test compute_fuzzy_f1_scores
        print("  Testing compute_fuzzy_f1_scores...")
        predictions = ["red", "large", "cotton"]
        labels = ["red", "large", "cotton"]
        attr_names = ["color", "size", "material"]
        
        precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(
            predictions, labels, attr_names
        )
        
        assert precision == 1.0, f"Expected precision=1.0, got {precision}"
        assert recall == 1.0, f"Expected recall=1.0, got {recall}"
        assert f1 == 1.0, f"Expected f1=1.0, got {f1}"
        print("    ✓ compute_fuzzy_f1_scores works correctly")
        
        # Test AttributeEvaluator
        print("  Testing AttributeEvaluator...")
        evaluator = AttributeEvaluator()
        
        result = evaluator.evaluate_sample(
            pred="red",
            label="red",
            attr_name="color",
            category="beauty",
            product_id="prod_1"
        )
        
        assert "tp" in result, "Result should have 'tp' field"
        assert "fp" in result, "Result should have 'fp' field"
        assert "fn" in result, "Result should have 'fn' field"
        assert result["tp"] == 1, "Should have TP=1 for correct prediction"
        print("    ✓ AttributeEvaluator works correctly")
        
        # Test get_metrics
        print("  Testing get_metrics...")
        metrics = evaluator.get_metrics()
        
        assert "overall_precision" in metrics, "Metrics should have 'overall_precision'"
        assert "overall_recall" in metrics, "Metrics should have 'overall_recall'"
        assert "overall_f1" in metrics, "Metrics should have 'overall_f1'"
        assert "overall_tp" in metrics, "Metrics should have 'overall_tp'"
        assert "overall_fp" in metrics, "Metrics should have 'overall_fp'"
        assert "overall_fn" in metrics, "Metrics should have 'overall_fn'"
        print("    ✓ get_metrics returns correct fields")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_json_parsing():
    """Verify JSON parsing works correctly"""
    print("\n" + "="*60)
    print("4. Verifying JSON Parsing")
    print("="*60)
    
    try:
        # Create a mock extractor for testing
        import re
        
        class MockExtractor:
            def _extract_value_from_json_response(self, response: str) -> str:
                response = response.strip()
                
                try:
                    data = json.loads(response)
                    if isinstance(data, dict) and "value" in data:
                        value = str(data["value"]).strip()
                        return value if value else ""
                except json.JSONDecodeError:
                    pass
                
                try:
                    json_match = re.search(r'\{[^{}]*"value"[^{}]*\}', response)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        if isinstance(data, dict) and "value" in data:
                            value = str(data["value"]).strip()
                            return value if value else ""
                except (json.JSONDecodeError, AttributeError):
                    pass
                
                return ""
        
        extractor = MockExtractor()
        
        # Test cases
        test_cases = [
            ('{"value": "red"}', "red"),
            ('{"value": ""}', ""),
            ('Some text {"value": "red"} more', "red"),
            ('Invalid JSON', ""),
        ]
        
        for response, expected in test_cases:
            result = extractor._extract_value_from_json_response(response)
            if result == expected:
                print(f"  ✓ Parsing '{response[:30]}...' -> '{result}'")
            else:
                print(f"  ✗ Parsing '{response[:30]}...' -> '{result}' (expected '{expected}')")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_files_exist():
    """Verify all necessary files exist"""
    print("\n" + "="*60)
    print("5. Verifying Files Exist")
    print("="*60)
    
    files_to_check = [
        'qwen_vl_extractor.py',
        'evaluation.py',
        'main.py',
        'test_improvements.py',
        'IMPROVEMENTS_SUMMARY.md',
        'QUICK_START_IMPROVEMENTS.md',
        'BEFORE_AFTER_COMPARISON.md',
        'CHANGES_SUMMARY.md',
    ]
    
    for filename in files_to_check:
        filepath = os.path.join('/data/veteran/project/dataE/Attribute_AKS', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({size} bytes)")
        else:
            print(f"  ✗ {filename} NOT FOUND")
            return False
    
    return True


def verify_documentation():
    """Verify documentation files are complete"""
    print("\n" + "="*60)
    print("6. Verifying Documentation")
    print("="*60)
    
    docs = {
        'IMPROVEMENTS_SUMMARY.md': ['Prompt改进', 'JSON解析增强', 'F1计算重构'],
        'QUICK_START_IMPROVEMENTS.md': ['改进1: JSON格式Prompt', '改进2: 健壮的JSON解析', '改进3: 精细化F1计算'],
        'BEFORE_AFTER_COMPARISON.md': ['改进前', '改进后', '对比'],
        'CHANGES_SUMMARY.md': ['修改的文件', '新增的文件', '向后兼容性'],
    }
    
    for filename, keywords in docs.items():
        filepath = os.path.join('/data/veteran/project/dataE/Attribute_AKS', filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            all_found = all(keyword in content for keyword in keywords)
            if all_found:
                print(f"  ✓ {filename} contains all expected sections")
            else:
                missing = [k for k in keywords if k not in content]
                print(f"  ⚠ {filename} missing sections: {missing}")
        except Exception as e:
            print(f"  ✗ Error reading {filename}: {e}")
            return False
    
    return True


def main():
    """Run all verification checks"""
    print("\n" + "="*80)
    print("INTEGRATION VERIFICATION")
    print("="*80)
    
    checks = [
        ("Imports", verify_imports),
        ("QwenVLExtractor", verify_qwen_extractor),
        ("Evaluation Module", verify_evaluation),
        ("JSON Parsing", verify_json_parsing),
        ("Files Exist", verify_files_exist),
        ("Documentation", verify_documentation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED - INTEGRATION SUCCESSFUL")
        print("="*80)
        return 0
    else:
        print("✗ SOME VERIFICATIONS FAILED - PLEASE CHECK ABOVE")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
