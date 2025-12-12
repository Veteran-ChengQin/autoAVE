"""
Test script to verify JSON format changes
Tests the new {attr_name: value} format for single attribute extraction
"""
import json
import re
from typing import Dict


class MockExtractor:
    """Mock extractor to test JSON parsing logic"""
    
    def _extract_value_from_json_response(self, response: str, attr_name: str) -> Dict[str, str]:
        """
        Extract attribute value from JSON-formatted response.
        
        Returns dict with attr_name as key and extracted value as value.
        """
        response = response.strip()
        
        # Try to parse as JSON
        try:
            # First, try direct JSON parsing
            data = json.loads(response)
            if isinstance(data, dict) and attr_name in data:
                value = str(data[attr_name]).strip()
                return {attr_name: (value if value else "")}
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to extract JSON from response (in case of extra text)
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if isinstance(data, dict) and attr_name in data:
                    value = str(data[attr_name]).strip()
                    return {attr_name: (value if value else "")}
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # If all JSON parsing fails, return empty value
        return {attr_name: ""}


def test_json_parsing():
    """Test JSON parsing with new format"""
    extractor = MockExtractor()
    
    print("\n" + "="*60)
    print("Testing JSON Parsing with {attr_name: value} format")
    print("="*60)
    
    test_cases = [
        # (response, attr_name, expected_result)
        ('{"color": "red"}', "color", {"color": "red"}),
        ('{"color": ""}', "color", {"color": ""}),
        ('Some text {"color": "red"} more', "color", {"color": "red"}),
        ('Invalid JSON', "color", {"color": ""}),
        ('{"size": "large", "color": "red"}', "color", {"color": "red"}),
        ('{"size": "large", "color": "red"}', "size", {"size": "large"}),
        ('{"color": "red color"}', "color", {"color": "red color"}),
        ('{"color": "   red   "}', "color", {"color": "red"}),
    ]
    
    passed = 0
    failed = 0
    
    for response, attr_name, expected in test_cases:
        result = extractor._extract_value_from_json_response(response, attr_name)
        if result == expected:
            print(f"✓ PASS: {response[:40]:40} -> {result}")
            passed += 1
        else:
            print(f"✗ FAIL: {response[:40]:40}")
            print(f"  Expected: {expected}")
            print(f"  Got: {result}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


def test_f1_calculation():
    """Test F1 calculation with new format"""
    from evaluation import compute_fuzzy_f1_scores, custom_fuzzy_match
    
    print("\n" + "="*60)
    print("Testing F1 Calculation")
    print("="*60)
    
    # Test 1: Perfect extraction
    print("\nTest 1: Perfect extraction")
    predictions = ["red", "large", "cotton"]
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Precision: {p}, Recall: {r}, F1: {f1}")
    assert p == 1.0 and r == 1.0 and f1 == 1.0, "Perfect extraction should have F1=1.0"
    print("  ✓ PASS")
    
    # Test 2: Some attributes not extracted
    print("\nTest 2: Some attributes not extracted (empty predictions)")
    predictions = ["red", "", "cotton"]
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Precision: {p}, Recall: {r}, F1: {f1}")
    print(f"  Attr F1: {attr_f1}")
    assert r < 1.0, "Should have lower recall when some attributes not extracted"
    print("  ✓ PASS")
    
    # Test 3: Partial match (fuzzy matching)
    print("\nTest 3: Partial match (fuzzy matching)")
    predictions = ["red color", "large", "cotton"]
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names, threshold=0.5)
    print(f"  Precision: {p}, Recall: {r}, F1: {f1}")
    print(f"  Attr F1: {attr_f1}")
    # "red color" vs "red": common prefix "red" = 3, threshold = 0.5*3 = 1.5, 3 >= 1.5 -> match
    assert attr_f1["color"] > 0.5, "Partial match should have reasonable F1"
    print("  ✓ PASS")
    
    print("\n" + "="*60)
    print("All F1 tests passed!")
    print("="*60)
    
    return True


def test_integration():
    """Test integration of extraction and evaluation"""
    from evaluation import compute_fuzzy_f1_scores
    
    print("\n" + "="*60)
    print("Testing Integration: Extraction -> Evaluation")
    print("="*60)
    
    extractor = MockExtractor()
    
    # Simulate extraction results
    print("\nSimulating extraction results...")
    
    # Sample 1: Successful extraction
    response1 = '{"color": "red"}'
    result1 = extractor._extract_value_from_json_response(response1, "color")
    print(f"  Sample 1: {response1} -> {result1}")
    
    # Sample 2: Failed extraction
    response2 = 'Cannot determine color'
    result2 = extractor._extract_value_from_json_response(response2, "color")
    print(f"  Sample 2: {response2} -> {result2}")
    
    # Sample 3: Partial match
    response3 = '{"color": "red color"}'
    result3 = extractor._extract_value_from_json_response(response3, "color")
    print(f"  Sample 3: {response3} -> {result3}")
    
    # Evaluate
    print("\nEvaluating extraction results...")
    predictions = [result1["color"], result2["color"], result3["color"]]
    labels = ["red", "red", "red"]
    attr_names = ["color", "color", "color"]
    
    p, r, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Attr F1: {attr_f1}")
    
    print("\n" + "="*60)
    print("Integration test passed!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("JSON FORMAT VERIFICATION TESTS")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_json_parsing()
    all_passed &= test_f1_calculation()
    all_passed &= test_integration()
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
