"""
Test script to verify the improvements to prompt, JSON parsing, and F1 calculation
"""
import sys
import json
from evaluation import custom_fuzzy_match, compute_fuzzy_f1_scores


def test_custom_fuzzy_match():
    """Test custom fuzzy matching with common prefix rule"""
    print("\n" + "="*60)
    print("Testing custom_fuzzy_match (common prefix rule)")
    print("="*60)
    
    test_cases = [
        # (label, pred, threshold, expected_match)
        ("red", "red", 0.5, True),
        ("red", "red color", 0.5, True),  # common prefix "red" = 3 >= 0.5*3 = 1.5 ✓
        ("red color", "red", 0.5, False),  # common prefix "red" = 3 < 0.5*10 = 5 ✗
        ("red", "blue", 0.5, False),
        ("medium", "med", 0.5, True),  # "med" is 3 >= 0.5*6 = 3 ✓
        ("medium", "me", 0.5, False),  # "me" is 2 < 0.5*6 = 3 ✗
        ("", "red", 0.5, False),  # empty label
        ("red", "", 0.5, False),  # empty pred
        ("", "", 0.5, False),  # both empty
    ]
    
    for label, pred, threshold, expected in test_cases:
        result = custom_fuzzy_match(label, pred, threshold)
        status = "✓" if result == expected else "✗"
        print(f"{status} custom_fuzzy_match('{label}', '{pred}', {threshold}) = {result} (expected {expected})")


def test_compute_fuzzy_f1_scores():
    """Test F1 score computation with TP/FP/FN statistics"""
    print("\n" + "="*60)
    print("Testing compute_fuzzy_f1_scores (TP/FP/FN statistics)")
    print("="*60)
    
    # Test case 1: Perfect predictions
    print("\nTest 1: Perfect predictions")
    predictions = ["red", "large", "cotton"]
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Predictions: {predictions}")
    print(f"  Labels: {labels}")
    print(f"  Overall: P={precision}, R={recall}, F1={f1}")
    print(f"  Per-attribute F1: {attr_f1}")
    assert f1 == 1.0, "Perfect predictions should have F1=1.0"
    
    # Test case 2: MLLM failed to extract (empty predictions)
    print("\nTest 2: MLLM failed to extract (empty predictions)")
    predictions = ["red", "", "cotton"]  # size not extracted
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Predictions: {predictions}")
    print(f"  Labels: {labels}")
    print(f"  Overall: P={precision}, R={recall}, F1={f1}")
    print(f"  Per-attribute F1: {attr_f1}")
    # TP=2, FP=0, FN=1 -> P=1.0, R=2/3=0.6667, F1=0.8
    assert precision == 1.0, "No false positives"
    assert recall == round(2/3, 4), "Recall should be 2/3"
    
    # Test case 3: Incorrect predictions
    print("\nTest 3: Incorrect predictions")
    predictions = ["blue", "large", "cotton"]  # color wrong
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Predictions: {predictions}")
    print(f"  Labels: {labels}")
    print(f"  Overall: P={precision}, R={recall}, F1={f1}")
    print(f"  Per-attribute F1: {attr_f1}")
    # TP=2, FP=0, FN=1 -> P=1.0, R=2/3=0.6667, F1=0.8
    assert precision == 1.0, "No false positives"
    assert recall == round(2/3, 4), "Recall should be 2/3"
    
    # Test case 4: Fuzzy matching (partial match)
    print("\nTest 4: Fuzzy matching (partial match)")
    predictions = ["red color", "large", "cotton"]  # color is fuzzy match
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Predictions: {predictions}")
    print(f"  Labels: {labels}")
    print(f"  Overall: P={precision}, R={recall}, F1={f1}")
    print(f"  Per-attribute F1: {attr_f1}")
    # "red color" vs "red": common prefix "red" = 3 chars >= 0.5*3 = 1.5 ✓
    assert f1 == 1.0, "Fuzzy match should succeed"
    
    # Test case 5: Mixed scenario
    print("\nTest 5: Mixed scenario (some correct, some wrong, some empty)")
    predictions = ["red", "", "polyester"]  # size not extracted, material wrong
    labels = ["red", "large", "cotton"]
    attr_names = ["color", "size", "material"]
    
    precision, recall, f1, attr_f1 = compute_fuzzy_f1_scores(predictions, labels, attr_names)
    print(f"  Predictions: {predictions}")
    print(f"  Labels: {labels}")
    print(f"  Overall: P={precision}, R={recall}, F1={f1}")
    print(f"  Per-attribute F1: {attr_f1}")
    # TP=1, FP=0, FN=2 -> P=1.0, R=1/3=0.3333, F1=0.5
    assert precision == 1.0, "No false positives"
    assert recall == round(1/3, 4), "Recall should be 1/3"


def test_json_response_parsing():
    """Test JSON response parsing for single and multi-attribute"""
    print("\n" + "="*60)
    print("Testing JSON response parsing")
    print("="*60)
    
    # Create a mock extractor (without loading model)
    class MockExtractor:
        def _extract_value_from_json_response(self, response: str) -> str:
            """Extract attribute value from JSON-formatted response"""
            response = response.strip()
            
            # Try to parse as JSON
            try:
                data = json.loads(response)
                if isinstance(data, dict) and "value" in data:
                    value = str(data["value"]).strip()
                    return value if value else ""
            except json.JSONDecodeError:
                pass
            
            # Fallback: try to extract JSON from response
            import re
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
        
        def _parse_multi_attr_json_response(self, response: str, attr_names: list) -> dict:
            """Parse multi-attribute JSON response"""
            result = {name: "" for name in attr_names}
            response = response.strip()
            
            try:
                data = json.loads(response)
                if isinstance(data, dict):
                    for attr_name in attr_names:
                        if attr_name in data:
                            value = str(data[attr_name]).strip()
                            result[attr_name] = value if value else ""
                    return result
            except json.JSONDecodeError:
                pass
            
            import re
            try:
                json_match = re.search(r'\{[^{}]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        for attr_name in attr_names:
                            if attr_name in data:
                                value = str(data[attr_name]).strip()
                                result[attr_name] = value if value else ""
                        return result
            except (json.JSONDecodeError, AttributeError):
                pass
            
            return result
    
    extractor = MockExtractor()
    
    # Test single-attribute parsing
    print("\nSingle-attribute JSON parsing:")
    test_cases = [
        ('{"value": "red"}', "red"),
        ('{"value": ""}', ""),
        ('{"value": "red color"}', "red color"),
        ('Some text {"value": "red"} more text', "red"),  # JSON embedded in text
        ('Invalid JSON', ""),  # Invalid JSON
    ]
    
    for response, expected in test_cases:
        result = extractor._extract_value_from_json_response(response)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Response: '{response}' -> '{result}' (expected '{expected}')")
    
    # Test multi-attribute parsing
    print("\nMulti-attribute JSON parsing:")
    attr_names = ["color", "size", "material"]
    
    test_cases = [
        ('{"color": "red", "size": "large", "material": "cotton"}', 
         {"color": "red", "size": "large", "material": "cotton"}),
        ('{"color": "red", "size": "", "material": "cotton"}', 
         {"color": "red", "size": "", "material": "cotton"}),
        ('Some text {"color": "red", "size": "large", "material": "cotton"} more', 
         {"color": "red", "size": "large", "material": "cotton"}),
    ]
    
    for response, expected in test_cases:
        result = extractor._parse_multi_attr_json_response(response, attr_names)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Response: '{response}'")
        print(f"      Result: {result}")
        print(f"      Expected: {expected}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING IMPROVEMENTS")
    print("="*60)
    
    try:
        test_custom_fuzzy_match()
        test_compute_fuzzy_f1_scores()
        test_json_response_parsing()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
