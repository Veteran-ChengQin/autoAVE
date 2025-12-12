#!/usr/bin/env python3
"""
Test script to verify JSON parsing fix for escaped quotes and Markdown code blocks.
"""

import json
import re
import ast
from typing import Dict, List


def _extract_value_from_json_response(response: str, attr_name: str) -> Dict[str, str]:
    """
    Extract attribute value from JSON-formatted response.
    
    Handles cases where MLLM fails to extract the target attribute.
    Returns dict with attr_name as key and extracted value as value.
    """
    response = response.strip()
    
    # Step 1: Remove Markdown code blocks (```json ... ```)
    response = re.sub(r'```(?:json)?\s*\n?', '', response)
    response = response.strip()
    
    # Step 2: Try to parse as JSON directly
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    
    # Step 3: Try to extract JSON object from response (in case of extra text)
    try:
        # Look for JSON object pattern
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            json_str = json_match.group(0)
            # Try direct JSON parsing first
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
            
            # If direct parsing fails, try ast.literal_eval
            # This handles cases where the JSON is wrapped in escaped quotes
            # e.g., {"\"Color\": \"black\""} parses as a set in Python
            try:
                data = ast.literal_eval(json_str)
                
                # Handle set (which occurs when JSON has escaped quotes)
                if isinstance(data, set):
                    # Combine all set elements into a single JSON object
                    result = {}
                    for item in data:
                        # Each item is a string like "key": "value"
                        # Wrap it in braces to make it valid JSON
                        try:
                            item_json = json.loads("{" + item + "}")
                            result.update(item_json)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if result:
                        return result
                
                # Handle dict
                elif isinstance(data, dict):
                    # If we got a dict from literal_eval, parse the inner JSON if it's a string
                    result = {}
                    for key, value in data.items():
                        if isinstance(value, str):
                            try:
                                # Try to parse string values as JSON
                                result[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                result[key] = value
                        else:
                            result[key] = value
                    return result
            except (ValueError, SyntaxError, TypeError):
                pass
    except (AttributeError, TypeError):
        pass
    
    # Step 4: Handle escaped quotes manually as last resort
    try:
        # Replace escaped quotes with regular quotes
        response_unescaped = response.replace('\\"', '"')
        # Try to parse
        data = json.loads(response_unescaped)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Step 5: If all JSON parsing fails, return empty value (attribute not extracted)
    print(f"Failed to parse JSON response for {attr_name}: {response}")
    return {attr_name: ""}


def test_parsing():
    """Test various JSON parsing scenarios."""
    
    test_cases = [
        # Case 1: Original failing case - Markdown code block with escaped quotes
        {
            "name": "Markdown code block with escaped quotes",
            "response": '```json\n{\n    "\\"Color\\": \\"black\\""\n}\n```',
            "attr_name": "Color",
            "expected": {"Color": "black"}
        },
        # Case 2: Simple JSON without Markdown
        {
            "name": "Simple JSON",
            "response": '{"color": "red"}',
            "attr_name": "color",
            "expected": {"color": "red"}
        },
        # Case 3: JSON with extra text
        {
            "name": "JSON with extra text",
            "response": 'Here is the result: {"size": "large"} Done.',
            "attr_name": "size",
            "expected": {"size": "large"}
        },
        # Case 4: Markdown code block without escaped quotes
        {
            "name": "Markdown code block",
            "response": '```json\n{"material": "cotton"}\n```',
            "attr_name": "material",
            "expected": {"material": "cotton"}
        },
        # Case 5: Multiple escaped quotes
        {
            "name": "Multiple escaped quotes",
            "response": '{\n    "\\"brand\\": \\"Nike\\"",\n    "\\"model\\": \\"Air Max\\""\n}',
            "attr_name": "brand",
            "expected": {"brand": "Nike", "model": "Air Max"}
        },
        # Case 6: Empty value
        {
            "name": "Empty value",
            "response": '{"color": ""}',
            "attr_name": "color",
            "expected": {"color": ""}
        },
        # Case 7: Markdown with escaped quotes and extra text
        {
            "name": "Markdown with escaped quotes and extra text",
            "response": 'The result is:\n```json\n{\n    "\\"price\\": \\"99.99\\""\n}\n```\nEnd of response',
            "attr_name": "price",
            "expected": {"price": "99.99"}
        },
    ]
    
    print("=" * 80)
    print("Testing JSON Parsing Fix")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {repr(test_case['response'][:100])}")
        
        result = _extract_value_from_json_response(test_case['response'], test_case['attr_name'])
        expected = test_case['expected']
        
        # Check if result matches expected
        if result == expected:
            print(f"✓ PASSED")
            print(f"  Result: {result}")
            passed += 1
        else:
            print(f"✗ FAILED")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = test_parsing()
    exit(0 if success else 1)
