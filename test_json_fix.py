#!/usr/bin/env python3
"""
Test script to verify the JSON parsing fix for malformed responses.
"""

import json
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_value_from_json_response(response: str, attr_name: str):
    """
    Test version of the fixed JSON extraction method.
    """
    import ast
    
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
            try:
                data = ast.literal_eval(json_str)
                
                # Handle set (which occurs when JSON has escaped quotes)
                if isinstance(data, set):
                    result = {}
                    for item in data:
                        try:
                            item_json = json.loads("{" + item + "}")
                            result.update(item_json)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    if result:
                        return result
                
                # Handle dict
                elif isinstance(data, dict):
                    result = {}
                    for key, value in data.items():
                        if isinstance(value, str):
                            try:
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
    
    # Step 4: Handle malformed JSON with manual parsing
    try:
        # Look for key-value patterns in the response
        # Handle cases like: "\"Brand\"": "\"Osensia\"\""
        # or: "\"Item Form\"": "\"box of lipsticks and eyeshadow palette"
        
        # Find all potential key-value pairs
        # Pattern: optional quotes + escaped quotes + key + escaped quotes + optional quotes + colon + space + value
        lines = response.split('\n')
        result = {}
        
        for line in lines:
            line = line.strip()
            if ':' in line and (line.startswith('"') or line.strip().startswith('"')):
                # Split on the first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Clean up key: remove outer quotes and unescape inner quotes
                    # Handle: "\"Brand\"" -> Brand
                    key = key_part.strip('"').strip("'")
                    if key.startswith('\\"') and key.endswith('\\"'):
                        key = key[2:-2]  # Remove \" from both ends
                    key = key.replace('\\"', '"')
                    
                    # Clean up value: remove quotes and handle malformed endings
                    # Handle: "\"Osensia\"\"" -> Osensia
                    # Handle: "\"box of lipsticks and eyeshadow palette" -> box of lipsticks and eyeshadow palette
                    value = value_part.strip().rstrip(',').rstrip('}').strip()
                    value = value.strip('"').strip("'")
                    if value.startswith('\\"'):
                        value = value[2:]  # Remove leading \"
                    if value.endswith('\\"'):
                        value = value[:-2]  # Remove trailing \"
                    # Handle extra quotes at the end
                    while value.endswith('"') and not value.endswith('\\"'):
                        value = value[:-1]
                    value = value.replace('\\"', '"')
                    
                    if key:  # Only add if key is not empty
                        result[key] = value
        
        if result:
            return result
    except (AttributeError, TypeError, IndexError):
        pass
    
    # Step 5: Handle escaped quotes manually as last resort
    try:
        # Replace escaped quotes with regular quotes
        response_unescaped = response.replace('\\"', '"')
        # Try to parse
        data = json.loads(response_unescaped)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Step 6: If all JSON parsing fails, return empty value
    logger.warning(f"Failed to parse JSON response for {attr_name}: {response}")
    return {attr_name: ""}

def test_malformed_json():
    """Test the fix with the provided malformed JSON examples."""
    
    # Test case 1: Extra quote at end of value (from actual log)
    test1 = '{\n    "\\"Brand\\"": "\\"Osensia\\"\\"\n}'
    print(f"Test 1 input: {repr(test1)}")
    result1 = _extract_value_from_json_response(test1, "Brand")
    print(f"Test 1 result: {result1}")
    print()
    
    # Test case 2: Missing closing quote (from actual log)
    test2 = '{\n    "\\"Item Form\\"": "\\"box of lipsticks and eyeshadow palette"\n}'
    print(f"Test 2 input: {repr(test2)}")
    result2 = _extract_value_from_json_response(test2, "Item Form")
    print(f"Test 2 result: {result2}")
    print()
    
    # Test case 3: Valid JSON (should still work)
    test3 = '{"Brand": "Nike"}'
    print(f"Test 3 input: {repr(test3)}")
    result3 = _extract_value_from_json_response(test3, "Brand")
    print(f"Test 3 result: {result3}")
    print()
    
    # Debug: Let's see what the regex actually matches
    import re
    pattern = r'["\']?\\*"?([^"\\:]+?)\\*"?["\']?\s*:\s*["\']?\\*"?([^"\\}]*?)\\*"*["\']?'
    print("Debug regex matches:")
    print("Test 1 matches:", re.findall(pattern, test1))
    print("Test 2 matches:", re.findall(pattern, test2))

if __name__ == "__main__":
    test_malformed_json()
