#!/usr/bin/env python3
"""Debug - correct approach."""

import json
import re
import ast

response = '```json\n{\n    "\\"Color\\": \\"black\\""\n}\n```'

# Remove Markdown
response = re.sub(r'```(?:json)?\s*\n?', '', response)
print("After removing Markdown:")
print(repr(response))

# Extract JSON
json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
if json_match:
    json_str = json_match.group(0)
    print("\nExtracted JSON string:")
    print(repr(json_str))
    
    # Parse as Python literal (this gives us a set)
    parsed = ast.literal_eval(json_str)
    print(f"\nAfter literal_eval: {parsed} (type: {type(parsed).__name__})")
    
    if isinstance(parsed, set):
        # Extract the string from the set
        string_content = list(parsed)[0]
        print(f"Extracted string: {repr(string_content)}")
        
        # The string contains: "Color": "black"
        # We need to wrap it in braces to make it valid JSON
        json_str_fixed = "{" + string_content + "}"
        print(f"After wrapping: {repr(json_str_fixed)}")
        
        # Now parse as JSON
        try:
            result = json.loads(json_str_fixed)
            print(f"Successfully parsed: {result}")
        except Exception as e:
            print(f"Failed: {e}")

print("\n" + "="*80)
print("Testing with multiple attributes:")

response2 = '{\n    "\\"brand\\": \\"Nike\\"",\n    "\\"model\\": \\"Air Max\\""\n}'
print(f"Input: {repr(response2)}")

parsed2 = ast.literal_eval(response2)
print(f"After literal_eval: {parsed2} (type: {type(parsed2).__name__})")

if isinstance(parsed2, set):
    string_content = list(parsed2)[0]
    print(f"Extracted string: {repr(string_content)}")
    
    json_str_fixed = "{" + string_content + "}"
    print(f"After wrapping: {repr(json_str_fixed)}")
    
    try:
        result = json.loads(json_str_fixed)
        print(f"Successfully parsed: {result}")
    except Exception as e:
        print(f"Failed: {e}")
