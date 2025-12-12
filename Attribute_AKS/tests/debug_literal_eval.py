#!/usr/bin/env python3
"""Debug literal_eval approach."""

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
    
    # Try literal_eval
    print("\nTrying ast.literal_eval...")
    try:
        data = ast.literal_eval(json_str)
        print(f"Success! Result: {data}")
        print(f"Type: {type(data)}")
        print(f"Is dict: {isinstance(data, dict)}")
        
        # Check the values
        for key, value in data.items():
            print(f"  Key: {repr(key)}, Value: {repr(value)}, Type: {type(value)}")
    except Exception as e:
        print(f"Failed: {e}")
