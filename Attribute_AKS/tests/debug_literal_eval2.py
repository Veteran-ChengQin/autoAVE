#!/usr/bin/env python3
"""Debug literal_eval approach - better strategy."""

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
    
    # Strategy: The JSON has escaped quotes inside a string value
    # The structure is: {"\"key\": \"value\""}
    # This is a dict with one key-value pair where the value is a JSON string
    
    # Try literal_eval to get the outer structure
    print("\nTrying ast.literal_eval...")
    try:
        data = ast.literal_eval(json_str)
        print(f"Result: {data}")
        print(f"Type: {type(data)}")
        
        # If it's a set, convert to dict by treating it as a single string value
        if isinstance(data, set):
            print("Got a set, extracting the string...")
            # The set contains one string element
            string_value = list(data)[0]
            print(f"String value: {repr(string_value)}")
            
            # Now parse this string as JSON
            try:
                inner_data = json.loads(string_value)
                print(f"Parsed inner JSON: {inner_data}")
            except Exception as e:
                print(f"Failed to parse inner JSON: {e}")
    except Exception as e:
        print(f"Failed: {e}")

print("\n" + "="*80)
print("Alternative approach: Direct string manipulation")

# The issue is that the JSON has escaped quotes
# Original: {"\"Color\": \"black\""}
# We want: {"Color": "black"}

# But we can't just replace \" with " because that would give us:
# {"Color": "black"} which is still invalid (missing outer quotes)

# Actually, looking at it more carefully:
# The outer structure is: { "string_value" }
# Where string_value contains: \"Color\": \"black\"
# This is NOT valid JSON at all!

# Let me check what the actual structure should be
print("\nAnalyzing the structure:")
print("Original response:", repr(response))

# The response contains a JSON object where the entire content is a string
# This is actually: a set containing a string in Python syntax
# The string is: "\"Color\": \"black\""
# Which when unescaped becomes: "Color": "black"

# So we need to:
# 1. Parse as Python literal (gets us the set with the string)
# 2. Extract the string
# 3. Parse the string as JSON

json_str = '{\n    "\\"Color\\": \\"black\\""\n}'
print("\nStep-by-step:")
print("1. Input:", repr(json_str))

# Parse as Python literal
parsed = ast.literal_eval(json_str)
print("2. After literal_eval:", parsed, type(parsed))

# Extract string
if isinstance(parsed, set):
    string_content = list(parsed)[0]
    print("3. Extracted string:", repr(string_content))
    
    # Parse as JSON
    try:
        result = json.loads(string_content)
        print("4. Parsed as JSON:", result)
    except Exception as e:
        print("4. Failed to parse as JSON:", e)
