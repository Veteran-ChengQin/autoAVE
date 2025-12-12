#!/usr/bin/env python3
"""Debug script to understand the escape sequence."""

import json
import re

# The problematic response
response = '```json\n{\n    "\\"Color\\": \\"black\\""\n}\n```'

print("Original response:")
print(repr(response))
print("\nActual string content:")
print(response)
print("\n" + "="*80)

# Step 1: Remove Markdown
response_step1 = re.sub(r'```(?:json)?\s*\n?', '', response)
print("After removing Markdown:")
print(repr(response_step1))
print(response_step1)
print("\n" + "="*80)

# The issue: the JSON has escaped quotes inside a string value
# Current: {"\"Color\": \"black\""}  - this is INVALID JSON
# It should be: {"Color": "black"}
# 
# The problem is that the entire JSON object is wrapped in quotes and escaped
# Let's try a different approach: use json.loads with the original escaped string

print("Approach 1: Try to parse the escaped JSON as a JSON string first")
try:
    # The response might be a JSON string containing JSON
    # Try to parse it as a string first
    unescaped = json.loads(response_step1)
    print(f"Parsed as JSON string: {repr(unescaped)}")
    # Now parse the unescaped content as JSON
    data = json.loads(unescaped)
    print(f"Successfully parsed inner JSON: {data}")
except Exception as e:
    print(f"Approach 1 failed: {e}")

print("\n" + "="*80)

print("Approach 2: Use ast.literal_eval")
try:
    import ast
    # Try to evaluate the string as a Python literal
    data = ast.literal_eval(response_step1)
    print(f"Parsed with literal_eval: {data}")
except Exception as e:
    print(f"Approach 2 failed: {e}")

print("\n" + "="*80)

print("Approach 3: Unescape and then parse")
try:
    # Use codecs to decode escape sequences
    import codecs
    unescaped = codecs.decode(response_step1, 'unicode_escape')
    print(f"After unicode_escape: {repr(unescaped)}")
    data = json.loads(unescaped)
    print(f"Successfully parsed: {data}")
except Exception as e:
    print(f"Approach 3 failed: {e}")

print("\n" + "="*80)

print("Approach 4: Manual character-by-character analysis")
# The structure is: {"\"key\": \"value\""}
# We need to extract the inner part and unescape it
try:
    # Find the JSON object
    json_match = re.search(r'\{[^{}]*\}', response_step1)
    if json_match:
        json_str = json_match.group(0)
        print(f"Extracted JSON: {repr(json_str)}")
        
        # Try to parse as-is first
        try:
            data = json.loads(json_str)
            print(f"Direct parse successful: {data}")
        except:
            # If direct parse fails, try to unescape
            import codecs
            unescaped = codecs.decode(json_str, 'unicode_escape')
            print(f"After unicode_escape: {repr(unescaped)}")
            data = json.loads(unescaped)
            print(f"Successfully parsed: {data}")
except Exception as e:
    print(f"Approach 4 failed: {e}")
