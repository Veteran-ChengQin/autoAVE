# QwenVL API Mode Usage Guide

## Overview

`QwenVLExtractor` now supports two modes of operation:
1. **Local Mode**: Uses a locally deployed Qwen-VL model
2. **API Mode**: Calls Qwen-VL API endpoints (e.g., Dashscope)

## API Mode Setup

### 1. Get API Key

First, obtain your API key from Dashscope (阿里云灵积):
- Visit: https://dashscope.aliyun.com/
- Register and get your API key

### 2. Set Environment Variable

```bash
export DASHSCOPE_API_KEY='your-api-key-here'
```

Or set it in your Python code:
```python
import os
os.environ['DASHSCOPE_API_KEY'] = 'your-api-key-here'
```

## Basic Usage

### Initialize Extractor in API Mode

```python
from qwen_vl_extractor import QwenVLExtractor

# API mode
extractor = QwenVLExtractor(
    mode="api",
    api_key="your-api-key-here",
    api_model="qwen-vl-plus"  # or "qwen-vl-max"
)
```

### Extract Single Attribute

```python
from PIL import Image

# Load your images
keyframes = [Image.open("frame1.jpg"), Image.open("frame2.jpg")]

# Extract attribute
result = extractor.extract_single_attr(
    keyframes=keyframes,
    attr_name="Color",
    title="Product Title",
    category="Product Category"
)

print(result)  # {"Color": "black"}
```

### Extract Multiple Attributes

```python
# Prepare keyframes for each attribute
attr_keyframes = {
    "Color": keyframes,
    "Material": keyframes,
    "Hair Type": keyframes
}

# Extract all attributes at once
results = extractor.extract_multi_attr(
    attr_keyframes=attr_keyframes,
    title="Women's Synthetic Wig",
    category="Hair Extensions & Wigs"
)

print(results)
# {
#   "Color": "black",
#   "Material": "synthetic fiber",
#   "Hair Type": "straight"
# }
```

## Complete Example with Video

```python
from qwen_vl_extractor import QwenVLExtractor
from video_utils import extract_frames
import os

# Get API key from environment
api_key = os.environ.get("DASHSCOPE_API_KEY")

# Initialize extractor
extractor = QwenVLExtractor(
    mode="api",
    api_key=api_key,
    api_model="qwen-vl-plus"
)

# Extract frames from video
video_path = "product_video.mp4"
frames = extract_frames(video_path, num_frames=8)

# Extract attributes
attributes = ["Color", "Hair Type", "Material"]
attr_keyframes = {attr: frames for attr in attributes}

results = extractor.extract_multi_attr(
    attr_keyframes=attr_keyframes,
    title="Women's Synthetic Wig",
    category="Hair Extensions & Wigs"
)

print("Extracted attributes:", results)
```

## Configuration Options

### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"local"` | Mode of operation: `"local"` or `"api"` |
| `model_path` | str | None | Path to local model (required for local mode) |
| `device` | str | `"cuda:0"` | Device for local model |
| `api_key` | str | None | API key (required for API mode) |
| `api_url` | str | Dashscope URL | API endpoint URL |
| `api_model` | str | `"qwen-vl-plus"` | Model name for API calls |

### Available API Models

- **qwen-vl-plus**: Balanced performance and cost
- **qwen-vl-max**: Higher accuracy, higher cost
- **qwen-vl-turbo**: Faster inference, lower cost

## API Request Format

The API mode sends requests in the following format:

```json
{
  "model": "qwen-vl-plus",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64_encoded_image>"
          }
        },
        {
          "type": "text",
          "text": "Extract the following attributes: Color, Material..."
        }
      ]
    }
  ],
  "stream": false
}
```

## Comparison: Local vs API Mode

| Feature | Local Mode | API Mode |
|---------|-----------|----------|
| **Setup** | Requires model download & GPU | Only needs API key |
| **Cost** | GPU infrastructure cost | Pay per API call |
| **Speed** | Depends on local GPU | Network latency + API processing |
| **Privacy** | Data stays local | Data sent to API |
| **Scalability** | Limited by local resources | Highly scalable |
| **Offline** | ✅ Works offline | ❌ Requires internet |

## Error Handling

```python
try:
    extractor = QwenVLExtractor(
        mode="api",
        api_key=api_key,
        api_model="qwen-vl-plus"
    )
    
    results = extractor.extract_multi_attr(
        attr_keyframes=attr_keyframes,
        title=title,
        category=category
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except requests.exceptions.RequestException as e:
    print(f"API call failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Batch Processing**: Use `extract_multi_attr()` instead of multiple `extract_single_attr()` calls to reduce API costs

2. **Frame Selection**: Limit frames to 8-16 per video for optimal balance between accuracy and cost

3. **Error Handling**: Always wrap API calls in try-except blocks to handle network issues

4. **Rate Limiting**: Be aware of API rate limits and implement appropriate retry logic

5. **API Key Security**: Never hardcode API keys; use environment variables or secure key management

## Troubleshooting

### Issue: "api_key is required for API mode"
**Solution**: Make sure to provide the `api_key` parameter when initializing in API mode

### Issue: API call timeout
**Solution**: Reduce the number of frames or increase the timeout in `_call_api()` method

### Issue: "Failed to parse JSON response"
**Solution**: Check the API response format; the model might be returning unexpected output

## Migration from Local to API Mode

If you're currently using local mode and want to switch to API mode:

```python
# Before (Local mode)
extractor = QwenVLExtractor(
    model_path="/path/to/model",
    device="cuda:0"
)

# After (API mode)
extractor = QwenVLExtractor(
    mode="api",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    api_model="qwen-vl-plus"
)

# The rest of your code remains the same!
# extract_single_attr() and extract_multi_attr() work identically
```

## Additional Resources

- [Dashscope Documentation](https://help.aliyun.com/zh/dashscope/)
- [Qwen-VL Model Card](https://github.com/QwenLM/Qwen-VL)
- [API Pricing](https://dashscope.aliyun.com/pricing)
