# Implementation Guide: Attribute-Conditioned Video Attribute Extraction with AKS

## System Overview

This document provides a detailed technical guide to the implemented system for attribute-conditioned video attribute extraction using Adaptive Keyframe Sampling (AKS).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VideoAVE Dataset                         │
│  (video_path, category, title, attributes[])               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Loader (data_loader.py)                   │
│  Flattens into (video, category, title, attr, value)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Frame Extraction (video_utils.py)                   │
│  Extract candidate frames @ 1 fps, cache to disk            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      Frame Scoring (frame_scorer.py)                        │
│  Score frames using BLIP-ITM with attribute-specific text   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│    AKS Sampling (aks_sampler.py)                            │
│  Judge & Split: Select M keyframes adaptively               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│   Qwen2.5-VL Extraction (qwen_vl_extractor.py)              │
│  Extract attribute value from keyframes                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      Evaluation (evaluation.py)                             │
│  Compute Fuzzy F1 scores and metrics                        │
└─────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Data Loader (`data_loader.py`)

**Purpose**: Load VideoAVE dataset and flatten into per-attribute samples.

**Key Classes**:
- `VideoAVEAttrDataset`: Main dataset class
  - Loads CSV files from VideoAVE
  - Parses JSON aspects (attributes)
  - Flattens into (video, category, title, attr_name, attr_value) tuples
  - Supports filtering by domain and split

**Usage**:
```python
dataset = VideoAVEAttrDataset(
    data_root="/path/to/VideoAVE/Dataset",
    domains=["beauty", "sports"],
    split="test",
    max_samples=100
)

sample = dataset[0]
# {
#   "product_id": "B07FXWHFRF",
#   "video_path": "https://...",
#   "category": "beauty",
#   "title": "Product Title",
#   "attr_name": "Brand",
#   "attr_value": "ZOYA"
# }
```

**Key Methods**:
- `__len__()`: Total number of attribute samples
- `__getitem__(idx)`: Get single sample
- `get_by_product(product_id)`: Get all attributes for a product
- `get_unique_products()`: List of unique products
- `get_unique_attributes()`: List of unique attribute names

### 2. Video Utilities (`video_utils.py`)

**Purpose**: Extract and cache video frames.

**Key Functions**:
- `extract_candidate_frames(video_path, fps, max_frames)`: Extract frames using decord
  - Returns: (List[PIL.Image], List[float]) timestamps
  - Handles video downloading and local files
  
- `get_frames_cached(video_path, cache_dir, fps, max_frames)`: Extract with caching
  - Checks cache first
  - Extracts if not cached
  - Saves to disk for future use

**Caching Strategy**:
- Uses MD5 hash of video path as cache key
- Stores frames as `.npy` files
- Stores timestamps as `.npy` files
- Avoids re-extraction on repeated calls

### 3. Frame Scorer (`frame_scorer.py`)

**Purpose**: Score frames based on text relevance using BLIP-ITM.

**Key Class**: `FrameScorer`
- Loads BLIP-ITM model from Hugging Face
- Scores frames in batches to avoid OOM
- Returns scores in range [0, 1]

**Usage**:
```python
scorer = FrameScorer(model_name="blip-itm-base", device="cuda:0")

text = "A video frame showing the product's color"
frames = [Image.open(f) for f in frame_paths]

scores = scorer.score(text, frames)
# Returns: List[float] of length len(frames)
```

**Scoring Process**:
1. Prepare text and images
2. Tokenize and process through BLIP
3. Compute image-text matching logits
4. Convert to probabilities via softmax
5. Return scores

### 4. AKS Sampler (`aks_sampler.py`)

**Purpose**: Adaptively select M keyframes from T candidates using Judge & Split.

**Key Class**: `AdaptiveKeyframeSampler`

**Algorithm**:
```
def select(scores, M):
    stack = [Segment(start=0, end=T, m=M, level=0)]
    final_segments = []
    
    while stack:
        seg = stack.pop()
        
        # Stopping conditions
        if seg.level >= max_level or seg.m <= 1:
            final_segments.append(seg)
            continue
        
        # Judge: Check for peaks
        s_all = mean(scores[seg.start:seg.end])
        s_top = mean(top_k_scores)
        
        if s_top - s_all >= threshold:
            # TOP strategy: select highest scores
            final_segments.append(seg)
        else:
            # BIN strategy: split and recurse
            mid = (seg.start + seg.end) // 2
            stack.append(Segment(mid, seg.end, seg.m//2, seg.level+1))
            stack.append(Segment(seg.start, mid, seg.m//2, seg.level+1))
    
    # Extract top-m frames from each segment
    selected = []
    for seg in final_segments:
        indices = sort_by_score(seg.start:seg.end)
        selected.extend(indices[:seg.m])
    
    return sorted(set(selected))
```

**Key Insight**:
- TOP strategy captures high-value frames (peaks)
- BIN strategy ensures temporal coverage (uniform distribution)
- Judge & Split balances both automatically

**Baseline Samplers**:
- `SimpleTopKSampler`: Select top-K by score
- `UniformSampler`: Uniform temporal sampling

### 5. Attribute Keyframe Selector (`attr_keyframe_selector.py`)

**Purpose**: End-to-end pipeline for selecting keyframes for each attribute.

**Key Class**: `AttrKeyframeSelector`

**Pipeline**:
```python
selector = AttrKeyframeSelector(
    frame_scorer=scorer,
    cache_dir=".cache",
    fps=1.0,
    max_frames=256,
    m_attr=8,
    max_level=4,
    s_threshold=0.6
)

keyframes, timestamps, indices = selector.select_keyframes_for_attr(
    video_path="video.mp4",
    attr_name="Color",
    title="Product Title",
    category="beauty"
)
```

**Query Construction**:
```
"A video frame that best shows the product's {attr_name}. 
Product title: {title}. Product category: {category}."
```

This encourages frames that:
- Show the product clearly
- Highlight the specific attribute
- Provide context (title, category)

### 6. Qwen2.5-VL Extractor (`qwen_vl_extractor.py`)

**Purpose**: Extract attribute values from keyframes using Qwen2.5-VL.

**Key Class**: `QwenVLExtractor`

**Two Modes**:

**Mode A: Single-Attribute (Current)**
```python
extractor = QwenVLExtractor(model_path="/path/to/qwen", device="cuda:0")

value = extractor.extract_single_attr(
    keyframes=[Image, Image, ...],
    attr_name="Color",
    title="Product Title",
    category="beauty"
)
# Returns: "Red"
```

**Mode B: Multi-Attribute (Future)**
```python
values = extractor.extract_multi_attr(
    attr_keyframes={
        "Color": [Image, ...],
        "Brand": [Image, ...],
    },
    title="Product Title",
    category="beauty"
)
# Returns: {"Color": "Red", "Brand": "Nike"}
```

**Prompt Template (Single)**:
```
System: You are an expert at extracting product attribute values 
from e-commerce videos. Answer ONLY the attribute value in natural 
language. Do not output the attribute name or any explanation.

User: I will show you several frames from a product video.
Product category: {category}
Product title: {title}
Attribute name: "{attr_name}"

Please answer ONLY the attribute value in natural language.
Do not output the attribute name or any explanation.

[Images...]
```

### 7. Evaluation (`evaluation.py`)

**Purpose**: Compute Fuzzy F1 and other metrics.

**Key Functions**:
- `fuzzy_f1_score(pred, label, threshold=0.5)`: Compute Fuzzy F1
  - Finds longest common substring (LCS)
  - Matches if: LCS_length > threshold * len(label)
  - Returns: 0.0 or 1.0 (binary)

- `longest_common_substring_length(s1, s2)`: Compute LCS using DP
  - Time: O(m*n), Space: O(m*n)

**Key Class**: `AttributeEvaluator`
- Accumulates results
- Computes metrics at multiple levels:
  - Overall (all samples)
  - Per-category
  - Per-attribute

**Metrics**:
- F1 Score: Mean of individual Fuzzy F1 scores
- Accuracy: Fraction of matches
- Count: Number of samples

### 8. Main Script (`main.py`)

**Purpose**: End-to-end inference and evaluation.

**Usage**:
```bash
python main.py \
    --split test \
    --domains beauty sports \
    --max_samples 100 \
    --output_dir ./results
```

**Pipeline**:
1. Load dataset
2. Initialize models (BLIP, Qwen2.5-VL)
3. For each sample:
   - Select keyframes using AKS
   - Extract attribute value using Qwen2.5-VL
   - Store result
4. Evaluate using Fuzzy F1
5. Save results and metrics to JSON

## Configuration Parameters

**Video Processing**:
- `FPS_CANDIDATE = 1`: Extract 1 frame per second
- `MAX_FRAMES = 256`: Limit to 256 candidate frames

**AKS Parameters**:
- `M_ATTR = 8`: Select 8 keyframes per attribute
- `MAX_LEVEL = 4`: Recursion depth for Judge & Split
- `S_THRESHOLD = 0.6`: Score threshold for TOP vs BIN

**Model Paths**:
- `QWEN_MODEL_PATH`: Path to Qwen2.5-VL checkpoint
- `BLIP_MODEL_NAME`: BLIP model ID ("blip-itm-base" or "blip-itm-large")

## Performance Considerations

### Memory Usage
- Frame extraction: ~100MB per video (256 frames @ 480p)
- BLIP-ITM: ~4GB (base model)
- Qwen2.5-VL: ~14GB (7B model in FP16)
- Total: ~18GB GPU VRAM recommended

### Speed
- Frame extraction: ~1-2 seconds per video
- BLIP scoring: ~0.5 seconds per attribute (8 frames)
- AKS sampling: ~0.01 seconds per attribute
- Qwen2.5-VL inference: ~2-5 seconds per attribute
- Total: ~3-8 seconds per attribute

### Optimization Tips
1. **Caching**: Frames are cached automatically
2. **Batching**: BLIP scores frames in batches
3. **Device**: Use GPU for both BLIP and Qwen2.5-VL
4. **Quantization**: Consider FP16 or INT8 for Qwen2.5-VL

## Testing

Run unit tests:
```bash
python test_system.py
```

Tests cover:
- Configuration loading
- Data loading
- AKS sampling
- Fuzzy F1 metric
- LCS computation

## Troubleshooting

### Issue: CUDA out of memory
**Solution**:
- Reduce `MAX_FRAMES` in config
- Use smaller BLIP model
- Reduce batch size in scorer

### Issue: Slow inference
**Solution**:
- Enable frame caching (automatic)
- Use fewer samples with `--max_samples`
- Reduce `M_ATTR` if acceptable

### Issue: Poor extraction quality
**Solution**:
- Increase `M_ATTR` (more frames)
- Adjust `S_THRESHOLD` (try 0.5 or 0.7)
- Check video quality
- Verify attribute names match dataset

## Future Improvements

1. **Multi-Attribute Mode**: Implement Mode B for efficiency
2. **Fine-tuning**: Train Qwen2.5-VL on VideoAVE
3. **Ensemble**: Combine multiple VL models
4. **Adaptive M**: Learn optimal M per attribute
5. **Streaming**: Process long videos incrementally

## References

- VideoAVE: https://github.com/...
- BLIP: https://github.com/salesforce/BLIP
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- AKS Paper: Adaptive Keyframe Sampling for Video Understanding
