# Attribute-Conditioned Video Attribute Extraction with AKS

This system implements attribute-conditioned video attribute value extraction using Adaptive Keyframe Sampling (AKS) and Qwen2.5-VL.

## Overview

The system pipeline:
1. **Frame Extraction**: Extract candidate frames from video at 1 fps
2. **AKS Scoring**: Score frames using BLIP-ITM based on attribute-specific queries
3. **Adaptive Sampling**: Use Judge & Split strategy to select M keyframes per attribute
4. **Value Extraction**: Feed keyframes to Qwen2.5-VL to extract attribute values
5. **Evaluation**: Evaluate using Fuzzy F1 metric (VideoAVE standard)

## System Architecture

### Core Components

- **`config.py`**: Configuration parameters
- **`data_loader.py`**: VideoAVE dataset loader with attribute flattening
- **`video_utils.py`**: Video frame extraction and caching
- **`frame_scorer.py`**: BLIP-ITM based frame-text relevance scoring
- **`aks_sampler.py`**: Adaptive Keyframe Sampling implementation
- **`attr_keyframe_selector.py`**: Attribute-aware keyframe selection pipeline
- **`qwen_vl_extractor.py`**: Qwen2.5-VL integration for attribute extraction
- **`evaluation.py`**: Fuzzy F1 evaluation metrics
- **`main.py`**: Main inference and evaluation script

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to adjust:
- Model paths (Qwen2.5-VL, BLIP-ITM)
- Video processing parameters (fps, max_frames)
- AKS parameters (M_ATTR, MAX_LEVEL, S_THRESHOLD)
- Device settings

Key parameters:
```python
FPS_CANDIDATE = 1           # Candidate frame extraction rate
MAX_FRAMES = 256            # Max candidate frames per video
M_ATTR = 8                  # Keyframes per attribute
MAX_LEVEL = 4               # AKS recursion depth
S_THRESHOLD = 0.6           # TOP vs BIN decision threshold
```

## Usage

### Basic Inference

```bash
python main.py \
    --split test \
    --domains beauty sports \
    --max_samples 100
```

### Options

- `--split`: Dataset split (train/test)
- `--domains`: Domains to evaluate (e.g., beauty sports)
- `--max_samples`: Limit number of samples (for debugging)
- `--output_dir`: Output directory for results

### Output

Results are saved as JSON:
- `results_*.json`: Predictions and ground truth
- `metrics_*.json`: Evaluation metrics

## Data Format

The system expects VideoAVE dataset format:
```csv
product_id,video_url,content_url,video_title,parent_title,aspects
B07FXWHFRF,https://...,https://...,Title,Parent,"{'Brand': 'ZOYA', 'Color': 'Red'}"
```

The data loader flattens this into per-attribute samples:
```python
{
    "product_id": "B07FXWHFRF",
    "video_path": "https://...",
    "category": "beauty",
    "title": "Product Title",
    "attr_name": "Brand",
    "attr_value": "ZOYA"
}
```

## AKS Algorithm

The Adaptive Keyframe Sampling uses a recursive Judge & Split strategy:

1. **Judge**: Analyze score distribution in current segment
   - If `max(top-k scores) - mean(all scores) > threshold`: Use TOP strategy
   - Otherwise: Use BIN strategy

2. **Split**: Recursively partition segments
   - TOP: Select highest-scoring frames directly
   - BIN: Split segment and recurse on sub-segments

3. **Termination**: Stop at max recursion depth or when m=1

This balances:
- **TOP strategy**: Captures high-value frames (peaks in scores)
- **BIN strategy**: Ensures temporal coverage (uniform distribution)

## Frame Scoring

Uses BLIP-ITM (Image-Text Matching) to score frame relevance:

Query template:
```
"A video frame that best shows the product's {attr_name}. 
Product title: {title}. Product category: {category}."
```

This encourages selection of frames that:
- Show the product clearly
- Highlight the specific attribute

## Attribute Value Extraction

Two modes supported:

### Mode A: Single-Attribute (Simple)
- Select keyframes for one attribute
- Pass to Qwen2.5-VL with attribute-specific prompt
- Extract value directly

### Mode B: Multi-Attribute (Efficient)
- Select keyframes for all attributes
- Merge and deduplicate frames
- Pass all to Qwen2.5-VL with multi-attribute prompt
- Parse structured response

Current implementation uses Mode A for simplicity.

## Evaluation Metrics

**Fuzzy F1**: Based on longest common substring
- Match if: `lcs_length > 0.5 * len(label)`
- Computed at attribute, category, and overall levels

Metrics reported:
- Overall F1 and Accuracy
- Per-category F1 and Accuracy
- Per-attribute F1 and Accuracy

## Example Results

Expected performance on VideoAVE:
- Overall F1: ~0.65-0.75 (with AKS)
- Baseline (uniform sampling): ~0.55-0.65

Improvement from AKS:
- Better temporal coverage (BIN strategy)
- Captures high-value frames (TOP strategy)
- Attribute-specific frame selection

## Troubleshooting

### Out of Memory
- Reduce `MAX_FRAMES` in config
- Reduce batch size in inference
- Use smaller BLIP model (blip-itm-base)

### Slow Inference
- Enable frame caching (automatic)
- Reduce number of samples with `--max_samples`
- Use GPU with more VRAM

### Poor Results
- Adjust `S_THRESHOLD` (higher = more TOP, lower = more BIN)
- Adjust `M_ATTR` (more frames = better coverage)
- Check video quality and frame extraction

## References

- VideoAVE Dataset: https://github.com/...
- BLIP-ITM: https://github.com/salesforce/BLIP
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- AKS Paper: Adaptive Keyframe Sampling for Video Understanding

## License

Same as VideoAVE dataset
