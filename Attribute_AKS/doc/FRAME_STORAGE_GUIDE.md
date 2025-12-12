# Frame Storage and CLIP Scoring Verification Guide

## Overview

This guide explains the new frame storage functionality and how to verify that CLIP scoring is working correctly.

## 1. Frame Storage System

### Purpose
The frame storage system allows you to save and review:
- **Candidate frames**: All frames extracted from the video
- **Keyframes**: Selected frames after AKS sampling
- **Scoring analysis**: Visual grid showing all frames with their scores

### Architecture

```
frame_storage/
└── {product_id}/
    └── {attribute_name}/
        ├── candidates/          # All extracted frames
        │   ├── frame_000_ts_0.00s.jpg
        │   ├── frame_001_ts_1.00s.jpg
        │   └── metadata.json    # Scores and timestamps
        ├── keyframes/           # Selected keyframes
        │   ├── keyframe_00_idx_000_ts_0.00s.jpg
        │   ├── keyframe_01_idx_005_ts_5.00s.jpg
        │   └── metadata.json    # Indices and scores
        └── analysis/            # Visual analysis
            ├── scoring_grid.jpg # Grid with all frames + scores
            └── score_stats.json # Statistics
```

### Usage

#### Enable Frame Storage

In `main.py`, frame storage is enabled by default:

```python
keyframe_selector = AttrKeyframeSelector(
    frame_scorer=frame_scorer,
    cache_dir=Config.FRAMES_CACHE_DIR,
    fps=Config.FPS_CANDIDATE,
    max_frames=Config.MAX_FRAMES,
    m_attr=Config.M_ATTR,
    max_level=Config.MAX_LEVEL,
    s_threshold=Config.S_THRESHOLD,
    storage_dir=os.path.join(Config.ATTRIBUTE_AKS_ROOT, "frame_storage"),
    enable_storage=True  # ← Enable storage
)
```

#### Disable Frame Storage

To disable frame storage (for faster processing):

```python
keyframe_selector = AttrKeyframeSelector(
    # ... other parameters ...
    enable_storage=False  # ← Disable storage
)
```

#### Access Stored Frames

```python
from frame_storage import FrameStorage

storage = FrameStorage(storage_root="./frame_storage")

# Get summary of all stored frames for a product
summary = storage.get_product_summary("product_id_123")
print(summary)
# Output:
# {
#   "product_id": "product_id_123",
#   "attributes": {
#     "Color": {"candidates": 64, "keyframes": 8, "analysis": true},
#     "Brand": {"candidates": 64, "keyframes": 8, "analysis": true}
#   }
# }
```

### Metadata Files

#### candidates/metadata.json
```json
{
  "product_id": "product_id_123",
  "attr_name": "Color",
  "num_frames": 64,
  "timestamps": [0.0, 1.0, 2.0, ...],
  "scores": [0.95, 0.87, 0.76, ...],
  "saved_at": "2025-11-30T02:06:18.123456"
}
```

#### keyframes/metadata.json
```json
{
  "product_id": "product_id_123",
  "attr_name": "Color",
  "num_keyframes": 8,
  "indices": [0, 5, 10, 15, 20, 25, 30, 35],
  "timestamps": [0.0, 5.0, 10.0, ...],
  "scores": [0.95, 0.92, 0.88, ...],
  "saved_at": "2025-11-30T02:06:18.123456"
}
```

#### analysis/score_stats.json
```json
{
  "product_id": "product_id_123",
  "attr_name": "Color",
  "num_candidates": 64,
  "num_keyframes": 8,
  "score_min": 0.12,
  "score_max": 0.98,
  "score_mean": 0.65,
  "score_std": 0.21,
  "scores": [0.95, 0.87, 0.76, ...],
  "keyframe_indices": [0, 5, 10, 15, 20, 25, 30, 35],
  "saved_at": "2025-11-30T02:06:18.123456"
}
```

## 2. CLIP Scoring Verification

### Problem Identified

Initially, CLIP scoring was returning identical scores for all frames. This was caused by:

1. **Incorrect input handling**: Passing the same text multiple times to the processor
2. **Sigmoid saturation**: Using sigmoid normalization on large logit values (>10), which maps all values to ~1.0

### Solution Implemented

The fix involved two changes to `frame_scorer.py`:

#### Change 1: Separate Text and Image Processing

Instead of:
```python
inputs = self.processor(
    text=[text] * len(batch_frames),  # ← Wrong: repeated text
    images=batch_frames,
    return_tensors="pt"
)
```

Now:
```python
text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
image_inputs = self.processor(images=batch_frames, return_tensors="pt")
```

#### Change 2: Min-Max Normalization Instead of Sigmoid

Instead of:
```python
batch_scores = torch.sigmoid(batch_scores)  # ← Causes saturation
```

Now:
```python
# Min-max normalization preserves relative differences
min_score = batch_scores.min()
max_score = batch_scores.max()
batch_scores = (batch_scores - min_score) / (max_score - min_score)
```

### Verification

Run the verification script:

```bash
python test_clip_fix.py
```

Expected output:
```
TEST 1: Query 'a red object'
Scores: ['1.0000', '0.2794', '0.1847', '0.0000', '0.1165']
✓ Scores are different (5 unique values)

TEST 2: Query 'a green object'
Scores: ['0.0400', '1.0000', '0.1711', '0.1674', '0.0000']
✓ Scores are different (5 unique values)

TEST 3: Verify query-image matching
✓ Red query scores red image higher than green image
✓ Green query scores green image higher than red image

SUMMARY
✓ CLIP scoring fix verified successfully!
```

### Score Interpretation

Scores are now normalized to [0, 1] range where:
- **1.0** = Highest matching frame in the batch
- **0.0** = Lowest matching frame in the batch
- **0.5** = Middle matching score

This allows the AKS algorithm to properly rank frames by relevance.

## 3. Demo

Run the demo to see frame storage in action:

```bash
python demo_frame_storage.py
```

This will:
1. Create 10 demo frames with different colors
2. Score them with CLIP for "a red object"
3. Save candidate frames
4. Select top 3 keyframes
5. Generate scoring analysis with visual grid
6. Print file structure

Output will be saved to `./demo_frames/demo_product_001/Color/`

## 4. Integration with Main Pipeline

The frame storage is automatically integrated into the main pipeline:

```bash
python main.py --split test --domains beauty --max_samples 5
```

This will:
1. Load 5 samples from the test set
2. Extract candidate frames for each attribute
3. Score frames with CLIP
4. Select keyframes using AKS
5. **Save all frames to `./frame_storage/`** for review
6. Extract attribute values using Qwen2.5-VL
7. Evaluate results

## 5. Quality Assurance Workflow

### Step 1: Run Inference
```bash
python main.py --split test --domains beauty --max_samples 10
```

### Step 2: Review Stored Frames
```
frame_storage/
├── product_001/
│   ├── Color/
│   │   ├── candidates/          # Review all extracted frames
│   │   ├── keyframes/           # Check selected keyframes
│   │   └── analysis/scoring_grid.jpg  # Visual verification
│   └── Brand/
│       ├── candidates/
│       ├── keyframes/
│       └── analysis/scoring_grid.jpg
└── product_002/
    └── ...
```

### Step 3: Verify Scoring
- Open `analysis/scoring_grid.jpg` to see all frames with scores
- Red-bordered frames are selected keyframes
- Verify that high-scoring frames match the query intent

### Step 4: Check Metadata
- Review `candidates/metadata.json` for score distribution
- Check `analysis/score_stats.json` for statistics
- Verify that scores are diverse (not all identical)

## 6. Troubleshooting

### Issue: All scores are identical

**Cause**: CLIP scoring not working correctly

**Solution**: Run verification script
```bash
python test_clip_fix.py
```

If test fails, check:
1. CLIP model is loaded correctly
2. GPU memory is available
3. Input images are valid

### Issue: Keyframes look wrong

**Cause**: AKS sampling not working correctly

**Solution**: 
1. Check `analysis/scoring_grid.jpg` - are high-scoring frames selected?
2. Review `score_stats.json` - is score distribution reasonable?
3. Verify query text in logs - is it appropriate for the attribute?

### Issue: Frame storage taking too much disk space

**Solution**: Disable frame storage for production
```python
enable_storage=False
```

Or clean up old frames:
```bash
rm -rf frame_storage/
```

## 7. Performance Notes

- Frame storage adds ~10-20% overhead per sample
- Each sample stores ~50-100 MB of images
- Scoring grid generation takes ~1-2 seconds per attribute
- Disable storage for faster processing if not needed for review

## 8. Files Modified

- `frame_scorer.py`: Fixed CLIP scoring
- `attr_keyframe_selector.py`: Added frame storage integration
- `frame_storage.py`: New module for frame storage
- `main.py`: Enabled frame storage by default

## 9. Next Steps

1. Run demo to verify everything works
2. Test on real VideoAVE data
3. Review stored frames for quality assurance
4. Adjust query templates if needed
5. Fine-tune AKS parameters if needed
