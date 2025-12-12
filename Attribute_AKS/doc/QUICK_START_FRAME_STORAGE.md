# Quick Start: Frame Storage and CLIP Scoring

## What Was Fixed

### Problem
CLIP scoring was returning identical scores (1.0000) for all frames, making keyframe selection impossible.

### Root Causes
1. **Input handling bug**: Passing repeated text to processor
2. **Sigmoid saturation**: Large logits (>10) all map to ~1.0

### Solution
- Separate text and image processing in CLIP
- Use min-max normalization instead of sigmoid

## Verification

### Quick Test (30 seconds)
```bash
cd /data/veteran/project/dataE/Attribute_AKS
python test_clip_fix.py
```

Expected output:
```
✓ Scores are different (5 unique values)
✓ Red query scores red image higher than green image
✓ CLIP scoring fix verified successfully!
```

### Full Demo (1 minute)
```bash
python demo_frame_storage.py
```

This creates:
- 10 demo frames with different colors
- Scores them with CLIP
- Saves candidate frames, keyframes, and analysis
- Generates visual scoring grid

Output location: `./demo_frames/demo_product_001/Color/`

## Using Frame Storage

### Enable in Your Code
```python
from attr_keyframe_selector import AttrKeyframeSelector
from frame_scorer import FrameScorer

scorer = FrameScorer(model_name="clip-vit-base", device="cuda:0")
selector = AttrKeyframeSelector(
    frame_scorer=scorer,
    storage_dir="./frame_storage",
    enable_storage=True  # ← Enable storage
)

# Now when you select keyframes, they'll be saved
keyframes, timestamps, indices = selector.select_keyframes_for_attr(
    video_path="video.mp4",
    attr_name="Color",
    title="Product Title",
    category="Category",
    product_id="product_123"  # ← Required for storage
)
```

### Review Stored Frames
```
frame_storage/
└── product_123/
    └── Color/
        ├── candidates/          # All extracted frames
        │   ├── frame_000_ts_0.00s.jpg
        │   ├── frame_001_ts_1.00s.jpg
        │   └── metadata.json    # Scores and timestamps
        ├── keyframes/           # Selected keyframes
        │   ├── keyframe_00_idx_000_ts_0.00s.jpg
        │   └── metadata.json
        └── analysis/
            ├── scoring_grid.jpg # Visual verification
            └── score_stats.json # Statistics
```

### Access Metadata
```python
import json

# Read candidate scores
with open("frame_storage/product_123/Color/candidates/metadata.json") as f:
    data = json.load(f)
    print(f"Scores: {data['scores']}")
    print(f"Min: {min(data['scores']):.4f}, Max: {max(data['scores']):.4f}")

# Read keyframe indices
with open("frame_storage/product_123/Color/keyframes/metadata.json") as f:
    data = json.load(f)
    print(f"Selected indices: {data['indices']}")
```

## Run Main Pipeline with Storage

```bash
python main.py --split test --domains beauty --max_samples 5
```

This will:
1. Load 5 samples
2. Extract frames
3. Score with CLIP (now working correctly!)
4. Select keyframes
5. **Save all frames to `./frame_storage/`**
6. Extract attribute values
7. Evaluate results

## Disable Storage (for speed)

If you don't need to review frames:

```python
selector = AttrKeyframeSelector(
    frame_scorer=scorer,
    enable_storage=False  # ← Disable storage
)
```

This saves ~10-20% processing time and disk space.

## Score Interpretation

Scores are now normalized to [0, 1]:
- **1.0** = Best matching frame in batch
- **0.5** = Medium matching
- **0.0** = Worst matching frame in batch

Example scores for "a red object":
```
Red image:    1.0000  ← Highest score
Dark red:     0.9331
Green:        0.2794
Yellow:       0.1165
Blue:         0.0000  ← Lowest score
```

## Troubleshooting

### Scores still identical?
```bash
python test_clip_fix.py
```

If test fails:
1. Check GPU memory: `nvidia-smi`
2. Verify CLIP model loaded: Check logs for "CLIP model loaded successfully"
3. Try restarting Python kernel

### Frames look wrong?
1. Open `frame_storage/{product_id}/{attr_name}/analysis/scoring_grid.jpg`
2. Check if high-scoring frames (red border) match the query
3. Review query text in logs
4. Verify input images are valid

### Storage taking too much space?
```bash
rm -rf frame_storage/
```

Or disable storage in code:
```python
enable_storage=False
```

## Files to Review

### Documentation
- `FRAME_STORAGE_GUIDE.md` - Complete guide
- `QUICK_START_FRAME_STORAGE.md` - This file

### Code
- `frame_storage.py` - Storage implementation
- `frame_scorer.py` - CLIP scoring fix
- `attr_keyframe_selector.py` - Integration

### Tests
- `test_clip_fix.py` - Verify CLIP scoring
- `demo_frame_storage.py` - Demo frame storage

## Next Steps

1. ✓ Run `test_clip_fix.py` to verify CLIP works
2. ✓ Run `demo_frame_storage.py` to see frame storage in action
3. ✓ Run `python main.py --max_samples 5` to test on real data
4. ✓ Review stored frames in `./frame_storage/`
5. ✓ Check logs for any issues
6. ✓ Adjust AKS parameters if needed

## Performance

- CLIP scoring: ~0.5-1.0 seconds per 64 frames
- Frame storage: ~1-2 seconds per attribute
- Total overhead: ~10-20% per sample
- Disk usage: ~50-100 MB per sample

## Support

For issues, check:
1. Logs in `./logs/attribute_aks.log`
2. Frame storage metadata JSON files
3. Scoring grid visualization
4. Test script output

Run tests to verify everything works:
```bash
python test_clip_fix.py
python demo_frame_storage.py
```
