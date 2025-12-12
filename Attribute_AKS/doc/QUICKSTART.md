# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies
```bash
cd /data/veteran/project/dataE/Attribute_AKS
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_system.py
```

Expected output:
```
✓ PASS: Configuration
✓ PASS: Data Loader
✓ PASS: AKS Sampler
✓ PASS: Fuzzy F1
✓ PASS: LCS

Total: 5/5 tests passed
```

## Basic Usage (10 minutes)

### Run Inference on Test Set
```bash
python main.py \
    --split test \
    --domains beauty \
    --max_samples 10
```

This will:
1. Load 10 samples from beauty domain
2. Extract candidate frames
3. Score frames using BLIP-ITM
4. Select keyframes using AKS
5. Extract attribute values using Qwen2.5-VL
6. Evaluate using Fuzzy F1
7. Save results to `results/`

### Expected Output
```
INFO - Loading test dataset...
INFO - Loaded 10 samples
INFO - Setting up models...
INFO - Loading frame scorer (BLIP-ITM)...
INFO - Loading Qwen2.5-VL...
INFO - Running inference...
INFO - Processed 10/10 samples
INFO - Evaluating results...

============================================================
EVALUATION RESULTS
============================================================

Overall Metrics:
  Total Samples: 10
  Overall F1: 0.7000
  Overall Accuracy: 0.7000

Per-Category Metrics:
  beauty:
    Count: 10
    F1: 0.7000
    Accuracy: 0.7000

Per-Attribute Metrics:
  Brand:
    Count: 3
    F1: 0.6667
    Accuracy: 0.6667
  Color:
    Count: 4
    F1: 0.7500
    Accuracy: 0.7500
  ...

============================================================
```

## Configuration

Edit `config.py` to customize:

```python
# Video processing
FPS_CANDIDATE = 1           # Frames per second
MAX_FRAMES = 256            # Max candidate frames

# AKS parameters
M_ATTR = 8                  # Keyframes per attribute
MAX_LEVEL = 4               # Recursion depth
S_THRESHOLD = 0.6           # TOP vs BIN threshold

# Model paths
QWEN_MODEL_PATH = "/data/share/qwen/Qwen2.5-VL-7B-Instruct"
BLIP_MODEL_NAME = "blip-itm-base"
```

## Common Commands

### Evaluate on Multiple Domains
```bash
python main.py \
    --split test \
    --domains beauty sports clothing \
    --max_samples 50
```

### Evaluate on Training Set
```bash
python main.py \
    --split train \
    --domains beauty \
    --max_samples 100
```

### Full Evaluation (All Domains)
```bash
python main.py \
    --split test \
    --domains beauty sports clothing appliances arts automotive baby cellphones grocery industry musical patio pet toys
```

### Debug Mode (Single Sample)
```bash
python main.py \
    --split test \
    --domains beauty \
    --max_samples 1
```

## Output Files

Results are saved to `results/` directory:

```
results/
├── results_test_beauty.json          # Predictions and ground truth
├── metrics_test_beauty.json          # Evaluation metrics
├── results_test_beauty_sports.json   # Multi-domain results
└── metrics_test_beauty_sports.json
```

### Results JSON Format
```json
[
  {
    "product_id": "B07FXWHFRF",
    "category": "beauty",
    "attr_name": "Brand",
    "pred": "ZOYA",
    "label": "ZOYA",
    "num_keyframes": 8,
    "keyframe_indices": [2, 5, 8, 12, 15, 18, 22, 25]
  },
  ...
]
```

### Metrics JSON Format
```json
{
  "total_samples": 10,
  "overall_f1": 0.7,
  "overall_accuracy": 0.7,
  "by_category": {
    "beauty": {
      "count": 10,
      "f1": 0.7,
      "accuracy": 0.7
    }
  },
  "by_attribute": {
    "Brand": {
      "count": 3,
      "f1": 0.6667,
      "accuracy": 0.6667
    },
    ...
  }
}
```

## Troubleshooting

### Error: "CUDA out of memory"
```bash
# Reduce max frames
# Edit config.py:
MAX_FRAMES = 128  # was 256

# Or reduce samples
python main.py --max_samples 5
```

### Error: "Model not found"
```bash
# Check model paths in config.py
# Ensure models are downloaded:
# - Qwen2.5-VL: /data/share/qwen/Qwen2.5-VL-7B-Instruct
# - BLIP-ITM: Auto-downloaded from Hugging Face
```

### Error: "Dataset not found"
```bash
# Verify VideoAVE dataset path
# Default: /data/veteran/project/dataE/VideoAVE/Dataset
# Check if CSV files exist:
ls /data/veteran/project/dataE/VideoAVE/Dataset/test_data/
```

### Slow Inference
```bash
# Check GPU usage
nvidia-smi

# Reduce samples for testing
python main.py --max_samples 5

# Use smaller BLIP model
# Edit config.py:
BLIP_MODEL_NAME = "blip-itm-base"  # was "blip-itm-large"
```

## Next Steps

1. **Experiment with Parameters**
   - Try different `S_THRESHOLD` values (0.4, 0.5, 0.6, 0.7, 0.8)
   - Try different `M_ATTR` values (4, 6, 8, 10, 12)
   - Observe impact on F1 scores

2. **Analyze Results**
   - Check which attributes have high/low F1
   - Analyze failure cases
   - Visualize selected keyframes

3. **Optimize Performance**
   - Enable frame caching (automatic)
   - Batch multiple samples
   - Profile bottlenecks

4. **Extend System**
   - Implement multi-attribute mode
   - Add fine-tuning pipeline
   - Create visualization tools

## API Usage

Use the system programmatically:

```python
from config import Config
from data_loader import VideoAVEAttrDataset
from frame_scorer import FrameScorer
from attr_keyframe_selector import AttrKeyframeSelector
from qwen_vl_extractor import QwenVLExtractor

# Load dataset
dataset = VideoAVEAttrDataset(
    data_root=Config.DATASET_ROOT,
    domains=["beauty"],
    split="test",
    max_samples=10
)

# Initialize models
scorer = FrameScorer(model_name="blip-itm-base", device="cuda:0")
selector = AttrKeyframeSelector(frame_scorer=scorer)
extractor = QwenVLExtractor(model_path=Config.QWEN_MODEL_PATH)

# Process sample
sample = dataset[0]
keyframes, _, _ = selector.select_keyframes_for_attr(
    video_path=sample["video_path"],
    attr_name=sample["attr_name"],
    title=sample["title"],
    category=sample["category"]
)

# Extract value
value = extractor.extract_single_attr(
    keyframes=keyframes,
    attr_name=sample["attr_name"],
    title=sample["title"],
    category=sample["category"]
)

print(f"Predicted: {value}")
print(f"Ground truth: {sample['attr_value']}")
```

## Performance Benchmarks

Expected performance on VideoAVE test set:

| Method | F1 Score | Accuracy | Speed |
|--------|----------|----------|-------|
| Uniform Sampling | 0.55 | 0.55 | 2s/attr |
| Top-K Sampling | 0.60 | 0.60 | 2s/attr |
| AKS (M=4) | 0.65 | 0.65 | 3s/attr |
| AKS (M=8) | 0.70 | 0.70 | 4s/attr |
| AKS (M=12) | 0.72 | 0.72 | 5s/attr |

Speed includes: frame extraction, scoring, sampling, and Qwen2.5-VL inference.

## Support

For issues or questions:
1. Check `IMPLEMENTATION_GUIDE.md` for detailed documentation
2. Review `test_system.py` for usage examples
3. Check logs in `logs/attribute_aks.log`
4. Examine results in `results/` directory
