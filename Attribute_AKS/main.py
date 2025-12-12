"""
Main script for attribute-conditioned video attribute extraction with AKS
"""
import os
import sys
import logging
import argparse
from typing import Optional
import json
import yaml
from dotenv import load_dotenv
from config import Config
from data_loader import VideoAVEAttrDataset, DataLoader
from frame_scorer import FrameScorer
from attr_keyframe_selector import AttrKeyframeSelector
from qwen_vl_extractor import QwenVLExtractor
from evaluation import AttributeEvaluator
load_dotenv()
# Create log directory first
os.makedirs(Config.LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.LOG_DIR, 'attribute_aks.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def setup_models(exp_config: dict):
    """Initialize all models based on experiment configuration"""
    logger.info("Setting up models...")
    
    input_mode = exp_config.get('input_mode', 'sampled_frames')
    model_mode = exp_config.get('model_mode', 'local')
    
    # Initialize keyframe selector only if needed
    keyframe_selector = None
    if input_mode in ['sampled_frames', 'all_frames']:
        # Frame scorer
        logger.info("Loading frame scorer (BLIP-ITM)...")
        frame_scorer = FrameScorer(
            model_name=exp_config.get('blip_model', Config.BLIP_MODEL_NAME),
            device=exp_config.get('device', Config.DEVICE)
        )
        
        # Keyframe selector
        logger.info("Initializing keyframe selector...")
        keyframe_selector = AttrKeyframeSelector(
            frame_scorer=frame_scorer,
            cache_dir=Config.FRAMES_CACHE_DIR,
            fps=exp_config.get('fps_candidate', Config.FPS_CANDIDATE),
            max_frames=exp_config.get('max_frames', Config.MAX_FRAMES),
            m_attr=exp_config.get('m_attr', Config.M_ATTR),
            max_level=exp_config.get('max_level', Config.MAX_LEVEL),
            s_threshold=exp_config.get('s_threshold', Config.S_THRESHOLD),
            storage_dir=os.path.join(Config.ATTRIBUTE_AKS_ROOT, "frame_storage"),
            enable_storage=True
        )
    
    # Qwen VL extractor
    logger.info(f"Loading Qwen VL in {model_mode} mode...")
    if model_mode == 'local':
        qwen_extractor = QwenVLExtractor(
            model_path=exp_config.get('model_path', Config.QWEN_MODEL_PATH),
            device=exp_config.get('device', Config.DEVICE),
            mode='local'
        )
    else:  # api mode
        api_key = exp_config.get('api_key') or os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("API key is required for API mode. Set 'api_key' in config or DASHSCOPE_API_KEY env var.")
        
        qwen_extractor = QwenVLExtractor(
            mode='api',
            api_key=api_key,
            api_url=exp_config.get('api_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'),
            api_model=exp_config.get('api_model', 'qwen-vl-plus')
        )
    
    return keyframe_selector, qwen_extractor


def infer_single_sample(sample: dict, keyframe_selector: Optional[AttrKeyframeSelector],
                       qwen_extractor: QwenVLExtractor, input_mode: str = "sampled_frames") -> dict:
    """
    Run inference on a single sample.
    
    Args:
        sample: Data sample with keys: video_path, category, title, attr_name, attr_value
        keyframe_selector: AttrKeyframeSelector instance (can be None for video mode)
        qwen_extractor: QwenVLExtractor instance
        input_mode: Input mode - "sampled_frames", "all_frames", or "video"
        
    Returns:
        Result dictionary with prediction and ground truth
    """
    video_path = sample["video_path"]
    category = sample["category"]
    title = sample["title"]
    attr_name = sample["attr_name"]
    attr_value = sample["attr_value"]
    product_id = sample.get("product_id", "")
    video_url = sample.get("video_url", "")  # For video mode
    
    logger.info(f"Processing: {product_id} - {attr_name} (mode: {input_mode})")
    
    try:
        if input_mode == "video":
            # Mode 4: Direct video URL input (API only)
            if not video_url:
                raise ValueError(f"video_url is required for video mode, but got empty for {product_id}")
            
            pred_dict = qwen_extractor.extract_single_attr_from_video(
                video_url=video_url,
                attr_name=attr_name,
                title=title,
                category=category
            )
            
            result = {
                "product_id": product_id,
                "category": category,
                "attr_name": attr_name,
                "pred": pred_dict,
                "label": attr_value,
                "input_mode": "video",
                "video_url": video_url
            }
            
        else:
            # Modes 1-3: Frame-based input
            if keyframe_selector is None:
                raise ValueError("keyframe_selector is required for frame-based modes")
            
            if input_mode == "sampled_frames":
                # Mode 1 & 2: Use AKS to select keyframes
                keyframes, timestamps, indices = keyframe_selector.select_keyframes_for_attr(
                    video_path, attr_name, title, category, product_id
                )
            elif input_mode == "all_frames":
                # Mode 3: Use all candidate frames without selection
                from video_utils import load_video
                keyframes = load_video(video_path, fps=keyframe_selector.fps, max_frames=keyframe_selector.max_frames)
                indices = list(range(len(keyframes)))
            else:
                raise ValueError(f"Unknown input_mode: {input_mode}")
            
            if not keyframes:
                logger.warning(f"No keyframes available for {attr_name}")
                return {
                    "product_id": product_id,
                    "category": category,
                    "attr_name": attr_name,
                    "pred": {},
                    "label": attr_value,
                    "error": "No keyframes available"
                }
            
            # Extract attribute value (returns {attr_name: value})
            pred_dict = qwen_extractor.extract_single_attr(
                keyframes, attr_name, title, category
            )
            
            result = {
                "product_id": product_id,
                "category": category,
                "attr_name": attr_name,
                "pred": pred_dict,
                "label": attr_value,
                "num_keyframes": len(keyframes),
                "keyframe_indices": indices if input_mode == "sampled_frames" else None,
                "input_mode": input_mode
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {attr_name}: {e}", exc_info=True)
        return {
            "product_id": product_id,
            "category": category,
            "attr_name": attr_name,
            "pred": {},
            "label": attr_value,
            "error": str(e)
        }


def infer_batch(dataset: VideoAVEAttrDataset, keyframe_selector: Optional[AttrKeyframeSelector],
               qwen_extractor: QwenVLExtractor, input_mode: str = "sampled_frames",
               max_samples: Optional[int] = None) -> list:
    """
    Run inference on a batch of samples.
    
    Args:
        dataset: VideoAVEAttrDataset instance
        keyframe_selector: AttrKeyframeSelector instance (can be None for video mode)
        qwen_extractor: QwenVLExtractor instance
        input_mode: Input mode - "sampled_frames", "all_frames", or "video"
        max_samples: Maximum samples to process
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    num_samples = len(dataset)
    
    for i in range(num_samples):
        sample = dataset[i]
        result = infer_single_sample(sample, keyframe_selector, qwen_extractor, input_mode)
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{num_samples} samples")
    
    return results


def evaluate_results(results: list, threshold: float = 0.5) -> dict:
    """
    Evaluate inference results with per-product aggregation (严谨的评估方式).
    
    This method aggregates results by product_id, computing F1 scores for each product
    as a whole (all attributes combined), which is the rigorous evaluation approach.
    
    Args:
        results: List of result dictionaries with format:
                 {
                     'product_id': str,
                     'category': str,
                     'attr_name': str,
                     'pred': {'attr_name': 'value'},
                     'label': 'value',
                     'num_keyframes': int,
                     'keyframe_indices': list,
                     ...
                 }
        threshold: Matching threshold for fuzzy matching (default 0.5)
        
    Returns:
        Metrics dictionary with per-product and overall metrics
    """
    from evaluation import compute_per_product_f1_scores
    
    if not results:
        logger.warning("No results to evaluate")
        return {}
    
    # Compute per-product F1 scores (严谨的评估方式)
    per_product_metrics, overall_metrics = compute_per_product_f1_scores(results, threshold=threshold)
    
    # Build metrics dictionary
    metrics = {
        "overall": overall_metrics,
        "per_product": per_product_metrics,
    }
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS (Per-Product Aggregation)")
    print("="*80)
    
    print(f"\n🔹 Overall Metrics (Aggregated by Product):")
    print(f"  Total Products: {overall_metrics['total_products']}")
    print(f"  Total Attributes: {overall_metrics['total_attributes']}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall: {overall_metrics['recall']:.4f}")
    print(f"  F1 Score: {overall_metrics['f1']:.4f}")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, FN: {overall_metrics['fn']}")
    
    # Print per-product metrics (sorted by F1 descending)
    if per_product_metrics:
        print(f"\n🔹 Per-Product Metrics (sorted by F1 descending):")
        sorted_products = sorted(
            per_product_metrics.items(),
            key=lambda x: x[1]["f1"],
            reverse=True
        )
        
        for product_id, metrics_dict in sorted_products:
            print(f"  {product_id}:")
            print(f"    Attributes: {metrics_dict['attr_count']}")
            print(f"    Precision: {metrics_dict['precision']:.4f}, Recall: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}")
            print(f"    TP: {metrics_dict['tp']}, FP: {metrics_dict['fp']}, FN: {metrics_dict['fn']}")
    
    print("\n" + "="*80)
    
    return metrics


def save_results(results: list, output_path: str) -> None:
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Attribute-conditioned video attribute extraction with AKS"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load experiment configuration
    logger.info(f"Loading experiment config from {args.config}")
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    logger.info(f"Starting attribute extraction")
    logger.info(f"Experiment: {exp_config.get('experiment_name', 'unnamed')}")
    logger.info(f"Config: {json.dumps(exp_config, indent=2)}")
    
    # Override output_dir if specified
    if args.output_dir:
        exp_config['output_dir'] = args.output_dir
    
    # Set default output_dir if not specified
    if 'output_dir' not in exp_config:
        exp_config['output_dir'] = os.path.join(Config.ATTRIBUTE_AKS_ROOT, "results")
    
    # Load dataset
    split = exp_config.get('split', 'test')
    domains = exp_config.get('domains', Config.EVAL_DOMAINS)
    max_samples = exp_config.get('max_samples', None)
    
    logger.info(f"Loading {split} dataset...")
    dataset = VideoAVEAttrDataset(
        data_root=Config.DATASET_ROOT,
        domains=domains,
        split=split,
        max_samples=max_samples
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Setup models
    keyframe_selector, qwen_extractor = setup_models(exp_config)
    
    # Run inference
    input_mode = exp_config.get('input_mode', 'sampled_frames')
    logger.info(f"Running inference with input_mode={input_mode}...")
    results = infer_batch(dataset, keyframe_selector, qwen_extractor, input_mode, max_samples)
    
    # Evaluate
    logger.info("Evaluating results...")
    threshold = exp_config.get('eval_threshold', 0.5)
    metrics = evaluate_results(results, threshold=threshold)
    
    # Save results
    exp_name = exp_config.get('experiment_name', 'exp')
    output_path = os.path.join(
        exp_config['output_dir'],
        f"results_{exp_name}.json"
    )
    save_results(results, output_path)
    
    # Save metrics
    metrics_path = os.path.join(
        exp_config['output_dir'],
        f"metrics_{exp_name}.json"
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save config used
    config_path = os.path.join(
        exp_config['output_dir'],
        f"config_{exp_name}.yaml"
    )
    with open(config_path, 'w') as f:
        yaml.dump(exp_config, f, indent=2)
    logger.info(f"Config saved to {config_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
