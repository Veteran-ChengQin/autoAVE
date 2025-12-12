"""
Main script for attribute-conditioned video attribute extraction with AKS
"""
import os
import sys
import logging
import argparse
from typing import Optional
import json

from config import Config
from data_loader import VideoAVEAttrDataset, DataLoader
from frame_scorer import FrameScorer
from attr_keyframe_selector import AttrKeyframeSelector
from qwen_vl_extractor import QwenVLExtractor
from evaluation import AttributeEvaluator

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


def setup_models():
    """Initialize all models"""
    logger.info("Setting up models...")
    
    # Frame scorer
    logger.info("Loading frame scorer (BLIP-ITM)...")
    frame_scorer = FrameScorer(
        model_name=Config.BLIP_MODEL_NAME,
        device=Config.DEVICE
    )
    
    # Keyframe selector
    logger.info("Initializing keyframe selector...")
    keyframe_selector = AttrKeyframeSelector(
        frame_scorer=frame_scorer,
        cache_dir=Config.FRAMES_CACHE_DIR,
        fps=Config.FPS_CANDIDATE,
        max_frames=Config.MAX_FRAMES,
        m_attr=Config.M_ATTR,
        max_level=Config.MAX_LEVEL,
        s_threshold=Config.S_THRESHOLD,
        storage_dir=os.path.join(Config.ATTRIBUTE_AKS_ROOT, "frame_storage"),
        enable_storage=True
    )
    
    # Qwen2.5-VL extractor
    logger.info("Loading Qwen2.5-VL...")
    qwen_extractor = QwenVLExtractor(
        model_path=Config.QWEN_MODEL_PATH,
        device=Config.DEVICE
    )
    
    return frame_scorer, keyframe_selector, qwen_extractor


def infer_single_sample(sample: dict, keyframe_selector: AttrKeyframeSelector,
                       qwen_extractor: QwenVLExtractor, mode: str = "single") -> dict:
    """
    Run inference on a single sample.
    
    Args:
        sample: Data sample with keys: video_path, category, title, attr_name, attr_value
        keyframe_selector: AttrKeyframeSelector instance
        qwen_extractor: QwenVLExtractor instance
        mode: "single" for single-attribute, "multi" for multi-attribute
        
    Returns:
        Result dictionary with prediction and ground truth
    """
    video_path = sample["video_path"]
    category = sample["category"]
    title = sample["title"]
    attr_name = sample["attr_name"]
    attr_value = sample["attr_value"]
    product_id = sample.get("product_id", "")
    
    logger.info(f"Processing: {product_id} - {attr_name}")
    
    try:
        # Select keyframes for this attribute
        keyframes, timestamps, indices = keyframe_selector.select_keyframes_for_attr(
            video_path, attr_name, title, category, product_id
        )
        
        if not keyframes:
            logger.warning(f"No keyframes selected for {attr_name}")
            return {
                "product_id": product_id,
                "category": category,
                "attr_name": attr_name,
                "pred": {},
                "label": attr_value,
                "error": "No keyframes selected"
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
            "keyframe_indices": indices,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {attr_name}: {e}")
        return {
            "product_id": product_id,
            "category": category,
            "attr_name": attr_name,
            "pred": {},
            "label": attr_value,
            "error": str(e)
        }


def infer_batch(dataset: VideoAVEAttrDataset, keyframe_selector: AttrKeyframeSelector,
               qwen_extractor: QwenVLExtractor, max_samples: Optional[int] = None) -> list:
    """
    Run inference on a batch of samples.
    
    Args:
        dataset: VideoAVEAttrDataset instance
        keyframe_selector: AttrKeyframeSelector instance
        qwen_extractor: QwenVLExtractor instance
        max_samples: Maximum samples to process
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    # num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    num_samples = len(dataset)
    
    for i in range(num_samples):
        sample = dataset[i]
        result = infer_single_sample(sample, keyframe_selector, qwen_extractor)
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
        "--mode", type=str, default="infer",
        choices=["infer", "eval"],
        help="Mode: 'infer' for inference, 'eval' for evaluation"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--domains", type=str, nargs="+", default=Config.EVAL_DOMAINS,
        help="Domains to evaluate on"
    )
    parser.add_argument(
        "--max_samples", type=int, default=2,
        help="Maximum samples to process (for debugging)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(Config.ATTRIBUTE_AKS_ROOT, "results"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting attribute extraction with AKS")
    logger.info(f"Config: {vars(args)}")
    
    # Load dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = VideoAVEAttrDataset(
        data_root=Config.DATASET_ROOT,
        domains=args.domains,
        split=args.split,
        max_samples=args.max_samples
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Setup models
    frame_scorer, keyframe_selector, qwen_extractor = setup_models()
    
    # Run inference
    logger.info("Running inference...")
    results = infer_batch(dataset, keyframe_selector, qwen_extractor, args.max_samples)
    
    # Evaluate
    logger.info("Evaluating results...")
    metrics = evaluate_results(results)
    
    # Save results
    output_path = os.path.join(
        args.output_dir,
        f"results_{args.split}_{'_'.join(args.domains)}.json"
    )
    save_results(results, output_path)
    
    # Save metrics
    metrics_path = os.path.join(
        args.output_dir,
        f"metrics_{args.split}_{'_'.join(args.domains)}.json"
    )
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
