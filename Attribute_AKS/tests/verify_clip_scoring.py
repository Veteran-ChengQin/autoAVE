"""
Diagnostic script to verify CLIP scoring is working correctly
"""
import os
import sys
import logging
import numpy as np
from PIL import Image
import torch

from frame_scorer import FrameScorer
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_images(num_images: int = 5) -> list:
    """Create test images with different colors"""
    images = []
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
    ]
    
    for i, color in enumerate(colors[:num_images]):
        img = Image.new('RGB', (224, 224), color=color)
        images.append(img)
    
    logger.info(f"Created {num_images} test images with different colors")
    return images


def test_clip_scoring_basic():
    """Test basic CLIP scoring with different queries"""
    logger.info("=" * 80)
    logger.info("TEST 1: Basic CLIP Scoring with Different Queries")
    logger.info("=" * 80)
    
    # Initialize scorer
    logger.info("Loading CLIP model...")
    scorer = FrameScorer(model_name="clip-vit-base", device=Config.DEVICE)
    
    # Create test images
    images = create_test_images(5)
    
    # Test different queries
    queries = [
        "a red object",
        "a green object",
        "a blue object",
        "a yellow object",
        "a magenta object",
    ]
    
    logger.info("\nScoring test images with different queries:")
    logger.info("-" * 80)
    
    for query in queries:
        logger.info(f"\nQuery: '{query}'")
        scores = scorer.score(query, images)
        
        logger.info(f"  Scores: {[f'{s:.4f}' for s in scores]}")
        logger.info(f"  Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {np.mean(scores):.4f}")
        logger.info(f"  Std: {np.std(scores):.4f}")
        
        # Check if scores are different
        if len(set([round(s, 4) for s in scores])) == 1:
            logger.warning("  ⚠️  WARNING: All scores are identical!")
        else:
            logger.info("  ✓ Scores are different (as expected)")


def test_clip_scoring_same_query():
    """Test CLIP scoring with same query on different images"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Same Query on Different Images")
    logger.info("=" * 80)
    
    # Initialize scorer
    logger.info("Loading CLIP model...")
    scorer = FrameScorer(model_name="clip-vit-base", device=Config.DEVICE)
    
    # Create test images
    images = create_test_images(5)
    
    # Single query
    query = "a red object"
    
    logger.info(f"\nQuery: '{query}'")
    logger.info("Scoring 5 images with different colors:")
    logger.info("-" * 80)
    
    scores = scorer.score(query, images)
    
    for i, score in enumerate(scores):
        logger.info(f"  Image {i}: {score:.4f}")
    
    logger.info(f"\nStatistics:")
    logger.info(f"  Min: {min(scores):.4f}")
    logger.info(f"  Max: {max(scores):.4f}")
    logger.info(f"  Mean: {np.mean(scores):.4f}")
    logger.info(f"  Std: {np.std(scores):.4f}")
    
    # Check if scores are different
    if len(set([round(s, 4) for s in scores])) == 1:
        logger.error("  ❌ ERROR: All scores are identical! CLIP scoring may be broken.")
        return False
    else:
        logger.info("  ✓ Scores are different (correct behavior)")
        return True


def test_clip_model_internals():
    """Test CLIP model internals to debug scoring"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: CLIP Model Internals Debug")
    logger.info("=" * 80)
    
    # Initialize scorer
    logger.info("Loading CLIP model...")
    scorer = FrameScorer(model_name="clip-vit-base", device=Config.DEVICE)
    
    # Create test images
    images = create_test_images(2)
    query = "a red object"
    
    logger.info(f"\nQuery: '{query}'")
    logger.info("Analyzing model outputs...")
    logger.info("-" * 80)
    
    # Manually process through model
    with torch.no_grad():
        # Prepare inputs
        inputs = scorer.processor(
            text=[query] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        ).to(scorer.device)
        
        logger.info(f"\nInput shapes:")
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key}: {val.shape}")
        
        # Forward pass
        outputs = scorer.model(**inputs)
        
        logger.info(f"\nOutput attributes:")
        for attr in dir(outputs):
            if not attr.startswith('_'):
                val = getattr(outputs, attr)
                if isinstance(val, torch.Tensor):
                    logger.info(f"  {attr}: {val.shape}")
        
        # Get logits
        logits = outputs.logits_per_image
        logger.info(f"\nlogits_per_image shape: {logits.shape}")
        logger.info(f"logits_per_image values:\n{logits}")
        
        # Apply sigmoid
        sigmoid_scores = torch.sigmoid(logits)
        logger.info(f"\nAfter sigmoid:\n{sigmoid_scores}")
        
        # Check if all values are the same
        if torch.allclose(sigmoid_scores, sigmoid_scores[0]):
            logger.error("  ❌ ERROR: All sigmoid scores are identical!")
            logger.error("  This suggests the model is not differentiating between images.")
        else:
            logger.info("  ✓ Sigmoid scores are different")


def test_batch_processing():
    """Test batch processing of images"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Batch Processing")
    logger.info("=" * 80)
    
    # Initialize scorer
    logger.info("Loading CLIP model...")
    scorer = FrameScorer(model_name="clip-vit-base", device=Config.DEVICE)
    
    # Create test images
    images = create_test_images(10)
    query = "a colored object"
    
    logger.info(f"\nQuery: '{query}'")
    logger.info(f"Scoring {len(images)} images in batches...")
    logger.info("-" * 80)
    
    scores = scorer.score(query, images)
    
    logger.info(f"\nScores: {[f'{s:.4f}' for s in scores]}")
    logger.info(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {np.mean(scores):.4f}")
    logger.info(f"Std: {np.std(scores):.4f}")
    
    if len(set([round(s, 4) for s in scores])) == 1:
        logger.error("  ❌ ERROR: All scores are identical!")
        return False
    else:
        logger.info("  ✓ Scores are different")
        return True


def main():
    logger.info("Starting CLIP Scoring Verification")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Torch version: {torch.__version__}")
    
    # Run tests
    test_clip_scoring_basic()
    
    result1 = test_clip_scoring_same_query()
    
    test_clip_model_internals()
    
    result2 = test_batch_processing()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    if result1 and result2:
        logger.info("✓ All tests passed! CLIP scoring appears to be working correctly.")
    else:
        logger.error("❌ Some tests failed. CLIP scoring may have issues.")
        logger.error("\nPossible issues:")
        logger.error("1. Model is not properly loaded")
        logger.error("2. Input preprocessing is incorrect")
        logger.error("3. Model output processing is incorrect")
        logger.error("4. Sigmoid normalization is causing issues")


if __name__ == "__main__":
    main()
