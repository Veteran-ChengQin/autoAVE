#!/usr/bin/env python
"""Test frame scorer loading and basic functionality"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from frame_scorer import FrameScorer
    from PIL import Image
    import numpy as np
    
    logger.info("Testing FrameScorer...")
    
    # Initialize frame scorer
    logger.info("Initializing FrameScorer with CLIP model...")
    scorer = FrameScorer(model_name="clip-vit-base", device="cuda:0")
    logger.info("✓ FrameScorer initialized successfully")
    
    # Create dummy frames for testing
    logger.info("Creating test frames...")
    dummy_frame = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    frames = [dummy_frame] * 3
    logger.info(f"✓ Created {len(frames)} test frames")
    
    # Test scoring
    logger.info("Testing frame scoring...")
    text = "A product photo showing the color red"
    scores = scorer.score(text, frames)
    logger.info(f"✓ Scoring successful, scores: {scores}")
    
    if len(scores) == len(frames):
        logger.info("✓ Score count matches frame count")
    else:
        logger.error(f"✗ Score count mismatch: {len(scores)} vs {len(frames)}")
        sys.exit(1)
    
    logger.info("\n✓ All tests passed!")
    
except Exception as e:
    logger.error(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
