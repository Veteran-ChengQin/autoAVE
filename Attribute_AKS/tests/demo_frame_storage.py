"""
Demo script showing frame storage and CLIP scoring verification
"""
import os
import logging
from PIL import Image
import json

from config import Config
from frame_scorer import FrameScorer
from frame_storage import FrameStorage
from video_utils import extract_candidate_frames

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_frames(num_frames: int = 10) -> tuple:
    """Create demo frames with different colors"""
    frames = []
    timestamps = []
    
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (128, 0, 0),      # Dark Red
        (0, 128, 0),      # Dark Green
        (0, 0, 128),      # Dark Blue
        (128, 128, 0),    # Olive
    ]
    
    for i in range(num_frames):
        color = colors[i % len(colors)]
        img = Image.new('RGB', (224, 224), color=color)
        frames.append(img)
        timestamps.append(float(i))
    
    return frames, timestamps


def main():
    logger.info("=" * 80)
    logger.info("FRAME STORAGE AND CLIP SCORING DEMO")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("\nInitializing components...")
    scorer = FrameScorer(model_name="clip-vit-base", device=Config.DEVICE)
    storage = FrameStorage(storage_root=os.path.join(Config.ATTRIBUTE_AKS_ROOT, "demo_frames"))
    
    # Create demo data
    logger.info("\nCreating demo frames...")
    frames, timestamps = create_demo_frames(10)
    
    # Score frames
    logger.info("\nScoring frames with CLIP...")
    query = "a red object"
    scores = scorer.score(query, frames)
    
    logger.info(f"Query: '{query}'")
    logger.info(f"Scores: {[f'{s:.4f}' for s in scores]}")
    logger.info(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {sum(scores)/len(scores):.4f}")
    
    # Save candidate frames
    logger.info("\nSaving candidate frames...")
    product_id = "demo_product_001"
    attr_name = "Color"
    
    candidates_dir = storage.save_candidate_frames(
        product_id, attr_name, frames, timestamps, scores
    )
    logger.info(f"Saved to: {candidates_dir}")
    
    # Select keyframes (top 3)
    logger.info("\nSelecting top 3 keyframes...")
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    keyframes = [frames[i] for i in sorted_indices]
    keyframe_timestamps = [timestamps[i] for i in sorted_indices]
    keyframe_scores = [scores[i] for i in sorted_indices]
    
    logger.info(f"Selected indices: {sorted_indices}")
    logger.info(f"Selected scores: {[f'{s:.4f}' for s in keyframe_scores]}")
    
    # Save keyframes
    logger.info("\nSaving keyframes...")
    keyframes_dir = storage.save_keyframes(
        product_id, attr_name, keyframes, keyframe_timestamps, sorted_indices, keyframe_scores
    )
    logger.info(f"Saved to: {keyframes_dir}")
    
    # Save scoring analysis
    logger.info("\nGenerating scoring analysis...")
    analysis_dir = storage.save_scoring_analysis(
        product_id, attr_name, frames, scores, timestamps, sorted_indices
    )
    logger.info(f"Saved to: {analysis_dir}")
    
    # Get product summary
    logger.info("\nGenerating product summary...")
    summary = storage.get_product_summary(product_id)
    logger.info(f"Summary:\n{json.dumps(summary, indent=2)}")
    
    # Print file structure
    logger.info("\n" + "=" * 80)
    logger.info("FILE STRUCTURE")
    logger.info("=" * 80)
    
    storage_root = storage.storage_root
    for root, dirs, files in os.walk(storage_root):
        level = root.replace(storage_root, '').count(os.sep)
        indent = ' ' * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            logger.info(f"{subindent}{file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✓ Candidate frames stored in: {candidates_dir}")
    logger.info(f"✓ Keyframes stored in: {keyframes_dir}")
    logger.info(f"✓ Analysis saved in: {analysis_dir}")
    logger.info(f"\nYou can now review the frames for quality assurance!")


if __name__ == "__main__":
    main()
