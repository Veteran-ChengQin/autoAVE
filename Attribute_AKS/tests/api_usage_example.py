"""
Example: Using QwenVLExtractor with API mode for video attribute extraction

This example demonstrates how to:
1. Initialize QwenVLExtractor in API mode
2. Extract frames from a video
3. Extract product attributes using the API
"""
import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vl_extractor import QwenVLExtractor
from video_utils import extract_candidate_frames
from PIL import Image
load_dotenv('./Attribute_AKS/.env')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_attributes_from_video_api(
    video_path: str,
    attributes: list,
    api_key: str,
    product_title: str = "",
    product_category: str = ""
):
    """
    Extract product attributes from video using API mode.
    
    Args:
        video_path: Path to video file
        attributes: List of attribute names to extract
        api_key: Dashscope API key
        product_title: Product title (optional)
        product_category: Product category (optional)
    
    Returns:
        Dictionary mapping attribute names to extracted values
    """
    
    # Initialize extractor in API mode
    extractor = QwenVLExtractor(
        mode="api",
        api_key=api_key,
        api_model="qwen-vl-plus"  # Options: "qwen-vl-plus", "qwen-vl-max"
    )
    
    logger.info(f"Extracting attributes from video: {video_path}")
    logger.info(f"Target attributes: {attributes}")
    
    # Extract frames from video
    # You can adjust num_frames based on video length and quality needs
    frames = extract_candidate_frames(video_path, num_frames=8)
    logger.info(f"Extracted {len(frames)} frames from video")
    
    # Method 1: Extract all attributes at once (recommended for efficiency)
    attr_keyframes = {attr: frames for attr in attributes}
    results = extractor.extract_multi_attr(
        attr_keyframes=attr_keyframes,
        title=product_title,
        category=product_category
    )
    
    return results


def extract_single_attribute_api(
    video_path: str,
    attribute_name: str,
    api_key: str,
    product_title: str = "",
    product_category: str = ""
):
    """
    Extract a single attribute from video using API mode.
    
    Args:
        video_path: Path to video file
        attribute_name: Name of attribute to extract
        api_key: Dashscope API key
        product_title: Product title (optional)
        product_category: Product category (optional)
    
    Returns:
        Dictionary with attribute name and extracted value
    """
    
    # Initialize extractor in API mode
    extractor = QwenVLExtractor(
        mode="api",
        api_key=api_key,
        api_model="qwen2.5-vl-7b-instruct"
    )
    
    logger.info(f"Extracting attribute '{attribute_name}' from video: {video_path}")
    
    # Extract frames from video
    frames = extract_single_attr_from_video(video_path, max_frames=8)
    logger.info(f"Extracted {len(frames)} frames from video")
    
    # Extract single attribute
    result = extractor.extract_single_attr(
        keyframes=frames,
        attr_name=attribute_name,
        title=product_title,
        category=product_category
    )
    
    return result


def main():
    """Main example function"""
    
    # Configuration
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("Please set DASHSCOPE_API_KEY environment variable")
        logger.info("Example: export DASHSCOPE_API_KEY='your-api-key-here'")
        return
    
    # Example video path
    video_path = "/data/veteran/project/dataE/Attribute_AKS/.cache/videos/21cc8d6834ae80202079b205ff2bc23d.mp4"
    
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        logger.info("Please update the video_path variable with a valid video file")
        return
    product_title = "Neutrogena Hydro Boost Hydrating Foundation Stick with Hyaluronic Acid, Oil-Free & Non-Comedogenic Moisturizing Makeup for Smooth Coverage & Radiant-Looking Skin, Classic Ivory, 0.29 oz"

    # Product information
    product_category = "Beauty"
    
    # Attributes to extract
    attributes = ['Item Form', 'Color', 'Skin Type', 'Finish Type']
    
    # Example 1: Extract multiple attributes at once
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Multi-attribute extraction")
    logger.info("=" * 60)
    
    # results = extract_attributes_from_video_api(
    #     video_path=video_path,
    #     attributes=attributes,
    #     api_key=api_key,
    #     product_title=product_title,
    #     product_category=product_category
    # )
    
    # logger.info("\nExtraction Results:")
    # for attr, value in results.items():
    #     logger.info(f"  {attr}: {value}")
    
    # # Example 2: Extract single attribute
    # logger.info("\n" + "=" * 60)
    # logger.info("Example 2: Single attribute extraction")
    # logger.info("=" * 60)
    
    result = extract_single_attribute_api(
        video_path=video_path,
        attribute_name="Color",
        api_key=api_key,
        product_title=product_title,
        product_category=product_category
    )
    
    logger.info(f"\nExtraction Result: {result}")


if __name__ == "__main__":
    main()
