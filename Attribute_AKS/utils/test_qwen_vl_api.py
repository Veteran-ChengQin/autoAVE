"""
Test script for QwenVLExtractor with API mode
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_vl_extractor import QwenVLExtractor
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_mode():
    """Test QwenVLExtractor in API mode"""
    
    # Initialize extractor in API mode
    # Note: Replace with your actual API key
    api_key = os.environ.get("DASHSCOPE_API_KEY", "your-api-key-here")
    
    extractor = QwenVLExtractor(
        mode="api",
        api_key=api_key,
        api_model="qwen-vl-plus"  # or "qwen-vl-max"
    )
    
    logger.info("QwenVLExtractor initialized in API mode")
    
    # Test with sample images
    # You can replace these with actual product images
    sample_image_path = "path/to/your/test/image.jpg"
    
    if os.path.exists(sample_image_path):
        image = Image.open(sample_image_path)
        keyframes = [image]
        
        # Test single attribute extraction
        result = extractor.extract_single_attr(
            keyframes=keyframes,
            attr_name="Color",
            title="Test Product",
            category="Test Category"
        )
        logger.info(f"Single attribute result: {result}")
        
        # Test multi-attribute extraction
        attr_keyframes = {
            "Color": keyframes,
            "Material": keyframes,
            "Size": keyframes
        }
        
        result = extractor.extract_multi_attr(
            attr_keyframes=attr_keyframes,
            title="Test Product",
            category="Test Category"
        )
        logger.info(f"Multi-attribute result: {result}")
    else:
        logger.warning(f"Sample image not found at {sample_image_path}")
        logger.info("Please update the sample_image_path variable with a valid image path")


def test_local_mode():
    """Test QwenVLExtractor in local mode"""
    
    model_path = "/path/to/your/local/qwen-vl-model"
    
    if os.path.exists(model_path):
        extractor = QwenVLExtractor(
            model_path=model_path,
            device="cuda:0",
            mode="local"
        )
        
        logger.info("QwenVLExtractor initialized in local mode")
        
        # Add your test code here similar to test_api_mode
    else:
        logger.warning(f"Model path not found: {model_path}")
        logger.info("Skipping local mode test")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Testing QwenVLExtractor")
    logger.info("=" * 50)
    
    # Test API mode
    logger.info("\n--- Testing API Mode ---")
    try:
        test_api_mode()
    except Exception as e:
        logger.error(f"API mode test failed: {e}", exc_info=True)
    
    # Test local mode (optional)
    logger.info("\n--- Testing Local Mode ---")
    try:
        test_local_mode()
    except Exception as e:
        logger.error(f"Local mode test failed: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 50)
    logger.info("Testing completed")
    logger.info("=" * 50)
