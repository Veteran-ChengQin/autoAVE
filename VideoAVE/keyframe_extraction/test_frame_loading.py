#!/usr/bin/env python3
"""
Test frame loading and processing after fixes
"""
import os
import sys
import json
import numpy as np
import logging

try:
    from PIL import Image
except ImportError:
    print("PIL not available, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_frame_data_format():
    """Test the format of loaded frame data"""
    logger = logging.getLogger(__name__)
    
    try:
        from config import Config
        config = Config()
        
        supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
        
        if not os.path.exists(supervision_file):
            logger.error(f"Supervision file not found: {supervision_file}")
            return False
        
        # Load supervision records
        with open(supervision_file, 'r') as f:
            supervision_records = json.load(f)
        
        logger.info(f"Found {len(supervision_records)} supervision records")
        
        if len(supervision_records) == 0:
            logger.warning("No supervision records found")
            return False
        
        # Test first record
        record = supervision_records[0]
        frames_file = record['frames_file']
        
        if not os.path.exists(frames_file):
            logger.error(f"Frames file not found: {frames_file}")
            return False
        
        # Load frames
        frames = np.load(frames_file)
        logger.info(f"Loaded frames with shape: {frames.shape}")
        logger.info(f"Frame dtype: {frames.dtype}")
        logger.info(f"Frame value range: {frames.min()} - {frames.max()}")
        
        # Test PIL conversion
        if len(frames) > 0:
            test_frame = frames[0]
            logger.info(f"Test frame shape: {test_frame.shape}")
            
            # Ensure correct format for PIL
            if test_frame.dtype != np.uint8:
                if test_frame.max() <= 1.0:
                    test_frame = (test_frame * 255).astype(np.uint8)
                else:
                    test_frame = np.clip(test_frame, 0, 255).astype(np.uint8)
            
            try:
                pil_image = Image.fromarray(test_frame)
                logger.info(f"✓ Successfully converted to PIL image: {pil_image.size}, mode: {pil_image.mode}")
                return True
            except Exception as e:
                logger.error(f"✗ Failed to convert to PIL image: {e}")
                return False
        else:
            logger.warning("No frames to test")
            return False
            
    except Exception as e:
        logger.error(f"Error testing frame data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_teacher_model_feature_extraction():
    """Test teacher model feature extraction with dummy data"""
    logger = logging.getLogger(__name__)
    
    try:
        from config import Config
        from teacher_model import TeacherModel
        
        config = Config()
        
        logger.info("Testing teacher model feature extraction...")
        
        # Create dummy PIL images
        dummy_frames = []
        for i in range(3):
            # Create a random RGB image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img_array)
            dummy_frames.append(pil_img)
        
        logger.info(f"Created {len(dummy_frames)} dummy frames")
        
        # Initialize teacher model
        logger.info("Loading teacher model...")
        teacher = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
        
        # Test feature extraction
        logger.info("Testing feature extraction...")
        features = teacher.extract_frame_features(dummy_frames)
        
        logger.info(f"✓ Feature extraction successful!")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing teacher model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🧪 FRAME LOADING TEST")
    print("Testing frame data format and teacher model feature extraction")
    
    tests = [
        ("Frame Data Format", test_frame_data_format),
        ("Teacher Model Feature Extraction", test_teacher_model_feature_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 50)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<35} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Frame loading and processing work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
