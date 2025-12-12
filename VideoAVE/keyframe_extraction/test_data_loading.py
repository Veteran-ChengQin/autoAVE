#!/usr/bin/env python3
"""
Test script for data loading after format changes
"""
import os
import sys
import json
import numpy as np
import logging

# Add keyframe_extraction to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'keyframe_extraction'))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_separated_format_loading():
    """Test loading data in separated format"""
    logger = logging.getLogger(__name__)
    
    try:
        import config
        config = config.Config()
        
        supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
        frames_dir = os.path.join(config.CACHE_DIR, "supervision_frames")
        
        logger.info("Testing separated format data loading...")
        
        # Check if files exist
        if not os.path.exists(supervision_file):
            logger.error(f"Supervision file not found: {supervision_file}")
            return False
        
        if not os.path.exists(frames_dir):
            logger.error(f"Frames directory not found: {frames_dir}")
            return False
        
        # Load supervision data
        with open(supervision_file, 'r') as f:
            supervision_records = json.load(f)
        
        logger.info(f"Loaded {len(supervision_records)} supervision records")
        
        # Test loading frames for first record
        if len(supervision_records) > 0:
            record = supervision_records[0]
            frames_file = record['frames_file']
            
            logger.info(f"Testing frame loading from: {frames_file}")
            
            if os.path.exists(frames_file):
                frames = np.load(frames_file)
                logger.info(f"✓ Successfully loaded frames with shape: {frames.shape}")
                logger.info(f"  Frame dtype: {frames.dtype}")
                logger.info(f"  Frame value range: {frames.min()} - {frames.max()}")
                
                # Test reconstructing full record
                full_record = {
                    'video_url': record['video_url'],
                    'title': record['title'],
                    'category': record['category'],
                    'ground_truth_attrs': record['ground_truth_attrs'],
                    'frames': frames,
                    'importance_scores': record['importance_scores'],
                    'num_frames': record['num_frames']
                }
                
                logger.info(f"✓ Successfully reconstructed full record")
                logger.info(f"  Video: {record['title'][:50]}...")
                logger.info(f"  Category: {record['category']}")
                logger.info(f"  Attributes: {list(record['ground_truth_attrs'].keys())}")
                logger.info(f"  Importance scores: {len(record['importance_scores'])} values")
                
                return True
            else:
                logger.error(f"Frames file not found: {frames_file}")
                return False
        else:
            logger.warning("No supervision records found")
            return False
            
    except Exception as e:
        logger.error(f"Error testing data loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_train_student_loading():
    """Test the actual train_student_model loading logic"""
    logger = logging.getLogger(__name__)
    
    try:
        import config
        config = config.Config()
        
        logger.info("Testing train_student_model data loading logic...")
        
        # Simulate the loading logic from train_student_model
        supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
        
        if not os.path.exists(supervision_file):
            logger.error(f"Supervision data not found at {supervision_file}")
            return False
        
        with open(supervision_file, 'r') as f:
            supervision_records = json.load(f)
        
        # Load frames from separate numpy files and reconstruct full data
        supervision_data = []
        logger.info(f"Loading frames for {len(supervision_records)} videos...")
        
        for i, record in enumerate(supervision_records):
            # Load frames from numpy file
            frames_file = record['frames_file']
            if os.path.exists(frames_file):
                logger.info(f"Loading frames for video {i+1}/{len(supervision_records)}")
                frames = np.load(frames_file)
                
                # Reconstruct full supervision data format
                full_record = {
                    'video_url': record['video_url'],
                    'title': record['title'],
                    'category': record['category'],
                    'ground_truth_attrs': record['ground_truth_attrs'],
                    'frames': frames,
                    'importance_scores': record['importance_scores'],
                    'num_frames': record['num_frames']
                }
                supervision_data.append(full_record)
            else:
                logger.warning(f"Frames file not found: {frames_file}")
        
        logger.info(f"✓ Successfully loaded {len(supervision_data)} videos with frames")
        
        # Test data structure
        if len(supervision_data) > 0:
            sample = supervision_data[0]
            required_keys = ['video_url', 'title', 'category', 'ground_truth_attrs', 'frames', 'importance_scores', 'num_frames']
            
            for key in required_keys:
                if key not in sample:
                    logger.error(f"Missing key in reconstructed data: {key}")
                    return False
            
            logger.info("✓ All required keys present in reconstructed data")
            logger.info(f"  Sample frames shape: {sample['frames'].shape}")
            logger.info(f"  Sample importance scores: {len(sample['importance_scores'])} values")
            
            return True
        else:
            logger.warning("No data loaded")
            return False
            
    except Exception as e:
        logger.error(f"Error testing train_student loading: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def show_data_info():
    """Show information about current data"""
    logger = logging.getLogger(__name__)
    
    try:
        from keyframe_extraction import Config
        config = Config()
        
        supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
        frames_dir = os.path.join(config.CACHE_DIR, "supervision_frames")
        
        print("\n" + "="*60)
        print("DATA INFORMATION")
        print("="*60)
        
        print(f"Cache directory: {config.CACHE_DIR}")
        print(f"Supervision file: {supervision_file}")
        print(f"Frames directory: {frames_dir}")
        
        # Check supervision file
        if os.path.exists(supervision_file):
            file_size = os.path.getsize(supervision_file)
            print(f"✓ Supervision file exists ({file_size / 1024:.1f} KB)")
            
            with open(supervision_file, 'r') as f:
                data = json.load(f)
            print(f"  Number of videos: {len(data)}")
            
            if len(data) > 0:
                total_frames = sum(item.get('num_frames', 0) for item in data)
                print(f"  Total frames: {total_frames}")
        else:
            print("✗ Supervision file not found")
        
        # Check frames directory
        if os.path.exists(frames_dir):
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.npy')]
            print(f"✓ Frames directory exists ({len(frame_files)} frame files)")
            
            # Calculate total size
            total_size = 0
            for file in frame_files:
                file_path = os.path.join(frames_dir, file)
                total_size += os.path.getsize(file_path)
            print(f"  Total frames size: {total_size / (1024**2):.1f} MB")
        else:
            print("✗ Frames directory not found")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error showing data info: {e}")

def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🧪 DATA LOADING TEST")
    print("Testing the modified data loading logic after format changes")
    
    # Show current data info
    show_data_info()
    
    # Run tests
    tests = [
        ("Separated Format Loading", test_separated_format_loading),
        ("Train Student Loading Logic", test_train_student_loading),
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
        print("\n🎉 All tests passed! The modified loading logic works correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
