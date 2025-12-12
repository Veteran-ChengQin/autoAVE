#!/usr/bin/env python3
"""
Convert large JSON file with embedded frames to separated format
"""
import os
import sys
import json
import numpy as np
import logging
from datetime import datetime

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'convert_json_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def convert_large_json(large_json_path: str, cache_dir: str = ".cache") -> bool:
    """
    Convert a large JSON file with embedded frames to separated format
    
    Args:
        large_json_path: Path to the large JSON file
        cache_dir: Cache directory for output
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Converting large JSON file: {large_json_path}")
        
        # Check if file exists
        if not os.path.exists(large_json_path):
            logger.error(f"File not found: {large_json_path}")
            return False
        
        # Create output directories
        supervision_file = os.path.join(cache_dir, "teacher_supervision_data.json")
        frames_dir = os.path.join(cache_dir, "supervision_frames")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get file size
        file_size = os.path.getsize(large_json_path)
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")
        
        if file_size > 10 * 1024**3:  # > 10GB
            logger.error("File too large to load into memory safely.")
            logger.info("Consider processing in smaller chunks or using streaming approach.")
            return False
        
        # Load JSON data
        logger.info("Loading JSON file into memory (this may take several minutes for large files)...")
        try:
            with open(large_json_path, 'r') as f:
                data = json.load(f)
        except MemoryError:
            logger.error("Not enough memory to load the JSON file.")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return False
        
        logger.info(f"Loaded {len(data)} records from JSON")
        
        # Convert to separated format
        supervision_data = []
        
        for i, item in enumerate(data):
            logger.info(f"Processing record {i+1}/{len(data)}")
            
            # Save frames separately as numpy files
            if 'frames' in item:
                frames_file = os.path.join(frames_dir, f"video_{i}_frames.npy")
                
                # Convert frames to numpy array if they're lists
                frames = item['frames']
                if isinstance(frames, list):
                    logger.info(f"  Converting frames list to numpy array...")
                    frames = np.array(frames, dtype=np.uint8)  # Assume uint8 for images
                
                # Save frames
                logger.info(f"  Saving frames to {frames_file}")
                np.save(frames_file, frames)
                
                # Create supervision record without frames
                supervision_record = {
                    'video_id': i,
                    'video_url': item.get('video_url', ''),
                    'title': item.get('title', ''),
                    'category': item.get('category', ''),
                    'ground_truth_attrs': item.get('ground_truth_attrs', {}),
                    'importance_scores': item.get('importance_scores', []),
                    'num_frames': item.get('num_frames', len(item.get('importance_scores', []))),
                    'frames_file': frames_file
                }
                supervision_data.append(supervision_record)
                
                # Log progress
                frames_shape = frames.shape if hasattr(frames, 'shape') else 'unknown'
                logger.info(f"  Video {i}: {frames_shape} frames saved")
        
        # Save lightweight supervision data
        logger.info("Saving supervision data...")
        with open(supervision_file, 'w') as f:
            json.dump(supervision_data, f, indent=2)
        
        logger.info("Conversion completed successfully!")
        logger.info(f"Supervision data saved to: {supervision_file}")
        logger.info(f"Frame data saved to: {frames_dir}")
        
        # Show size comparison
        new_json_size = os.path.getsize(supervision_file)
        
        # Calculate frames directory size
        frames_dir_size = 0
        for root, dirs, files in os.walk(frames_dir):
            for file in files:
                file_path = os.path.join(root, file)
                frames_dir_size += os.path.getsize(file_path)
        
        logger.info("\n" + "="*60)
        logger.info("SIZE COMPARISON")
        logger.info("="*60)
        logger.info(f"Original JSON file: {file_size / (1024**2):.1f} MB")
        logger.info(f"New supervision JSON: {new_json_size / (1024**2):.1f} MB")
        logger.info(f"Frames directory: {frames_dir_size / (1024**2):.1f} MB")
        logger.info(f"Total new size: {(new_json_size + frames_dir_size) / (1024**2):.1f} MB")
        logger.info(f"JSON size reduction: {(1 - new_json_size/file_size)*100:.1f}%")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting JSON file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert large JSON file to separated format')
    parser.add_argument('json_file', help='Path to the large JSON file')
    parser.add_argument('--cache_dir', default='.cache', help='Cache directory for output (default: .cache)')
    parser.add_argument('--backup', action='store_true', help='Create backup of original file')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check input file
    if not os.path.exists(args.json_file):
        logger.error(f"Input file not found: {args.json_file}")
        return 1
    
    # Create backup if requested
    if args.backup:
        backup_path = args.json_file + '.backup'
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup: {backup_path}")
            import shutil
            shutil.copy2(args.json_file, backup_path)
        else:
            logger.info(f"Backup already exists: {backup_path}")
    
    # Convert file
    logger.info("Starting conversion...")
    success = convert_large_json(args.json_file, args.cache_dir)
    
    if success:
        logger.info("✅ Conversion completed successfully!")
        logger.info(f"You can now delete the original large JSON file: {args.json_file}")
        logger.info(f"The new format is stored in: {args.cache_dir}")
        return 0
    else:
        logger.error("❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
