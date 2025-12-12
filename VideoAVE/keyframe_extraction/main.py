"""
Main entry point for keyframe extraction system
"""
import os
import sys
import argparse
import logging
import json
import torch
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keyframe_extraction import (
    Config, TeacherModel, VideoLoader, FrameScorer, 
    FrameImportancePredictor, StudentTrainer, 
    load_videoave_data, run_evaluation_pipeline
)

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'keyframe_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def generate_teacher_supervision(config: Config, max_videos_per_domain: int = 10):
    """Generate teacher supervision data (frame importance scores)"""
    logger = logging.getLogger(__name__)
    logger.info("Starting teacher supervision generation...")
    
    # Initialize components
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    video_loader = VideoLoader(config.CACHE_DIR)
    frame_scorer = FrameScorer(
        teacher_model, video_loader, 
        config.BASELINE_FRAMES, config.MONTE_CARLO_SAMPLES, 
        config.CACHE_DIR
    )
    
    # Load video data
    video_data = load_videoave_data(
        config.DATASET_ROOT, 
        domains=config.PROTOTYPE_DOMAINS,
        max_samples_per_domain=max_videos_per_domain
    )
    
    logger.info(f"Loaded {len(video_data)} videos for supervision generation")
    
    # Generate frame importance scores
    scored_data = frame_scorer.batch_compute_scores(
        video_data, config.TARGET_FPS, config.MAX_FRAMES
    )
    
    # Save results - separate frames and supervision data to avoid huge JSON files
    supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
    frames_dir = os.path.join(config.CACHE_DIR, "supervision_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Prepare lightweight supervision data (without frames)
    supervision_data = []
    
    for i, item in enumerate(scored_data):
        # Save frames separately as numpy files
        if 'frames' in item:
            frames_file = os.path.join(frames_dir, f"video_{i}_frames.npy")
            np.save(frames_file, item['frames'])
            
            # Create supervision record without frames
            supervision_record = {
                'video_id': i,
                'video_url': item['video_url'],
                'title': item['title'],
                'category': item['category'],
                'ground_truth_attrs': item['ground_truth_attrs'],
                'importance_scores': item['importance_scores'],
                'num_frames': item['num_frames'],
                'frames_file': frames_file  # Reference to frames file
            }
            supervision_data.append(supervision_record)
    
    # Save lightweight supervision data as JSON
    with open(supervision_file, 'w') as f:
        json.dump(supervision_data, f, indent=2)
    
    logger.info(f"Teacher supervision data saved to {supervision_file}")
    logger.info(f"Frame data saved to {frames_dir} directory")
    logger.info(f"Total supervision records: {len(supervision_data)}")
    
    return scored_data

def train_student_model(config: Config, supervision_data: list = None):
    """Train the student frame importance predictor"""
    logger = logging.getLogger(__name__)
    logger.info("Starting student model training...")
    
    # Load supervision data if not provided
    if supervision_data is None:
        supervision_file = os.path.join(config.CACHE_DIR, "teacher_supervision_data.json")
        if not os.path.exists(supervision_file):
            logger.error(f"Supervision data not found at {supervision_file}")
            logger.info("Please run teacher supervision generation first")
            return None
        
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
        
        logger.info(f"Successfully loaded {len(supervision_data)} videos with frames")
    
    # Initialize models
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    student_model = FrameImportancePredictor(config.VISION_DIM, config.HIDDEN_DIM)
    trainer = StudentTrainer(student_model, teacher_model, config.DEVICE)
    
    # Train the model
    history = trainer.train(
        supervision_data,
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE
    )
    
    # Save the trained model
    model_path = os.path.join(config.CACHE_DIR, "student_model.pth")
    trainer.save_model(model_path)
    
    # Save training history
    history_path = os.path.join(config.CACHE_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Student model saved to {model_path}")
    logger.info(f"Training history saved to {history_path}")
    
    return trainer

def evaluate_system(config: Config, max_videos_per_domain: int = 20):
    """Evaluate the keyframe selection system"""
    logger = logging.getLogger(__name__)
    logger.info("Starting system evaluation...")
    
    # Initialize teacher model
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    
    # Load student model if available
    student_trainer = None
    model_path = os.path.join(config.CACHE_DIR, "student_model.pth")
    
    if os.path.exists(model_path):
        logger.info("Loading trained student model...")
        student_model = FrameImportancePredictor(config.VISION_DIM, config.HIDDEN_DIM)
        student_trainer = StudentTrainer(student_model, teacher_model, config.DEVICE)
        student_trainer.load_model(model_path)
    else:
        logger.warning("No trained student model found. Will only evaluate baseline strategies.")
    
    # Run evaluation
    results = run_evaluation_pipeline(
        config, teacher_model, student_trainer, max_videos_per_domain
    )
    
    # Save results
    results_path = os.path.join(config.CACHE_DIR, "evaluation_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    results_json = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Keyframe Extraction System for VideoAVE')
    parser.add_argument('--mode', choices=['supervision', 'train', 'evaluate', 'full'], 
                       default='train', help='Mode to run')
    parser.add_argument('--max_videos', type=int, default=5, 
                       help='Maximum videos per domain for supervision/training')
    parser.add_argument('--eval_videos', type=int, default=10,
                       help='Maximum videos per domain for evaluation')
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = Config()
    config.DEVICE = args.device
    
    # Create cache directories
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.FEATURES_CACHE_DIR, exist_ok=True)
    os.makedirs(config.SCORES_CACHE_DIR, exist_ok=True)
    
    logger.info(f"Starting keyframe extraction system in {args.mode} mode")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Cache directory: {config.CACHE_DIR}")
    
    try:
        if args.mode in ['supervision', 'full']:
            logger.info("=" * 60)
            logger.info("STEP 1: GENERATING TEACHER SUPERVISION")
            logger.info("=" * 60)
            supervision_data = generate_teacher_supervision(config, args.max_videos)
        else:
            supervision_data = None
        
        if args.mode in ['train', 'full']:
            logger.info("=" * 60)
            logger.info("STEP 2: TRAINING STUDENT MODEL")
            logger.info("=" * 60)
            student_trainer = train_student_model(config, supervision_data)
        
        if args.mode in ['evaluate', 'full']:
            logger.info("=" * 60)
            logger.info("STEP 3: EVALUATING SYSTEM")
            logger.info("=" * 60)
            results = evaluate_system(config, args.eval_videos)
        
        logger.info("=" * 60)
        logger.info("KEYFRAME EXTRACTION SYSTEM COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in keyframe extraction system: {e}")
        raise

if __name__ == "__main__":
    main()
