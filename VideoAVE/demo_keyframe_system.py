#!/usr/bin/env python3
"""
Demo script for keyframe extraction system
This script demonstrates the basic functionality with a small dataset
"""
import os
import sys
import logging

# Add keyframe_extraction to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'keyframe_extraction'))

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def demo_teacher_supervision():
    """Demo teacher supervision generation"""
    print("\n" + "="*60)
    print("DEMO: TEACHER SUPERVISION GENERATION")
    print("="*60)
    
    from keyframe_extraction import Config, TeacherModel, VideoLoader, FrameScorer, load_videoave_data
    
    # Initialize config
    config = Config()
    config.DEVICE = "cuda:0"  # Adjust if needed
    
    # Create cache directory
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Cache directory: {config.CACHE_DIR}")
    
    # Load a small sample of data
    print("\nLoading sample data...")
    video_data = load_videoave_data(
        config.DATASET_ROOT,
        domains=['beauty'],  # Just beauty domain
        max_samples_per_domain=2  # Just 2 videos for demo
    )
    
    print(f"Loaded {len(video_data)} videos")
    for i, video in enumerate(video_data):
        print(f"  Video {i+1}: {video['title'][:50]}...")
        print(f"    Attributes: {list(video['ground_truth_attrs'].keys())}")
    
    # Initialize components
    print("\nInitializing models...")
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    video_loader = VideoLoader(config.CACHE_DIR)
    frame_scorer = FrameScorer(
        teacher_model, video_loader,
        baseline_size=8,  # Smaller for demo
        num_mc_samples=2,  # Fewer samples for demo
        cache_dir=config.CACHE_DIR
    )
    
    # Generate scores for one video
    print("\nGenerating frame importance scores...")
    sample_video = video_data[0]
    
    try:
        frames, scores = frame_scorer.compute_frame_importance_scores(
            sample_video['video_url'],
            sample_video['title'],
            sample_video['category'],
            sample_video['ground_truth_attrs'],
            target_fps=1.0,  # Lower FPS for demo
            max_frames=16    # Fewer frames for demo
        )
        
        print(f"✓ Generated scores for {len(scores)} frames")
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Top 3 frame indices: {sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating scores: {e}")
        return False

def demo_student_training():
    """Demo student model training with minimal data"""
    print("\n" + "="*60)
    print("DEMO: STUDENT MODEL TRAINING")
    print("="*60)
    
    from keyframe_extraction import (
        Config, TeacherModel, FrameImportancePredictor, 
        StudentTrainer, load_videoave_data
    )
    import numpy as np
    
    config = Config()
    config.DEVICE = "cuda:0"
    
    # Create mock supervision data for demo
    print("Creating mock supervision data...")
    mock_data = []
    
    for i in range(3):  # 3 mock videos
        num_frames = 8
        frames = np.random.randint(0, 255, (num_frames, 224, 224, 3), dtype=np.uint8)
        scores = np.random.random(num_frames).tolist()
        
        mock_data.append({
            'video_url': f'mock_video_{i}',
            'title': f'Mock Product {i}',
            'category': 'beauty',
            'ground_truth_attrs': {'Color': 'Red', 'Brand': 'TestBrand'},
            'frames': frames,
            'importance_scores': scores,
            'num_frames': num_frames
        })
    
    print(f"Created {len(mock_data)} mock videos with supervision data")
    
    # Initialize models
    print("Initializing models...")
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    student_model = FrameImportancePredictor(config.VISION_DIM, config.HIDDEN_DIM)
    trainer = StudentTrainer(student_model, teacher_model, config.DEVICE)
    
    # Train with minimal epochs
    print("Training student model...")
    try:
        history = trainer.train(
            mock_data,
            num_epochs=2,  # Very few epochs for demo
            batch_size=4,
            learning_rate=1e-3
        )
        
        print("✓ Training completed!")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        return None

def demo_evaluation():
    """Demo evaluation with baseline strategies"""
    print("\n" + "="*60)
    print("DEMO: EVALUATION COMPARISON")
    print("="*60)
    
    from keyframe_extraction import Config, TeacherModel, VideoLoader, KeyframeEvaluator, load_videoave_data
    
    config = Config()
    config.DEVICE = "cuda:0"
    
    # Load sample data
    print("Loading evaluation data...")
    video_data = load_videoave_data(
        config.DATASET_ROOT,
        domains=['beauty'],
        max_samples_per_domain=2  # Small sample for demo
    )
    
    # Initialize components
    teacher_model = TeacherModel(config.TEACHER_MODEL_PATH, config.DEVICE)
    video_loader = VideoLoader(config.CACHE_DIR)
    evaluator = KeyframeEvaluator(teacher_model, video_loader)
    
    # Evaluate baseline strategies only (no student model)
    print("Evaluating baseline strategies...")
    
    try:
        results = evaluator.compare_strategies(
            video_data,
            student_trainer=None,  # No student model for demo
            top_k_values=[4, 8]    # Fewer K values for demo
        )
        
        print("✓ Evaluation completed!")
        evaluator.print_comparison_results(results)
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return False

def main():
    """Run demo"""
    setup_logging()
    
    print("🎬 KEYFRAME EXTRACTION SYSTEM DEMO")
    print("This demo shows the basic functionality with minimal data")
    print("For full functionality, use: python keyframe_extraction/main.py")
    
    # Check if we should run demos
    print("\nAvailable demos:")
    print("1. Teacher supervision generation")
    print("2. Student model training (with mock data)")
    print("3. Evaluation comparison")
    
    # choice = input("\nWhich demo would you like to run? (1/2/3/all): ").strip().lower()
    choice = '2'
    
    if choice in ['1', 'all']:
        success = demo_teacher_supervision()
        if not success:
            print("Teacher supervision demo failed. Check your GPU and model paths.")
    
    if choice in ['2', 'all']:
        trainer = demo_student_training()
        if trainer is None:
            print("Student training demo failed.")
    
    if choice in ['3', 'all']:
        success = demo_evaluation()
        if not success:
            print("Evaluation demo failed.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("For production use:")
    print("cd keyframe_extraction")
    print("python main.py --mode full --max_videos 10 --eval_videos 20")

if __name__ == "__main__":
    main()
