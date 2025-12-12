"""
Evaluation utilities for keyframe selection
"""
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from PIL import Image
import logging
from tqdm import tqdm

from .teacher_model import TeacherModel, fuzzy_f1_score
from .student_model import StudentTrainer
from .video_loader import VideoLoader

logger = logging.getLogger(__name__)

class KeyframeEvaluator:
    """
    Evaluator for keyframe selection strategies
    """
    def __init__(self, teacher_model: TeacherModel, video_loader: VideoLoader):
        self.teacher = teacher_model
        self.video_loader = video_loader
    
    def evaluate_selection_strategy(self, video_data: List[Dict[str, Any]], 
                                  selection_strategy: str, student_trainer: StudentTrainer = None,
                                  top_k_values: List[int] = [4, 8, 16]) -> Dict[str, Any]:
        """
        Evaluate a keyframe selection strategy
        
        Args:
            video_data: List of video data dictionaries
            selection_strategy: Strategy name ('uniform', 'random', 'student', 'first_k')
            student_trainer: Student trainer (required for 'student' strategy)
            top_k_values: List of K values to test
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'strategy': selection_strategy,
            'results_by_k': {},
            'overall_results': {}
        }
        
        for k in top_k_values:
            logger.info(f"Evaluating {selection_strategy} strategy with K={k}")
            
            f1_scores = []
            
            for video_info in tqdm(video_data, desc=f"Evaluating K={k}"):
                try:
                    # Get video frames
                    video_path, frames, timestamps = self.video_loader.get_cached_frames(
                        video_info['video_url'], target_fps=2.0, max_frames=64
                    )
                    
                    if len(frames) == 0:
                        continue
                    
                    # Convert to PIL images
                    pil_frames = self.video_loader.frames_to_pil(frames)
                    
                    # Select keyframes based on strategy
                    selected_indices = self._select_keyframes(
                        pil_frames, selection_strategy, k, student_trainer
                    )
                    
                    if len(selected_indices) == 0:
                        continue
                    
                    # Get selected frames
                    selected_frames = [pil_frames[i] for i in selected_indices]
                    
                    # Get ground truth attributes
                    ground_truth = video_info['ground_truth_attrs']
                    attribute_names = list(ground_truth.keys())
                    
                    # Predict attributes using teacher model
                    predictions = self.teacher.predict_attributes(
                        selected_frames, 
                        video_info['title'], 
                        video_info['category'],
                        mode="attribute_conditioned",
                        attributes=attribute_names
                    )
                    
                    # Compute F1 score
                    f1 = fuzzy_f1_score(predictions, ground_truth)
                    f1_scores.append(f1)
                    
                except Exception as e:
                    logger.error(f"Error evaluating video {video_info['video_url']}: {e}")
                    continue
            
            # Compute statistics for this K
            if f1_scores:
                results['results_by_k'][k] = {
                    'mean_f1': np.mean(f1_scores),
                    'std_f1': np.std(f1_scores),
                    'median_f1': np.median(f1_scores),
                    'num_videos': len(f1_scores),
                    'all_f1_scores': f1_scores
                }
                
                logger.info(f"K={k}: Mean F1 = {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            else:
                results['results_by_k'][k] = {
                    'mean_f1': 0.0,
                    'std_f1': 0.0,
                    'median_f1': 0.0,
                    'num_videos': 0,
                    'all_f1_scores': []
                }
        
        # Compute overall statistics
        all_f1_scores = []
        for k_results in results['results_by_k'].values():
            all_f1_scores.extend(k_results['all_f1_scores'])
        
        if all_f1_scores:
            results['overall_results'] = {
                'mean_f1': np.mean(all_f1_scores),
                'std_f1': np.std(all_f1_scores),
                'median_f1': np.median(all_f1_scores),
                'total_evaluations': len(all_f1_scores)
            }
        
        return results
    
    def _select_keyframes(self, frames: List[Image.Image], strategy: str, k: int, 
                         student_trainer: StudentTrainer = None) -> List[int]:
        """
        Select keyframes using specified strategy
        
        Args:
            frames: List of PIL Images
            strategy: Selection strategy
            k: Number of frames to select
            student_trainer: Student trainer (for 'student' strategy)
            
        Returns:
            List of selected frame indices
        """
        num_frames = len(frames)
        k = min(k, num_frames)  # Don't select more frames than available
        
        if strategy == 'uniform':
            # Uniformly spaced frames
            if k == 1:
                return [num_frames // 2]
            indices = np.linspace(0, num_frames - 1, k, dtype=int)
            return indices.tolist()
        
        elif strategy == 'random':
            # Random selection
            indices = np.random.choice(num_frames, k, replace=False)
            return sorted(indices.tolist())
        
        elif strategy == 'first_k':
            # First K frames
            return list(range(min(k, num_frames)))
        
        elif strategy == 'student':
            # Use student model predictions
            if student_trainer is None:
                raise ValueError("Student trainer required for 'student' strategy")
            
            # Get importance scores from student model
            importance_scores = student_trainer.predict_frame_importance(frames)
            
            # Select top-K frames
            top_indices = np.argsort(importance_scores)[-k:]
            return sorted(top_indices.tolist())
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def compare_strategies(self, video_data: List[Dict[str, Any]], 
                          student_trainer: StudentTrainer = None,
                          top_k_values: List[int] = [4, 8, 16]) -> Dict[str, Any]:
        """
        Compare multiple keyframe selection strategies
        
        Args:
            video_data: List of video data dictionaries
            student_trainer: Student trainer (for student strategy)
            top_k_values: List of K values to test
            
        Returns:
            Dictionary with comparison results
        """
        strategies = ['uniform', 'random', 'first_k']
        if student_trainer is not None:
            strategies.append('student')
        
        comparison_results = {}
        
        for strategy in strategies:
            logger.info(f"Evaluating {strategy} strategy...")
            results = self.evaluate_selection_strategy(
                video_data, strategy, student_trainer, top_k_values
            )
            comparison_results[strategy] = results
        
        # Create summary comparison
        summary = self._create_comparison_summary(comparison_results, top_k_values)
        comparison_results['summary'] = summary
        
        return comparison_results
    
    def _create_comparison_summary(self, results: Dict[str, Any], 
                                 top_k_values: List[int]) -> Dict[str, Any]:
        """Create summary comparison table"""
        summary = {
            'by_k': {},
            'overall': {}
        }
        
        # Summary by K value
        for k in top_k_values:
            summary['by_k'][k] = {}
            for strategy, strategy_results in results.items():
                if strategy == 'summary':
                    continue
                
                k_results = strategy_results['results_by_k'].get(k, {})
                summary['by_k'][k][strategy] = {
                    'mean_f1': k_results.get('mean_f1', 0.0),
                    'std_f1': k_results.get('std_f1', 0.0),
                    'num_videos': k_results.get('num_videos', 0)
                }
        
        # Overall summary
        for strategy, strategy_results in results.items():
            if strategy == 'summary':
                continue
            
            overall_results = strategy_results['overall_results']
            summary['overall'][strategy] = {
                'mean_f1': overall_results.get('mean_f1', 0.0),
                'std_f1': overall_results.get('std_f1', 0.0),
                'total_evaluations': overall_results.get('total_evaluations', 0)
            }
        
        return summary
    
    def print_comparison_results(self, comparison_results: Dict[str, Any]):
        """Print formatted comparison results"""
        summary = comparison_results['summary']
        
        print("\n" + "="*80)
        print("KEYFRAME SELECTION STRATEGY COMPARISON")
        print("="*80)
        
        # Results by K
        print("\nResults by K value:")
        print("-" * 60)
        
        for k, k_results in summary['by_k'].items():
            print(f"\nK = {k} frames:")
            print(f"{'Strategy':<15} {'Mean F1':<10} {'Std F1':<10} {'# Videos':<10}")
            print("-" * 50)
            
            # Sort by mean F1 descending
            sorted_strategies = sorted(k_results.items(), 
                                     key=lambda x: x[1]['mean_f1'], reverse=True)
            
            for strategy, metrics in sorted_strategies:
                print(f"{strategy:<15} {metrics['mean_f1']:<10.4f} "
                      f"{metrics['std_f1']:<10.4f} {metrics['num_videos']:<10}")
        
        # Overall results
        print("\n\nOverall Results:")
        print("-" * 60)
        print(f"{'Strategy':<15} {'Mean F1':<10} {'Std F1':<10} {'# Evaluations':<15}")
        print("-" * 55)
        
        sorted_overall = sorted(summary['overall'].items(), 
                              key=lambda x: x[1]['mean_f1'], reverse=True)
        
        for strategy, metrics in sorted_overall:
            print(f"{strategy:<15} {metrics['mean_f1']:<10.4f} "
                  f"{metrics['std_f1']:<10.4f} {metrics['total_evaluations']:<15}")
        
        print("\n" + "="*80)

def run_evaluation_pipeline(config, teacher_model: TeacherModel, 
                           student_trainer: StudentTrainer = None,
                           max_videos_per_domain: int = 20) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline
    
    Args:
        config: Configuration object
        teacher_model: Teacher model instance
        student_trainer: Trained student model (optional)
        max_videos_per_domain: Maximum videos per domain for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    from .frame_scoring import load_videoave_data
    
    # Load evaluation data
    logger.info("Loading evaluation data...")
    video_data = load_videoave_data(
        config.DATASET_ROOT, 
        domains=config.PROTOTYPE_DOMAINS,
        max_samples_per_domain=max_videos_per_domain
    )
    
    # Create evaluator
    video_loader = VideoLoader(config.CACHE_DIR)
    evaluator = KeyframeEvaluator(teacher_model, video_loader)
    
    # Run comparison
    logger.info("Running strategy comparison...")
    results = evaluator.compare_strategies(
        video_data, student_trainer, config.TOP_K_FRAMES
    )
    
    # Print results
    evaluator.print_comparison_results(results)
    
    return results
