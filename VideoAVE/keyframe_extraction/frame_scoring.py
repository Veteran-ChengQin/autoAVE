"""
Frame importance scoring using teacher supervision
"""
import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from PIL import Image
import logging
from tqdm import tqdm
import hashlib
import pickle

from .teacher_model import TeacherModel, fuzzy_f1_score
from .video_loader import VideoLoader, create_frame_subsets

logger = logging.getLogger(__name__)

class FrameScorer:
    def __init__(self, teacher_model: TeacherModel, video_loader: VideoLoader, 
                 baseline_size: int = 16, num_mc_samples: int = 3, cache_dir: str = ".cache"):
        self.teacher = teacher_model
        self.video_loader = video_loader
        self.baseline_size = baseline_size
        self.num_mc_samples = num_mc_samples
        self.cache_dir = cache_dir
        
        # Create cache directories
        self.scores_cache_dir = os.path.join(cache_dir, "frame_scores")
        os.makedirs(self.scores_cache_dir, exist_ok=True)
    
    def compute_frame_importance_scores(self, video_url: str, title: str, category: str, 
                                      ground_truth_attrs: Dict[str, str], 
                                      target_fps: float = 2.0, max_frames: int = 64) -> Tuple[np.ndarray, List[float]]:
        """
        Compute importance scores for all frames in a video using teacher supervision
        
        Args:
            video_url: URL or path to video
            title: Product title
            category: Product category  
            ground_truth_attrs: Ground truth attribute-value pairs
            target_fps: Target sampling rate
            max_frames: Maximum frames to extract
            
        Returns:
            frames: Array of frames (T, H, W, 3)
            importance_scores: List of importance scores for each frame
        """
        # Generate cache key
        cache_key = hashlib.md5(
            f"{video_url}_{title}_{category}_{str(ground_truth_attrs)}_{target_fps}_{max_frames}".encode()
        ).hexdigest()
        cache_file = os.path.join(self.scores_cache_dir, f"{cache_key}.pkl")
        
        # Check cache
        if os.path.exists(cache_file):
            logger.info(f"Loading cached frame scores for {video_url}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['frames'], data['scores']
        
        # Extract frames
        logger.info(f"Computing frame importance scores for {video_url}")
        video_path, frames, timestamps = self.video_loader.get_cached_frames(
            video_url, target_fps, max_frames
        )
        
        # Convert to PIL images
        pil_frames = self.video_loader.frames_to_pil(frames)
        num_frames = len(pil_frames)
        
        if num_frames == 0:
            return frames, []
        
        # Get attribute names for attribute-conditioned evaluation
        attribute_names = list(ground_truth_attrs.keys())
        
        # Compute importance score for each frame
        importance_scores = []
        
        for frame_idx in tqdm(range(num_frames), desc="Computing frame scores"):
            delta_scores = []
            
            # Generate multiple Monte Carlo samples
            subsets = create_frame_subsets(num_frames, self.baseline_size, frame_idx, self.num_mc_samples)
            
            for baseline_indices, with_target_indices in subsets:
                # Evaluate baseline subset (without target frame)
                baseline_frames = [pil_frames[i] for i in baseline_indices]
                baseline_pred = self.teacher.predict_attributes(
                    baseline_frames, title, category, 
                    mode="attribute_conditioned", attributes=attribute_names
                )
                baseline_f1 = fuzzy_f1_score(baseline_pred, ground_truth_attrs)
                
                # Evaluate subset with target frame
                with_target_frames = [pil_frames[i] for i in with_target_indices]
                with_target_pred = self.teacher.predict_attributes(
                    with_target_frames, title, category,
                    mode="attribute_conditioned", attributes=attribute_names
                )
                with_target_f1 = fuzzy_f1_score(with_target_pred, ground_truth_attrs)
                
                # Compute delta (marginal contribution)
                delta = with_target_f1 - baseline_f1
                delta_scores.append(delta)
            
            # Average across Monte Carlo samples
            avg_delta = np.mean(delta_scores)
            importance_scores.append(avg_delta)
            
            logger.debug(f"Frame {frame_idx}: delta = {avg_delta:.4f}")
        
        # Normalize scores to [0, 1] range within this video
        importance_scores = np.array(importance_scores)
        if len(importance_scores) > 0:
            min_score = importance_scores.min()
            max_score = importance_scores.max()
            if max_score > min_score:
                importance_scores = (importance_scores - min_score) / (max_score - min_score)
            else:
                importance_scores = np.ones_like(importance_scores) * 0.5
        
        # Apply temperature scaling (optional)
        gamma = 1.2  # gamma > 1 makes high-scoring frames more prominent
        importance_scores = importance_scores ** gamma
        
        # Cache results
        cache_data = {
            'frames': frames,
            'scores': importance_scores.tolist(),
            'timestamps': timestamps.tolist()
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Computed importance scores for {num_frames} frames")
        return frames, importance_scores.tolist()
    
    def batch_compute_scores(self, video_data: List[Dict[str, Any]], 
                           target_fps: float = 2.0, max_frames: int = 64) -> List[Dict[str, Any]]:
        """
        Batch compute frame importance scores for multiple videos
        
        Args:
            video_data: List of video data dictionaries with keys:
                       'video_url', 'title', 'category', 'ground_truth_attrs'
            target_fps: Target sampling rate
            max_frames: Maximum frames per video
            
        Returns:
            List of dictionaries with computed scores and features
        """
        results = []
        
        for video_info in tqdm(video_data, desc="Processing videos"):
            try:
                frames, scores = self.compute_frame_importance_scores(
                    video_info['video_url'],
                    video_info['title'], 
                    video_info['category'],
                    video_info['ground_truth_attrs'],
                    target_fps,
                    max_frames
                )
                
                result = {
                    'video_url': video_info['video_url'],
                    'title': video_info['title'],
                    'category': video_info['category'],
                    'ground_truth_attrs': video_info['ground_truth_attrs'],
                    'frames': frames,
                    'importance_scores': scores,
                    'num_frames': len(scores)
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing video {video_info['video_url']}: {e}")
                continue
        
        return results

def load_videoave_data(dataset_path: str, domains: List[str] = None, 
                      max_samples_per_domain: int = None) -> List[Dict[str, Any]]:
    """
    Load VideoAVE dataset for frame scoring
    
    Args:
        dataset_path: Path to VideoAVE dataset directory
        domains: List of domain names to include (e.g., ['beauty', 'sports'])
        max_samples_per_domain: Maximum samples per domain (for prototyping)
        
    Returns:
        List of video data dictionaries
    """
    import pandas as pd
    import ast
    
    video_data = []
    
    if domains is None:
        domains = ['beauty', 'sports']  # Default prototype domains
    
    for domain in domains:
        # Load training data for the domain
        train_file = os.path.join(dataset_path, "train_data", f"{domain}_train.csv")
        
        if not os.path.exists(train_file):
            logger.warning(f"Training file not found: {train_file}")
            continue
        
        df = pd.read_csv(train_file)
        logger.info(f"Loaded {len(df)} samples from {domain} domain")
        
        # Limit samples for prototyping
        if max_samples_per_domain:
            df = df.head(max_samples_per_domain)
            logger.info(f"Limited to {len(df)} samples for prototyping")
        
        for _, row in df.iterrows():
            try:
                # Parse aspects (ground truth attributes)
                aspects = ast.literal_eval(row['aspects'])
                # 可能有空字典
                if aspects == {}:
                    continue

                video_info = {
                    'video_url': row['content_url'],
                    'title': row['video_title'],
                    'category': domain,
                    'ground_truth_attrs': aspects,
                    'product_id': row['product_id']
                }
                video_data.append(video_info)
                
            except Exception as e:
                logger.warning(f"Error parsing row: {e}")
                continue
    
    logger.info(f"Loaded total {len(video_data)} videos from {len(domains)} domains")
    return video_data
