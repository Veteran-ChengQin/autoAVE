"""
Video loading and frame sampling utilities
"""
import os
import cv2
import numpy as np
import torch
import hashlib
import requests
from decord import VideoReader, cpu
from PIL import Image
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoLoader:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_video(self, url: str, dest_path: str) -> None:
        """Download video from URL to local path"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Video downloaded to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to download video from {url}: {e}")
            raise
    
    def get_video_path(self, video_url: str) -> str:
        """Get local path for video, download if necessary"""
        if video_url.startswith(('http://', 'https://')):
            # Generate hash for URL to create unique filename
            video_hash = hashlib.md5(video_url.encode('utf-8')).hexdigest()
            video_path = os.path.join(self.cache_dir, f'{video_hash}.mp4')
            
            if not os.path.exists(video_path):
                self.download_video(video_url, video_path)
            return video_path
        else:
            # Local file path
            return video_url
    
    def sample_frames_uniform(self, video_path: str, target_fps: float = 2.0, 
                            max_frames: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample frames uniformly from video
        
        Args:
            video_path: Path to video file
            target_fps: Target frames per second
            max_frames: Maximum number of frames to extract
            
        Returns:
            frames: Array of shape (T, H, W, 3) with RGB frames
            timestamps: Array of shape (T,) with frame timestamps
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            # Calculate sampling interval
            frame_interval = max(1, int(video_fps / target_fps))
            
            # Generate frame indices
            indices = list(range(0, total_frames, frame_interval))
            
            # Limit to max_frames
            if len(indices) > max_frames:
                # Uniformly sample max_frames from the indices
                step = len(indices) / max_frames
                indices = [indices[int(i * step)] for i in range(max_frames)]
            
            # Extract frames
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
            timestamps = np.array([vr.get_frame_timestamp(idx)[0] for idx in indices])
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames, timestamps
            
        except Exception as e:
            logger.error(f"Failed to extract frames from {video_path}: {e}")
            raise
    
    def frames_to_pil(self, frames: np.ndarray) -> List[Image.Image]:
        """Convert numpy frames to PIL Images"""
        return [Image.fromarray(frame) for frame in frames]
    
    def get_cached_frames(self, video_url: str, target_fps: float = 2.0, 
                         max_frames: int = 64) -> Tuple[str, np.ndarray, np.ndarray]:
        """
        Get frames with caching support
        
        Returns:
            video_path: Local path to video file
            frames: Array of frames
            timestamps: Array of timestamps
        """
        # Generate cache key
        cache_key = hashlib.md5(f"{video_url}_{target_fps}_{max_frames}".encode()).hexdigest()
        frames_cache = os.path.join(self.cache_dir, f"{cache_key}_frames.npy")
        timestamps_cache = os.path.join(self.cache_dir, f"{cache_key}_timestamps.npy")
        
        # Check cache
        if os.path.exists(frames_cache) and os.path.exists(timestamps_cache):
            frames = np.load(frames_cache)
            timestamps = np.load(timestamps_cache)
            video_path = self.get_video_path(video_url)
            logger.info(f"Loaded cached frames for {video_url}")
            return video_path, frames, timestamps
        
        # Extract frames
        video_path = self.get_video_path(video_url)
        frames, timestamps = self.sample_frames_uniform(video_path, target_fps, max_frames)
        
        # Save to cache
        np.save(frames_cache, frames)
        np.save(timestamps_cache, timestamps)
        
        return video_path, frames, timestamps

def create_frame_subsets(total_frames: int, baseline_size: int, 
                        target_frame_idx: int, num_samples: int = 3) -> List[Tuple[List[int], List[int]]]:
    """
    Create frame subsets for computing frame importance scores
    
    Args:
        total_frames: Total number of frames available
        baseline_size: Size of baseline subset
        target_frame_idx: Index of the target frame to evaluate
        num_samples: Number of Monte Carlo samples
        
    Returns:
        List of (subset_without_target, subset_with_target) pairs
    """
    available_frames = list(range(total_frames))
    available_frames.remove(target_frame_idx)  # Remove target frame
    
    subsets = []
    for _ in range(num_samples):
        # Sample baseline frames (without target)
        if len(available_frames) >= baseline_size:
            baseline_frames = np.random.choice(available_frames, baseline_size, replace=False).tolist()
        else:
            baseline_frames = available_frames.copy()
        
        # Create subset with target frame
        with_target = baseline_frames + [target_frame_idx]
        
        subsets.append((baseline_frames, with_target))
    
    return subsets
