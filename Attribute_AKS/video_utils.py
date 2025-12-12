"""
Video frame extraction utilities
"""
import os
import logging
import tempfile
import requests
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


def download_video(url: str) -> str:
    """
    Download video from URL to temporary file.
    
    Args:
        url: Video URL
        
    Returns:
        Path to downloaded temporary file
    """
    try:
        logger.info(f"Downloading video from {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        
        # Download in chunks
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        
        temp_file.close()
        logger.info(f"Downloaded video to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to download video from {url}: {e}")
        raise


def extract_candidate_frames(video_path: str, fps: float = 1.0, 
                            max_frames: int = 256) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract candidate frames from video at specified fps.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames to extract
        
    Returns:
        frames: List of PIL Images
        timestamps: List of timestamps (in seconds) for each frame
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        logger.error("decord not installed. Install with: pip install decord")
        raise
    
    # Handle URL vs local file
    local_video_path = video_path
    temp_file = None
    
    try:
        # Check if it's a URL
        if video_path.startswith(('http://', 'https://')):
            temp_file = download_video(video_path)
            local_video_path = temp_file
        
        vr = VideoReader(local_video_path, ctx=cpu(0))
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        
        logger.info(f"Video: {total_frames} frames @ {video_fps} fps")
        
        # Calculate sampling interval
        frame_interval = max(1, int(video_fps / fps))
        
        # Generate frame indices
        indices = list(range(0, total_frames, frame_interval))
        
        # Limit to max_frames
        if len(indices) > max_frames:
            # Uniformly subsample
            step = len(indices) / max_frames
            indices = [indices[int(i * step)] for i in range(max_frames)]
        
        logger.info(f"Extracting {len(indices)} frames")
        
        # Extract frames
        frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)
        
        # Convert to PIL Images
        frames = [Image.fromarray(frame) for frame in frames_np]
        
        # Get timestamps
        timestamps = []
        for idx in indices:
            ts = vr.get_frame_timestamp(idx)
            # ts is typically a tuple/list of (start_time, end_time), take the start time
            try:
                if isinstance(ts, (tuple, list)) and len(ts) > 0:
                    timestamps.append(float(ts[0]))
                elif hasattr(ts, '__len__') and len(ts) > 0:
                    # Handle numpy arrays or other array-like objects
                    timestamps.append(float(ts.flat[0]))
                else:
                    timestamps.append(float(ts) if ts is not None else float(idx) / video_fps)
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Failed to extract timestamp for frame {idx}: {e}, using frame index")
                timestamps.append(float(idx) / video_fps)
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames, timestamps
        
    except Exception as e:
        logger.error(f"Failed to extract frames from {video_path}: {e}")
        raise
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")


def get_video_hash(video_path: str) -> str:
    """Generate hash for video path"""
    return hashlib.md5(video_path.encode()).hexdigest()


def cache_frames(frames: List[Image.Image], timestamps: List[float], 
                cache_dir: str, video_path: str) -> None:
    """
    Cache extracted frames to disk.
    
    Args:
        frames: List of PIL Images
        timestamps: List of timestamps
        cache_dir: Directory to save cache
        video_path: Original video path (used for cache key)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    video_hash = get_video_hash(video_path)
    frames_file = os.path.join(cache_dir, f"{video_hash}_frames.npy")
    timestamps_file = os.path.join(cache_dir, f"{video_hash}_timestamps.npy")
    
    # Convert frames to numpy array
    frames_np = np.array([np.array(f) for f in frames])
    
    # Save
    np.save(frames_file, frames_np)
    np.save(timestamps_file, np.array(timestamps))
    
    logger.info(f"Cached frames to {frames_file}")


def load_cached_frames(cache_dir: str, video_path: str) -> Optional[Tuple[List[Image.Image], List[float]]]:
    """
    Load cached frames if available.
    
    Returns:
        (frames, timestamps) or None if not cached
    """
    video_hash = get_video_hash(video_path)
    frames_file = os.path.join(cache_dir, f"{video_hash}_frames.npy")
    timestamps_file = os.path.join(cache_dir, f"{video_hash}_timestamps.npy")
    
    if os.path.exists(frames_file) and os.path.exists(timestamps_file):
        try:
            frames_np = np.load(frames_file)
            timestamps_np = np.load(timestamps_file)
            
            # Ensure timestamps are a flat list of floats
            if timestamps_np.ndim > 1:
                timestamps = timestamps_np.flatten().tolist()
            else:
                timestamps = timestamps_np.tolist()
            
            # Convert to Python floats
            timestamps = [float(t) for t in timestamps]
            
            frames = [Image.fromarray(f) for f in frames_np]
            logger.info(f"Loaded cached frames from {frames_file}")
            return frames, timestamps
        except Exception as e:
            logger.warning(f"Failed to load cached frames: {e}")
    
    return None


def get_frames_cached(video_path: str, cache_dir: str, fps: float = 1.0,
                     max_frames: int = 256) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract frames with caching support.
    
    Args:
        video_path: Path to video file
        cache_dir: Directory for caching
        fps: Frames per second to extract
        max_frames: Maximum frames to extract
        
    Returns:
        frames: List of PIL Images
        timestamps: List of timestamps
    """
    # Try to load from cache
    cached = load_cached_frames(cache_dir, video_path)
    if cached is not None:
        return cached
    
    # Extract frames
    frames, timestamps = extract_candidate_frames(video_path, fps, max_frames)
    
    # Cache for future use
    cache_frames(frames, timestamps, cache_dir, video_path)
    
    return frames, timestamps
