"""
Attribute-aware keyframe selection using AKS
"""
import logging
from typing import List, Tuple, Optional
from PIL import Image

from aks_sampler import AdaptiveKeyframeSampler
from frame_scorer import FrameScorer
from video_utils import get_frames_cached
from frame_storage import FrameStorage

logger = logging.getLogger(__name__)


class AttrKeyframeSelector:
    """
    Selects keyframes for each attribute using AKS.
    
    Pipeline:
    1. Extract candidate frames from video
    2. For each attribute, construct query text
    3. Score frames using VL scorer (BLIP-ITM)
    4. Use AKS to adaptively select M keyframes
    """
    
    def __init__(self, frame_scorer: FrameScorer, cache_dir: str = ".cache",
                 fps: float = 1.0, max_frames: int = 256,
                 m_attr: int = 8, max_level: int = 4, s_threshold: float = 0.6,
                 storage_dir: str = "./frame_storage", enable_storage: bool = True):
        """
        Args:
            frame_scorer: FrameScorer instance for scoring frames
            cache_dir: Directory for caching frames
            fps: Frames per second for candidate extraction
            max_frames: Maximum candidate frames
            m_attr: Number of keyframes per attribute
            max_level: Max recursion level for AKS
            s_threshold: Score threshold for AKS Judge & Split
            storage_dir: Directory for storing frames for review
            enable_storage: Whether to save frames for review
        """
        self.scorer = frame_scorer
        self.cache_dir = cache_dir
        self.fps = fps
        self.max_frames = max_frames
        self.m_attr = m_attr
        self.enable_storage = enable_storage
        
        self.sampler = AdaptiveKeyframeSampler(max_level=max_level, s_threshold=s_threshold)
        
        if enable_storage:
            self.storage = FrameStorage(storage_dir)
        else:
            self.storage = None
    
    def construct_query(self, attr_name: str, title: str, category: str) -> str:
        """
        Construct text query for scoring frames related to an attribute.
        
        Args:
            attr_name: Attribute name (e.g., "Color", "Brand")
            title: Product title
            category: Product category
            
        Returns:
            Query text for frame scoring
        """
        query = (
            f"A video frame that best shows the product's {attr_name}. "
            f"Product title: {title}. Product category: {category}."
        )
        return query
    
    def select_keyframes_for_attr(self, video_path: str, attr_name: str,
                                 title: str, category: str,
                                 product_id: str = None) -> Tuple[List[Image.Image], List[float], List[int]]:
        """
        Select keyframes for a specific attribute.
        
        Args:
            video_path: Path to video file
            attr_name: Attribute name
            title: Product title
            category: Product category
            product_id: Product identifier for storage (optional)
            
        Returns:
            keyframes: List of selected PIL Images
            timestamps: List of timestamps for selected frames
            indices: List of indices in original candidate frames
        """
        try:
            # Step 1: Extract candidate frames
            logger.info(f"Extracting candidate frames from {video_path}")
            candidate_frames, candidate_timestamps = get_frames_cached(
                video_path, self.cache_dir, self.fps, self.max_frames
            )
            
            if not candidate_frames:
                logger.warning(f"No frames extracted from {video_path}")
                return [], [], []
            
            # Step 2: Construct query
            query = self.construct_query(attr_name, title, category)
            logger.info(f"Query for {attr_name}: {query}")
            
            # Step 3: Score frames
            logger.info(f"Scoring {len(candidate_frames)} frames")
            scores = self.scorer.score(query, candidate_frames)
            
            if not scores:
                logger.warning(f"No scores computed for {attr_name}")
                return [], [], []
            
            # Convert scores to Python floats for logging and ensure proper format
            scores_flat = []
            for s in scores:
                if isinstance(s, (list, tuple)):
                    scores_flat.append(float(s[0]) if s else 0.0)
                else:
                    scores_flat.append(float(s))
            
            logger.info(f"Scores: min={min(scores_flat):.3f}, max={max(scores_flat):.3f}, mean={sum(scores_flat)/len(scores_flat):.3f}")
            
            # Save candidate frames if storage is enabled
            if self.storage and product_id:
                self.storage.save_candidate_frames(
                    product_id, attr_name, candidate_frames, candidate_timestamps, scores_flat
                )
            
            # Step 4: AKS sampling
            logger.info(f"Selecting {self.m_attr} keyframes using AKS")
            selected_indices = self.sampler.select(scores_flat, self.m_attr)
            
            # Step 5: Extract selected frames
            keyframes = [candidate_frames[i] for i in selected_indices]
            timestamps = [candidate_timestamps[i] for i in selected_indices]
            keyframe_scores = [scores_flat[i] for i in selected_indices]
            
            logger.info(f"Selected {len(keyframes)} keyframes at indices {selected_indices}")
            
            # Save keyframes if storage is enabled
            if self.storage and product_id:
                self.storage.save_keyframes(
                    product_id, attr_name, keyframes, timestamps, selected_indices, keyframe_scores
                )
                # Save scoring analysis
                self.storage.save_scoring_analysis(
                    product_id, attr_name, candidate_frames, scores_flat, candidate_timestamps, selected_indices
                )
            
            return keyframes, timestamps, selected_indices
            
        except Exception as e:
            logger.error(f"Error selecting keyframes for {attr_name}: {e}")
            raise
    
    def select_keyframes_for_product(self, video_path: str, title: str, category: str,
                                    attr_names: List[str], product_id: str = None) -> dict:
        """
        Select keyframes for all attributes of a product.
        
        Args:
            video_path: Path to video file
            title: Product title
            category: Product category
            attr_names: List of attribute names
            product_id: Product identifier for storage (optional)
            
        Returns:
            Dictionary mapping attr_name -> (keyframes, timestamps, indices)
        """
        result = {}
        
        for attr_name in attr_names:
            logger.info(f"Processing attribute: {attr_name}")
            keyframes, timestamps, indices = self.select_keyframes_for_attr(
                video_path, attr_name, title, category, product_id
            )
            result[attr_name] = {
                "keyframes": keyframes,
                "timestamps": timestamps,
                "indices": indices,
            }
        
        return result
