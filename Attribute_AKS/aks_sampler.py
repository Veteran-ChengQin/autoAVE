"""
Adaptive Keyframe Sampling (AKS) implementation
Based on the Judge & Split + TOP/BIN strategy from AKS paper
"""
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """Represents a temporal segment for recursive sampling"""
    start: int   # left closed
    end: int     # right open
    m: int       # number of frames to select in this segment
    level: int   # current recursion level


class AdaptiveKeyframeSampler:
    """
    Adaptive Keyframe Sampling using Judge & Split strategy.
    
    The algorithm recursively partitions the video timeline based on score distribution:
    - If scores show clear peaks (high variance), use TOP strategy (select highest scores)
    - Otherwise, use BIN strategy (partition and cover uniformly)
    """
    
    def __init__(self, max_level: int = 4, s_threshold: float = 0.6):
        """
        Args:
            max_level: Maximum recursion depth
            s_threshold: Score threshold for TOP vs BIN decision
                        (difference between top-k mean and overall mean)
        """
        self.max_level = max_level
        self.s_threshold = s_threshold
    
    def select(self, scores: List[float], M: int) -> List[int]:
        """
        Select M keyframes from T candidate frames using adaptive sampling.
        
        Args:
            scores: List of relevance scores for each frame (length T)
            M: Number of keyframes to select
            
        Returns:
            List of selected frame indices, sorted in temporal order
        """
        T = len(scores)
        
        # Edge cases
        if M >= T:
            return list(range(T))
        if M <= 0:
            return []
        
        scores = np.array(scores, dtype=np.float32)
        
        # Initialize with full segment
        stack = [Segment(start=0, end=T, m=M, level=0)]
        final_segments = []
        
        # Recursive Judge & Split
        while stack:
            seg = stack.pop()
            s, e, m, lvl = seg.start, seg.end, seg.m, seg.level
            
            # Extract scores for this segment
            seg_scores = scores[s:e]
            
            # Stopping conditions
            if lvl >= self.max_level or m <= 1:
                # Reached max depth or only need 1 frame: finalize segment
                final_segments.append(seg)
                continue
            
            # Judge: Check if segment has clear high-score peaks
            s_all = float(np.mean(seg_scores))
            
            # Get top-m scores in this segment
            top_k = min(m, len(seg_scores))
            top_scores = sorted(seg_scores, reverse=True)[:top_k]
            s_top = float(np.mean(top_scores))
            
            # Decision: TOP vs BIN
            if s_top - s_all >= self.s_threshold:
                # Clear peaks detected: use TOP strategy (select highest scores)
                # Don't split further, just mark for top-k selection
                final_segments.append(seg)
            else:
                # No clear peaks: use BIN strategy (split and cover)
                # Recursively split the segment
                mid = (s + e) // 2
                
                # Prevent infinite recursion on very small segments
                if mid <= s or mid >= e:
                    final_segments.append(seg)
                    continue
                
                # Distribute m frames between left and right segments
                m_left = m // 2
                m_right = m - m_left
                
                # Push to stack (LIFO order)
                stack.append(Segment(start=mid, end=e, m=m_right, level=lvl + 1))
                stack.append(Segment(start=s, end=mid, m=m_left, level=lvl + 1))
        
        # Extract selected frames from final segments
        selected = []
        for seg in final_segments:
            s, e, m, _ = seg.start, seg.end, seg.m, seg.level
            
            # Get frame indices in this segment
            segment_indices = list(range(s, e))
            
            # Sort by score (descending) and select top m
            segment_indices_sorted = sorted(
                segment_indices, 
                key=lambda i: scores[i], 
                reverse=True
            )
            selected.extend(segment_indices_sorted[:m])
        
        # Remove duplicates and sort temporally
        selected = sorted(set(selected))
        
        # If we have fewer than M due to deduplication, add more frames
        if len(selected) < M:
            remaining = [i for i in range(T) if i not in selected]
            remaining_sorted = sorted(
                remaining, 
                key=lambda i: scores[i], 
                reverse=True
            )
            selected.extend(remaining_sorted[:(M - len(selected))])
            selected = sorted(set(selected))
        
        return selected[:M]  # Ensure exactly M frames


class SimpleTopKSampler:
    """Baseline: Simple top-K sampling without adaptive logic"""
    
    def select(self, scores: List[float], M: int) -> List[int]:
        """Select top M frames by score"""
        T = len(scores)
        if M >= T:
            return list(range(T))
        
        scores = np.array(scores, dtype=np.float32)
        indices = np.argsort(-scores)[:M]
        return sorted(indices.tolist())


class UniformSampler:
    """Baseline: Uniform sampling"""
    
    def select(self, scores: List[float], M: int) -> List[int]:
        """Select M frames uniformly distributed"""
        T = len(scores)
        if M >= T:
            return list(range(T))
        
        step = T / M
        indices = [int(i * step) for i in range(M)]
        return indices
