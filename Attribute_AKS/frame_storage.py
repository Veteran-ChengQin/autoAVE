"""
Frame storage and visualization utilities for debugging and review
"""
import os
import logging
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class FrameStorage:
    """
    Store candidate frames and keyframes for review and debugging.
    Organizes frames by product and attribute for easy inspection.
    """
    
    def __init__(self, storage_root: str = "./frame_storage"):
        """
        Args:
            storage_root: Root directory for storing frames
        """
        self.storage_root = storage_root
        os.makedirs(storage_root, exist_ok=True)
        logger.info(f"FrameStorage initialized at {storage_root}")
    
    def _create_product_dir(self, product_id: str) -> str:
        """Create directory structure for a product"""
        product_dir = os.path.join(self.storage_root, product_id)
        os.makedirs(product_dir, exist_ok=True)
        return product_dir
    
    def _create_attr_dir(self, product_id: str, attr_name: str) -> str:
        """Create directory structure for an attribute"""
        attr_dir = os.path.join(self.storage_root, product_id, attr_name)
        os.makedirs(attr_dir, exist_ok=True)
        return attr_dir
    
    def save_candidate_frames(self, product_id: str, attr_name: str,
                             frames: List[Image.Image], timestamps: List[float],
                             scores: List[float] = None) -> str:
        """
        Save candidate frames extracted from video.
        
        Args:
            product_id: Product identifier
            attr_name: Attribute name
            frames: List of PIL Images
            timestamps: List of timestamps for each frame
            scores: Optional list of scores for each frame
            
        Returns:
            Path to candidate frames directory
        """
        attr_dir = self._create_attr_dir(product_id, attr_name)
        candidates_dir = os.path.join(attr_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)
        
        # Save frames
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            try:
                ts_float = float(ts)  # Ensure ts is a Python float
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to convert timestamp {ts} (type: {type(ts)}): {e}")
                ts_float = float(i)  # Fallback to frame index
            filename = f"frame_{i:03d}_ts_{ts_float:.2f}s.jpg"
            filepath = os.path.join(candidates_dir, filename)
            frame.save(filepath, quality=95)
        
        # Save metadata
        # Ensure scores are properly converted to floats
        scores_list = []
        if scores:
            for s in scores:
                if isinstance(s, (list, tuple)):
                    # If score is a list/tuple, take the first element
                    scores_list.append(float(s[0]) if s else 0.0)
                else:
                    scores_list.append(float(s))
        
        metadata = {
            "product_id": product_id,
            "attr_name": attr_name,
            "num_frames": len(frames),
            "timestamps": [float(t) for t in timestamps],
            "scores": scores_list,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(candidates_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(frames)} candidate frames to {candidates_dir}")
        return candidates_dir
    
    def save_keyframes(self, product_id: str, attr_name: str,
                      keyframes: List[Image.Image], timestamps: List[float],
                      indices: List[int], scores: List[float] = None) -> str:
        """
        Save selected keyframes.
        
        Args:
            product_id: Product identifier
            attr_name: Attribute name
            keyframes: List of selected PIL Images
            timestamps: List of timestamps for selected frames
            indices: List of indices in original candidate frames
            scores: Optional list of scores for keyframes
            
        Returns:
            Path to keyframes directory
        """
        attr_dir = self._create_attr_dir(product_id, attr_name)
        keyframes_dir = os.path.join(attr_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Save frames
        for i, (frame, ts, idx) in enumerate(zip(keyframes, timestamps, indices)):
            ts_float = float(ts)  # Ensure ts is a Python float
            idx_int = int(idx)    # Ensure idx is a Python int
            filename = f"keyframe_{i:02d}_idx_{idx_int:03d}_ts_{ts_float:.2f}s.jpg"
            filepath = os.path.join(keyframes_dir, filename)
            frame.save(filepath, quality=95)
        
        # Save metadata
        # Ensure scores are properly converted to floats
        scores_list = []
        if scores:
            for s in scores:
                if isinstance(s, (list, tuple)):
                    scores_list.append(float(s[0]) if s else 0.0)
                else:
                    scores_list.append(float(s))
        
        metadata = {
            "product_id": product_id,
            "attr_name": attr_name,
            "num_keyframes": len(keyframes),
            "indices": [int(idx) for idx in indices],
            "timestamps": [float(t) for t in timestamps],
            "scores": scores_list,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(keyframes_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(keyframes)} keyframes to {keyframes_dir}")
        return keyframes_dir
    
    def save_scoring_analysis(self, product_id: str, attr_name: str,
                             candidate_frames: List[Image.Image],
                             scores: List[float],
                             timestamps: List[float],
                             keyframe_indices: List[int]) -> str:
        """
        Save a visual analysis of frame scoring.
        Creates a grid showing all candidate frames with their scores.
        
        Args:
            product_id: Product identifier
            attr_name: Attribute name
            candidate_frames: List of all candidate frames
            scores: List of scores for each frame
            timestamps: List of timestamps
            keyframe_indices: Indices of selected keyframes
            
        Returns:
            Path to analysis image
        """
        attr_dir = self._create_attr_dir(product_id, attr_name)
        analysis_dir = os.path.join(attr_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create grid visualization
        num_frames = len(candidate_frames)
        cols = min(4, num_frames)  # 4 columns max
        rows = (num_frames + cols - 1) // cols
        
        # Resize frames for grid
        frame_size = 200
        resized_frames = [f.resize((frame_size, frame_size)) for f in candidate_frames]
        
        # Create grid image
        grid_width = cols * frame_size + (cols - 1) * 10
        grid_height = rows * (frame_size + 40) + (rows - 1) * 10
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Paste frames and add text
        for i, (frame, score, ts, idx) in enumerate(zip(resized_frames, scores, timestamps, range(num_frames))):
            row = i // cols
            col = i % cols
            x = col * (frame_size + 10)
            y = row * (frame_size + 50)
            
            # Paste frame
            grid_img.paste(frame, (x, y))
            
            # Add text (score and timestamp)
            draw = ImageDraw.Draw(grid_img)
            text = f"#{idx} Score:{score:.3f} T:{ts:.1f}s"
            
            # Highlight keyframes
            if idx in keyframe_indices:
                # Draw red border for keyframes
                for j in range(3):
                    draw.rectangle(
                        [(x - j, y - j), (x + frame_size + j, y + frame_size + j)],
                        outline='red'
                    )
                text = f"★ #{idx} Score:{score:.3f} T:{ts:.1f}s"
            
            # Draw text
            text_y = y + frame_size + 5
            draw.text((x, text_y), text, fill='black')
        
        # Save grid
        grid_path = os.path.join(analysis_dir, "scoring_grid.jpg")
        grid_img.save(grid_path, quality=95)
        
        # Save score statistics
        # Ensure scores are properly converted to floats
        scores_float = []
        for s in scores:
            if isinstance(s, (list, tuple)):
                scores_float.append(float(s[0]) if s else 0.0)
            else:
                scores_float.append(float(s))
        
        stats = {
            "product_id": product_id,
            "attr_name": attr_name,
            "num_candidates": num_frames,
            "num_keyframes": len(keyframe_indices),
            "score_min": float(min(scores_float)),
            "score_max": float(max(scores_float)),
            "score_mean": float(sum(scores_float) / len(scores_float)),
            "score_std": float((sum((s - sum(scores_float)/len(scores_float))**2 for s in scores_float) / len(scores_float))**0.5) if len(scores_float) > 1 else 0.0,
            "scores": scores_float,
            "keyframe_indices": [int(idx) for idx in keyframe_indices],
            "saved_at": datetime.now().isoformat()
        }
        
        stats_path = os.path.join(analysis_dir, "score_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved scoring analysis to {analysis_dir}")
        return analysis_dir
    
    def get_product_summary(self, product_id: str) -> Dict:
        """
        Get summary of all stored frames for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with summary information
        """
        product_dir = os.path.join(self.storage_root, product_id)
        
        if not os.path.exists(product_dir):
            return {}
        
        summary = {
            "product_id": product_id,
            "attributes": {}
        }
        
        # Scan attributes
        for attr_name in os.listdir(product_dir):
            attr_path = os.path.join(product_dir, attr_name)
            if not os.path.isdir(attr_path):
                continue
            
            attr_info = {
                "candidates": 0,
                "keyframes": 0,
                "analysis": False
            }
            
            # Count candidates
            candidates_dir = os.path.join(attr_path, "candidates")
            if os.path.exists(candidates_dir):
                attr_info["candidates"] = len([f for f in os.listdir(candidates_dir) if f.endswith('.jpg')])
            
            # Count keyframes
            keyframes_dir = os.path.join(attr_path, "keyframes")
            if os.path.exists(keyframes_dir):
                attr_info["keyframes"] = len([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])
            
            # Check analysis
            analysis_dir = os.path.join(attr_path, "analysis")
            if os.path.exists(analysis_dir):
                attr_info["analysis"] = os.path.exists(os.path.join(analysis_dir, "scoring_grid.jpg"))
            
            summary["attributes"][attr_name] = attr_info
        
        return summary
