"""
Frame-text relevance scoring using BLIP-ITM or CLIP
"""
import os
import torch
import logging
from typing import List, Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class FrameScorer:
    """
    Score frames based on their relevance to a text query.
    Uses BLIP-ITM (Image-Text Matching) for scoring.
    """
    
    def __init__(self, model_name: str = "blip-itm-base", device: str = "cuda:0"):
        """
        Args:
            model_name: Model to use ("blip-itm-base", "blip-itm-large", or "clip-vit-base")
            device: Device to run model on
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        self._load_model(model_name)
    
    def _load_model(self, model_name: str) -> None:
        """Load the scoring model"""
        try:
            # Try CLIP first (more reliable with current torch version)
            logger.info("Loading CLIP model for frame scoring...")
            self._load_clip_model("clip-vit-base")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _load_blip_model(self, model_name: str) -> None:
        """Load BLIP-ITM model from local path"""
        try:
            from transformers import AutoProcessor, BlipForImageTextRetrieval
            
            # Map short names to local paths
            model_mapping = {
                "blip-itm-base": "/data/share/vllm/blip-itm-base-coco",
                "blip-itm-large": "/data/share/vllm/blip-itm-large-coco",
            }
            
            model_path = model_mapping.get(model_name, model_name)
            
            logger.info(f"Loading BLIP model from: {model_path}")
            
            # Check if path exists
            if not os.path.exists(model_path):
                logger.warning(f"BLIP model path not found: {model_path}. Falling back to CLIP...")
                self._load_clip_model("clip-vit-base")
                return
            
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = BlipForImageTextRetrieval.from_pretrained(
                model_path,
                dtype=torch.float16
            ).to(self.device)
            self.model.eval()
            logger.info("BLIP model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load BLIP-ITM: {e}. Falling back to CLIP...")
            try:
                self._load_clip_model("clip-vit-base")
            except Exception as e2:
                logger.error(f"Failed to load both BLIP and CLIP: {e2}")
                raise
    
    def _load_clip_model(self, model_name: str) -> None:
        """Load CLIP model using standard HuggingFace approach"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Map short names to HuggingFace model IDs
            model_mapping = {
                "clip-vit-base": "openai/clip-vit-base-patch32",
                "clip-vit-large": "openai/clip-vit-large-patch14",
            }
            
            model_id = model_mapping.get(model_name, "openai/clip-vit-base-patch32")
            
            logger.info(f"Loading CLIP model: {model_id}")
            
            # Load processor and model using standard approach
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    @torch.no_grad()
    def score(self, text: str, frames: List[Image.Image]) -> List[float]:
        """
        Score frames based on relevance to text.
        
        Args:
            text: Query text
            frames: List of PIL Images
            
        Returns:
            List of scores (one per frame), typically in range [0, 1]
        """
        if not frames:
            return []
        
        try:
            # Determine which scoring method to use based on what was loaded
            if self.model is None:
                logger.error("Model not loaded")
                raise RuntimeError("Model not loaded")
            
            # Try CLIP first (more reliable)
            try:
                return self._score_clip(text, frames)
            except Exception as e:
                logger.warning(f"CLIP scoring failed: {e}, trying BLIP...")
                try:
                    return self._score_blip(text, frames)
                except Exception as e2:
                    logger.error(f"Both CLIP and BLIP scoring failed: {e2}")
                    raise
        except Exception as e:
            logger.error(f"Error scoring frames: {e}")
            raise
    
    @torch.no_grad()
    def _score_blip(self, text: str, frames: List[Image.Image]) -> List[float]:
        """Score using BLIP-ITM"""
        scores = []
        
        # Process in batches to avoid OOM
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare inputs
            inputs = self.processor(
                images=batch_frames,
                text=[text] * len(batch_frames),
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Extract ITM scores (image-text matching logits)
            # BLIP returns logits, we convert to probabilities
            logits = outputs.logits_per_image if hasattr(outputs, 'logits_per_image') else outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of match
            
            scores.extend(probs.cpu().numpy().tolist())
        
        return scores
    
    @torch.no_grad()
    def _score_clip(self, text: str, frames: List[Image.Image]) -> List[float]:
        """Score using CLIP from transformers"""
        scores = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Process text and images separately to ensure correct shape
            # This avoids issues with the processor when handling multiple images
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
            image_inputs = self.processor(images=batch_frames, return_tensors="pt")
            
            # Move to device
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Get embeddings
            text_features = self.model.get_text_features(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs.get('attention_mask')
            )
            image_features = self.model.get_image_features(
                pixel_values=image_inputs['pixel_values']
            )
            
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            logit_scale = self.model.logit_scale.exp()
            batch_scores = (image_features @ text_features.t() * logit_scale).squeeze(-1)
            
            # Convert to numpy (keep raw logits, don't apply sigmoid)
            batch_scores = batch_scores.cpu().numpy()
            
            # Normalize to [0, 1] range using min-max normalization
            # This preserves the relative differences between scores
            if len(batch_scores) > 1:
                min_score = batch_scores.min()
                max_score = batch_scores.max()
                if max_score > min_score:
                    batch_scores = (batch_scores - min_score) / (max_score - min_score)
                else:
                    batch_scores = np.ones_like(batch_scores) * 0.5
            else:
                batch_scores = np.array([0.5])
            
            # Ensure it's a 1D array
            if batch_scores.ndim == 0:
                batch_scores = np.array([batch_scores.item()])
            elif batch_scores.ndim > 1:
                batch_scores = batch_scores.flatten()
            
            # Convert to list and extend
            batch_scores_list = batch_scores.tolist()
            if isinstance(batch_scores_list, list):
                scores.extend(batch_scores_list)
            else:
                scores.append(batch_scores_list)
        
        return scores
    
    def score_batch(self, text_list: List[str], frames: List[Image.Image]) -> np.ndarray:
        """
        Score multiple texts against multiple frames.
        
        Args:
            text_list: List of query texts
            frames: List of PIL Images
            
        Returns:
            Array of shape (len(text_list), len(frames)) with scores
        """
        scores_matrix = []
        
        for text in text_list:
            scores = self.score(text, frames)
            scores_matrix.append(scores)
        
        return np.array(scores_matrix)
