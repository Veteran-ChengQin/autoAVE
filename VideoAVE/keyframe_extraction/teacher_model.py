"""
Teacher model wrapper for VideoAVE evaluation
"""
import torch
import numpy as np
import os
import ast
import re
from typing import List, Dict, Any, Tuple
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TeacherModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Load model and processor
        logger.info(f"Loading teacher model from {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Get vision backbone for feature extraction
        self.vision_backbone = self.model.visual
        
        logger.info("Teacher model loaded successfully")
    
    def extract_frame_features(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Extract visual features from frames using the full model
        
        Args:
            frames: List of PIL Images
            
        Returns:
            features: Tensor of shape (num_frames, feature_dim)
        """
        with torch.no_grad():
            # Use a simple dummy text for feature extraction
            dummy_text = "Describe this image."
            
            # Process each frame individually to get consistent features
            all_features = []
            
            for frame in frames:
                try:
                    # Create messages for the model
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": dummy_text},
                            {"type": "image", "image": frame}
                        ]}
                    ]
                    
                    # Apply chat template
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # Process inputs
                    inputs = self.processor(
                        text=[text],
                        images=[frame],
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get hidden states from the model (without generation)
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(**inputs, output_hidden_states=True)
                        # Use the last hidden state as features
                        hidden_states = outputs.hidden_states[-1]  # Last layer
                        # Average pool over sequence length to get frame-level features
                        frame_feature = hidden_states.mean(dim=1)  # (1, hidden_dim)
                        all_features.append(frame_feature.cpu())
                        
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error extracting features for frame: {e}")
                    # Use zero features as fallback
                    zero_feature = torch.zeros(1, self.model.config.hidden_size)
                    all_features.append(zero_feature)
            
            # Concatenate all frame features
            if all_features:
                features = torch.cat(all_features, dim=0)  # (num_frames, feature_dim)
            else:
                # Return empty tensor if no frames
                features = torch.empty(0, self.model.config.hidden_size)
            
            return features
    
    def predict_attributes(self, frames: List[Image.Image], title: str, 
                          category: str, mode: str = "attribute_conditioned",
                          attributes: List[str] = None) -> Dict[str, str]:
        """
        Predict attributes for given frames
        
        Args:
            frames: List of PIL Images
            title: Product title
            category: Product category
            mode: "attribute_conditioned" or "generalized"
            attributes: List of attribute names (for attribute_conditioned mode)
            
        Returns:
            Dictionary of predicted attribute-value pairs
        """
        try:
            # Create prompt based on mode
            if mode == "attribute_conditioned" and attributes:
                attr_str = ", ".join(attributes)
                prompt = (f"You are a professional product analyst specializing in attribute extraction "
                         f"for leading e-commerce platforms. Your task is to extract the values of the "
                         f"following attributes for the product in this video: {attr_str}. "
                         f"Answer it in this format only: 'attribute1': 'attribute1_value', "
                         f"'attribute2': 'attribute2_value', ... Choose one specific value for each attribute.")
            else:
                prompt = ("You are a professional product analyst specializing in attribute extraction "
                         "for leading e-commerce platforms. Your task is to extract the attributes "
                         "of the product in this video. Answer it in this format only: "
                         "'attribute1': 'attribute1_value', 'attribute2': 'attribute2_value', ... "
                         "Choose one specific value for each attribute.")
            
            # Create video from frames (save temporarily)
            temp_video_path = self._frames_to_temp_video(frames)
            
            # Run inference
            response = self._inference(temp_video_path, prompt)
            
            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            # Parse response to dictionary
            pred_attrs = self._parse_response(response)
            
            return pred_attrs
            
        except Exception as e:
            logger.error(f"Error in attribute prediction: {e}")
            return {}
    
    def _frames_to_temp_video(self, frames: List[Image.Image], fps: int = 2) -> str:
        """Convert frames to temporary video file"""
        import tempfile
        import cv2
        
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for frame in frames:
            frame_array = np.array(frame)
            if frame_array.shape[2] == 3:  # RGB
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            frame_arrays.append(frame_array)
        
        # Write video
        if frame_arrays:
            height, width = frame_arrays[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            for frame_array in frame_arrays:
                out.write(frame_array)
            
            out.release()
        
        return temp_path
    
    def _inference(self, video_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        """Run inference on video with prompt"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"video": video_path, "total_pixels": 20480 * 28 * 28, "min_pixels": 224*224},
                ]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            fps_inputs = video_kwargs.get('fps', None)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                fps=fps_inputs,
                padding=False,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            # Clean up
            del inputs, output_ids, generated_ids, image_inputs, video_inputs
            torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse model response to extract attribute-value pairs"""
        try:
            # Try to find patterns like 'key': 'value'
            pattern = r"'([^']+)':\s*'([^']+)'"
            matches = re.findall(pattern, response)
            
            if matches:
                return dict(matches)
            
            # Fallback: try to parse as dictionary
            try:
                # Clean response and try to evaluate as dict
                cleaned = response.strip()
                if not cleaned.startswith('{'):
                    cleaned = '{' + cleaned + '}'
                return ast.literal_eval(cleaned)
            except:
                pass
            
            logger.warning(f"Could not parse response: {response}")
            return {}
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

def fuzzy_f1_score(pred_attrs: Dict[str, str], gold_attrs: Dict[str, str]) -> float:
    """
    Compute fuzzy F1 score based on VideoAVE evaluation criteria
    
    Args:
        pred_attrs: Predicted attribute-value pairs
        gold_attrs: Ground truth attribute-value pairs
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not gold_attrs:
        return 1.0 if not pred_attrs else 0.0
    
    if not pred_attrs:
        return 0.0
    
    def fuzzy_match(pred_val: str, gold_val: str) -> bool:
        """Check if predicted value matches gold value using fuzzy matching"""
        pred_val = str(pred_val).lower().strip()
        gold_val = str(gold_val).lower().strip()
        
        # Common substring length >= 50% of gold value length
        common_len = len(os.path.commonprefix([pred_val, gold_val]))
        return common_len >= (len(gold_val) / 2)
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_gold_keys = set()
    
    # Check each predicted attribute
    for pred_key, pred_val in pred_attrs.items():
        if pred_key in gold_attrs:
            if fuzzy_match(pred_val, gold_attrs[pred_key]):
                tp += 1
            else:
                fp += 1  # Wrong value
            matched_gold_keys.add(pred_key)
        else:
            fp += 1  # Extra attribute
    
    # Count missing attributes
    for gold_key in gold_attrs:
        if gold_key not in matched_gold_keys:
            fn += 1
    
    # Calculate F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1
