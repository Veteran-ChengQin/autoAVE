"""
Qwen2.5-VL integration for attribute value extraction
"""
import logging
import re
from typing import List, Dict, Optional
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class QwenVLExtractor:
    """
    Uses Qwen2.5-VL to extract attribute values from video frames.
    Supports both single-attribute and multi-attribute modes.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Args:
            model_path: Path to Qwen2.5-VL model
            device: Device to run model on
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Qwen2.5-VL model"""
        try:
            from transformers import AutoProcessor
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            
            logger.info(f"Loading Qwen2.5-VL from {self.model_path}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Load model with optimized settings
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,  # Required for Qwen models
            }
            
            # Skip flash_attention_2 for now to avoid dependency issues
            logger.info("Using default attention implementation")
            
            # Load the model directly
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            self.model.eval()
            
            logger.info("Qwen2.5-VL loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            logger.error("Please ensure you have the latest transformers version and qwen_vl_utils installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            raise
    
    def extract_single_attr(self, keyframes: List[Image.Image], attr_name: str,
                           title: str, category: str) -> str:
        """
        Extract value for a single attribute (Plan A: simple approach).
        
        Args:
            keyframes: List of PIL Images for this attribute
            attr_name: Attribute name
            title: Product title
            category: Product category
            
        Returns:
            Predicted attribute value as string
        """
        if not keyframes:
            logger.warning(f"No keyframes provided for {attr_name}")
            return ""
        
        # Construct prompt - request JSON format response
        system_prompt = (
            "You are an expert at extracting product attribute values from e-commerce videos. "
            "You MUST respond in valid JSON format only."
        )
        
        user_prompt = (
            f"I will show you several frames from a product video.\n"
            f"Product category: {category}\n"
            f"Product title: {title}\n"
            f"Attribute name: \"{attr_name}\"\n\n"
            f"Extract the attribute value from the video frames.\n"
            f"Respond ONLY with valid JSON in this exact format:\n"
            f'{{"value": "<extracted_value>"}}\n\n'
            f"If the attribute cannot be determined, respond with:\n"
            f'{{"value": ""}}'
        )
        
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame} for frame in keyframes
                    ] + [
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information using qwen_vl_utils
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                logger.warning("qwen_vl_utils not available, using fallback processing")
                image_inputs = keyframes
                video_inputs = None
            except Exception as e:
                logger.warning(f"process_vision_info failed: {e}, using fallback processing")
                image_inputs = keyframes
                video_inputs = None
            
            # Prepare model inputs
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except TypeError:
                # Fallback for different processor API
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
            # Trim generated ids to only new tokens (following reference pattern)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Extract the value from response
            value = self._extract_value_from_response(generated_text)
            logger.info(f"Extracted {attr_name}: {value}")
            
            return value
            
        except Exception as e:
            logger.error(f"Error extracting {attr_name}: {e}", exc_info=True)
            return ""
    
    def extract_multi_attr(self, attr_keyframes: Dict[str, List[Image.Image]],
                          title: str, category: str) -> Dict[str, str]:
        """
        Extract values for multiple attributes at once (Plan B: efficient approach).
        
        Args:
            attr_keyframes: Dict mapping attr_name -> list of PIL Images
            title: Product title
            category: Product category
            
        Returns:
            Dict mapping attr_name -> predicted value
        """
        if not attr_keyframes:
            logger.warning("No attributes provided")
            return {}
        
        # Merge all keyframes (with deduplication by time)
        all_frames = []
        attr_names = list(attr_keyframes.keys())
        
        for attr_name in attr_names:
            all_frames.extend(attr_keyframes[attr_name])
        
        # Limit total frames
        max_total_frames = 32
        if len(all_frames) > max_total_frames:
            step = len(all_frames) / max_total_frames
            all_frames = [all_frames[int(i * step)] for i in range(max_total_frames)]
        
        # Construct prompt
        system_prompt = (
            "You are an expert at extracting product attribute values from e-commerce videos."
        )
        
        attr_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(attr_names)])
        
        # Format example output
        example_format = ', '.join([f"'{name}': '<value>'" for name in attr_names])
        
        user_prompt = (
            f"I will show you several frames from a product video.\n"
            f"Product category: {category}\n"
            f"Product title: {title}\n\n"
            f"Please extract the values for the following attributes from the video content only:\n\n"
            f"{attr_list}\n\n"
            f"Answer in EXACTLY this format:\n"
            f"{example_format}\n\n"
            f"Do not mention any other attributes."
        )
        
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame} for frame in all_frames
                    ] + [
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information using qwen_vl_utils
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
            except ImportError:
                logger.warning("qwen_vl_utils not available, using fallback processing")
                image_inputs = all_frames
                video_inputs = None
            except Exception as e:
                logger.warning(f"process_vision_info failed: {e}, using fallback processing")
                image_inputs = all_frames
                video_inputs = None
            
            # Prepare model inputs
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except TypeError:
                # Fallback for different processor API
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
            # Trim generated ids to only new tokens (following reference pattern)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
    
    # Process vision information using qwen_vl_utils
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        logger.warning("qwen_vl_utils not available, using fallback processing")
        image_inputs = keyframes
        video_inputs = None
    except Exception as e:
        logger.warning(f"process_vision_info failed: {e}, using fallback processing")
        image_inputs = keyframes
        video_inputs = None
    
    # Prepare model inputs
    try:
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except TypeError:
        # Fallback for different processor API
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    
    inputs = inputs.to(self.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
    
    # Trim generated ids to only new tokens (following reference pattern)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode
    generated_text = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Extract the value from response (JSON format)
    value = self._extract_value_from_json_response(generated_text)
    logger.info(f"Extracted {attr_name}: {value}")
    
    return value
            
except Exception as e:
    logger.error(f"Error extracting {attr_name}: {e}", exc_info=True)
    return ""
    
def _extract_value_from_json_response(self, response: str) -> str:
    """
    Extract attribute value from JSON-formatted response.
    
    Handles cases where MLLM fails to extract the target attribute.
    """
    import json
    
    response = response.strip()
    
    # Try to parse as JSON
    try:
        # First, try direct JSON parsing
        data = json.loads(response)
        if isinstance(data, dict) and "value" in data:
            value = str(data["value"]).strip()
            return value if value else ""
    except json.JSONDecodeError:
        pass
    
    # Fallback: try to extract JSON from response (in case of extra text)
    try:
        # Look for JSON object pattern
        import re
        json_match = re.search(r'\{[^{}]*"value"[^{}]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            if isinstance(data, dict) and "value" in data:
                value = str(data["value"]).strip()
                return value if value else ""
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # If all JSON parsing fails, return empty string (attribute not extracted)
    logger.warning(f"Failed to parse JSON response: {response}")
    return ""

def extract_multi_attr(self, attr_keyframes: Dict[str, List[Image.Image]],
                      title: str, category: str) -> Dict[str, str]:
    """
    Extract values for multiple attributes at once (Plan B: efficient approach).
    
    Args:
        attr_keyframes: Dict mapping attr_name -> list of PIL Images
        title: Product title
        category: Product category
        
    Returns:
        Dict mapping attr_name -> predicted value
    """
    if not attr_keyframes:
        logger.warning("No attributes provided")
        return {}
    
    # Merge all keyframes (with deduplication by time)
    all_frames = []
    attr_names = list(attr_keyframes.keys())
    
    for attr_name in attr_names:
        all_frames.extend(attr_keyframes[attr_name])
    
    # Limit total frames
    max_total_frames = 32
    if len(all_frames) > max_total_frames:
        step = len(all_frames) / max_total_frames
        all_frames = [all_frames[int(i * step)] for i in range(max_total_frames)]
    
    # Construct prompt
    system_prompt = (
        "You are an expert at extracting product attribute values from e-commerce videos."
    )
    
    attr_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(attr_names)])
    
    # Format example output
    example_format = ', '.join([f"'{name}': '<value>'" for name in attr_names])
    
    user_prompt = (
        f"I will show you several frames from a product video.\n"
        f"Product category: {category}\n"
        f"Product title: {title}\n\n"
        f"Please extract the values for the following attributes from the video content only:\n\n"
        f"{attr_list}\n\n"
        f"Answer in EXACTLY this format:\n"
        f"{example_format}\n\n"
        f"Do not mention any other attributes."
    )
    
    try:
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame} for frame in all_frames
                ] + [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
            # Try to extract key-value pairs from response
            for attr_name in attr_names:
                # Look for patterns like 'attr_name': 'value' or attr_name: value
                pattern = rf"'{attr_name}':\s*'([^']*)'|{re.escape(attr_name)}:\s*'([^']*)'|{re.escape(attr_name)}:\s*([^\n,]*)"
                match = re.search(pattern, response, re.IGNORECASE)
                
                if match:
                    # Get the first non-None group
                    value = next((g for g in match.groups() if g is not None), "")
                    result[attr_name] = value.strip()
        
        except Exception as e:
            logger.warning(f"Error parsing response: {e}", exc_info=True)
        
        return result
