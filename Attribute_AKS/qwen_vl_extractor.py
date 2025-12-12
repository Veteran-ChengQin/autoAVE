"""
Qwen2.5-VL integration for attribute value extraction
"""
import logging
import re
import json
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
                           title: str, category: str) -> Dict[str, str]:
        """
        Extract value for a single attribute with JSON response format.
        
        Args:
            keyframes: List of PIL Images for this attribute
            attr_name: Attribute name
            title: Product title
            category: Product category
            
        Returns:
            Dictionary with attr_name as key and extracted value as value
            e.g., {"color": "red"} or {"color": ""} if not extracted
        """
        if not keyframes:
            logger.warning(f"No keyframes provided for {attr_name}")
            return {attr_name: ""}
        
        # Construct prompt - request JSON format response
        user_prompt = (
            f"I will show you several frames from a product video.\n"
            f"Product category: {category}\n"
            f"Product title: {title}\n"
            f"Attribute name: \"{attr_name}\"\n\n"
            f"Extract the attribute value from the video frames.\n"
            f"Respond ONLY with valid JSON in this exact format:\n"
            '{"<attr_name>": "<extracted_value>"}\n\n'
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
            
            # Extract the value from JSON response
            result = self._extract_value_from_json_response(generated_text, attr_name)
            logger.info(f"Extracted {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting {attr_name}: {e}", exc_info=True)
            return {attr_name: ""}
    
    def extract_multi_attr(self, attr_keyframes: Dict[str, List[Image.Image]],
                          title: str, category: str) -> Dict[str, str]:
        """
        Extract values for multiple attributes at once with JSON response format.
        
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
        
        # Construct prompt with JSON format
        attr_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(attr_names)])
        
        # Build example JSON format
        example_json = {name: "<value>" for name in attr_names}
        example_json_str = json.dumps(example_json)
        
        user_prompt = (
            f"I will show you several frames from a product video.\n"
            f"Product category: {category}\n"
            f"Product title: {title}\n\n"
            f"Please extract the values for the following attributes from the video content only:\n\n"
            f"{attr_list}\n\n"
            f"Respond ONLY with valid JSON in this exact format:\n"
            f"{example_json_str}\n\n"
            f"If an attribute cannot be determined, use empty string as value.\n"
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
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # Parse JSON response
            result = self._parse_multi_attr_json_response(generated_text, attr_names)
            logger.info(f"Extracted attributes: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}", exc_info=True)
            return {name: "" for name in attr_names}
    
    def _extract_value_from_json_response(self, response: str, attr_name: str) -> Dict[str, str]:
        """
        Extract attribute value from JSON-formatted response.
        
        Handles cases where MLLM fails to extract the target attribute.
        Returns dict with attr_name as key and extracted value as value.
        
        Args:
            response: MLLM response text
            attr_name: Target attribute name
            
        Returns:
            Dictionary like {attr_name: "value"} or {attr_name: ""} if not extracted
        """
        import ast
        
        response = response.strip()
        
        # Step 1: Remove Markdown code blocks (```json ... ```)
        response = re.sub(r'```(?:json)?\s*\n?', '', response)
        response = response.strip()
        
        # Step 2: Try to parse as JSON directly
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        
        # Step 3: Try to extract JSON object from response (in case of extra text)
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                json_str = json_match.group(0)
                # Try direct JSON parsing first
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass
                
                # If direct parsing fails, try ast.literal_eval
                # This handles cases where the JSON is wrapped in escaped quotes
                # e.g., {"\"Color\": \"black\""} parses as a set in Python
                try:
                    data = ast.literal_eval(json_str)
                    
                    # Handle set (which occurs when JSON has escaped quotes)
                    if isinstance(data, set):
                        # Combine all set elements into a single JSON object
                        result = {}
                        for item in data:
                            # Each item is a string like "key": "value"
                            # Wrap it in braces to make it valid JSON
                            try:
                                item_json = json.loads("{" + item + "}")
                                result.update(item_json)
                            except (json.JSONDecodeError, ValueError):
                                pass
                        if result:
                            return result
                    
                    # Handle dict
                    elif isinstance(data, dict):
                        # If we got a dict from literal_eval, parse the inner JSON if it's a string
                        result = {}
                        for key, value in data.items():
                            if isinstance(value, str):
                                try:
                                    # Try to parse string values as JSON
                                    result[key] = json.loads(value)
                                except (json.JSONDecodeError, ValueError):
                                    result[key] = value
                            else:
                                result[key] = value
                        return result
                except (ValueError, SyntaxError, TypeError):
                    pass
        except (AttributeError, TypeError):
            pass
        
        # Step 4: Handle escaped quotes manually as last resort
        try:
            # Replace escaped quotes with regular quotes
            response_unescaped = response.replace('\\"', '"')
            # Try to parse
            data = json.loads(response_unescaped)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Step 5: If all JSON parsing fails, return empty value (attribute not extracted)
        logger.warning(f"Failed to parse JSON response for {attr_name}: {response}")
        return {attr_name: ""}
    
    def _parse_multi_attr_json_response(self, response: str, attr_names: List[str]) -> Dict[str, str]:
        """
        Parse multi-attribute JSON response.
        
        Handles cases where MLLM fails to extract specific attributes.
        """
        import ast
        
        result = {name: "" for name in attr_names}
        
        response = response.strip()
        
        # Step 1: Remove Markdown code blocks (```json ... ```)
        response = re.sub(r'```(?:json)?\s*\n?', '', response)
        response = response.strip()
        
        # Step 2: Try to parse as JSON directly
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                for attr_name in attr_names:
                    if attr_name in data:
                        value = str(data[attr_name]).strip()
                        result[attr_name] = value if value else ""
                return result
        except json.JSONDecodeError:
            pass
        
        # Step 3: Try to extract JSON object from response (in case of extra text)
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
            if json_match:
                json_str = json_match.group(0)
                # Try direct JSON parsing first
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        for attr_name in attr_names:
                            if attr_name in data:
                                value = str(data[attr_name]).strip()
                                result[attr_name] = value if value else ""
                        return result
                except json.JSONDecodeError:
                    pass
                
                # If direct parsing fails, try ast.literal_eval
                # This handles cases where the JSON is wrapped in escaped quotes
                try:
                    data = ast.literal_eval(json_str)
                    
                    # Handle set (which occurs when JSON has escaped quotes)
                    if isinstance(data, set):
                        # Combine all set elements into a single JSON object
                        combined = {}
                        for item in data:
                            # Each item is a string like "key": "value"
                            # Wrap it in braces to make it valid JSON
                            try:
                                item_json = json.loads("{" + item + "}")
                                combined.update(item_json)
                            except (json.JSONDecodeError, ValueError):
                                pass
                        
                        # Extract requested attributes
                        for attr_name in attr_names:
                            if attr_name in combined:
                                value = str(combined[attr_name]).strip()
                                result[attr_name] = value if value else ""
                        
                        if any(result[name] for name in attr_names):
                            return result
                    
                    # Handle dict
                    elif isinstance(data, dict):
                        for attr_name in attr_names:
                            if attr_name in data:
                                value = str(data[attr_name]).strip()
                                result[attr_name] = value if value else ""
                        return result
                except (ValueError, SyntaxError, TypeError):
                    pass
        except (AttributeError, TypeError):
            pass
        
        # Step 4: Handle escaped quotes manually as last resort
        try:
            # Replace escaped quotes with regular quotes
            response_unescaped = response.replace('\\"', '"')
            # Try to parse
            data = json.loads(response_unescaped)
            if isinstance(data, dict):
                for attr_name in attr_names:
                    if attr_name in data:
                        value = str(data[attr_name]).strip()
                        result[attr_name] = value if value else ""
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Step 5: If all JSON parsing fails, log warning
        logger.warning(f"Failed to parse multi-attribute JSON response: {response}")
        return result
