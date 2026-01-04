"""
Qwen2.5-VL integration for attribute value extraction
"""
import logging
import re
import json
import base64
import io
import os
import time
import pickle
from typing import List, Dict, Optional, Literal
from pathlib import Path
from PIL import Image
import torch
import requests

logger = logging.getLogger(__name__)


class QwenVLExtractor:
    """
    Uses Qwen2.5-VL to extract attribute values from video frames.
    Supports both single-attribute and multi-attribute modes.
    Supports both local model and API-based inference.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None, 
                 device: str = "cuda:0",
                 mode: Literal["local", "api"] = "local",
                 api_key: Optional[str] = None,
                 api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                 api_model: str = "qwen-vl-plus",
                 cache_dir: Optional[str] = None):
        """
        Args:
            model_path: Path to Qwen2.5-VL model (required for local mode)
            device: Device to run model on (for local mode)
            mode: "local" for local model inference, "api" for API calls
            api_key: API key for API mode
            api_url: API endpoint URL
            api_model: Model name for API calls
            cache_dir: Directory to store OSS URL cache (for API mode)
        """
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        
        # API configuration
        self.api_key = api_key
        self.api_url = api_url
        self.api_model = api_model
        
        # OSS URL cache configuration
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".qwen_vl_cache")
        self.oss_cache_file = os.path.join(self.cache_dir, "oss_url_cache.pkl")
        self._oss_url_cache = self._load_oss_cache()
        
        if self.mode == "local":
            if not model_path:
                raise ValueError("model_path is required for local mode")
            self._load_model()
        elif self.mode == "api":
            if not api_key:
                raise ValueError("api_key is required for API mode")
            logger.info(f"Using API mode with model: {self.api_model}")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'api'")
    
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
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string for API calls."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    
    def _load_oss_cache(self) -> Dict[str, str]:
        """Load OSS URL cache from disk"""
        try:
            if os.path.exists(self.oss_cache_file):
                with open(self.oss_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded OSS URL cache with {len(cache)} entries")
                return cache
            else:
                logger.info("No existing OSS URL cache found, starting with empty cache")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load OSS URL cache: {e}, starting with empty cache")
            return {}
    
    def _save_oss_cache(self) -> None:
        """Save OSS URL cache to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self.oss_cache_file, 'wb') as f:
                pickle.dump(self._oss_url_cache, f)
            logger.debug(f"Saved OSS URL cache with {len(self._oss_url_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save OSS URL cache: {e}")
    
    def _get_cached_oss_url(self, product_id: str) -> Optional[str]:
        """Get cached OSS URL for a product_id"""
        return self._oss_url_cache.get(product_id)
    
    def _cache_oss_url(self, product_id: str, oss_url: str) -> None:
        """Cache OSS URL for a product_id"""
        self._oss_url_cache[product_id] = oss_url
        self._save_oss_cache()
        logger.info(f"Cached OSS URL for product_id {product_id}: {oss_url}")
    
    def _get_upload_policy(self) -> Dict:
        """获取文件上传凭证"""
        url = "https://dashscope.aliyuncs.com/api/v1/uploads"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "action": "getPolicy",
            "model": self.api_model
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to get upload policy: {response.text}")
        
        return response.json()['data']
    
    def _upload_file_to_oss(self, policy_data: Dict, file_path: str) -> str:
        """将文件上传到临时存储OSS"""
        file_name = Path(file_path).name
        key = f"{policy_data['upload_dir']}/{file_name}"
        
        try:
            with open(file_path, 'rb') as file:
                files = {
                    'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
                    'Signature': (None, policy_data['signature']),
                    'policy': (None, policy_data['policy']),
                    'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
                    'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
                    'key': (None, key),
                    'success_action_status': (None, '200'),
                    'file': (file_name, file)
                }
                
                response = requests.post(policy_data['upload_host'], files=files, timeout=60)
                if response.status_code != 200:
                    raise Exception(f"Failed to upload file: {response.text}")
            
            return f"oss://{key}"
        except Exception as e:
            logger.error(f"Failed to upload file {file_path} to OSS: {e}")
            raise
    
    def _upload_video_and_get_url(self, file_path: str, product_id: Optional[str] = None) -> str:
        """上传视频文件并获取OSS URL，支持缓存功能"""
        # 如果提供了product_id，先检查缓存
        if product_id:
            cached_url = self._get_cached_oss_url(product_id)
            if cached_url:
                logger.info(f"Using cached OSS URL for product_id {product_id}: {cached_url}")
                return cached_url
        
        try:
            logger.info(f"Uploading video to OSS: {file_path}")
            # 1. 获取上传凭证
            policy_data = self._get_upload_policy()
            # 2. 上传文件到OSS
            oss_url = self._upload_file_to_oss(policy_data, file_path)
            logger.info(f"Video uploaded successfully, OSS URL: {oss_url}")
            
            # 如果提供了product_id，缓存结果
            if product_id:
                self._cache_oss_url(product_id, oss_url)
            
            return oss_url
        except Exception as e:
            logger.error(f"Failed to upload video {file_path}: {e}")
            raise
    
    def _call_api(self, messages: List[Dict], stream: bool = False, max_retries: int = 3) -> str:
        """
        Call Qwen VL API with retry logic and increased timeout.
        
        Args:
            messages: List of message dicts with role and content
            stream: Whether to use streaming mode
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-OssResourceResolve": "enable"
        }
        
        payload = {
            "model": self.api_model,
            "temperature": 0.2,
            "messages": messages,
            "stream": stream
        }
        
        if stream:
            payload["stream_options"] = {"include_usage": True}
        
        # Increased timeout from 60 to 180 seconds
        timeout = 180
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        full_response += content
                                except json.JSONDecodeError:
                                    continue
                    return full_response
                else:
                    # Handle non-streaming response
                    result = response.json()
                    return result['choices'][0]['message']['content']
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Retry on timeout or connection errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"API call timeout/connection error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                # Don't retry on other request exceptions
                logger.error(f"API call failed: {e}")
                raise
    
    def extract_single_attr_from_video(self, video_url: str, attr_name: str,
                                       title: str, category: str, product_id: Optional[str] = None, max_frames: int = 16) -> Dict[str, str]:
        """
        Extract value for a single attribute directly from video URL.
        Supports both API mode and local mode.
        
        Args:
            video_url: URL of the video (http/https URL or local file path)
            attr_name: Attribute name
            title: Product title
            category: Product category
            product_id: Product ID for OSS URL caching (optional)
            max_frames: Maximum number of frames to extract from video
            
        Returns:
            Dictionary with attr_name as key and extracted value as value
        """
        if self.mode == "local":
            # Local mode: extract frames and process them
            return self._extract_single_attr_from_video_local(video_url, attr_name, title, category, max_frames)
        elif self.mode == "api":
            # API mode: use existing implementation
            return self._extract_single_attr_from_video_api(video_url, attr_name, title, category, product_id)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _extract_single_attr_from_video_local(self, video_url: str, attr_name: str,
                                            title: str, category: str, max_frames: int = 16) -> Dict[str, str]:
        """
        Extract value for a single attribute from video using local model.
        """
        try:
            from video_utils import extract_candidate_frames
            
            # Extract frames from video (use reasonable defaults)
            logger.info(f"Extracting frames from video: {video_url}")
            frames, timestamps = extract_candidate_frames(video_url, fps=1.0, max_frames=max_frames)
            
            if not frames:
                logger.warning(f"No frames extracted from video: {video_url}")
                return {attr_name: ""}
            
            logger.info(f"Extracted {len(frames)} frames, processing with local model")
            
            # Use existing extract_single_attr method for frame processing
            return self.extract_single_attr(frames, attr_name, title, category)
            
        except Exception as e:
            logger.error(f"Error extracting {attr_name} from video in local mode: {e}", exc_info=True)
            return {attr_name: ""}
    
    def _extract_single_attr_from_video_api(self, video_url: str, attr_name: str,
                                          title: str, category: str, product_id: Optional[str] = None) -> Dict[str, str]:
        """
        Extract value for a single attribute from video using API mode.
        """
        
        # Construct prompt - request JSON format response
        user_prompt = (
            f"This video is mainly used to introduce products in the {category} category.\n"
            f"Product title: {title}\n"
            f"Please extract the value of the product attribute \"{attr_name}\" from the video.\n\n"
            f"Please return only valid JSON format as follows:\n"
            '{"{attr_name}": "<extracted attribute value>"}\n\n'
            f"If the attribute value cannot be determined, please use an empty string."
        )
        
        try:
            # Check if video_url is a local file path or remote URL
            if os.path.exists(video_url):
                # Local file - upload to OSS and get temporary URL (with caching)
                logger.info(f"Processing local video file: {video_url}")
                video_data_url = self._upload_video_and_get_url(video_url, product_id)
            else:
                # Remote URL - use directly
                video_data_url = video_url
            
            # Prepare API request with video URL or base64 data
            content = [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video_data_url
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
            
            messages = [{"role": "user", "content": content}]
            
            # Call API
            generated_text = self._call_api(messages, stream=False)
            
            # Extract the value from JSON response
            result = self._extract_value_from_json_response(generated_text, attr_name)
            logger.info(f"Extracted {result} from video")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting {attr_name} from video in API mode: {e}", exc_info=True)
            return {attr_name: ""}
    
    def extract_multi_attr_from_video(self, video_url: str, attr_names: List[str],
                                      title: str, category: str, product_id: Optional[str] = None, max_frames: int = 16) -> Dict[str, str]:
        """
        Extract values for multiple attributes directly from video URL.
        Supports both API mode and local mode.
        
        Args:
            video_url: URL of the video (http/https URL or local file path)
            attr_names: List of attribute names
            title: Product title
            category: Product category
            product_id: Product ID for OSS URL caching (optional)
            max_frames: Maximum number of frames to extract from video
            
        Returns:
            Dict mapping attr_name -> predicted value
        """
        if self.mode == "local":
            # Local mode: extract frames and process them
            return self._extract_multi_attr_from_video_local(video_url, attr_names, title, category, max_frames)
        elif self.mode == "api":
            # API mode: use existing implementation
            return self._extract_multi_attr_from_video_api(video_url, attr_names, title, category, product_id)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _extract_multi_attr_from_video_local(self, video_url: str, attr_names: List[str],
                                           title: str, category: str, max_frames: int = 16) -> Dict[str, str]:
        """
        Extract values for multiple attributes from video using local model.
        """
        if not attr_names:
            logger.warning("No attributes provided")
            return {}
        
        try:
            from video_utils import extract_candidate_frames
            
            # Extract frames from video (use reasonable defaults)
            logger.info(f"Extracting frames from video: {video_url}")
            frames, timestamps = extract_candidate_frames(video_url, fps=1.0, max_frames=max_frames)
            
            if not frames:
                logger.warning(f"No frames extracted from video: {video_url}")
                return {name: "" for name in attr_names}
            
            logger.info(f"Extracted {len(frames)} frames, processing with local model")
            
            # Create attr_keyframes dict for multi-attribute processing
            attr_keyframes = {attr_name: frames for attr_name in attr_names}
            
            # Use existing extract_multi_attr method for frame processing
            return self.extract_multi_attr(attr_keyframes, title, category)
            
        except Exception as e:
            logger.error(f"Error extracting attributes from video in local mode: {e}", exc_info=True)
            return {name: "" for name in attr_names}
    
    def _extract_multi_attr_from_video_api(self, video_url: str, attr_names: List[str],
                                         title: str, category: str, product_id: Optional[str] = None) -> Dict[str, str]:
        """
        Extract values for multiple attributes from video using API mode.
        """
        
        if not attr_names:
            logger.warning("No attributes provided")
            return {}
        
        # Construct prompt with JSON format
        attr_list_str = str(attr_names)
        
        # Build example JSON format
        example_json = {name: "<value>" for name in attr_names}
        example_json_str = json.dumps(example_json, ensure_ascii=False)
        
        user_prompt = (
            f"This video is mainly used to introduce products in the {category} category.\n"
            f"Product title: {title}\n\n"
            f"Please extract the values of these product attributes from the video, attribute list: {attr_list_str}\n\n"
            f"Please return only valid JSON format as follows:\n"
            f"{example_json_str}\n\n"
            f"If an attribute value cannot be determined, please use an empty string as the value.\n"
            f"Do not mention other attributes."
        )
        
        try:
            # Check if video_url is a local file path or remote URL
            if os.path.exists(video_url):
                # Local file - upload to OSS and get temporary URL (with caching)
                logger.info(f"Processing local video file: {video_url}")
                video_data_url = self._upload_video_and_get_url(video_url, product_id)
            else:
                # Remote URL - use directly
                video_data_url = video_url
            
            # Prepare API request with video URL or base64 data
            content = [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": video_data_url
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
            
            messages = [{"role": "user", "content": content}]
            
            # Call API
            generated_text = self._call_api(messages, stream=False)
            
            # Parse JSON response
            result = self._parse_multi_attr_json_response(generated_text, attr_names)
            logger.info(f"Extracted attributes from video: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting attributes from video in API mode: {e}", exc_info=True)
            return {name: "" for name in attr_names}
    
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
            f"Target attribute: {attr_name}\n\n"
            f"Extract the {attr_name} value from the video frames.\n"
            f"Respond with ONLY valid JSON using this EXACT format (no extra text, no markdown):\n"
            f'{{"{attr_name}": "value_here"}}\n\n'
            f"Example response: {{\"{attr_name}\": \"red\"}}\n"
        )
        
        try:
            if self.mode == "api":
                # API mode: encode images as base64
                content = []
                for frame in keyframes:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_to_base64(frame)
                        }
                    })
                content.append({"type": "text", "text": user_prompt})
                
                messages = [{"role": "user", "content": content}]
                
                # Call API
                generated_text = self._call_api(messages, stream=False)
                
            else:
                # Local mode: use transformers
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
            if self.mode == "api":
                # API mode: encode images as base64
                content = []
                for frame in all_frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": self._encode_image_to_base64(frame)
                        }
                    })
                content.append({"type": "text", "text": user_prompt})
                
                messages = [{"role": "user", "content": content}]
                
                # Call API
                generated_text = self._call_api(messages, stream=False)
                
            else:
                # Local mode: use transformers
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
        
        # Step 4: Handle malformed JSON with manual parsing
        try:
            # Look for key-value patterns in the response
            # Handle cases like: "\"Brand\"": "\"Osensia\"\""
            # or: "\"Item Form\"": "\"box of lipsticks and eyeshadow palette"
            
            # Find all potential key-value pairs
            # Pattern: optional quotes + escaped quotes + key + escaped quotes + optional quotes + colon + space + value
            lines = response.split('\n')
            result = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line and (line.startswith('"') or line.strip().startswith('"')):
                    # Split on the first colon
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0].strip()
                        value_part = parts[1].strip()
                        
                        # Clean up key: remove outer quotes and unescape inner quotes
                        # Handle: "\"Brand\"" -> Brand
                        key = key_part.strip('"').strip("'")
                        if key.startswith('\\"') and key.endswith('\\"'):
                            key = key[2:-2]  # Remove \" from both ends
                        key = key.replace('\\"', '"')
                        
                        # Clean up value: remove quotes and handle malformed endings
                        # Handle: "\"Osensia\"\"" -> Osensia
                        # Handle: "\"box of lipsticks and eyeshadow palette" -> box of lipsticks and eyeshadow palette
                        value = value_part.strip().rstrip(',').rstrip('}').strip()
                        value = value.strip('"').strip("'")
                        if value.startswith('\\"'):
                            value = value[2:]  # Remove leading \"
                        if value.endswith('\\"'):
                            value = value[:-2]  # Remove trailing \"
                        # Handle extra quotes at the end
                        while value.endswith('"') and not value.endswith('\\"'):
                            value = value[:-1]
                        value = value.replace('\\"', '"')
                        
                        if key:  # Only add if key is not empty
                            result[key] = value
            
            if result:
                return result
        except (AttributeError, TypeError, IndexError):
            pass
        
        # Step 5: Handle escaped quotes manually as last resort
        try:
            # Replace escaped quotes with regular quotes
            response_unescaped = response.replace('\\"', '"')
            # Try to parse
            data = json.loads(response_unescaped)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Step 6: If all JSON parsing fails, return empty value (attribute not extracted)
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
