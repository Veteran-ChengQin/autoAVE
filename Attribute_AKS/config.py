"""
Configuration for Attribute-conditioned Video Attribute Extraction with AKS
"""
import os

class Config:
    # Data paths
    VIDEOAVE_ROOT = "/data/veteran/project/dataE/VideoAVE"
    DATASET_ROOT = os.path.join(VIDEOAVE_ROOT, "Dataset")
    ATTRIBUTE_AKS_ROOT = "/data/veteran/project/dataE/Attribute_AKS"
    
    # Model paths
    QWEN_MODEL_PATH = "/data/share/qwen/Qwen2.5-VL-7B-Instruct"
    BLIP_MODEL_NAME = "blip-itm-base"  # or "blip-itm-large"
    
    # Video processing
    FPS_CANDIDATE = 1  # frames per second for candidate frame extraction
    MAX_FRAMES = 256  # maximum candidate frames per video
    
    # AKS (Adaptive Keyframe Sampling)
    M_ATTR = 8  # frames per attribute
    MAX_LEVEL = 4  # maximum recursion level for Judge & Split
    S_THRESHOLD = 0.6  # score threshold for TOP vs BIN decision
    
    # Qwen2.5-VL inference
    MAX_KEYFRAMES_PER_ATTR = 8  # max frames to pass to VLM per attribute
    
    # Training
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    DEVICE = "cuda:0"
    
    # Evaluation
    EVAL_DOMAINS = ["beauty"]  # domains to evaluate on
    
    # Cache
    CACHE_DIR = os.path.join(ATTRIBUTE_AKS_ROOT, ".cache")
    FRAMES_CACHE_DIR = os.path.join(CACHE_DIR, "frames")
    SCORES_CACHE_DIR = os.path.join(CACHE_DIR, "scores")
    
    # Logging
    LOG_DIR = os.path.join(ATTRIBUTE_AKS_ROOT, "logs")
