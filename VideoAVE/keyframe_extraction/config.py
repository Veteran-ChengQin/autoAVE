"""
Configuration file for keyframe extraction system
"""
import os

class Config:
    # Data paths
    VIDEOAVE_ROOT = "/data/veteran/project/dataE/VideoAVE"
    DATASET_ROOT = os.path.join(VIDEOAVE_ROOT, "Dataset")
    
    # Model paths
    TEACHER_MODEL_PATH = "/data/share/qwen/Qwen2.5-VL-7B-Instruct"
    
    # Video processing
    TARGET_FPS = 2  # frames per second for sampling
    MAX_FRAMES = 48  # maximum frames per video
    
    # Teacher supervision
    BASELINE_FRAMES = 16  # number of frames in baseline subset
    MONTE_CARLO_SAMPLES = 3  # number of MC samples per frame
    
    # Student model
    VISION_DIM = 1024  # dimension of vision features (CLIP-ViT)
    HIDDEN_DIM = 512  # hidden dimension of MLP
    
    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 10
    DEVICE = "cuda:0"
    
    # Evaluation
    TOP_K_FRAMES = [4, 8, 16]  # different frame budgets to test
    
    # Domains to use for prototype (start with 2 domains)
    PROTOTYPE_DOMAINS = ["beauty", "sports"]
    
    # Cache
    CACHE_DIR = ".cache"
    FEATURES_CACHE_DIR = os.path.join(CACHE_DIR, "features")
    SCORES_CACHE_DIR = os.path.join(CACHE_DIR, "scores")
