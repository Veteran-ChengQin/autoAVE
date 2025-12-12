#!/usr/bin/env python
"""Quick test to verify CLIP model loading"""
import os
import sys
import torch

# Disable torch.load safety check
os.environ["TRANSFORMERS_UNSAFE_LOAD"] = "1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    from transformers import CLIPProcessor, CLIPModel
    
    model_id = "openai/clip-vit-base-patch32"
    print(f"\nAttempting to load CLIP model: {model_id}")
    
    processor = CLIPProcessor.from_pretrained(model_id)
    print("✓ Processor loaded")
    
    model = CLIPModel.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("✓ Model loaded")
    
    model = model.to("cuda:0")
    print("✓ Model moved to GPU")
    
    model.eval()
    print("✓ Model set to eval mode")
    
    print("\n✓ CLIP model loaded successfully!")
    
except Exception as e:
    print(f"\n✗ Error loading CLIP model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
