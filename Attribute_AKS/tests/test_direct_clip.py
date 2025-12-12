"""
Direct test of CLIP scoring without FrameScorer wrapper
"""
import logging
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images(num_images: int = 5) -> list:
    """Create test images with different colors"""
    images = []
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
    ]
    
    for color in colors[:num_images]:
        img = Image.new('RGB', (224, 224), color=color)
        images.append(img)
    
    return images


def main():
    logger.info("=" * 80)
    logger.info("DIRECT CLIP SCORING TEST")
    logger.info("=" * 80)
    
    device = "cuda:0"
    model_id = "openai/clip-vit-base-patch32"
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    
    images = create_test_images(5)
    text = "a red object"
    
    logger.info(f"\nQuery: '{text}'")
    logger.info("Images: Red, Green, Blue, Yellow, Magenta")
    
    with torch.no_grad():
        # Process text and images separately
        text_inputs = processor(text=text, return_tensors="pt", padding=True)
        image_inputs = processor(images=images, return_tensors="pt")
        
        # Move to device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        # Get embeddings
        text_features = model.get_text_features(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs.get('attention_mask')
        )
        image_features = model.get_image_features(
            pixel_values=image_inputs['pixel_values']
        )
        
        logger.info(f"\nText features shape: {text_features.shape}")
        logger.info(f"Image features shape: {image_features.shape}")
        
        # Normalize embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        logit_scale = model.logit_scale.exp()
        logger.info(f"Logit scale: {logit_scale.item():.4f}")
        
        # Raw logits
        logits = image_features @ text_features.t() * logit_scale
        logger.info(f"\nRaw logits shape: {logits.shape}")
        logger.info(f"Raw logits:\n{logits}")
        
        # After squeeze
        squeezed = logits.squeeze(-1)
        logger.info(f"\nAfter squeeze shape: {squeezed.shape}")
        logger.info(f"After squeeze:\n{squeezed}")
        
        # After sigmoid
        sigmoid_scores = torch.sigmoid(squeezed)
        logger.info(f"\nAfter sigmoid:\n{sigmoid_scores}")
        
        # Convert to numpy
        scores_np = sigmoid_scores.cpu().numpy()
        logger.info(f"\nFinal scores: {[f'{s:.4f}' for s in scores_np]}")
        logger.info(f"Min: {scores_np.min():.4f}, Max: {scores_np.max():.4f}")
        logger.info(f"Std: {scores_np.std():.4f}")
        
        if np.allclose(scores_np, scores_np[0]):
            logger.error("❌ All scores are identical!")
        else:
            logger.info("✓ Scores are different")


if __name__ == "__main__":
    main()
