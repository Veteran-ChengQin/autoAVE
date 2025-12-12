"""
Detailed debugging of CLIP scoring issue
"""
import os
import logging
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_images(num_images: int = 3) -> list:
    """Create test images with different colors"""
    images = []
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
    ]
    
    for i, color in enumerate(colors[:num_images]):
        img = Image.new('RGB', (224, 224), color=color)
        images.append(img)
    
    logger.info(f"Created {num_images} test images with different colors")
    return images


def main():
    logger.info("=" * 80)
    logger.info("DETAILED CLIP DEBUGGING")
    logger.info("=" * 80)
    
    device = "cuda:0"
    
    # Load model
    logger.info("\nLoading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    
    # Create test images
    images = create_test_images(3)
    query = "a red object"
    
    logger.info(f"\nQuery: '{query}'")
    logger.info("Images: Red, Green, Blue")
    
    # Test 1: Raw logits
    logger.info("\n" + "-" * 80)
    logger.info("TEST 1: Raw logits from model")
    logger.info("-" * 80)
    
    with torch.no_grad():
        inputs = processor(
            text=[query] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        logger.info(f"Input keys: {inputs.keys()}")
        logger.info(f"pixel_values shape: {inputs['pixel_values'].shape}")
        logger.info(f"input_ids shape: {inputs['input_ids'].shape}")
        
        outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        logger.info(f"\nlogits_per_image shape: {logits_per_image.shape}")
        logger.info(f"logits_per_image:\n{logits_per_image}")
        
        # Test 2: After sigmoid
        logger.info("\n" + "-" * 80)
        logger.info("TEST 2: After sigmoid normalization")
        logger.info("-" * 80)
        
        sigmoid_scores = torch.sigmoid(logits_per_image)
        logger.info(f"sigmoid_scores:\n{sigmoid_scores}")
        
        # Test 3: Squeeze operation
        logger.info("\n" + "-" * 80)
        logger.info("TEST 3: Squeeze operation")
        logger.info("-" * 80)
        
        squeezed = logits_per_image.squeeze(-1)
        logger.info(f"After squeeze(-1): shape={squeezed.shape}")
        logger.info(f"Values:\n{squeezed}")
        
        # Test 4: Check if all values are same
        logger.info("\n" + "-" * 80)
        logger.info("TEST 4: Value analysis")
        logger.info("-" * 80)
        
        values = sigmoid_scores.cpu().numpy().flatten()
        logger.info(f"Unique values: {np.unique(values)}")
        logger.info(f"Min: {values.min():.6f}, Max: {values.max():.6f}")
        logger.info(f"Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        
        if np.allclose(values, values[0]):
            logger.error("❌ All values are identical!")
        else:
            logger.info("✓ Values are different")
        
        # Test 5: Check model embeddings
        logger.info("\n" + "-" * 80)
        logger.info("TEST 5: Model embeddings")
        logger.info("-" * 80)
        
        # Get image embeddings
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
        logger.info(f"Image features shape: {image_features.shape}")
        logger.info(f"Image features (first 5 dims):\n{image_features[:, :5]}")
        
        # Get text embeddings
        text_features = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        logger.info(f"Text features shape: {text_features.shape}")
        logger.info(f"Text features (first 5 dims):\n{text_features[:, :5]}")
        
        # Check if image embeddings are different
        logger.info("\nImage embeddings similarity:")
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                sim = torch.cosine_similarity(
                    image_features[i:i+1], 
                    image_features[j:j+1]
                ).item()
                logger.info(f"  Image {i} vs {j}: {sim:.4f}")
        
        # Test 6: Manual similarity computation
        logger.info("\n" + "-" * 80)
        logger.info("TEST 6: Manual similarity computation")
        logger.info("-" * 80)
        
        # Normalize embeddings
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logger.info(f"Normalized image features norm: {image_features_norm.norm(dim=-1)}")
        logger.info(f"Normalized text features norm: {text_features_norm.norm(dim=-1)}")
        
        # Compute similarity
        logit_scale = model.logit_scale.exp()
        logger.info(f"Logit scale: {logit_scale.item():.4f}")
        
        similarity = image_features_norm @ text_features_norm.t() * logit_scale
        logger.info(f"Manual similarity computation:\n{similarity}")
        
        # Test 7: Check if model is in eval mode
        logger.info("\n" + "-" * 80)
        logger.info("TEST 7: Model state")
        logger.info("-" * 80)
        
        logger.info(f"Model training: {model.training}")
        logger.info(f"Model device: {next(model.parameters()).device}")
        
        # Check gradients
        logger.info(f"Requires grad: {next(model.parameters()).requires_grad}")


if __name__ == "__main__":
    main()
