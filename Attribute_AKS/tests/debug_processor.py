"""
Debug CLIP processor behavior
"""
import logging
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images(num_images: int = 3) -> list:
    """Create test images with different colors"""
    images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for color in colors[:num_images]:
        img = Image.new('RGB', (224, 224), color=color)
        images.append(img)
    
    return images


def main():
    logger.info("=" * 80)
    logger.info("DEBUGGING CLIP PROCESSOR")
    logger.info("=" * 80)
    
    device = "cuda:0"
    model_id = "openai/clip-vit-base-patch32"
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    
    images = create_test_images(3)
    text = "a red object"
    
    # Test 1: Pass text as string
    logger.info("\n" + "-" * 80)
    logger.info("TEST 1: text as string")
    logger.info("-" * 80)
    
    inputs1 = processor(text=text, images=images, return_tensors="pt", padding=True)
    logger.info(f"Input keys: {inputs1.keys()}")
    logger.info(f"input_ids shape: {inputs1['input_ids'].shape}")
    logger.info(f"input_ids:\n{inputs1['input_ids']}")
    
    with torch.no_grad():
        outputs1 = model(**{k: v.to(device) for k, v in inputs1.items()})
        logits1 = outputs1.logits_per_image
        logger.info(f"logits_per_image shape: {logits1.shape}")
        logger.info(f"logits_per_image:\n{logits1}")
    
    # Test 2: Pass text as list with one element
    logger.info("\n" + "-" * 80)
    logger.info("TEST 2: text as list with one element")
    logger.info("-" * 80)
    
    inputs2 = processor(text=[text], images=images, return_tensors="pt", padding=True)
    logger.info(f"Input keys: {inputs2.keys()}")
    logger.info(f"input_ids shape: {inputs2['input_ids'].shape}")
    logger.info(f"input_ids:\n{inputs2['input_ids']}")
    
    with torch.no_grad():
        outputs2 = model(**{k: v.to(device) for k, v in inputs2.items()})
        logits2 = outputs2.logits_per_image
        logger.info(f"logits_per_image shape: {logits2.shape}")
        logger.info(f"logits_per_image:\n{logits2}")
    
    # Test 3: Pass text as list with repeated elements
    logger.info("\n" + "-" * 80)
    logger.info("TEST 3: text as list with repeated elements")
    logger.info("-" * 80)
    
    inputs3 = processor(text=[text] * len(images), images=images, return_tensors="pt", padding=True)
    logger.info(f"Input keys: {inputs3.keys()}")
    logger.info(f"input_ids shape: {inputs3['input_ids'].shape}")
    logger.info(f"input_ids:\n{inputs3['input_ids']}")
    
    with torch.no_grad():
        outputs3 = model(**{k: v.to(device) for k, v in inputs3.items()})
        logits3 = outputs3.logits_per_image
        logger.info(f"logits_per_image shape: {logits3.shape}")
        logger.info(f"logits_per_image:\n{logits3}")
    
    # Test 4: Separate text and image processing
    logger.info("\n" + "-" * 80)
    logger.info("TEST 4: Separate text and image processing")
    logger.info("-" * 80)
    
    text_inputs = processor(text=text, return_tensors="pt", padding=True)
    image_inputs = processor(images=images, return_tensors="pt")
    
    logger.info(f"Text input_ids shape: {text_inputs['input_ids'].shape}")
    logger.info(f"Image pixel_values shape: {image_inputs['pixel_values'].shape}")
    
    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=text_inputs['input_ids'].to(device),
            attention_mask=text_inputs['attention_mask'].to(device)
        )
        image_features = model.get_image_features(
            pixel_values=image_inputs['pixel_values'].to(device)
        )
        
        logger.info(f"Text features shape: {text_features.shape}")
        logger.info(f"Image features shape: {image_features.shape}")
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = model.logit_scale.exp()
        logits4 = image_features @ text_features.t() * logit_scale
        
        logger.info(f"logits shape: {logits4.shape}")
        logger.info(f"logits:\n{logits4}")


if __name__ == "__main__":
    main()
