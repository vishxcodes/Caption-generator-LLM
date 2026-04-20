import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

def load_nsfw_model():
    print("Loading NSFW Image Classification Model...")
    try:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
        model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection").to(device)
        print(f"✅ NSFW Model loaded successfully on {device}.")
        return processor, model, device
    except Exception as e:
        print(f"❌ Failed to load NSFW model: {e}")
        return None, None, None

def detect_nsfw(image: Image.Image, processor, model, device) -> dict:
    if processor is None or model is None:
        raise RuntimeError("NSFW model components are not loaded.")

    # Prepare image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    
    # Extract labels and confidence
    id2label = model.config.id2label
    details = {id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    # Generally Falconsai/nsfw_image_detection maps:
    # 0 -> 'normal'
    # 1 -> 'nsfw'
    
    nsfw_score = details.get("nsfw", 0.0)
    # the user asked for a simple safe check, e.g. safe if < 0.5
    safe = bool(nsfw_score <= 0.5)
    
    return {
        "safe": safe,
        "nsfw_score": round(nsfw_score, 4),
        "details": {k: round(v, 4) for k, v in details.items()}
    }
