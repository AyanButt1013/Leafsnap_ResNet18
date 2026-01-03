from PIL import Image
import io
import torchvision.transforms as transforms

def preprocess_image(image_bytes: bytes):
    # Convert bytes to a file-like object
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply the same transforms as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # add batch dimension