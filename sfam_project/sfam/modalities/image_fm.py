import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

class ImageFM:
    """
    Image Feature Manager (Robust):
    Pads images to square before resizing to preserve aspect ratio.
    Crucial for geometry-dependent biometrics.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def process(self, image_path: str):
        # 1. Load & Standardize
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path.convert("RGB")

        # 2. Letterbox (Pad to Square)
        # Prevents "squishing" faces which distorts geometric features
        w, h = img.size
        max_dim = max(w, h)
        padding = (0, 0, max_dim - w, max_dim - h) # Left, Top, Right, Bottom
        img = ImageOps.expand(img, padding) 

        # 3. Tensor Pipeline
        img = img.resize(self.target_size, Image.Resampling.BICUBIC)
        tensor = F.to_tensor(img)
        tensor = F.normalize(tensor, self.mean, self.std)

        return tensor.unsqueeze(0)

# Default instance
processor = ImageFM()