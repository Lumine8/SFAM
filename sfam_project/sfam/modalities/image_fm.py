import torch
import torchvision.transforms as T
from PIL import Image

class ImageFM:
    """
    Image Feature Manager:
    Converts raw images into the spatial tensor format required by SFAM.
    """
    def __init__(self, target_size=(224, 224)):
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, image_path: str):
        """
        Input: Path to an image (str) or PIL Image
        Output: Tensor of shape [1, 3, 224, 224]
        """
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            img = image_path.convert("RGB")
            
        # Transform and add batch dimension
        tensor = self.transform(img).unsqueeze(0)
        return tensor

# Create a default instance for easy import
processor = ImageFM()