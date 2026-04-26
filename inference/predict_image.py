"""
Image Inference — Predict environmental risk from a single image.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from models.image_model import ImageFeatureExtractor


class ImagePredictor:
    """Predict environmental risk score from a floor/environment image."""

    def __init__(self, checkpoint_path='checkpoints/image_model.pth', device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = ImageFeatureExtractor(feature_dim=512, pretrained=False, freeze_backbone=False)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Image model loaded from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint found at {checkpoint_path}, using random weights")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path):
        """Predict risk score from an image file path."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        risk_score = self.model(tensor).item()
        return risk_score

    @torch.no_grad()
    def predict_from_pil(self, pil_image):
        """Predict risk score from a PIL Image object."""
        img = pil_image.convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        risk_score = self.model(tensor).item()
        return risk_score

    @torch.no_grad()
    def extract_features(self, image_path):
        """Extract feature vector from an image."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        features = self.model.extract_features(tensor)
        return features


if __name__ == '__main__':
    predictor = ImagePredictor()
    print("Image predictor ready.")
