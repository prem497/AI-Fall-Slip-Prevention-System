"""
Fusion Inference — Combined image + video risk prediction.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models.fusion_model import FusionModel


class FusionPredictor:
    """Predict combined fall/slip risk from image + video inputs."""

    def __init__(self, checkpoint_path='checkpoints/fusion_model.pth',
                 num_frames=16, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_frames = num_frames
        self.model = FusionModel(image_feature_dim=512, video_feature_dim=512,
                                  pretrained=False, freeze_backbones=False, num_frames=num_frames)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Fusion model loaded from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint found, using random weights (demo mode)")

        self.model.to(self.device)
        self.model.eval()

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.frame_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int) \
            if total >= self.num_frames else \
            np.pad(np.arange(total), (0, self.num_frames - total), mode='wrap')

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                          else np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

    @torch.no_grad()
    def predict(self, image_path, video_path):
        """Predict combined risk score from image + video file paths."""
        # Process image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)

        # Process video
        frames = self._extract_frames(video_path)
        vid_tensor = torch.stack([self.frame_transform(f) for f in frames], dim=0)
        vid_tensor = vid_tensor.unsqueeze(0).to(self.device)

        risk_score = self.model(img_tensor, vid_tensor).item()
        assessment = self.model.get_risk_assessment(risk_score)
        return assessment

    @torch.no_grad()
    def predict_from_data(self, pil_image, video_path):
        """Predict from PIL image + video path."""
        img = pil_image.convert('RGB')
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)

        frames = self._extract_frames(video_path)
        vid_tensor = torch.stack([self.frame_transform(f) for f in frames], dim=0)
        vid_tensor = vid_tensor.unsqueeze(0).to(self.device)

        risk_score = self.model(img_tensor, vid_tensor).item()
        assessment = self.model.get_risk_assessment(risk_score)
        return assessment


if __name__ == '__main__':
    predictor = FusionPredictor()
    print("Fusion predictor ready (demo mode).")
