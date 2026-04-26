"""
Video Inference — Predict motion risk from a video clip.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from torchvision import transforms
from models.video_model import VideoFeatureExtractor


class VideoPredictor:
    """Predict fall/slip risk score from a video clip."""

    def __init__(self, checkpoint_path='checkpoints/video_model.pth',
                 num_frames=16, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_frames = num_frames
        self.model = VideoFeatureExtractor(feature_dim=512, num_frames=num_frames,
                                           pretrained=False, freeze_backbone=False)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"✅ Video model loaded from {checkpoint_path}")
        else:
            print(f"⚠️ No checkpoint found at {checkpoint_path}, using random weights")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _extract_frames(self, video_path):
        """Extract num_frames frames uniformly from a video."""
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
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        cap.release()
        return frames

    @torch.no_grad()
    def predict(self, video_path):
        """Predict risk score from a video file."""
        frames = self._extract_frames(video_path)
        tensor = torch.stack([self.transform(f) for f in frames], dim=0)  # (T, C, H, W)
        tensor = tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        risk_score = self.model(tensor).item()
        return risk_score

    @torch.no_grad()
    def extract_features(self, video_path):
        """Extract temporal feature vector from a video."""
        frames = self._extract_frames(video_path)
        tensor = torch.stack([self.transform(f) for f in frames], dim=0)
        tensor = tensor.unsqueeze(0).to(self.device)
        features = self.model.extract_features(tensor)
        return features


if __name__ == '__main__':
    predictor = VideoPredictor()
    print("Video predictor ready.")
