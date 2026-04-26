"""
Fusion Model — Late Fusion of Image + Video Features → Risk Score.

Architecture:
    Image Features (512-d) + Video Features (512-d) → Concatenate (1024-d)
    → Fully Connected Layers → Sigmoid → Risk Score (0 to 1)
"""

import torch
import torch.nn as nn
from .image_model import ImageFeatureExtractor
from .video_model import VideoFeatureExtractor


class FusionModel(nn.Module):
    """
    Late-fusion model combining environmental (image) and motion (video) features.

    Pipeline:
        1. Image → EfficientNet → 512-d features
        2. Video → ResNet+LSTM → 512-d features
        3. Concatenate → 1024-d
        4. FC layers → Risk Score (0..1)

    Args:
        image_feature_dim (int): Image feature vector size.
        video_feature_dim (int): Video feature vector size.
        pretrained (bool): Use pretrained backbones.
        freeze_backbones (bool): Freeze both backbones initially.
        num_frames (int): Number of frames per video clip.
    """

    def __init__(self, image_feature_dim=512, video_feature_dim=512,
                 pretrained=True, freeze_backbones=True, num_frames=16):
        super().__init__()

        # ── Sub-models ──
        self.image_model = ImageFeatureExtractor(
            feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbones
        )
        self.video_model = VideoFeatureExtractor(
            feature_dim=video_feature_dim,
            num_frames=num_frames,
            pretrained=pretrained,
            freeze_backbone=freeze_backbones
        )

        fusion_dim = image_feature_dim + video_feature_dim  # 1024

        # ── Fusion Fully Connected Network ──
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, video_frames):
        """
        Full forward pass.
        Args:
            image: (B, 3, 224, 224) — environment/floor image
            video_frames: (B, T, 3, 224, 224) — video clip frames
        Returns:
            risk_score: (B, 1) — fall/slip risk probability
        """
        img_features = self.image_model.extract_features(image)
        vid_features = self.video_model.extract_features(video_frames)

        fused = torch.cat([img_features, vid_features], dim=1)
        risk_score = self.fusion_fc(fused)
        return risk_score

    def get_risk_assessment(self, risk_score):
        """
        Convert raw risk score to human-readable assessment.
        Args:
            risk_score (float): Risk probability 0..1
        Returns:
            dict with status, message, and color
        """
        score = risk_score if isinstance(risk_score, float) else risk_score.item()

        if score < 0.3:
            return {
                'score': score,
                'status': '✅ SAFE',
                'message': 'Environment is safe. No immediate fall/slip risk detected.',
                'level': 'low',
                'color': 'green'
            }
        elif score < 0.6:
            return {
                'score': score,
                'status': '⚠️ WARNING',
                'message': 'Moderate risk detected. Possible hazards or unstable motion observed.',
                'level': 'medium',
                'color': 'orange'
            }
        else:
            return {
                'score': score,
                'status': '🚨 HIGH RISK',
                'message': 'High fall/slip risk! Floor may be slippery and/or motion is unstable.',
                'level': 'high',
                'color': 'red'
            }

    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_backbones(self, image_layers=3, video_layers=2):
        """Selectively unfreeze backbone layers."""
        self.image_model.unfreeze_backbone(image_layers)
        self.video_model.unfreeze_backbone(video_layers)

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    model = FusionModel(pretrained=False)
    img = torch.randn(4, 3, 224, 224)
    vid = torch.randn(4, 16, 3, 224, 224)
    risk = model(img, vid)
    print(f"Risk score shape: {risk.shape}")  # (4, 1)
    print(f"Sample risk: {risk[0].item():.4f}")
    print(f"Assessment: {model.get_risk_assessment(risk[0])}")
    print(f"Params: {model.get_num_params()}")
