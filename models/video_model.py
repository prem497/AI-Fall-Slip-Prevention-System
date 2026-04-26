"""
Video Model — ResNet + LSTM Temporal Motion Feature Extractor.

Architecture: Video → 16 Frames → ResNet-18 (per frame) → LSTM → Feature Vector
Purpose: Capture temporal motion patterns indicating pre-fall/slip behavior
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoFeatureExtractor(nn.Module):
    """
    ResNet-18 frame-level feature extractor + LSTM for temporal modeling.

    Pipeline:
        1. Extract 16 frames from video clip
        2. Each frame → ResNet-18 → 512-d feature
        3. Sequence of 16 features → LSTM → temporal feature vector

    Args:
        feature_dim (int): Output feature dimension.
        num_frames (int): Number of frames per clip.
        hidden_size (int): LSTM hidden state size.
        num_lstm_layers (int): Number of LSTM layers.
        pretrained (bool): Use ImageNet pretrained ResNet.
        freeze_backbone (bool): Freeze ResNet layers.
        bidirectional (bool): Use bidirectional LSTM.
    """

    def __init__(self, feature_dim=512, num_frames=16, hidden_size=256,
                 num_lstm_layers=2, pretrained=True, freeze_backbone=True,
                 bidirectional=True):
        super().__init__()

        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # ── Frame-level Feature Extractor (ResNet-18) ──
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.resnet_features = resnet.fc.in_features  # 512

        # Remove final FC layer — keep as feature extractor
        self.frame_encoder = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.frame_encoder.parameters():
                param.requires_grad = False

        # ── Temporal LSTM ──
        self.lstm = nn.LSTM(
            input_size=self.resnet_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3 if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # ── Feature Projection ──
        self.feature_head = nn.Sequential(
            nn.Linear(lstm_output_size, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Standalone classifier (used during video-only pretraining)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def encode_frames(self, frames):
        """
        Encode each frame independently through ResNet.
        Args:
            frames: (B, T, C, H, W) — batch of frame sequences
        Returns:
            frame_features: (B, T, 512)
        """
        B, T, C, H, W = frames.shape
        # Reshape to process all frames at once
        x = frames.reshape(B * T, C, H, W)       # (B*T, C, H, W)
        x = self.frame_encoder(x)                  # (B*T, 512, 1, 1)
        x = x.flatten(start_dim=1)                 # (B*T, 512)
        x = x.reshape(B, T, -1)                    # (B, T, 512)
        return x

    def extract_features(self, frames):
        """Extract temporal feature vector without classification.
        Args:
            frames: (B, T, C, H, W)
        Returns:
            features: (B, feature_dim)
        """
        frame_features = self.encode_frames(frames)  # (B, T, 512)
        lstm_out, (h_n, _) = self.lstm(frame_features)

        # Use the last hidden state (concatenated for bidirectional)
        if self.bidirectional:
            # h_n shape: (num_layers * 2, B, hidden_size)
            temporal_feature = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            temporal_feature = h_n[-1]

        features = self.feature_head(temporal_feature)
        return features

    def forward(self, frames):
        """Full forward pass.
        Args:
            frames: (B, T, C, H, W)
        Returns:
            risk_score: (B, 1)
        """
        features = self.extract_features(frames)
        risk_score = self.classifier(features)
        return risk_score

    def unfreeze_backbone(self, num_layers=2):
        """Unfreeze the last N layers of ResNet for fine-tuning."""
        layers = list(self.frame_encoder.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    model = VideoFeatureExtractor(feature_dim=512, pretrained=False)
    x = torch.randn(4, 16, 3, 224, 224)  # (B, T, C, H, W)
    features = model.extract_features(x)
    risk = model(x)
    print(f"Feature shape: {features.shape}")   # (4, 512)
    print(f"Risk shape:    {risk.shape}")         # (4, 1)
    print(f"Params: {model.get_num_params()}")
