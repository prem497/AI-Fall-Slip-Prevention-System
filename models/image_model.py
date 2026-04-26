"""
Image Model — EfficientNet-based Environment Risk Feature Extractor.

Architecture: Input Image → EfficientNet-B0 (pretrained) → Feature Vector (1280-d)
Purpose: Detect environmental hazards (wet floors, obstacles, poor lighting)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageFeatureExtractor(nn.Module):
    """
    EfficientNet-B0 backbone for extracting environmental risk features
    from floor/environment images.

    Args:
        feature_dim (int): Output feature dimension.
        pretrained (bool): Use ImageNet pretrained weights.
        freeze_backbone (bool): Freeze EfficientNet layers initially.
    """

    def __init__(self, feature_dim=512, pretrained=True, freeze_backbone=True):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)

        # EfficientNet-B0 final features = 1280
        self.backbone_features = 1280

        # Remove the classifier head — keep only feature layers
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Feature projection head
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Standalone classifier (used during image-only pretraining)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def extract_features(self, x):
        """Extract feature vector without classification.
        Args:
            x: (B, 3, 224, 224)
        Returns:
            features: (B, feature_dim)
        """
        backbone_out = self.backbone(x)       # (B, 1280, 1, 1)
        features = self.feature_head(backbone_out)  # (B, feature_dim)
        return features

    def forward(self, x):
        """Full forward pass with risk prediction.
        Args:
            x: (B, 3, 224, 224)
        Returns:
            risk_score: (B, 1) — probability 0..1
        """
        features = self.extract_features(x)
        risk_score = self.classifier(features)
        return risk_score

    def unfreeze_backbone(self, num_layers=3):
        """Unfreeze the last N layers of the backbone for fine-tuning."""
        children = list(self.backbone.children())
        # EfficientNet backbone is wrapped in Sequential
        if len(children) == 1 and isinstance(children[0], nn.Sequential):
            layers = list(children[0].children())
        else:
            layers = children

        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    model = ImageFeatureExtractor(feature_dim=512, pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    features = model.extract_features(x)
    risk = model(x)
    print(f"Feature shape: {features.shape}")   # (4, 512)
    print(f"Risk shape:    {risk.shape}")         # (4, 1)
    print(f"Params: {model.get_num_params()}")
