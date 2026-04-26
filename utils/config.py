"""
Configuration for the Fall & Slip Prevention System.
"""

import os
import torch
import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    raw_image_dir: str = 'data/raw/images'
    raw_video_dir: str = 'data/raw/videos'
    processed_image_dir: str = 'data/processed/images'
    processed_video_dir: str = 'data/processed/videos'
    train_dir: str = 'data/splits/train'
    val_dir: str = 'data/splits/val'
    test_dir: str = 'data/splits/test'
    img_size: int = 224
    num_frames: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    num_classes: int = 1  # Binary risk score
    class_names: List[str] = field(default_factory=lambda: ['safe', 'fall'])
    pretrained: bool = True
    freeze_backbone: bool = True
    image_feature_dim: int = 512
    video_feature_dim: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 10
    loss: str = 'bce'  # Binary Cross-Entropy


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_dict(self):
        """Convert to flat dict for Trainer."""
        return {
            'device': self.device,
            'lr': self.train.lr,
            'weight_decay': self.train.weight_decay,
            'epochs': self.train.epochs,
            'batch_size': self.train.batch_size,
            'patience': self.train.patience,
        }

    def save(self, path):
        import dataclasses
        with open(path, 'w') as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            d = json.load(f)
        c = cls()
        c.data = DataConfig(**d.get('data', {}))
        c.model = ModelConfig(**d.get('model', {}))
        c.train = TrainConfig(**d.get('train', {}))
        c.seed = d.get('seed', 42)
        c.device = d.get('device', 'cpu')
        return c


DEFAULT_CONFIG = Config()
