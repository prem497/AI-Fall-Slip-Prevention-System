"""
Train Video Model — ResNet + LSTM for temporal motion analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from models.video_model import VideoFeatureExtractor
from datasets.video_dataset import FallVideoDataset
from training.trainer import Trainer


def train_video_model():
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 50,
        'batch_size': 8,  # Smaller batch for video
        'patience': 10,
    }

    print("\n🎬 Loading Video Datasets...")
    train_set = FallVideoDataset('data/splits/train', split='train', num_frames=16, img_size=224)
    val_set = FallVideoDataset('data/splits/val', split='val', num_frames=16, img_size=224)

    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")
    print(f"  Distribution: {train_set.get_class_distribution()}")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    print("\n🏗️  Building ResNet + LSTM Video Model...")
    model = VideoFeatureExtractor(feature_dim=512, num_frames=16, pretrained=True, freeze_backbone=True)
    params = model.get_num_params()
    print(f"  Total: {params['total']:,} | Trainable: {params['trainable']:,}")

    trainer = Trainer(model, train_loader, val_loader, config, model_type='video')
    return trainer.fit()


if __name__ == '__main__':
    train_video_model()
