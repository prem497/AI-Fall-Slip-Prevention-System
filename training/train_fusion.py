"""
Train Fusion Model — Combines pretrained image + video feature extractors.

Steps:
1. Load pretrained image and video model checkpoints
2. Remove classifiers, use as feature extractors
3. Train the fusion FC layers on concatenated features
4. Optionally fine-tune end-to-end
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from models.fusion_model import FusionModel
from datasets.fusion_dataset import FallFusionDataset
from training.trainer import Trainer


def load_pretrained_weights(fusion_model, image_ckpt=None, video_ckpt=None):
    """Load pretrained weights into the fusion model's sub-models."""
    if image_ckpt and os.path.exists(image_ckpt):
        print(f"  📥 Loading image weights: {image_ckpt}")
        ckpt = torch.load(image_ckpt, map_location='cpu')
        fusion_model.image_model.load_state_dict(ckpt['model_state_dict'], strict=False)

    if video_ckpt and os.path.exists(video_ckpt):
        print(f"  📥 Loading video weights: {video_ckpt}")
        ckpt = torch.load(video_ckpt, map_location='cpu')
        fusion_model.video_model.load_state_dict(ckpt['model_state_dict'], strict=False)

    return fusion_model


def train_fusion_model():
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 30,
        'batch_size': 8,
        'patience': 8,
    }

    print("\n🔀 Loading Fusion Datasets...")
    train_set = FallFusionDataset('data/splits/train', split='train', img_size=224, num_frames=16)
    val_set = FallFusionDataset('data/splits/val', split='val', img_size=224, num_frames=16)

    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    print("\n🏗️  Building Fusion Model...")
    model = FusionModel(image_feature_dim=512, video_feature_dim=512,
                        pretrained=True, freeze_backbones=True, num_frames=16)

    # Load pretrained sub-model weights
    model = load_pretrained_weights(
        model,
        image_ckpt='checkpoints/image_model.pth',
        video_ckpt='checkpoints/video_model.pth'
    )

    params = model.get_num_params()
    print(f"  Total: {params['total']:,} | Trainable: {params['trainable']:,}")

    trainer = Trainer(model, train_loader, val_loader, config, model_type='fusion')
    history = trainer.fit()

    # ── Optional: End-to-end fine-tuning ──
    print("\n🔓 Unfreezing backbones for end-to-end fine-tuning...")
    model.unfreeze_backbones(image_layers=2, video_layers=2)
    config['lr'] = 1e-5
    config['epochs'] = 10
    config['patience'] = 5

    params = model.get_num_params()
    print(f"  Trainable after unfreeze: {params['trainable']:,}")

    trainer2 = Trainer(model, train_loader, val_loader, config, model_type='fusion')
    trainer2.best_val_loss = trainer.best_val_loss
    history2 = trainer2.fit()

    return history, history2


if __name__ == '__main__':
    train_fusion_model()
