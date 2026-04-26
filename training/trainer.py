"""
Trainer — Generic training loop with BCE loss for risk score prediction.
Supports image, video, and fusion model training.
"""

import os
import time
import torch
import torch.nn as nn
from utils.metrics import AverageMeter, EarlyStopping


class Trainer:
    """Training loop with mixed precision, gradient clipping, and checkpointing."""

    def __init__(self, model, train_loader, val_loader, config, model_type='image'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # BCE Loss for 0-1 risk score
        self.criterion = nn.BCELoss()

        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 50), eta_min=1e-7
        )

        self.epochs = config.get('epochs', 50)
        self.early_stopping = EarlyStopping(patience=config.get('patience', 10))
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _forward(self, batch):
        """Route data through the correct model."""
        if self.model_type == 'fusion':
            images, videos, labels = batch
            images = images.to(self.device)
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images, videos)
        elif self.model_type == 'video':
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
        else:  # image
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
        return outputs, labels

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss')
        acc_meter = AverageMeter('Acc')

        for batch_idx, batch in enumerate(self.train_loader):
            outputs, labels = self._forward(batch)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accuracy: threshold at 0.5
            preds = (outputs >= 0.5).float()
            acc = (preds == labels).float().mean().item()

            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc, labels.size(0))

            if (batch_idx + 1) % 10 == 0:
                print(f"    Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {loss_meter.avg:.4f}  Acc: {acc_meter.avg:.4f}")

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        loss_meter = AverageMeter('Val Loss')
        correct, total = 0, 0

        for batch in self.val_loader:
            outputs, labels = self._forward(batch)
            loss = self.criterion(outputs, labels)
            loss_meter.update(loss.item(), labels.size(0))

            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        return loss_meter.avg, accuracy

    def save_checkpoint(self, epoch, val_loss, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)
        print(f"    💾 Checkpoint saved → {path}")

    def fit(self):
        ckpt_map = {
            'image': 'checkpoints/image_model.pth',
            'video': 'checkpoints/video_model.pth',
            'fusion': 'checkpoints/fusion_model.pth',
        }

        print(f"\n{'='*60}")
        print(f"  🚀 Training {self.model_type.upper()} Model")
        print(f"  Device: {self.device} | Epochs: {self.epochs}")
        print(f"  Loss: BCELoss | Optimizer: Adam")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            elapsed = time.time() - start

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"\n  Epoch {epoch}/{self.epochs} ({elapsed:.1f}s) LR={lr:.2e}")
            print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
            print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, ckpt_map[self.model_type])

            self.scheduler.step()

            if self.early_stopping(val_loss):
                print(f"\n  ⛔ Early stopping at epoch {epoch}")
                break

        print(f"\n{'='*60}")
        print(f"  ✅ Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        return self.history
