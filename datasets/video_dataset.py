"""
Video Dataset — Loads video clips, extracts 16 frames, returns with risk labels.
Labels: 0.0 (safe) to 1.0 (fall risk) for BCE loss training.
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class FallVideoDataset(Dataset):
    """
    Dataset for video clips labeled as safe (0) or fall-risk (1).

    Directory structure:
        root_dir/
            safe/        → label 0.0
            fall/        → label 1.0
    """

    CLASS_MAP = {'safe': 0.0, 'no_fall': 0.0, 'fall': 1.0, 'hazardous': 1.0}
    VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')

    def __init__(self, root_dir, split='train', num_frames=16,
                 img_size=224, augment=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.samples = []

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.CLASS_MAP.get(class_name.lower(), None)
            if label is None:
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.VIDEO_EXT):
                    self.samples.append((os.path.join(class_dir, fname), label))

        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.augment:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.img_size + 16, self.img_size + 16)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _sample_frames(self, video_path):
        """Sample num_frames frames uniformly from a video."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            # Return black frames as fallback
            return [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                    for _ in range(self.num_frames)]

        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total)
            indices = np.pad(indices, (0, self.num_frames - total), mode='wrap')

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        cap.release()
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            frames: (T, C, H, W) — T=num_frames
            label: (1,) float tensor
        """
        path, label = self.samples[idx]
        raw_frames = self._sample_frames(path)

        frames = torch.stack([self.transform(f) for f in raw_frames], dim=0)  # (T, C, H, W)
        return frames, torch.tensor([label], dtype=torch.float32)

    def get_class_distribution(self):
        dist = {'safe': 0, 'fall': 0}
        for _, l in self.samples:
            dist['fall' if l >= 0.5 else 'safe'] += 1
        return dist
