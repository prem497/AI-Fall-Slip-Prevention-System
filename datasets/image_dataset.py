"""
Image Dataset — Loads floor/environment images with risk labels.
Labels: 0.0 (safe) to 1.0 (hazardous) for BCE loss training.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FallImageDataset(Dataset):
    """
    Dataset for environment images labeled as safe (0) or hazardous (1).

    Directory structure:
        root_dir/
            safe/        → label 0.0
            hazardous/   → label 1.0
    """

    CLASS_MAP = {'safe': 0.0, 'no_fall': 0.0, 'hazardous': 1.0, 'fall': 1.0}

    def __init__(self, root_dir, split='train', img_size=224, augment=True):
        self.root_dir = root_dir
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
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), label))

        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.augment:
            return transforms.Compose([
                transforms.Resize((self.img_size + 32, self.img_size + 32)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15),
            ])
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

    def get_class_distribution(self):
        dist = {'safe': 0, 'hazardous': 0}
        for _, l in self.samples:
            dist['hazardous' if l >= 0.5 else 'safe'] += 1
        return dist
