"""
Fusion Dataset — Paired image + video samples for fusion model training.
"""

import os
import torch
from torch.utils.data import Dataset
from .image_dataset import FallImageDataset
from .video_dataset import FallVideoDataset


class FallFusionDataset(Dataset):
    """
    Provides paired (image, video, label) samples for the fusion model.

    Directory structure:
        root_dir/
            images/
                safe/ ...
                hazardous/ ...
            videos/
                safe/ ...
                fall/ ...
    """

    def __init__(self, root_dir, split='train', img_size=224,
                 num_frames=16, augment=True):
        image_dir = os.path.join(root_dir, 'images')
        video_dir = os.path.join(root_dir, 'videos')

        self.image_dataset = FallImageDataset(
            root_dir=image_dir, split=split, img_size=img_size, augment=augment)
        self.video_dataset = FallVideoDataset(
            root_dir=video_dir, split=split, num_frames=num_frames,
            img_size=img_size, augment=augment)

        self.length = min(len(self.image_dataset), len(self.video_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image, img_label = self.image_dataset[idx]
        video_frames, vid_label = self.video_dataset[idx]
        # Use max label (worst-case risk)
        label = torch.max(img_label, vid_label)
        return image, video_frames, label
