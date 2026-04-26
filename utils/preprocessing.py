"""
Preprocessing utilities for the Fall & Slip Prevention System.
Handles image and video preprocessing, data splitting, and augmentation helpers.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset directory into train/val/test splits.
    
    Args:
        source_dir (str): Source directory containing class subdirectories.
        output_dir (str): Output directory for split data.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        seed (int): Random seed for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    random.seed(seed)
    splits = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = sorted(os.listdir(class_path))
        random.shuffle(files)

        n = len(files)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        split_files = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        for split_name, split_list in split_files.items():
            dest_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in split_list:
                src = os.path.join(class_path, fname)
                dst = os.path.join(dest_dir, fname)
                shutil.copy2(src, dst)

        print(f"  [{class_name}] Train: {len(split_files['train'])}, "
              f"Val: {len(split_files['val'])}, Test: {len(split_files['test'])}")


def resize_image(image_path, output_path, size=(224, 224)):
    """
    Resize a single image and save it.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple): Target (width, height).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, Image.LANCZOS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def preprocess_images(input_dir, output_dir, size=(224, 224)):
    """
    Batch resize all images in a directory tree.
    
    Args:
        input_dir (str): Input directory with class subdirectories.
        output_dir (str): Output directory for processed images.
        size (tuple): Target (width, height).
    """
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src = os.path.join(root, fname)
                rel_path = os.path.relpath(src, input_dir)
                dst = os.path.join(output_dir, rel_path)
                resize_image(src, dst, size)

    print(f"Preprocessed images saved to: {output_dir}")


def normalize_frame(frame, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalize a frame (numpy array) with ImageNet statistics.
    
    Args:
        frame (np.ndarray): Input frame in RGB, values 0-255.
        mean (tuple): Channel means.
        std (tuple): Channel standard deviations.
        
    Returns:
        np.ndarray: Normalized frame as float32.
    """
    frame = frame.astype(np.float32) / 255.0
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c] - mean[c]) / std[c]
    return frame


def compute_dataset_statistics(image_dir):
    """
    Compute per-channel mean and std for a dataset directory.
    
    Args:
        image_dir (str): Directory containing images (may have subdirectories).
        
    Returns:
        tuple: (mean, std) as lists of 3 floats each.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = np.array(Image.open(os.path.join(root, fname)).convert('RGB'))
                img = img.astype(np.float64) / 255.0
                pixel_sum += img.reshape(-1, 3).sum(axis=0)
                pixel_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
                num_pixels += img.shape[0] * img.shape[1]

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def create_class_balanced_sampler(dataset):
    """
    Create a weighted random sampler for class-balanced training.
    
    Args:
        dataset: A dataset object with .samples attribute (list of (path, label) tuples).
        
    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter

    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)

    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
