"""
Unit tests for checking model architectures, datasets, and inference pipeline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from models import ImageFeatureExtractor, VideoFeatureExtractor, FusionModel
from datasets import FallImageDataset, FallVideoDataset, FallFusionDataset

class TestFallPreventionSystem(unittest.TestCase):
    
    def test_image_model(self):
        model = ImageFeatureExtractor(feature_dim=512, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(0 <= out[0].item() <= 1)

    def test_video_model(self):
        model = VideoFeatureExtractor(feature_dim=512, num_frames=16, pretrained=False)
        x = torch.randn(2, 16, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))

    def test_fusion_model(self):
        model = FusionModel(image_feature_dim=512, video_feature_dim=512, pretrained=False)
        img = torch.randn(2, 3, 224, 224)
        vid = torch.randn(2, 16, 3, 224, 224)
        out = model(img, vid)
        self.assertEqual(out.shape, (2, 1))

    def test_datasets(self):
        # Requires dummy data from setup_data.py
        if os.path.exists('data/splits/train/images'):
            ds = FallImageDataset('data/splits/train/images', split='train', img_size=224)
            self.assertGreater(len(ds), 0)
            img, label = ds[0]
            self.assertEqual(img.shape, (3, 224, 224))

if __name__ == '__main__':
    unittest.main()
