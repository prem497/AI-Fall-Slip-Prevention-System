"""
Utility to setup the dataset structure and generate dummy data for testing.
Ensures consistency between Image, Video, and Fusion training pipelines.
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch

def setup_all():
    # 1. Define paths
    splits = ['train', 'val', 'test']
    categories = {
        'images': ['safe', 'hazardous'],
        'videos': ['safe', 'fall']
    }
    
    base_path = 'data/splits'
    
    print("--- Initializing Directories ---")
    for split in splits:
        for dtype, cats in categories.items():
            for cat in cats:
                path = os.path.join(base_path, split, dtype, cat)
                os.makedirs(path, exist_ok=True)
    
    print("DONE: Directory structure created.")

    print("\n--- Generating Dummy Images ---")
    for split in ['train', 'val']:
        for cat in categories['images']:
            for i in range(5):
                path = f'data/splits/{split}/images/{cat}/dummy_{i}.jpg'
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(img).save(path)
    print("DONE: Dummy images generated.")

    print("\n--- Generating Dummy Videos ---")
    for split in ['train', 'val']:
        for cat in categories['videos']:
            for i in range(3):
                path = f'data/splits/{split}/videos/{cat}/dummy_{i}.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(path, fourcc, 10.0, (224, 224))
                for _ in range(16):
                    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    out.write(frame)
                out.release()
    print("DONE: Dummy videos generated.")

    print("\n--- Saving Dummy Checkpoints ---")
    from models import ImageFeatureExtractor, VideoFeatureExtractor, FusionModel
    os.makedirs('checkpoints', exist_ok=True)
    
    device = 'cpu'
    
    img_model = ImageFeatureExtractor(pretrained=False).to(device)
    torch.save({'model_state_dict': img_model.state_dict()}, 'checkpoints/image_model.pth')
    
    vid_model = VideoFeatureExtractor(pretrained=False).to(device)
    torch.save({'model_state_dict': vid_model.state_dict()}, 'checkpoints/video_model.pth')
    
    fusion_model = FusionModel(pretrained=False).to(device)
    torch.save({'model_state_dict': fusion_model.state_dict()}, 'checkpoints/fusion_model.pth')
    print("DONE: Dummy weights saved.")

if __name__ == "__main__":
    setup_all()
    print("\nSystem Ready. You can now run 'python main.py --mode app'")
