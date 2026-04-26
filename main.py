"""
🛡️ AI Fall & Slip Prevention System (Elderly + Child Safety)
Main entry point — Training pipeline and CLI interface.

Usage:
    python main.py --mode train_image
    python main.py --mode train_video
    python main.py --mode train_fusion
    python main.py --mode predict --image path/to/img.jpg --video path/to/vid.mp4
    python main.py --mode app
"""

import argparse
import sys
import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_banner():
    banner = """
    AI Fall & Slip Prevention System
    Elderly + Child Safety | Deep Learning Powered

    Architecture:
    - Image  -> EfficientNet-B0 -> 512-d features
    - Video  -> ResNet-18 + Bi-LSTM -> 512-d features
    - Fusion -> Concatenate -> FC -> Sigmoid -> Risk Score
    """
    print(banner)


def train_image():
    from training.train_image import train_image_model
    return train_image_model()


def train_video():
    from training.train_video import train_video_model
    return train_video_model()


def train_fusion():
    from training.train_fusion import train_fusion_model
    return train_fusion_model()


def train_all():
    """Full training pipeline: Image → Video → Fusion."""
    print("\n" + "=" * 60)
    print("  PHASE 1: Training Image Model (EfficientNet)")
    print("=" * 60)
    train_image()

    print("\n" + "=" * 60)
    print("  PHASE 2: Training Video Model (ResNet + LSTM)")
    print("=" * 60)
    train_video()

    print("\n" + "=" * 60)
    print("  PHASE 3: Training Fusion Model")
    print("=" * 60)
    train_fusion()

    print("\n✅ Full training pipeline complete!")


def predict(image_path, video_path):
    """Run inference with the fusion model."""
    from inference.predict_fusion import FusionPredictor

    predictor = FusionPredictor()

    if image_path and video_path:
        result = predictor.predict(image_path, video_path)
    else:
        print("❌ Both --image and --video are required for fusion prediction.")
        return

    print(f"\n{'=' * 50}")
    print(f"  🎯 Risk Assessment")
    print(f"{'=' * 50}")
    print(f"  Risk Score: {result['score']:.2f}")
    print(f"  Status:     {result['status']}")
    print(f"  Message:    {result['message']}")
    print(f"{'=' * 50}")

    # Output in requested format
    if result['score'] >= 0.6:
        print(f'\n  Risk Score: {result["score"]:.2f} → HIGH RISK. Floor slippery and motion unstable.')
    elif result['score'] >= 0.3:
        print(f'\n  Risk Score: {result["score"]:.2f} → WARNING. Moderate hazard detected.')
    else:
        print(f'\n  Risk Score: {result["score"]:.2f} → SAFE. No significant risk.')


def run_app():
    """Launch the Streamlit dashboard."""
    print("\nLaunching Streamlit App...")
    os.system("python -m streamlit run app/streamlit_app.py")


def show_model_info():
    """Display model architecture info."""
    from models import ImageFeatureExtractor, VideoFeatureExtractor, FusionModel

    print("\n📊 Model Architecture Summary\n")

    img_model = ImageFeatureExtractor(feature_dim=512, pretrained=False)
    vid_model = VideoFeatureExtractor(feature_dim=512, pretrained=False)
    fusion_model = FusionModel(pretrained=False)

    for name, model in [("Image (EfficientNet)", img_model),
                         ("Video (ResNet+LSTM)", vid_model),
                         ("Fusion", fusion_model)]:
        params = model.get_num_params()
        print(f"  {name}:")
        print(f"    Total params:     {params['total']:>12,}")
        print(f"    Trainable params: {params['trainable']:>12,}")
        print()

    # Quick forward pass test
    print("  🧪 Forward pass test:")
    img_input = torch.randn(1, 3, 224, 224)
    vid_input = torch.randn(1, 16, 3, 224, 224)

    with torch.no_grad():
        img_out = img_model(img_input)
        vid_out = vid_model(vid_input)
        fusion_out = fusion_model(img_input, vid_input)

    print(f"    Image risk:  {img_out.item():.4f}")
    print(f"    Video risk:  {vid_out.item():.4f}")
    print(f"    Fusion risk: {fusion_out.item():.4f}")
    print("\n  ✅ All models working correctly!")


def main():
    print_banner()
    set_seed(42)

    parser = argparse.ArgumentParser(description="AI Fall & Slip Prevention System")
    parser.add_argument('--mode', type=str, default='info',
                        choices=['train_image', 'train_video', 'train_fusion',
                                 'train_all', 'predict', 'app', 'info'],
                        help='Operation mode')
    parser.add_argument('--image', type=str, default=None, help='Path to image for prediction')
    parser.add_argument('--video', type=str, default=None, help='Path to video for prediction')

    args = parser.parse_args()

    if args.mode == 'train_image':
        train_image()
    elif args.mode == 'train_video':
        train_video()
    elif args.mode == 'train_fusion':
        train_fusion()
    elif args.mode == 'train_all':
        train_all()
    elif args.mode == 'predict':
        predict(args.image, args.video)
    elif args.mode == 'app':
        run_app()
    elif args.mode == 'info':
        show_model_info()


if __name__ == '__main__':
    main()
