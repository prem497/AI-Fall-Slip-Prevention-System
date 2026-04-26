"""
Inference package for Fall & Slip Prevention System.
"""
from .predict_image import ImagePredictor
from .predict_video import VideoPredictor
from .predict_fusion import FusionPredictor
from .pipeline import FallSlipPipeline

__all__ = ['ImagePredictor', 'VideoPredictor', 'FusionPredictor', 'FallSlipPipeline']
