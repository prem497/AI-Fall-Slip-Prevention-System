"""
Inference Pipeline — Unified prediction interface for the Fall & Slip Prevention System.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict_image import ImagePredictor
from inference.predict_video import VideoPredictor
from inference.predict_fusion import FusionPredictor


class FallSlipPipeline:
    """
    Unified inference pipeline supporting image-only, video-only, and fusion prediction.
    """

    def __init__(self, mode='fusion'):
        self.mode = mode
        if mode == 'image':
            self.predictor = ImagePredictor()
        elif mode == 'video':
            self.predictor = VideoPredictor()
        else:
            self.predictor = FusionPredictor()

    def predict(self, image_path=None, video_path=None):
        """
        Run prediction based on available inputs.

        Returns:
            dict with score, status, message, level, color
        """
        if self.mode == 'fusion' and image_path and video_path:
            return self.predictor.predict(image_path, video_path)

        elif self.mode == 'image' and image_path:
            score = self.predictor.predict(image_path)
            return self._format_result(score, 'environment')

        elif self.mode == 'video' and video_path:
            score = self.predictor.predict(video_path)
            return self._format_result(score, 'motion')

        else:
            raise ValueError(f"Invalid inputs for mode '{self.mode}'")

    def _format_result(self, score, source):
        if score < 0.3:
            return {
                'score': score,
                'status': '✅ SAFE',
                'message': f'{source.title()} analysis: No significant risk detected.',
                'level': 'low', 'color': 'green'
            }
        elif score < 0.6:
            return {
                'score': score,
                'status': '⚠️ WARNING',
                'message': f'{source.title()} analysis: Moderate risk. Exercise caution.',
                'level': 'medium', 'color': 'orange'
            }
        else:
            return {
                'score': score,
                'status': '🚨 HIGH RISK',
                'message': f'{source.title()} analysis: High risk detected! Immediate attention needed.',
                'level': 'high', 'color': 'red'
            }


if __name__ == '__main__':
    print("Pipeline ready. Use FallSlipPipeline(mode='fusion') for full prediction.")
