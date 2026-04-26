"""
Video utilities for the Fall & Slip Prevention System.
Handles video I/O, frame extraction, and clip generation.
"""

import os
import cv2
import numpy as np


def extract_frames(video_path, output_dir=None, sample_rate=1, max_frames=None):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames, frame_idx, saved = [], 0, 0
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            if output_dir:
                cv2.imwrite(os.path.join(output_dir, f"frame_{saved:06d}.jpg"), frame)
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return saved if output_dir else frames


def get_video_info(video_path):
    """Get metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    cap.release()
    return info


def sample_uniform_frames(video_path, num_frames=16):
    """Sample frames uniformly across a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


def create_video_clips(video_path, clip_length=16, stride=8):
    """Split a video into overlapping clips."""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    clips = []
    for start in range(0, len(all_frames) - clip_length + 1, stride):
        clips.append(all_frames[start:start + clip_length])
    return clips


def compute_optical_flow(frames):
    """Compute dense optical flow between consecutive frames."""
    flows = []
    for i in range(len(frames) - 1):
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flows.append(mag)
    return flows


def compute_motion_energy(frames):
    """Compute motion energy between consecutive frames."""
    energies = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i].astype(float) - frames[i + 1].astype(float))
        energies.append(np.mean(diff))
    return energies
