"""
🛡️ AI Fall & Slip Prevention System — Streamlit Dashboard
Elderly + Child Safety | Real-time Risk Assessment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import cv2


import numpy as np
import tempfile
from PIL import Image
from torchvision import transforms

from models.fusion_model import FusionModel
from models.image_model import ImageFeatureExtractor
from models.video_model import VideoFeatureExtractor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AI Fall & Slip Prevention System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .risk-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    .risk-safe {
        background: linear-gradient(135deg, #0d9488, #14b8a6);
        color: white;
    }
    .risk-warning {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
    }
    .risk-danger {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: white;
    }
    .risk-card h2 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
    }
    .risk-card h3 {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .risk-card p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .metric-card h4 {
        font-size: 0.85rem;
        font-weight: 500;
        color: #a5b4fc;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .arch-box {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 1.5rem;
        border-radius: 14px;
        color: #e2e8f0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        margin: 1rem 0;
        border-left: 4px solid #6366f1;
    }

    .info-badge {
        display: inline-block;
        background: #312e81;
        color: #a5b4fc;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def load_models():
    """Load all three models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_model = ImageFeatureExtractor(feature_dim=512, pretrained=True, freeze_backbone=False)
    video_model = VideoFeatureExtractor(feature_dim=512, num_frames=16, pretrained=True, freeze_backbone=False)
    fusion_model = FusionModel(image_feature_dim=512, video_feature_dim=512,
                                pretrained=True, freeze_backbones=False, num_frames=16)

    # Load checkpoints if available
    for model, path, name in [
        (image_model, 'checkpoints/image_model.pth', 'Image'),
        (video_model, 'checkpoints/video_model.pth', 'Video'),
        (fusion_model, 'checkpoints/fusion_model.pth', 'Fusion'),
    ]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)

    image_model.to(device).eval()
    video_model.to(device).eval()
    fusion_model.to(device).eval()

    return image_model, video_model, fusion_model, device


def get_transforms():
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    frame_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return img_transform, frame_transform


def extract_video_frames(video_path, num_frames=16):
    """Extract frames uniformly from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    if total <= 0:
        cap.release()
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames, {'fps': 0, 'frames': 0, 'duration': 0}

    indices = np.linspace(0, total - 1, num_frames, dtype=int) \
        if total >= num_frames else \
        np.pad(np.arange(total), (0, num_frames - total), mode='wrap')

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret
                      else np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames, {'fps': round(fps, 1), 'frames': total, 'duration': round(duration, 1)}


def get_risk_assessment(score):
    """Convert score to risk assessment."""
    if score < 0.3:
        return {
            'status': '✅ SAFE', 'level': 'safe', 'css_class': 'risk-safe',
            'message': 'Environment is safe. No immediate fall/slip risk detected.',
            'detail': 'Floor conditions appear stable and motion patterns are normal.'
        }
    elif score < 0.6:
        return {
            'status': '⚠️ WARNING', 'level': 'warning', 'css_class': 'risk-warning',
            'message': 'Moderate risk detected. Possible hazards observed.',
            'detail': 'Potential wet surface or slightly unstable movement detected. Exercise caution.'
        }
    else:
        return {
            'status': '🚨 HIGH RISK', 'level': 'danger', 'css_class': 'risk-danger',
            'message': 'High fall/slip risk! Floor slippery and/or motion unstable.',
            'detail': 'Immediate attention required. Clear hazards and assist the person.'
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI Fall & Slip Prevention System</h1>
        <p>Elderly + Child Safety • Deep Learning Powered • Real-time Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    image_model, video_model, fusion_model, device = load_models()
    img_transform, frame_transform = get_transforms()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        mode = st.selectbox("Prediction Mode", ["🔀 Fusion (Image + Video)", "📸 Image Only", "🎬 Video Only"])

        st.markdown("---")
        st.markdown("### 📐 Architecture")
        st.markdown("""
        <div class="arch-box">
        Image → EfficientNet → 512-d<br>
        Video → ResNet → LSTM → 512-d<br>
        Concat → FC Layers → σ → Risk
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Risk Thresholds")
        st.markdown("""
        <span class="info-badge">🟢 Safe: 0.0 — 0.3</span><br>
        <span class="info-badge">🟡 Warning: 0.3 — 0.6</span><br>
        <span class="info-badge">🔴 High Risk: 0.6 — 1.0</span>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
        with col2:
            st.metric("Frames", "16")

    # ── Main Content ──
    tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📋 About", "🏗️ Architecture"])

    # ── TAB 1: PREDICTION ──
    with tab1:
        is_fusion = "Fusion" in mode
        is_image_only = "Image" in mode and not is_fusion
        is_video_only = "Video" in mode

        col_img, col_vid = st.columns(2)

        uploaded_image = None
        uploaded_video = None

        with col_img:
            if is_fusion or is_image_only:
                st.markdown("#### 📸 Upload Environment Image")
                uploaded_image = st.file_uploader(
                    "Floor / room image (JPG, PNG)", type=['jpg', 'jpeg', 'png'],
                    key='img_upload'
                )
                if uploaded_image:
                    img = Image.open(uploaded_image).convert('RGB')
                    st.image(img, caption="Uploaded Image", use_container_width=True)

        with col_vid:
            if is_fusion or is_video_only:
                st.markdown("#### 🎬 Upload Activity Video")
                uploaded_video = st.file_uploader(
                    "Activity video clip (MP4, AVI)", type=['mp4', 'avi', 'mov', 'mkv'],
                    key='vid_upload'
                )
                if uploaded_video:
                    st.video(uploaded_video)

        st.markdown("---")

        # ── Predict Button ──
        can_predict = False
        if is_fusion and uploaded_image and uploaded_video:
            can_predict = True
        elif is_image_only and uploaded_image:
            can_predict = True
        elif is_video_only and uploaded_video:
            can_predict = True

        if can_predict:
            if st.button("🚀 Analyze Risk", type="primary", use_container_width=True):
                with st.spinner("🔍 Running deep learning analysis..."):

                    risk_score = 0.0
                    img_score = None
                    vid_score = None

                    # ── Image Processing ──
                    if uploaded_image and (is_fusion or is_image_only):
                        img = Image.open(uploaded_image).convert('RGB')
                        img_tensor = img_transform(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            img_score = image_model(img_tensor).item()

                    # ── Video Processing ──
                    if uploaded_video and (is_fusion or is_video_only):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(uploaded_video.read())
                            tmp_path = tmp.name

                        frames, vid_info = extract_video_frames(tmp_path, num_frames=16)
                        vid_tensor = torch.stack([frame_transform(f) for f in frames], dim=0)
                        vid_tensor = vid_tensor.unsqueeze(0).to(device)

                        with torch.no_grad():
                            vid_score = video_model(vid_tensor).item()

                        os.unlink(tmp_path)

                    # ── Fusion or Individual Score ──
                    if is_fusion and img_score is not None and vid_score is not None:
                        # Re-run through fusion model
                        uploaded_image.seek(0)
                        img = Image.open(uploaded_image).convert('RGB')
                        img_tensor = img_transform(img).unsqueeze(0).to(device)

                        uploaded_video.seek(0)
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(uploaded_video.read())
                            tmp_path = tmp.name
                        frames, _ = extract_video_frames(tmp_path, num_frames=16)
                        vid_tensor = torch.stack([frame_transform(f) for f in frames], dim=0)
                        vid_tensor = vid_tensor.unsqueeze(0).to(device)

                        with torch.no_grad():
                            risk_score = fusion_model(img_tensor, vid_tensor).item()
                        os.unlink(tmp_path)

                    elif img_score is not None:
                        risk_score = img_score
                    elif vid_score is not None:
                        risk_score = vid_score

                    # ── Display Results ──
                    assessment = get_risk_assessment(risk_score)

                    st.markdown(f"""
                    <div class="risk-card {assessment['css_class']}">
                        <h2>{risk_score:.2f}</h2>
                        <h3>{assessment['status']}</h3>
                        <p>{assessment['message']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed Explanation
                    st.markdown(f"**📝 Explanation:** {assessment['detail']}")

                    # ── Sub-scores ──
                    if is_fusion:
                        st.markdown("#### 📊 Component Scores")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Environment Risk</h4>
                                <h2>{img_score:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with mc2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Motion Risk</h4>
                                <h2>{vid_score:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        with mc3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Fused Risk</h4>
                                <h2>{risk_score:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Sample Frame Grid ──
                    if uploaded_video and 'frames' in dir():
                        st.markdown("#### 🎞️ Sampled Frames (16 uniform samples)")
                        frame_cols = st.columns(8)
                        for i, frame in enumerate(frames[:8]):
                            with frame_cols[i]:
                                st.image(frame, caption=f"F{i+1}", use_container_width=True)
                        frame_cols2 = st.columns(8)
                        for i, frame in enumerate(frames[8:16]):
                            with frame_cols2[i]:
                                st.image(frame, caption=f"F{i+9}", use_container_width=True)

                    # Output example format
                    st.markdown("---")
                    if risk_score >= 0.6:
                        env_msg = "Floor slippery" if (img_score or 0) >= 0.5 else "Floor conditions uncertain"
                        motion_msg = "motion unstable" if (vid_score or 0) >= 0.5 else "motion slightly irregular"
                        st.error(f'**Risk Score: {risk_score:.2f} → HIGH RISK.** {env_msg} and {motion_msg}.')
                    elif risk_score >= 0.3:
                        st.warning(f'**Risk Score: {risk_score:.2f} → WARNING.** Moderate hazard detected.')
                    else:
                        st.success(f'**Risk Score: {risk_score:.2f} → SAFE.** No significant risk detected.')

        elif not can_predict:
            if is_fusion:
                st.info("📤 Please upload both an **image** and a **video** for fusion analysis.")
            elif is_image_only:
                st.info("📤 Please upload an **image** for environment risk analysis.")
            else:
                st.info("📤 Please upload a **video** for motion risk analysis.")

    # ── TAB 2: ABOUT ──
    with tab2:
        st.markdown("### 🎯 System Overview")
        st.markdown("""
        The **AI Fall & Slip Prevention System** is a deep learning solution designed to
        **predict fall/slip risk BEFORE it happens**, protecting elderly and children.

        #### Key Features
        - 🧠 **Pure Deep Learning** — No NLP, no APIs, no external AI services
        - 📸 **Environment Analysis** — Detects wet floors, obstacles, poor lighting
        - 🎬 **Motion Analysis** — Tracks body movement patterns indicating instability
        - 🔀 **Multi-modal Fusion** — Combines both signals for accurate prediction
        - ⚡ **Real-time** — Optimized for fast inference

        #### Risk Categories
        | Score Range | Status | Action |
        |------------|--------|--------|
        | 0.0 — 0.3 | ✅ Safe | No action needed |
        | 0.3 — 0.6 | ⚠️ Warning | Exercise caution |
        | 0.6 — 1.0 | 🚨 High Risk | Immediate attention |
        """)

    # ── TAB 3: ARCHITECTURE ──
    with tab3:
        st.markdown("### 🏗️ Model Architecture")

        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                    FUSION MODEL                         │
        │                                                         │
        │  ┌──────────────┐        ┌──────────────────────────┐  │
        │  │  IMAGE PATH  │        │      VIDEO PATH          │  │
        │  │              │        │                          │  │
        │  │  224×224 RGB │        │  16 Frames × 224×224     │  │
        │  │      ↓       │        │          ↓               │  │
        │  │ EfficientNet │        │  ResNet-18 (per frame)   │  │
        │  │   (B0)       │        │          ↓               │  │
        │  │      ↓       │        │   Bi-LSTM (2 layers)    │  │
        │  │  512-d feat  │        │          ↓               │  │
        │  │              │        │      512-d feat          │  │
        │  └──────┬───────┘        └──────────┬───────────────┘  │
        │         │                           │                   │
        │         └─────────┬─────────────────┘                   │
        │                   │                                     │
        │            Concatenate (1024-d)                         │
        │                   ↓                                     │
        │          FC 1024 → 512 → 256 → 128                     │
        │                   ↓                                     │
        │              Sigmoid → Risk Score (0..1)                │
        └─────────────────────────────────────────────────────────┘
        ```
        """)

        st.markdown("#### 🔧 Technical Details")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("""
            **Image Model**
            - Backbone: EfficientNet-B0
            - Pretrained: ImageNet
            - Output: 512-d features
            - Purpose: Environmental risk
            """)
        with mc2:
            st.markdown("""
            **Video Model**
            - Frame Encoder: ResNet-18
            - Temporal: Bi-LSTM (2 layers)
            - Input: 16 frames/clip
            - Purpose: Motion analysis
            """)
        with mc3:
            st.markdown("""
            **Fusion Model**
            - Strategy: Late concatenation
            - FC Layers: 1024→512→256→128→1
            - Loss: BCE
            - Optimizer: Adam
            """)


if __name__ == '__main__':
    main()
