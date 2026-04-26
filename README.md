# 🛡️ AI Fall & Slip Prevention System (Elderly + Child Safety)

A **deep learning** system that predicts fall/slip risk **BEFORE** it happens, designed to protect elderly and children.

> **Pure Deep Learning** — No NLP, no APIs, no external AI services.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                    FUSION MODEL                        │
│                                                        │
│  Image (224×224)          Video (16 frames × 224×224)  │
│       ↓                          ↓                     │
│  EfficientNet-B0         ResNet-18 (per frame)         │
│       ↓                          ↓                     │
│  512-d features          Bi-LSTM (2 layers)            │
│       ↓                          ↓                     │
│       │                     512-d features             │
│       └──────── Concatenate ─────┘                     │
│                     ↓                                  │
│           FC: 1024 → 512 → 256 → 128 → 1              │
│                     ↓                                  │
│              Sigmoid → Risk Score (0..1)               │
└────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Fall & Slip Prevention System/
├── models/
│   ├── image_model.py       # EfficientNet-B0 feature extractor
│   ├── video_model.py       # ResNet-18 + Bi-LSTM temporal model
│   ├── fusion_model.py      # Late fusion → risk score
│   └── __init__.py
├── datasets/
│   ├── image_dataset.py     # Floor/environment image loader
│   ├── video_dataset.py     # Video clip loader (16 frames)
│   ├── fusion_dataset.py    # Paired image + video loader
│   └── __init__.py
├── utils/
│   ├── config.py            # Configuration dataclasses
│   ├── preprocessing.py     # Data splitting, resizing
│   ├── video_utils.py       # Frame extraction, optical flow
│   └── metrics.py           # Evaluation metrics, early stopping
├── training/
│   ├── train_image.py       # Train EfficientNet image model
│   ├── train_video.py       # Train ResNet+LSTM video model
│   ├── train_fusion.py      # Train fusion model (with fine-tuning)
│   └── trainer.py           # Generic training loop (BCE loss)
├── inference/
│   ├── predict_image.py     # Image-only prediction
│   ├── predict_video.py     # Video-only prediction
│   ├── predict_fusion.py    # Combined prediction
│   └── pipeline.py          # Unified inference pipeline
├── app/
│   └── streamlit_app.py     # Streamlit web dashboard
├── checkpoints/             # Saved model weights (.pth)
├── data/                    # Dataset directories
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Models
```bash
python main.py --mode info
```

### 3. Prepare Dataset
Place your data in the following structure:
```
data/splits/
├── train/
│   ├── safe/         (images or videos)
│   └── fall/         (images or videos)
├── val/
│   ├── safe/
│   └── fall/
└── test/
    ├── safe/
    └── fall/
```

For the fusion model, use:
```
data/splits/train/
├── images/
│   ├── safe/
│   └── hazardous/
└── videos/
    ├── safe/
    └── fall/
```

### 4. Train Models
```bash
# Train individually
python main.py --mode train_image
python main.py --mode train_video
python main.py --mode train_fusion

# Or train all in sequence
python main.py --mode train_all
```

### 5. Run Prediction
```bash
python main.py --mode predict --image path/to/floor.jpg --video path/to/activity.mp4
```

### 6. Launch Streamlit App
```bash
python main.py --mode app
# or directly:
streamlit run app/streamlit_app.py
```

---

## 🎯 Output Example

```
Risk Score: 0.82 → HIGH RISK. Floor slippery and motion unstable.
```

---

## 🧠 Model Details

| Model | Backbone | Output | Parameters |
|-------|----------|--------|------------|
| Image | EfficientNet-B0 (pretrained) | 512-d features | ~4.7M |
| Video | ResNet-18 + Bi-LSTM | 512-d features | ~12.5M |
| Fusion | FC Network | Risk score (0..1) | ~17.5M total |

### Training Strategy
1. **Phase 1**: Train image model with frozen EfficientNet backbone
2. **Phase 2**: Train video model with frozen ResNet backbone
3. **Phase 3**: Train fusion FC layers with frozen sub-models
4. **Phase 4** (optional): End-to-end fine-tuning with unfrozen backbones

### Loss & Optimizer
- **Loss**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Early Stopping**: Patience = 10

---

## 📊 Risk Thresholds

| Score | Status | Description |
|-------|--------|-------------|
| 0.0 — 0.3 | ✅ Safe | No significant risk detected |
| 0.3 — 0.6 | ⚠️ Warning | Moderate hazards, exercise caution |
| 0.6 — 1.0 | 🚨 High Risk | Immediate attention required |

---

## 🛠️ Tech Stack

- **PyTorch** — Deep learning framework
- **EfficientNet-B0** — Image feature extraction
- **ResNet-18 + LSTM** — Video temporal modeling
- **Streamlit** — Web dashboard
- **OpenCV** — Video processing

---

## 📝 License

This project is for educational and research purposes.
