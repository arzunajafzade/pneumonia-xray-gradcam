# 🫁 Pneumonia Detection from Chest X-Rays — with Grad-CAM Explainability

A deep learning system that detects pneumonia from chest X-ray images using **transfer learning** (ResNet50 / EfficientNetB0) and explains *why* it made each prediction using **Grad-CAM**.

**🔗 Live Demo:** [pneumonia-xray-gradcam.streamlit.app](https://pneumonia-xray-gradcam-69gph29zfb4bx2vplvftma.streamlit.app/)



---

## ✨ Overview

Deep learning models often act as "black boxes" — they output a prediction with no indication of *why*. This is especially risky in medical imaging, where a clinician needs to trust and verify a model's reasoning before acting on it.

This project addresses that gap:

- **Classifies** chest X-rays as `NORMAL` or `PNEUMONIA`
- **Explains** each prediction with a Grad-CAM heatmap, highlighting the exact lung regions that drove the model's decision
- **Deployed** as an interactive Streamlit web app — upload an X-ray and get an instant prediction + visual explanation

> ⚠️ **Disclaimer:** This is an educational/portfolio project, not a certified medical device. It has not been clinically validated and should never be used for real diagnostic decisions.

---

## 🧠 How It Works

| Stage | Approach |
|---|---|
| **Backbone** | ResNet50 or EfficientNetB0, pretrained on ImageNet |
| **Training** | Two-stage transfer learning — (1) train classification head with frozen backbone, (2) fine-tune top backbone layers at a low learning rate |
| **Class imbalance** | Computed `class_weight` to prevent bias toward the majority class |
| **Explainability** | Grad-CAM on the last convolutional layer, overlaid on the original X-ray |

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| AUC-ROC | 0.93 |
| Precision | 0.85 |
| Recall (Sensitivity) | 0.91 |

*Recall was prioritized during model selection, since missing an actual pneumonia case (false negative) is the costlier error in a clinical context.*

---

## 🚀 Try It Live

No installation needed — just open the demo and upload a chest X-ray (or use one of the built-in sample images):

👉 **[pneumonia-xray-gradcam.streamlit.app](https://pneumonia-xray-gradcam-69gph29zfb4bx2vplvftma.streamlit.app/)**

---

## 💻 Run It Locally

### 1. Clone the repository

```bash
git clone https://github.com/arzunajafzade/pneumonia-xray-gradcam.git
cd pneumonia-xray-gradcam
```

### 2. Install dependencies

> Requires **Python 3.10–3.12** (TensorFlow does not yet support 3.13+).

```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model

The trained model is already included in `models/best_model.keras`, so this step is only needed if you want to retrain from scratch.

Download the dataset from Kaggle: [**Chest X-Ray Images (Pneumonia)**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Unzip it into the following structure:

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Point the training script to it and run:

```bash
export PNEUMONIA_DATA_DIR=/full/path/to/chest_xray   # Windows: set PNEUMONIA_DATA_DIR=...
python train.py
```

### 4. Run predictions from the command line

```bash
python predict.py --image path/to/xray.jpeg --output result.png
```

### 5. Run the web app locally

```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
├── streamlit_app.py       # Interactive web app (Streamlit)
├── config.py               # Hyperparameters and paths
├── data_pipeline.py         # Data loading, augmentation, class balancing
├── model.py                  # Transfer learning model definition
├── train.py                   # Two-stage training script
├── gradcam.py                  # Grad-CAM implementation
├── predict.py                   # CLI inference + explanation
├── models/
│   └── best_model.keras          # Trained model weights
├── sample_images/                  # A few demo X-rays for the web app
└── requirements.txt
```

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model training and inference
- **ResNet50 / EfficientNetB0** — pretrained backbones (transfer learning)
- **Grad-CAM** — model explainability
- **Streamlit** — web app deployment
- **Python 3.12**

---

## 🔮 Possible Extensions

- Multimodal fusion: combine the X-ray with patient metadata (age, symptoms)
- K-fold cross-validation for more robust evaluation
- Model quantization for mobile/edge deployment

---

## 📄 License

This project is intended for educational purposes. Dataset credit: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney (Kaggle).
